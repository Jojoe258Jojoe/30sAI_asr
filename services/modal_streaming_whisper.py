# modal_streaming_whisper_severity.py
#
# Real-time streaming Whisper service with severity-based model routing.
#
# Architecture:
#   Audio → SeverityClassifier (wav2vec2 → MLP) → mild | moderate | severe
#         → matching fine-tuned Whisper LoRA model → streaming transcription
#
# Model loading strategy (volume cache first, HuggingFace fallback):
#   Cold start #1 — downloads classifier + 3 Whisper models from HuggingFace Hub,
#                   saves each to its own subdirectory on the Modal volume.
#   Cold start #2+ — loads directly from the volume cache (no network call).
#   Volume layout:
#     /models/classifier/          ← wav2vec2 severity classifier
#     /models/whisper_mild/        ← fine-tuned Whisper (mild)
#     /models/whisper_moderate/    ← fine-tuned Whisper (moderate)
#     /models/whisper_severe/      ← fine-tuned Whisper (severe)
#
# Streaming modes:
#   WebSocket /ws         — token-level streaming via TextIteratorStreamer
#   POST /transcribe_file — chunked HF pipeline (arbitrary-length audio, NDJSON)
#
# Deploy:  modal deploy modal_streaming_whisper_severity.py
# Secrets: modal secret create huggingface-secret HF_TOKEN=hf_...

import modal
from pathlib import Path
import numpy as np
import asyncio
import json
import re
import time
from threading import Thread
from typing import Optional, Any
import io

try:
    from fastapi import UploadFile, File, Query
except ModuleNotFoundError:
    UploadFile = Any

    def File(*args, **kwargs):
        return None

    def Query(*args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODAL_APP_NAME = "streaming_whisper_severity"

# ── Severity classifier (wav2vec2-based) ─────────────────────────────────────
CLASSIFIER_REPO   = "jojo007unfi/whisper-severity-classifier"
WAV2VEC_BACKBONE  = "facebook/wav2vec2-base"
LABELS            = ["mild", "moderate", "severe"]
CLIP_SECONDS      = 8          # classifier uses only first 8 s of audio

# ── Per-severity fine-tuned Whisper repos (LoRA PEFT checkpoints) ─────────────
# Update these to match your actual HuggingFace repo IDs
WHISPER_REPOS = {
    "mild":     "jojo007unfi/whisper-mild",
    "moderate": "jojo007unfi/whisper-moderate",
    "severe":   "jojo007unfi/whisper-severe",
}
WHISPER_BASE_MODEL = "openai/whisper-large-v3"

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16_000
LANGUAGE       = "en"
TASK           = "transcribe"
MAX_NEW_TOKENS = 225

# ── Streaming thresholds ─────────────────────────────────────────────────────
MIN_CHUNK_SEC       = 1.0    # don't process fragments under 1 s
MAX_CHUNK_SEC       = 6.0
BUFFER_TRIMMING_SEC = 20.0

# ── Silence / VAD detection ───────────────────────────────────────────────────
SILENCE_RMS_THRESHOLD = 0.04   # tail-silence gate  (raised from 0.012)
SPEECH_RMS_THRESHOLD  = 0.02   # whole-buffer VAD gate — skip Whisper if below
SILENCE_TAIL_SEC      = 0.4    # how much tail to inspect for silence

# ── WebSocket behaviour ──────────────────────────────────────────────────────
PARTIAL_THROTTLE_SEC = 0.10
HEARTBEAT_SEC        = 4.0

# ── Beam sizes ───────────────────────────────────────────────────────────────
BEAM_SIZE_STREAM = 1   # greedy — required for TextIteratorStreamer
BEAM_SIZE_FINAL  = 1   # also greedy — beam search over-generates for impaired speech

# ── Whisper generation constraints ───────────────────────────────────────────
# Tighter settings reduce confabulation on dysarthric/unclear audio
NO_REPEAT_NGRAM   = 5      # block 5-gram repeats (was 3)
REPETITION_PENALTY = 1.8   # stronger repetition penalty (was 1.3)
LOG_PROB_THRESHOLD = -1.0  # drop tokens Whisper is uncertain about
COMPRESSION_RATIO  = 1.35  # reject outputs that are too repetitive

# ── Modal infra ──────────────────────────────────────────────────────────────
GPU          = "A10G"    # 24 GB VRAM — fits large-v3 in float16 + classifier
SCALEDOWN    = 60 * 5
TASK_TIMEOUT = 60 * 30

# ── Volume cache — one subdirectory per model ────────────────────────────────
VOLUME_MOUNT = Path("/models")
VOLUME_CACHE = {
    "classifier": VOLUME_MOUNT / "classifier",
    "mild":       VOLUME_MOUNT / "whisper_mild",
    "moderate":   VOLUME_MOUNT / "whisper_moderate",
    "severe":     VOLUME_MOUNT / "whisper_severe",
}

CORS_ORIGINS = [
    "https://30sai.netlify.app",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]


# ---------------------------------------------------------------------------
# Profanity filter
# ---------------------------------------------------------------------------

import os as _os

_raw_profanity  = _os.environ.get("PROFANITY_LIST_JSON")
_PROFANITY_LIST = json.loads(_raw_profanity) if _raw_profanity else []
_PROF_RE        = (
    re.compile(r"\b(" + "|".join(re.escape(w) for w in _PROFANITY_LIST) + r")\b",
               re.IGNORECASE)
    if _PROFANITY_LIST else None
)

def _clean(text: str) -> str:
    if _PROF_RE is None:
        return text
    return _PROF_RE.sub("[inaudible]", text)


# ---------------------------------------------------------------------------
# Severity Classifier — architecture must match training exactly
# ---------------------------------------------------------------------------

def _build_classifier_arch(num_labels: int = 3, hidden_dim: int = 256, dropout: float = 0.3):
    """Rebuilds the SeverityClassifier nn.Module from the training notebook."""
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    class SeverityClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Wav2Vec2Model.from_pretrained(WAV2VEC_BACKBONE)
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.classifier = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_labels),
            )

        def forward(self, input_values):
            hidden = self.encoder(input_values).last_hidden_state  # (B, T, 768)
            pooled = hidden.mean(dim=1)                            # (B, 768)
            return self.classifier(pooled)                         # (B, num_labels)

    return SeverityClassifier()


def load_severity_classifier(hf_token: str, device: str, volume=None):
    """
    Loads the wav2vec2 severity classifier.
    Priority:
      1. VOLUME_CACHE['classifier'] — saved on a previous cold start (fastest)
      2. HuggingFace Hub            — download, save to volume for next time
    Returns (model, feature_extractor, config).
    """
    import torch, shutil
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor

    cache          = VOLUME_CACHE["classifier"]
    cached_weights = cache / "best_classifier.pt"
    cached_config  = cache / "config.json"

    if cached_weights.exists() and cached_config.exists():
        print(f"[classifier] Loading from volume cache: {cache}")
        weights_path = str(cached_weights)
        config_path  = str(cached_config)
        source = "volume_cache"
    else:
        print(f"[classifier] Cache miss — downloading from {CLASSIFIER_REPO} …")
        weights_path = hf_hub_download(CLASSIFIER_REPO, "best_classifier.pt", token=hf_token)
        config_path  = hf_hub_download(CLASSIFIER_REPO, "config.json",        token=hf_token)
        source = "huggingface_hub"

    with open(config_path) as f:
        cfg = json.load(f)

    clf = _build_classifier_arch(num_labels=cfg["num_labels"])
    clf.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    clf.to(device).eval()

    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_BACKBONE)

    if source == "huggingface_hub":
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_path, cached_weights)
        shutil.copy2(config_path,  cached_config)
        print(f"[classifier] Saved to volume cache: {cache}")
        if volume:
            volume.commit()

    print(f"[classifier] Ready  source={source}  "
          f"labels={cfg.get('id2label')}  best_val_acc={cfg.get('best_val_acc')}%")
    return clf, feat_ext, cfg


def load_whisper_model(severity: str, hf_token: str, device: str, volume=None):
    """
    Loads a fine-tuned Whisper model for the given severity level.
    Priority:
      1. VOLUME_CACHE[severity] — merged full checkpoint saved on a previous cold start
      2. HuggingFace Hub        — download LoRA adapter, merge into base, save to volume
    Returns (model, processor).
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    repo  = WHISPER_REPOS[severity]
    cache = VOLUME_CACHE[severity]

    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE_MODEL, token=hf_token, language=LANGUAGE, task=TASK
    )

    if _is_valid_hf_checkpoint(cache):
        print(f"[whisper/{severity}] Loading from volume cache: {cache}")
        model = WhisperForConditionalGeneration.from_pretrained(
            str(cache), torch_dtype=torch.float16, device_map="auto",
        )
        source = "volume_cache"
    else:
        print(f"[whisper/{severity}] Cache miss — downloading from {repo} …")
        try:
            from peft import PeftModel
            base  = WhisperForConditionalGeneration.from_pretrained(
                WHISPER_BASE_MODEL, token=hf_token,
                torch_dtype=torch.float16, device_map="auto",
            )
            model = PeftModel.from_pretrained(base, repo, token=hf_token)
            model = model.merge_and_unload()   # fuse LoRA weights — faster inference
            print(f"[whisper/{severity}] Loaded as LoRA PEFT + merged")
        except Exception as peft_err:
            print(f"[whisper/{severity}] PEFT load failed ({peft_err}), "
                  f"falling back to full checkpoint …")
            model = WhisperForConditionalGeneration.from_pretrained(
                repo, token=hf_token,
                torch_dtype=torch.float16, device_map="auto",
            )
        source = "huggingface_hub"

        # Persist merged weights so subsequent cold starts skip the download
        cache.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(cache), safe_serialization=True)
        processor.save_pretrained(str(cache))
        print(f"[whisper/{severity}] Saved merged checkpoint to volume cache: {cache}")
        if volume:
            volume.commit()

    # Configure generation defaults
    model.config.forced_decoder_ids              = None
    model.config.suppress_tokens                 = []
    model.generation_config.forced_decoder_ids   = None
    model.generation_config.task                 = TASK
    model.generation_config.language             = LANGUAGE
    model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM
    model.generation_config.repetition_penalty   = REPETITION_PENALTY

    print(f"[whisper/{severity}] Ready  source={source}")
    return model, processor


def _is_valid_hf_checkpoint(path: Path) -> bool:
    """True if path contains a saved HF Whisper checkpoint."""
    if not path.exists():
        return False
    has_config  = (path / "config.json").exists()
    has_weights = (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin").exists()
    )
    return has_config and has_weights


# ---------------------------------------------------------------------------
# Severity prediction — runs on each incoming audio clip
# ---------------------------------------------------------------------------

def predict_severity(
    audio_array: np.ndarray,
    classifier,
    feat_ext,
    device: str,
    confidence_threshold: float = 0.50,
) -> tuple[str, float]:
    """
    Predicts mild / moderate / severe from a raw float32 audio array.
    Falls back to 'moderate' when max confidence is below the threshold.
    Returns (severity_label, confidence).
    """
    import torch

    max_length = SAMPLE_RATE * CLIP_SECONDS
    clip = audio_array[:max_length].copy()
    if len(clip) < max_length:
        clip = np.pad(clip, (0, max_length - len(clip)))

    inputs = feat_ext(
        clip, sampling_rate=SAMPLE_RATE,
        return_tensors="pt", padding=True,
        max_length=max_length, truncation=True,
    ).input_values.to(device)

    with torch.no_grad():
        logits = classifier(inputs)
        probs  = torch.softmax(logits, dim=-1).cpu().squeeze()
        pred   = logits.argmax(1).item()

    severity   = LABELS[pred]
    confidence = float(probs[pred])

    if confidence < confidence_threshold:
        print(f"[classifier] Low confidence ({confidence:.1%}) → defaulting to 'moderate'")
        severity = "moderate"

    print(f"[classifier] → {severity}  ({confidence:.1%})  "
          f"scores={dict(zip(LABELS, [f'{p:.1%}' for p in probs.tolist()]))}")
    return severity, confidence


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

def _tail_is_silent(audio: np.ndarray) -> bool:
    tail_samples = int(SILENCE_TAIL_SEC * SAMPLE_RATE)
    if len(audio) < tail_samples:
        return False
    rms = float(np.sqrt(np.mean(audio[-tail_samples:].astype(np.float64) ** 2)))
    return rms < SILENCE_RMS_THRESHOLD


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

# Phrases Whisper commonly hallucinates on silence/noise
_HALLUCINATION_PATTERNS = re.compile(
    r'^[\s]*('
    r'(so[\s,]*){3,}'           # "so so so so..."
    r'|thank\s+you[\s.]*$'      # lone "thank you"
    r'|(thank\s+you[\s,]*){2,}' # repeated "thank you"
    r'|(yes[\s,]*){3,}'         # "yes yes yes..."
    r'|(okay[\s,]*){2,}'        # "okay okay..."
    r'|(hi[\s,]*){3,}'          # "hi hi hi..."
    r'|(oh[\s,]*){3,}'          # "oh oh oh..."
    r')[\s.!?]*$',
    re.IGNORECASE,
)

# Separate check for any single word repeated 5+ times
_REPEATED_WORD = re.compile(r'^[\s]*(\b\w+\b)[\s,]*(?:\1[\s,]*){4,}[\s.!?]*$', re.IGNORECASE)

_HALLUCINATION_WORDS = {
    # Whisper's most common noise hallucinations
    'crowing', 'crickets', 'chirping', 'meowing', 'barking',
    'applause', 'laughter', 'music', 'silence',
}

def _filter_hallucinations(text: str) -> str:
    """Returns empty string if text looks like a Whisper hallucination."""
    t = text.strip()
    if not t:
        return ''
    if _HALLUCINATION_PATTERNS.match(t) or _REPEATED_WORD.match(t):
        print(f"[vad] Hallucination filtered: {t[:60]!r}")
        return ''
    # Check if all content words are hallucination markers
    words = re.findall(r'\b\w+\b', t.lower())
    if words and all(w in _HALLUCINATION_WORDS for w in words):
        print(f"[vad] Hallucination word-list filtered: {t[:60]!r}")
        return ''
    return text


# Short filler words Whisper commonly fuses without a case boundary.
# Sorted longest-first so the prefix regex is greedy-safe.
_FILLER_WORDS = sorted(
    ['yeah', 'yes', 'okay', 'please', 'thank', 'sorry', 'right', 'sure',
     'great', 'hello', 'hey', 'the', 'and', 'but', 'so', 'um', 'uh', 'oh',
     'ah', 'no'],
    key=len, reverse=True,
)
# Pre-build a regex that greedily peels a known filler word from the LEFT of a
# lowercase token, used only when no camelCase or dot boundary is present.
_FILLER_SET = set(_FILLER_WORDS)
_FILLER_PREFIX_RE = re.compile(
    r'^(' + '|'.join(re.escape(w) for w in _FILLER_WORDS) + r')(.+)$',
    re.IGNORECASE,
)


def _preprocess_token(token: str) -> list[str]:
    """
    Split a whitespace-free fused token into its constituent sub-tokens BEFORE
    the same-pattern repetition collapser runs.  Three strategies, in order:

    1. Dot boundaries  -- "Nice.Nice.Nice.Thank" -> ["Nice.", "Nice.", "Nice.", "Thank"]
    2. camelCase       -- "soThank" -> ["so", "Thank"]
                         "sososoThank" -> ["sososo", "Thank"]
    3. Filler prefix   -- "soyeahyeahyeahyeahthe" -> ["so", "yeah", "yeah", "yeah", "yeah", "the"]
                         "sothank"               -> ["so", "thank"]
                         "theyesyesthe"          -> ["the", "yes", "yes", "the"]

    Each sub-token is recursively preprocessed so nested cases are handled.
    Returns [token] unchanged if no split is found.
    """
    # strategy 1: split after a dot that is followed by a letter
    dot_parts = re.split(r'(?<=\.)(?=[A-Za-z])', token)
    if len(dot_parts) > 1:
        result: list[str] = []
        for p in dot_parts:
            result.extend(_preprocess_token(p))
        return result

    # strategy 2: camelCase — insert a split at every lower->upper boundary
    camel_parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', token).split()
    if len(camel_parts) > 1:
        result = []
        for p in camel_parts:
            result.extend(_preprocess_token(p))
        return result

    # strategy 3: peel a known filler prefix off a lowercase-fused token.
    # But first check whether the WHOLE token is already a collapsed repetition
    # (e.g. "hellohellohellohel") — if so, return that result directly instead
    # of letting the filler-prefix splitter recursively leave a dangling tail.
    collapsed = _collapse_fused_token(token)
    if collapsed != token:
        return [collapsed]

    m = _FILLER_PREFIX_RE.match(token)
    if m:
        prefix, rest = m.group(1), m.group(2)
        # only split if the remainder starts with another filler or is long
        # enough to plausibly contain one — avoids splitting real words like "those"
        if _FILLER_PREFIX_RE.match(rest) or rest.lower() in _FILLER_SET or len(rest) >= 4:
            return [prefix] + _preprocess_token(rest)

    return [token]


def _collapse_fused_token(token: str) -> str:
    """
    Detects a single whitespace-free token that is actually the same syllable/word
    repeated back-to-back, including truncated endings (Whisper sometimes stops
    mid-repetition at a chunk boundary).

    Examples:
        "heyheyheyHey"                        -> "hey"
        "heyheyheyheyheyheyheyheyheyhey"      -> "hey"
        "hellohellohello...hel" (truncated)   -> "hello"
        "HellohelloHello"                     -> "Hello"
        "ordinary"                            -> "ordinary"  (unchanged)

    Strategy: for each candidate pattern length (2 to len//2):
      1. Full match  -- entire token is (pattern)+
      2. Partial match -- at least 2 full copies of pattern, then a strict
         non-empty prefix of pattern as a tail (the truncated last syllable).
    Returns on the first pattern that fits, preserving original casing.
    """
    bare = token.lower()
    n    = len(bare)
    for pat_len in range(2, n // 2 + 1):
        pat = bare[:pat_len]
        # full repetition
        if re.fullmatch(f'(?:{re.escape(pat)})+', bare, re.IGNORECASE):
            return token[:pat_len]
        # truncated tail: >= 2 full copies + a strict prefix of the pattern
        full_reps = n // pat_len
        if full_reps >= 2:
            full_len  = pat_len * full_reps
            remainder = bare[full_len:]
            if remainder and len(remainder) < pat_len and pat.startswith(remainder):
                return token[:pat_len]
    return token


def _deduplicate_words(text: str) -> str:
    """
    Reduces word-level repetition that slips past the model's n-gram penalty.

    Rule 0 — Fused-token collapse (pre-pass, per whitespace token).
              "heyheyheyHey" → "hey"   (Whisper glued the tokens without spaces)

    Rule 1 — No consecutive duplicates (any run length collapses to one).
              "hello hello hello" → "hello"
              Works across punctuation: "hey, hey!" → "hey,"

    Rule 2 — No word appears more than twice in a single sentence.
              Sentence boundaries: . ! ?
              Case-insensitive; punctuation stripped for the word key.

    Preserves original casing and punctuation for tokens that survive.
    """
    if not text:
        return text

    # ── Rule 0: split fused tokens, then collapse internally-repeated ones ─────
    expanded: list[str] = []
    for raw in text.split():
        for sub in _preprocess_token(raw):
            expanded.append(_collapse_fused_token(sub))
    text = ' '.join(expanded)

    # ── Rule 1a: collapse consecutive duplicate words (handles , ; trailing punct) ─
    text = re.sub(
        r'\b(\w+)[,;]?(\s+\1[,;]?)+\b',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )
    # ── Rule 1b: collapse consecutive duplicate words with trailing dot ──────
    #   "Nice. Nice. Nice." → "Nice."   "Okay. Okay." → "Okay."
    text = re.sub(
        r'\b(\w+)\.(\s+\1\.)+',
        r'\1.',
        text,
        flags=re.IGNORECASE,
    )

    # ── Rule 2: cap each word to 2 occurrences per sentence ──────────────────
    sentence_parts = re.split(r'(?<=[.!?])\s+', text)
    cleaned_parts: list[str] = []

    for sentence in sentence_parts:
        tokens = sentence.split()
        seen: dict[str, int] = {}
        kept: list[str] = []
        for token in tokens:
            key = token.lower().strip('.,!?;:\'"')
            if not key:
                kept.append(token)
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= 2:
                kept.append(token)
        cleaned_parts.append(' '.join(kept))

    return ' '.join(cleaned_parts)


# ---------------------------------------------------------------------------
# StreamingASRProcessor  (severity-aware)
# ---------------------------------------------------------------------------

class StreamingASRProcessor:
    """
    Rolling audio buffer that routes each chunk to the correct Whisper model
    based on severity predicted by the classifier.

    On the first call the classifier runs once per WebSocket session to pick
    a model.  That same model is reused for the entire session (a caller's
    severity does not change mid-call).  Set `repredict_every_n` to a positive
    integer if you want the classifier to re-run periodically.
    """

    def __init__(self, whisper_models: dict, classifier, feat_ext, device: str,
                 min_chunk_size: float = MIN_CHUNK_SEC,
                 buffer_trimming_sec: float = BUFFER_TRIMMING_SEC,
                 repredict_every_n: int = 0):
        self.whisper_models      = whisper_models   # dict: severity → (model, processor)
        self.classifier          = classifier
        self.feat_ext            = feat_ext
        self.device              = device
        self.min_chunk_size      = min_chunk_size
        self.buffer_trimming_sec = buffer_trimming_sec
        self.repredict_every_n   = repredict_every_n

        self.audio_buffer        = np.array([], dtype=np.float32)
        self.committed_text      = ""
        self.buffer_time_offset  = 0.0

        # Severity routing state
        self._severity           = None   # set on first chunk
        self._confidence         = None
        self._chunks_processed   = 0

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def severity(self) -> Optional[str]:
        return self._severity

    @property
    def model(self):
        sev = self._severity or "moderate"
        return self.whisper_models[sev][0]

    @property
    def processor(self):
        sev = self._severity or "moderate"
        return self.whisper_models[sev][1]

    # ── Audio ingestion ──────────────────────────────────────────────────────

    def insert_audio_chunk(self, audio: np.ndarray):
        self.audio_buffer = np.append(self.audio_buffer, audio)

        # Classify on first chunk, then optionally re-classify every N chunks
        should_classify = (
            self._severity is None
            or (self.repredict_every_n > 0
                and self._chunks_processed % self.repredict_every_n == 0)
        )
        if should_classify and len(self.audio_buffer) >= SAMPLE_RATE * 1.0:
            self._severity, self._confidence = predict_severity(
                self.audio_buffer, self.classifier, self.feat_ext, self.device
            )
        self._chunks_processed += 1

    def _should_process(self) -> bool:
        duration = len(self.audio_buffer) / SAMPLE_RATE
        if duration < self.min_chunk_size:
            return False
        if duration >= MAX_CHUNK_SEC:
            return True
        return _tail_is_silent(self.audio_buffer)

    # ── Core: token-streaming transcription ─────────────────────────────────

    def process_iter_streaming(self, token_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        """
        Runs model.generate() with TextIteratorStreamer in a background thread,
        forwards every token into token_queue via the async event loop.
        Puts sentinel None when generation is complete.
        Returns the full text of this chunk or None.
        """
        if not self._should_process():
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
            return None

        # ── Whole-buffer VAD gate ────────────────────────────────────────────
        # If the entire buffer is below the speech threshold, skip Whisper —
        # sending silence/noise produces hallucinated "So", "Thank you" etc.
        buf_rms = float(np.sqrt(np.mean(self.audio_buffer.astype(np.float64) ** 2)))
        if buf_rms < SPEECH_RMS_THRESHOLD:
            self._trim_buffer()
            loop.call_soon_threadsafe(token_queue.put_nowait, None)
            return None

        import torch
        from transformers import TextIteratorStreamer

        model     = self.model
        processor = self.processor
        audio     = self.audio_buffer.copy()
        device    = next(model.parameters()).device

        inputs = processor(
            audio, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device, dtype=torch.float16)
        attention_mask  = inputs.attention_mask.to(device)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
            decode_kwargs={"clean_up_tokenization_spaces": True},
        )

        gen_kwargs = dict(
            input_features              = input_features,
            attention_mask              = attention_mask,
            streamer                    = streamer,
            language                    = LANGUAGE,
            task                        = TASK,
            max_new_tokens              = MAX_NEW_TOKENS,
            num_beams                   = BEAM_SIZE_STREAM,  # must be 1 for TextIteratorStreamer
            do_sample                   = False,
            temperature                 = 0.0,
            no_repeat_ngram_size        = NO_REPEAT_NGRAM,
            repetition_penalty          = REPETITION_PENALTY,
            compression_ratio_threshold = COMPRESSION_RATIO,
            condition_on_prev_tokens    = False,
        )

        gen_thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        gen_thread.start()

        chunk_text = ""
        for token in streamer:
            cleaned    = _clean(token)
            chunk_text += cleaned
            loop.call_soon_threadsafe(token_queue.put_nowait, cleaned)

        gen_thread.join()
        loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

        chunk_text = _deduplicate_words(_filter_hallucinations(chunk_text))
        self.committed_text += chunk_text
        self._trim_buffer()
        return chunk_text.strip() or None

    # ── Final transcription (beam search, no streamer) ───────────────────────

    def finish(self) -> Optional[dict]:
        if len(self.audio_buffer) == 0:
            return None

        # VAD gate — don't run Whisper on silent tail audio
        buf_rms = float(np.sqrt(np.mean(self.audio_buffer.astype(np.float64) ** 2)))
        if buf_rms < SPEECH_RMS_THRESHOLD:
            self.audio_buffer = np.array([], dtype=np.float32)
            return None

        import torch

        model     = self.model
        processor = self.processor
        device    = next(model.parameters()).device

        saved_min, self.min_chunk_size = self.min_chunk_size, 0.0

        inputs = processor(
            self.audio_buffer, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device, dtype=torch.float16)
        attention_mask  = inputs.attention_mask.to(device)

        with torch.no_grad():
            ids = model.generate(
                input_features,
                attention_mask              = attention_mask,
                language                    = LANGUAGE,
                task                        = TASK,
                num_beams                   = BEAM_SIZE_FINAL,
                max_new_tokens              = MAX_NEW_TOKENS,
                temperature                 = 0.0,
                no_repeat_ngram_size        = NO_REPEAT_NGRAM,
                repetition_penalty          = REPETITION_PENALTY,
                compression_ratio_threshold = COMPRESSION_RATIO,
                condition_on_prev_tokens    = False,
            )

        text     = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        text     = _deduplicate_words(_filter_hallucinations(_clean(text).strip()))
        duration = len(self.audio_buffer) / SAMPLE_RATE
        start    = self.buffer_time_offset

        self.min_chunk_size      = saved_min
        self.audio_buffer        = np.array([], dtype=np.float32)
        self.buffer_time_offset += duration

        if not text:
            return None
        return {
            "start":      start,
            "end":        start + duration,
            "text":       text,
            "severity":   self._severity,
            "confidence": round(self._confidence or 0.0, 3),
            "is_partial": False,
        }

    # ── Buffer management ────────────────────────────────────────────────────

    def _trim_buffer(self):
        duration = len(self.audio_buffer) / SAMPLE_RATE
        if duration > self.buffer_trimming_sec:
            keep_samples            = int(self.buffer_trimming_sec * SAMPLE_RATE)
            self.buffer_time_offset += duration - self.buffer_trimming_sec
            self.audio_buffer       = self.audio_buffer[-keep_samples:]


# ---------------------------------------------------------------------------
# WebSocketConnection  (unchanged from original — works with new processor)
# ---------------------------------------------------------------------------

class WebSocketConnection:
    def __init__(self, websocket, processor: StreamingASRProcessor):
        self.websocket       = websocket
        self.processor       = processor
        self.last_sent_text  = ""
        self.last_partial_ts = 0.0
        self._closed         = False
        self._partial_buffer = ""

    async def _heartbeat(self):
        while not self._closed:
            await asyncio.sleep(HEARTBEAT_SEC)
            try:
                await self.websocket.send_text(
                    json.dumps({"type": "ping", "ts": time.time()})
                )
            except Exception:
                break

    async def _send_token(self, token: str):
        if not token.strip():
            return
        now = time.monotonic()
        if now - self.last_partial_ts < PARTIAL_THROTTLE_SEC:
            return
        self.last_partial_ts  = now
        self._partial_buffer += token
        await self.websocket.send_text(json.dumps({
            "type":       "partial",
            "token":      token,
            "text":       self._partial_buffer.strip(),
            "severity":   self.processor.severity,
            "is_partial": True,
        }))

    async def _send_final(self, result: Optional[dict]):
        if not result or not result.get("text", "").strip():
            return
        if result["text"] == self.last_sent_text:
            return
        self.last_sent_text  = result["text"]
        self._partial_buffer = ""
        await self.websocket.send_text(json.dumps(result))

    async def process_audio_stream(self):
        loop           = asyncio.get_running_loop()
        heartbeat_task = asyncio.create_task(self._heartbeat())

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.receive(), timeout=HEARTBEAT_SEC * 3
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    raise

                msg_type = message.get("type") if isinstance(message, dict) else None

                # ── Audio bytes ─────────────────────────────────────────────
                if msg_type == "websocket.receive" and "bytes" in message:
                    audio_bytes = message["bytes"]
                    if not audio_bytes:
                        continue

                    audio_array = (
                        np.frombuffer(audio_bytes, dtype=np.int16)
                        .astype(np.float32) / 32768.0
                    )
                    self.processor.insert_audio_chunk(audio_array)

                    if not self.processor._should_process():
                        continue

                    token_queue: asyncio.Queue = asyncio.Queue()
                    gen_future = loop.run_in_executor(
                        None,
                        self.processor.process_iter_streaming,
                        token_queue, loop,
                    )

                    while True:
                        try:
                            token = await asyncio.wait_for(token_queue.get(), timeout=30.0)
                        except asyncio.TimeoutError:
                            break
                        if token is None:
                            break
                        await self._send_token(token)

                    await gen_future

                # ── Control messages ────────────────────────────────────────
                elif msg_type == "websocket.receive" and "text" in message:
                    try:
                        control = json.loads(message["text"])
                    except Exception:
                        continue

                    action = control.get("action")

                    if action == "finalize":
                        final_result = await loop.run_in_executor(None, self.processor.finish)
                        await self._send_final(final_result)
                        await self.websocket.send_text(json.dumps({"type": "done"}))
                        break

                    elif action == "reset":
                        self.processor.__init__(
                            self.processor.whisper_models,
                            self.processor.classifier,
                            self.processor.feat_ext,
                            self.processor.device,
                            min_chunk_size      = self.processor.min_chunk_size,
                            buffer_trimming_sec = self.processor.buffer_trimming_sec,
                        )
                        self.last_sent_text  = ""
                        self.last_partial_ts = 0.0
                        self._partial_buffer = ""
                        await self.websocket.send_text(json.dumps({"type": "reset_ack"}))

                elif msg_type == "websocket.disconnect":
                    break

                elif isinstance(message, str):
                    try:
                        control = json.loads(message)
                        if control.get("action") == "finalize":
                            final_result = await loop.run_in_executor(None, self.processor.finish)
                            await self._send_final(final_result)
                            await self.websocket.send_text(json.dumps({"type": "done"}))
                            break
                    except Exception:
                        pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Error in streaming: {e}")
            try:
                await self.websocket.send_text(json.dumps({"error": str(e)}))
            except Exception:
                pass
        finally:
            self._closed = True
            heartbeat_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(heartbeat_task), timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass


# ---------------------------------------------------------------------------
# Modal image & app setup
# ---------------------------------------------------------------------------

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi[standard]",
        "numpy",
        "librosa",
        "torch==2.6.0",
        "transformers>=4.39.0",
        "accelerate>=0.30.0",
        "peft>=0.11.0",            # for LoRA Whisper checkpoints
        "soundfile",
        "websockets",
        "uvicorn",
        "huggingface_hub>=0.23.4",
        "scikit-learn",            # classifier dependency
    )
)

app       = modal.App(MODAL_APP_NAME)
volume    = modal.Volume.from_name("asr-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")  # HF_TOKEN set in Modal secrets


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

@app.cls(
    image          = cuda_image,
    gpu            = GPU,
    scaledown_window       = SCALEDOWN,
    enable_memory_snapshot = True,
    volumes        = {VOLUME_MOUNT: volume},
    secrets        = [hf_secret],
    max_containers = 10,
    timeout        = TASK_TIMEOUT,
)
class StreamingWhisperService:
    """
    Severity-routed streaming Whisper service.

    Cold start #1 (no volume cache):
      Downloads classifier + 3 Whisper models from HuggingFace Hub, saves each
      to the Modal volume so subsequent cold starts skip the download entirely.
      Whisper LoRA adapters are merged into the base model before saving, so
      the cached copy is a plain full checkpoint (fast to load, no PEFT needed).

    Cold start #2+ (volume cache hit):
      Loads all models directly from /models/* on the volume — no network calls.

    Per WebSocket session:
      - Classifier runs on the first 1+ second of audio to pick a Whisper model
      - That model handles the entire session (severity does not change mid-call)
      - Severity label + confidence are included in every response payload
    """

    @modal.enter()
    def enter(self):
        import torch
        import os

        # Reload volume so this container sees files written by earlier cold starts
        try:
            volume.reload()
        except Exception as e:
            print(f"[enter] Volume reload skipped: {e}")

        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not set. Run: modal secret create huggingface-secret HF_TOKEN=hf_..."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[enter] Device: {self.device}")

        # Load severity classifier (volume cache first, HF Hub fallback)
        self.classifier, self.feat_ext, self.clf_config = load_severity_classifier(
            hf_token, self.device, volume=volume
        )

        # Load all three Whisper models (volume cache first, HF Hub fallback)
        self.whisper_models = {}
        for severity in LABELS:
            model, processor = load_whisper_model(severity, hf_token, self.device, volume=volume)
            self.whisper_models[severity] = (model, processor)

        # Final commit ensures all newly cached models are persisted together
        try:
            volume.commit()
            print("[enter] Volume committed — all model caches are persistent.")
        except Exception as e:
            print(f"[enter] Volume commit skipped: {e}")

        print(f"[enter] All models ready — classifier + {list(self.whisper_models.keys())}")

    # ── File transcription ───────────────────────────────────────────────────

    @modal.fastapi_endpoint(docs=True, method="POST")
    async def transcribe_file(
        self,
        wav: Any               = File(..., description="WAV audio file (16kHz mono)"),
        language: Optional[str]= Query(None),
        word_timestamps: bool  = Query(False),
    ):
        """
        Classifies severity from the first 8 s of audio, then runs the
        matching Whisper model over the full file.
        Returns NDJSON segments, each including severity + confidence fields.
        """
        import librosa
        from fastapi.responses import StreamingResponse
        from transformers import pipeline as hf_pipeline

        wav_bytes            = await wav.read()
        audio_array, _       = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE)
        severity, confidence = predict_severity(
            audio_array, self.classifier, self.feat_ext, self.device
        )

        model, processor = self.whisper_models[severity]

        pipe = hf_pipeline(
            task              = "automatic-speech-recognition",
            model             = model,
            tokenizer         = processor.tokenizer,
            feature_extractor = processor.feature_extractor,
            chunk_length_s    = 30,
            stride_length_s   = (3, 3),
            generate_kwargs   = {
                "language":                    language or LANGUAGE,
                "task":                        TASK,
                "num_beams":                   BEAM_SIZE_FINAL,
                "temperature":                 0.0,
                "no_repeat_ngram_size":        NO_REPEAT_NGRAM,
                "repetition_penalty":          REPETITION_PENALTY,
                "compression_ratio_threshold": COMPRESSION_RATIO,
                "condition_on_prev_tokens":    False,
            },
            return_timestamps = "word" if word_timestamps else True,
        )

        def _segment_generator():
            result = pipe({"array": audio_array, "sampling_rate": SAMPLE_RATE})
            chunks = result.get("chunks") or []
            extra  = {"severity": severity, "confidence": round(confidence, 3)}
            if chunks:
                for chunk in chunks:
                    ts   = chunk.get("timestamp", (0.0, 0.0)) or (0.0, 0.0)
                    text = _deduplicate_words(_clean(chunk.get("text", "")).strip())
                    if not text:
                        continue
                    yield json.dumps({
                        "start": ts[0] if ts[0] is not None else 0.0,
                        "end":   ts[1] if ts[1] is not None else 0.0,
                        "text":  text,
                        **extra,
                    }) + "\n"
            else:
                text = _deduplicate_words(_clean(result.get("text", "")).strip())
                if text:
                    duration = len(audio_array) / SAMPLE_RATE
                    yield json.dumps({"start": 0.0, "end": duration, "text": text, **extra}) + "\n"

        return StreamingResponse(_segment_generator(), media_type="application/x-ndjson")

    # ── WebSocket streaming ──────────────────────────────────────────────────

    @modal.asgi_app()
    def streaming_endpoint(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware

        web_app = FastAPI(title="30sAI Severity-Routed Whisper", version="4.0.0")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins     = CORS_ORIGINS,
            allow_credentials = True,
            allow_methods     = ["*"],
            allow_headers     = ["*"],
            expose_headers    = ["*"],
        )

        @web_app.get("/health")
        async def health_check():
            return {
                "status":          "healthy",
                "classifier":      CLASSIFIER_REPO,
                "whisper_repos":   WHISPER_REPOS,
                "labels":          LABELS,
                "best_val_acc":    self.clf_config.get("best_val_acc"),
                "sample_rate":     SAMPLE_RATE,
                "beam_size_stream":BEAM_SIZE_STREAM,
                "beam_size_final": BEAM_SIZE_FINAL,
                "streaming_mode":  "TextIteratorStreamer (token-level)",
                "volume":          "asr-models",
                "volume_cache":    {k: str(v) for k, v in VOLUME_CACHE.items()},
                "version":         "4.0.0",
            }

        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket connection accepted")

            processor  = StreamingASRProcessor(
                whisper_models = self.whisper_models,
                classifier     = self.classifier,
                feat_ext       = self.feat_ext,
                device         = self.device,
            )
            connection = WebSocketConnection(websocket, processor)

            # Immediately inform the client which model will be used
            # (updated once severity is predicted from first audio chunk)
            session_task = asyncio.create_task(connection.process_audio_stream())

            try:
                await session_task
            except asyncio.CancelledError:
                session_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(session_task), timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                raise
            except WebSocketDisconnect:
                print("WebSocket disconnected by client.")
            except Exception as e:
                print(f"WebSocket handler error: {e}")
                try:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                except Exception:
                    pass
            finally:
                connection._closed = True
                task = asyncio.current_task()
                if task is None or not task.cancelled():
                    loop = asyncio.get_running_loop()
                    try:
                        final = await asyncio.wait_for(
                            loop.run_in_executor(None, processor.finish), timeout=10.0
                        )
                        if final:
                            final["is_partial"] = False
                            await websocket.send_text(json.dumps(final))
                    except Exception:
                        pass
                try:
                    await websocket.close()
                except Exception:
                    pass

        return web_app


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def test_streaming():
    print("=" * 60)
    print("Severity-Routed Streaming Whisper  v4.0.0")
    print(f"  Classifier   : {CLASSIFIER_REPO}")
    for sev, repo in WHISPER_REPOS.items():
        print(f"  Whisper/{sev:<8}: {repo}")
    print(f"  GPU          : {GPU}")
    print("=" * 60)
    base = f"https://{app.app_id}--streaming-whisper-severity-service"
    print(f"\nFile upload (NDJSON):")
    print(f'  curl -X POST "{base}-transcribe-file.modal.run" -F "wav=@audio.wav"')
    print(f"\nWebSocket:")
    print(f"  wss://{app.app_id}--streaming-whisper-severity-service-streaming-endpoint.modal.run/ws")
    print(f"\nHealth:")
    print(f"  {base}-streaming-endpoint.modal.run/health")
    print("\nCold start downloads: classifier + 3 Whisper models from HuggingFace Hub.")
    print("Cold start #1: downloads classifier + 3 Whisper models from HuggingFace Hub,")
    print("               saves merged checkpoints to the 'asr-models' volume.")
    print("Cold start #2+: loads everything from volume cache — no network calls.")
