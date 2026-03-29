"""
Whisper Transcription Service
Handles speech-to-text conversion with VAD and enhancement
"""

import modal
import numpy as np
import io
import base64
import time
import re
import torch
from typing import Optional, List, Tuple

app = modal.App("whisper-service", create_if_missing=True)

# Volume for models
models_volume = modal.Volume.from_name("aac-models", create_if_missing=True)
MODELS_PATH = "/mnt/aac-models"
WHISPER_PATH = "/mnt/aac-models/whisper"

# Image with dependencies
whisper_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.1.0",
        "numpy==1.24.3",
        "scipy==1.11.4",
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "webrtcvad==2.0.10",
        "transformers==4.35.0",
        "accelerate==0.24.1",
        "peft==0.7.0",
    )
)

# Secret for HuggingFace
hf_secret = modal.Secret.from_name("huggingface", optional=True)


def enhance_audio(pcm_int16: np.ndarray) -> np.ndarray:
    """Pre-emphasis + normalize for better Whisper accuracy"""
    audio = pcm_int16.astype(np.float32) / 32767.0
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    peak = np.abs(audio).max()
    if peak > 0.01:
        audio = audio / peak * 0.95
    return audio


def vad_segments(pcm_bytes: bytes, sr: int = 16000) -> List[bytes]:
    """Split PCM into voiced segments using WebRTC VAD"""
    import webrtcvad
    
    vad = webrtcvad.Vad(1)  # mode 1 = gentle
    frame_ms = 30
    bpf = int(sr * frame_ms / 1000) * 2
    sil_thr = 900 // frame_ms
    min_fr = (300 * sr * 2 // 1000) // bpf

    buf = bytearray(pcm_bytes)
    seg = []
    segs = []
    sil = 0
    speaking = False
    
    while len(buf) >= bpf:
        frame = bytes(buf[:bpf])
        buf = buf[bpf:]
        
        if vad.is_speech(frame, sr):
            seg.append(frame)
            sil = 0
            speaking = True
        elif speaking:
            sil += 1
            if sil >= sil_thr:
                if len(seg) >= min_fr:
                    pad = bytes(len(seg[0]))
                    segs.append(b"".join([pad] * 7 + seg + [pad] * 7))
                seg = []
                sil = 0
                speaking = False
    
    if len(seg) >= min_fr:
        pad = bytes(len(seg[0]))
        segs.append(b"".join([pad] * 7 + seg + [pad] * 7))
    
    return segs


def clean_profanity(text: str) -> str:
    """Replace profanity with [inaudible]"""
    profanity_pattern = re.compile(
        r'\b(fuck|fucking|fucked|fucker|fucks|shit|shits|shitty|bitch|bitches|'
        r'asshole|assholes|dumbass|jackass|bastard|bastards|cunt|cunts|'
        r'goddamn|piss|pissed|cock|dick|prick|whore|slut|nigger|nigga)\b',
        re.IGNORECASE
    )
    return profanity_pattern.sub('[inaudible]', text)


@app.cls(
    gpu="any",
    image=whisper_image,
    scaledown_window=300,
    secrets=[hf_secret],
    volumes={MODELS_PATH: models_volume},
)
class WhisperService:
    """Whisper transcription service with VAD"""
    
    @modal.enter()
    def load(self):
        """Load Whisper model from HuggingFace or volume"""
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try loading from volume first
        import os
        model_loaded = False
        
        if os.path.exists(WHISPER_PATH) and os.listdir(WHISPER_PATH):
            try:
                print(f"Loading Whisper from volume: {WHISPER_PATH}")
                self.processor = WhisperProcessor.from_pretrained(WHISPER_PATH)
                self.model = WhisperForConditionalGeneration.from_pretrained(WHISPER_PATH)
                self.model = self.model.to(self.device)
                self.model.eval()
                model_loaded = True
                print("Whisper loaded from volume")
            except Exception as e:
                print(f"Failed to load from volume: {e}")
        
        if not model_loaded:
            # Load from HuggingFace
            model_name = "openai/whisper-base"  # or use fine-tuned model
            print(f"Loading Whisper from HuggingFace: {model_name}")
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Whisper loaded from HuggingFace")
    
    @modal.method()
    def transcribe(self, audio_b64: str) -> dict:
        """
        Transcribe base64-encoded PCM audio
        
        Args:
            audio_b64: Base64 encoded 16kHz PCM int16 audio
            
        Returns:
            dict with transcript, confidence, processing_ms
        """
        start_time = time.time()
        
        if not audio_b64:
            return {"transcript": "", "confidence": 0, "processing_ms": 0}
        
        pcm_bytes = base64.b64decode(audio_b64)
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # Skip silent audio
        if np.abs(arr).max() < 300:
            return {"transcript": "", "confidence": 0, "processing_ms": 0}
        
        # Get voiced segments
        segs = vad_segments(pcm_bytes)
        if not segs:
            return {"transcript": "", "confidence": 0, "processing_ms": 0}
        
        # Use longest segment
        seg = max(segs, key=len)
        audio = enhance_audio(np.frombuffer(seg, dtype=np.int16))
        
        # Process with Whisper
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.get("attention_mask", torch.ones(input_features.shape[:2], dtype=torch.long)).to(self.device)
        
        gen_kwargs = {
            "attention_mask": attention_mask,
            "language": "en",
            "task": "transcribe",
            "num_beams": 5,
            "no_repeat_ngram_size": 5,
            "repetition_penalty": 1.3,
            "temperature": 0.0,
            "max_new_tokens": 225,
        }
        
        # Optional prompt
        try:
            prompt_ids = self.processor.get_prompt_ids(
                "Professional customer service call. Mobile money PIN reset.",
                return_tensors="pt"
            ).to(self.device)
            gen_kwargs["prompt_ids"] = prompt_ids
        except Exception:
            pass
        
        with torch.no_grad():
            ids = self.model.generate(input_features, **gen_kwargs)
        
        transcript = self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        
        # Remove repeated sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', transcript) if s.strip()]
        seen = set()
        deduped = []
        for s in sentences:
            key = s.lower()[:40]
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        
        transcript = ". ".join(deduped).strip()
        transcript = clean_profanity(transcript)
        
        # Remove disfluency markers
        transcript = re.sub(r"\[(REP|PROLONG|PARTIAL|FILLER|BLOCK)\]\s*", "", transcript)
        
        processing_ms = int((time.time() - start_time) * 1000)
        
        return {
            "transcript": transcript.strip(),
            "confidence": 0.85,  # Placeholder confidence
            "processing_ms": processing_ms
        }
