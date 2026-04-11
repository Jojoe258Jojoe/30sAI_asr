# modal_streaming_whisper.py
# Deploys a real-time streaming Whisper service on Modal with WebSocket support
#
# To deploy:
#   modal deploy modal_streaming_whisper.py
#
# To test with WebSocket client:
#   python streaming_client.py

import modal
from pathlib import Path
import numpy as np
import asyncio
import json
import time
import logging
from typing import Optional, Dict, Any
import io
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import UploadFile, File, Query

MODAL_APP_NAME = "streaming_whisper"

SAMPLE_RATE = 16000
BEAM_SIZE = 5
MODEL_MOUNT_DIR = Path("/models")
MODEL_DOWNLOAD_DIR = Path("downloads")
MIN_CHUNK_SIZE = 1.0  # seconds
BUFFER_TRIMMING_SEC = 15

GPU = 'L4'
SCALEDOWN = 60 * 5  # 5 minutes for streaming service

HUGGINGFACE_REPO = "cdli/whisper-large-v3_finetuned_ugandan_english_nonstandard_speech_v1.0"
MODEL_ID = HUGGINGFACE_REPO


def _patch_mel_bins(model_path, n_mels=128):
    """
    Patch both config.json and preprocessor_config.json in the converted model
    directory to use the correct number of mel bins.

    faster-whisper derives the mel filterbank size from config.json (num_mel_bins),
    NOT from the feature_extractor attribute at runtime.  If config.json still says
    80, the spectrogram will be 80-bin regardless of any runtime patching, and the
    encoder (which was fine-tuned with 128 bins) will reject it.
    """
    import json as _json

    for cfg_name in ("config.json", "preprocessor_config.json"):
        cfg_path = model_path / cfg_name
        if not cfg_path.exists():
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)

        changed = False
        for key in ("num_mel_bins", "n_mels"):
            if key in cfg and cfg[key] != n_mels:
                print(f"Patching {cfg_name}: {key} {cfg[key]} -> {n_mels}")
                cfg[key] = n_mels
                changed = True

        if changed:
            with open(cfg_path, "w", encoding="utf-8") as f:
                _json.dump(cfg, f, ensure_ascii=False, indent=2)
        else:
            print(f"{cfg_name}: mel bins already correct ({n_mels})")


def maybe_download_and_convert_model(model_storage_dir, model_id):
    """Download and convert model to CTranslate2 format if not available locally."""
    import ctranslate2

    print(f"Checking for cached model in storage directory: {model_storage_dir}, model ID: {model_id}")

    subdir = model_id.replace("/", "_") + "_ct2"
    model_path = model_storage_dir / subdir
    model_bin_path = model_path / "model.bin"
    print(f"Looking for model at: {model_path}")

    if not model_path.exists() or not model_bin_path.exists():
        print(f"Model not found in cache or incomplete -- downloading from HuggingFace: {model_id}")
        model_path.mkdir(parents=True, exist_ok=True)

        converter = ctranslate2.converters.TransformersConverter(
            model_name_or_path=model_id,
        )
        print(f"Converting to CTranslate2 format (float16)...")
        converter.convert(str(model_path), quantization="float16", force=True)
        print(f"Model successfully converted and saved to {model_path}")
    else:
        print(f"Found cached model at {model_path}")

    # Always patch mel bin counts — even on a cached model the config may be wrong.
    _patch_mel_bins(model_path, n_mels=128)

    return str(model_path)


class HypothesisBuffer:
    """Manages partial transcriptions and handles word-level streaming."""

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new_words, offset):
        """
        Insert new words with timestamp offset.
        NOTE: offset is NOT pre-added here — words are stored with their raw
        model-relative timestamps. The offset is only applied in _format_output
        so it is never double-counted.
        """
        # Store words with raw timestamps (no offset baked in)
        self.new = [(a, b, t) for a, b, t in new_words
                    if a > self.last_commited_time - 0.1]

        # Remove duplicate n-grams at the seam to avoid repeating text
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for j in range(i):
                                self.new.pop(0)
                            break

    def flush(self):
        """
        Commit the longest stable prefix — words that appear identically in both
        the current transcription pass (self.new) and the previous one (self.buffer).

        FIX: comparison is now case-insensitive and strips punctuation so that minor
        Whisper variations (capitalisation, trailing comma) don't block commitment.
        """
        import re

        def _normalise(w):
            return re.sub(r"[^\w]", "", w).lower()

        commit = []
        while self.new:
            if len(self.buffer) == 0:
                break
            na, nb, nt = self.new[0]
            bt = self.buffer[0][2]
            if _normalise(nt) == _normalise(bt):
                # Use the text form from the newer pass (self.new) as it is fresher
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """Remove committed words whose end timestamp is before `time`."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Return the current unconfirmed buffer (partial words)."""
        return self.buffer


class StreamingASRProcessor:
    """Processes audio chunks in real-time and returns partial transcriptions."""

    def __init__(self, whisper_model, min_chunk_size=MIN_CHUNK_SIZE,
                 buffer_trimming_sec=BUFFER_TRIMMING_SEC):
        self.whisper_model = whisper_model
        self.min_chunk_size = min_chunk_size
        self.buffer_trimming_sec = buffer_trimming_sec

        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = 0.0
        self.commited = []

        self.beam_size = BEAM_SIZE
        self.vad_filter = True

    def insert_audio_chunk(self, audio: np.ndarray):
        """Append audio chunk to buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def get_prompt(self):
        """Generate prompt from recent context for better continuity."""
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1
        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        return " ".join(prompt[::-1])

    def _run_transcribe(self):
        """Run faster-whisper on the current audio buffer. Returns word list."""
        prompt = self.get_prompt()

        try:
            segments, _ = self.whisper_model.transcribe(
                self.audio_buffer,
                beam_size=self.beam_size,
                language=None,
                task="transcribe",
                initial_prompt=prompt if prompt else None,
                vad_filter=self.vad_filter,
                word_timestamps=True,
                condition_on_previous_text=False,
            )
        except (RuntimeError, ValueError) as e:
            if "Invalid input features shape" in str(e):
                print(f"Feature mismatch detected: {e}. Reinitialising FeatureExtractor with n_mels=128.")
                try:
                    fe = self.whisper_model.feature_extractor
                    FeatureExtractor = type(fe)
                    self.whisper_model.feature_extractor = FeatureExtractor(feature_size=128)
                    print(f"FeatureExtractor ({FeatureExtractor.__name__}) replaced. Retrying.")
                    segments, _ = self.whisper_model.transcribe(
                        self.audio_buffer,
                        beam_size=self.beam_size,
                        language=None,
                        task="transcribe",
                        initial_prompt=prompt if prompt else None,
                        vad_filter=self.vad_filter,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                    )
                except Exception as retry_e:
                    print(f"Retry transcription also failed: {retry_e}")
                    return []
            else:
                raise

        timestamped_words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    timestamped_words.append((word.start, word.end, word.word))
        return timestamped_words

    def process_iter(self):
        """
        Process current audio buffer and return new transcription.
        Returns a result dict, or None if there is nothing new to report.
        """
        if len(self.audio_buffer) / SAMPLE_RATE < self.min_chunk_size:
            return None

        timestamped_words = self._run_transcribe()

        # Pass raw (no-offset) timestamps to the buffer; offset is applied in
        # _format_output so it is added exactly once.
        self.transcript_buffer.insert(timestamped_words, self.buffer_time_offset)
        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)

        result = self._format_output(committed, is_partial=True)

        # FIX: if nothing was committed yet (first chunk, or words not stable),
        # fall back to sending the unconfirmed buffer as a clearly-marked partial
        # result so downstream services always receive progress.
        if result is None:
            partial_words = self.transcript_buffer.complete()
            if partial_words:
                result = self._format_output(partial_words, is_partial=True)

        if len(self.audio_buffer) / SAMPLE_RATE > self.buffer_trimming_sec:
            self._trim_buffer()

        return result

    def _format_output(self, words, is_partial=True):
        """
        Format timestamped words into a result dict.

        FIX: buffer_time_offset is applied here exactly once.  The words coming
        from HypothesisBuffer are stored with raw model-relative timestamps
        (offset NOT pre-baked), so we add it only at this final step.
        """
        if not words:
            return None

        start_time = words[0][0] + self.buffer_time_offset
        end_time   = words[-1][1] + self.buffer_time_offset
        text = " ".join(w[2] for w in words)

        return {
            'start': start_time,
            'end': end_time,
            'text': text,
            'is_partial': is_partial,
            'words': [
                (w[0] + self.buffer_time_offset,
                 w[1] + self.buffer_time_offset,
                 w[2]) for w in words
            ]
        }

    def _trim_buffer(self):
        """Trim audio buffer to prevent memory growth."""
        if self.commited:
            trim_time = self.commited[-1][1]
            cut_samples = int((trim_time - self.buffer_time_offset) * SAMPLE_RATE)
            if cut_samples > 0:
                self.audio_buffer = self.audio_buffer[cut_samples:]
                self.buffer_time_offset = trim_time
                self.transcript_buffer.pop_commited(trim_time)

    def finish(self):
        """
        Force-transcribe any remaining audio and return the final result.

        FIX 1: process_iter() is called with min_chunk_size=0 to transcribe
                whatever audio is still in the buffer before draining it.
        FIX 2: buffer length is captured BEFORE clearing so the offset update
                is always correct (previously the buffer was cleared first,
                making len(audio_buffer) always 0 in the offset calculation).
        """
        # Force-transcribe remaining audio regardless of minimum chunk size
        if len(self.audio_buffer) > 0:
            saved_min = self.min_chunk_size
            self.min_chunk_size = 0.0
            committed_result = self.process_iter()
            self.min_chunk_size = saved_min
        else:
            committed_result = None

        # Anything still sitting in the unconfirmed buffer is now final too
        incomplete = self.transcript_buffer.complete()
        fallback_result = self._format_output(incomplete, is_partial=False) if incomplete else None

        # Prefer the freshly committed result; fall back to unconfirmed words
        result = committed_result or fallback_result
        if result:
            result['is_partial'] = False

        # FIX: capture duration BEFORE clearing the buffer
        remaining_duration = len(self.audio_buffer) / SAMPLE_RATE
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset += remaining_duration

        return result


class WebSocketConnection:
    """Manages a WebSocket connection for streaming audio."""

    def __init__(self, websocket, processor):
        self.websocket = websocket
        self.processor = processor
        self.last_sent_text = ""

    async def send_transcription(self, result):
        """Send a transcription result, skipping exact duplicates."""
        if not result:
            return
        # Deduplicate confirmed results; always forward partials so the
        # frontend shows live progress even before words are committed.
        if not result.get('is_partial', True) or result['text'] != self.last_sent_text:
            self.last_sent_text = result['text']
            await self.websocket.send_text(json.dumps(result))

    async def process_audio_stream(self):
        """Main loop: receive audio chunks and stream transcriptions back."""
        try:
            while True:
                message = await self.websocket.receive()

                if isinstance(message, dict):
                    msg_type = message.get('type')
                else:
                    msg_type = None

                if msg_type == 'websocket.receive':
                    if 'bytes' in message:
                        audio_bytes = message['bytes']
                        audio_array = (
                            np.frombuffer(audio_bytes, dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        self.processor.insert_audio_chunk(audio_array)
                        try:
                            result = self.processor.process_iter()
                        except Exception as chunk_e:
                            print(f"Error while transcribing chunk: {chunk_e}")
                            result = None
                        await self.send_transcription(result)

                    elif 'text' in message:
                        text_payload = message.get('text')
                        try:
                            control = json.loads(text_payload)
                        except Exception:
                            continue
                        if control.get('action') == 'finalize':
                            final_result = self.processor.finish()
                            await self.send_transcription(final_result)
                            break

                elif msg_type == 'websocket.disconnect':
                    break

                elif isinstance(message, str):
                    try:
                        control = json.loads(message)
                        if control.get('action') == 'finalize':
                            final_result = self.processor.finish()
                            await self.send_transcription(final_result)
                            break
                    except Exception:
                        pass

                else:
                    # FIX: unknown message type → log and continue (not break).
                    # A single malformed ping/keepalive no longer kills the stream.
                    print(f"Ignoring unexpected WebSocket message type: {type(message)}")
                    continue

        except Exception as e:
            print(f"Error in streaming: {e}")
            try:
                await self.websocket.send_text(json.dumps({'error': str(e)}))
            except Exception:
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
        "torch",
        "ctranslate2==4.7.1",
        "faster_whisper==1.2.1",
        "transformers==4.57.4",
        "soundfile",
        "websockets",
        "uvicorn",
    )
)

app = modal.App(MODAL_APP_NAME)
volume = modal.Volume.from_name(MODAL_APP_NAME, create_if_missing=True)


@app.cls(
    image=cuda_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    enable_memory_snapshot=True,
    volumes={MODEL_MOUNT_DIR: volume},
    max_containers=10,
    timeout=600,  # 10 min — default 300 s kills long streaming sessions
)
class StreamingWhisperService:
    """Real-time streaming Whisper service with WebSocket support."""

    model_id = MODEL_ID

    @modal.enter()
    def enter(self):
        print(f"Loading Whisper model: {self.model_id}")

        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_and_convert_model(model_dir, self.model_id)

        try:
            self.whisper_model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
                cpu_threads=4,
                num_workers=1,
            )
            print("Model loaded successfully via ctranslate2-converted path")

        except Exception as e:
            print("Failed to load converted model, falling back to direct HF path:", e)
            self.whisper_model = WhisperModel(
                self.model_id,
                device="cuda",
                compute_type="float16",
                cpu_threads=4,
                num_workers=1,
            )
            print("Model loaded successfully via direct HuggingFace path")

        # The memory snapshot may have captured the model with an 80-mel feature
        # extractor. Reinitialise here — enter() runs on every restore — so the
        # extractor is always correct regardless of snapshot state.
        fe = self.whisper_model.feature_extractor
        self.whisper_model.feature_extractor = type(fe)(feature_size=128)
        print(f"Feature extractor set to feature_size=128 (was: {getattr(fe, 'feature_size', '?')})")

        self.processor = None

    def get_processor(self):
        """Get or create a streaming processor (used for single-connection reuse)."""
        if self.processor is None:
            self.processor = StreamingASRProcessor(
                self.whisper_model,
                min_chunk_size=MIN_CHUNK_SIZE,
                buffer_trimming_sec=BUFFER_TRIMMING_SEC
            )
        return self.processor

    @modal.fastapi_endpoint(docs=True, method="POST")
    async def transcribe_file(self,
                              wav: UploadFile = File(..., description="WAV audio file (16kHz mono)"),
                              language: Optional[str] = Query(None, description="Optional language code"),
                              word_timestamps: bool = Query(False, description="Include word-level timestamps")):
        """Non-streaming file transcription endpoint."""
        import librosa

        wav_bytes = await wav.read()
        audio_array, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE)

        segments, _ = self.whisper_model.transcribe(
            audio_array,
            beam_size=BEAM_SIZE,
            language=language,
            task="transcribe",
            vad_filter=True,
            word_timestamps=word_timestamps,
        )

        transcription = ""
        all_segments = []
        all_words = []

        for segment in segments:
            transcription += segment.text + " "
            all_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'confidence': float(np.exp(segment.avg_logprob))
            })
            if segment.words:
                for word in segment.words:
                    all_words.append({
                        'start': word.start,
                        'end': word.end,
                        'word': word.word,
                        'probability': word.probability if hasattr(word, 'probability') else None
                    })

        return {
            'result': 'success',
            'transcription': transcription.strip(),
            'segments': all_segments,
            'words': all_words if word_timestamps else None,
            'language_detected': None
        }

    @modal.asgi_app()
    def streaming_endpoint(self):
        """WebSocket endpoint for real-time streaming with CORS support."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware

        web_app = FastAPI(title="30sAI Whisper Streaming Service", version="1.0.0")

        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "https://30sai.netlify.app",
                "http://localhost:8000",
                "http://localhost:3000",
                "http://127.0.0.1:8000",
                "http://127.0.0.1:3000",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        @web_app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model": self.model_id,
                "sample_rate": SAMPLE_RATE,
                "beam_size": BEAM_SIZE
            }

        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time audio streaming."""
            await websocket.accept()
            print("WebSocket connection accepted")

            # Each connection gets its own fresh processor — never shared state.
            processor = StreamingASRProcessor(
                self.whisper_model,
                min_chunk_size=MIN_CHUNK_SIZE,
                buffer_trimming_sec=BUFFER_TRIMMING_SEC
            )
            connection = WebSocketConnection(websocket, processor)

            try:
                await connection.process_audio_stream()
            except WebSocketDisconnect:
                print("WebSocket disconnected")
            except Exception as e:
                print(f"Error in WebSocket handler: {e}")
                try:
                    await websocket.send_text(json.dumps({'error': str(e)}))
                except Exception:
                    pass
            finally:
                final = processor.finish()
                if final:
                    try:
                        await websocket.send_text(json.dumps(final))
                    except Exception:
                        pass
                try:
                    await websocket.close()
                except Exception:
                    pass

        return web_app


@app.local_entrypoint()
def test_streaming():
    print("=" * 60)
    print("Testing Streaming Whisper Service")
    print("=" * 60)
    url = "https://" + app.app_id + "--streaming-whisper-service-transcribe-file.modal.run"
    print(f"\nFile upload endpoint: {url}")
    print(f'curl -X POST "{url}" -F "wav=@your_audio.wav"')
    print("\nWebSocket endpoint:")
    print("wss://" + app.app_id + "--streaming-whisper-service-streaming-endpoint.modal.run/ws")