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

# Customize for your use case - can use any Whisper model
MODEL_ID = HUGGINGFACE_REPO
# MODEL_ID = "openai/whisper-large-v3-turbo"  # Alternative

def maybe_download_and_convert_model(model_storage_dir, model_id):
    """Download and convert model to CTranslate2 format if not available locally."""
    import ctranslate2

    print(f"Checking for cached model in storage directory: {model_storage_dir}, model ID: {model_id}")

    subdir = model_id.replace("/", "_") + "_ct2"
    model_path = model_storage_dir / subdir
    print(f"Looking for model at: {model_path}")

    if not model_path.exists():
        print(f"Model not found in cache -- downloading from HuggingFace: {model_id}")
        model_path.mkdir(parents=True)

        converter = ctranslate2.converters.TransformersConverter(
            model_name_or_path=model_id,
        )
        print(f"Converting to CTranslate2 format (float16)...")
        converter.convert(str(model_path), quantization="float16", force=True)
        print(f"Model successfully converted and saved to {model_path}")
    else:
        print(f"Found cached model at {model_path}")

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
        """Insert new words with timestamp offset."""
        # Add offset to timestamps
        new = [(a + offset, b + offset, t) for a, b, t in new_words]
        
        # Filter words that are after last committed time
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]
        
        # Remove duplicate n-grams (avoid repeating same text)
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1, i+1))
                        if c == tail:
                            for j in range(i):
                                self.new.pop(0)
                            break

    def flush(self):
        """Return committed chunk - the longest common prefix of last two inserts."""
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            
            if len(self.buffer) == 0:
                break
                
            if nt == self.buffer[0][2]:
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
        """Remove committed words before timestamp."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Return incomplete buffer."""
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
        self.buffer_time_offset = 0
        self.commited = []
        
        # Streaming configuration
        self.beam_size = BEAM_SIZE
        self.vad_filter = True
        
    def insert_audio_chunk(self, audio: np.ndarray):
        """Append audio chunk to buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)
        
    def get_prompt(self):
        """Generate prompt from recent context for better continuity."""
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1
            
        # Get prompt from recent text
        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 character prompt limit
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
            
        return " ".join(prompt[::-1])
    
    def process_iter(self):
        """Process current audio buffer and return new transcription."""
        if len(self.audio_buffer) / SAMPLE_RATE < self.min_chunk_size:
            return None
            
        prompt = self.get_prompt()
        
        # Run transcription on current buffer
        segments, _ = self.whisper_model.transcribe(
            self.audio_buffer,
            beam_size=self.beam_size,
            language=None,  # Auto-detect
            task="transcribe",
            initial_prompt=prompt if prompt else None,
            vad_filter=self.vad_filter,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        
        # Extract words with timestamps
        timestamped_words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    timestamped_words.append((word.start, word.end, word.word))
        
        # Insert into hypothesis buffer
        self.transcript_buffer.insert(timestamped_words, self.buffer_time_offset)
        
        # Get committed words
        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)
        
        # Format output
        result = self._format_output(committed)
        
        # Trim buffer if too long
        if len(self.audio_buffer) / SAMPLE_RATE > self.buffer_trimming_sec:
            self._trim_buffer()
            
        return result
    
    def _format_output(self, words):
        """Format timestamped words into readable output."""
        if not words:
            return None
            
        start_time = words[0][0] + self.buffer_time_offset
        end_time = words[-1][1] + self.buffer_time_offset
        text = " ".join(w[2] for w in words)
        
        return {
            'start': start_time,
            'end': end_time,
            'text': text,
            'is_partial': True,
            'words': [(w[0] + self.buffer_time_offset, 
                      w[1] + self.buffer_time_offset, 
                      w[2]) for w in words]
        }
    
    def _trim_buffer(self):
        """Trim audio buffer to prevent memory issues."""
        if self.commited:
            # Trim to last committed word
            trim_time = self.commited[-1][1]
            cut_samples = int((trim_time - self.buffer_time_offset) * SAMPLE_RATE)
            
            if cut_samples > 0:
                self.audio_buffer = self.audio_buffer[cut_samples:]
                self.buffer_time_offset = trim_time
                
                # Clean up transcript buffer
                self.transcript_buffer.pop_commited(trim_time)
    
    def finish(self):
        """Get final transcription when streaming ends."""
        # Process any remaining audio
        result = None
        if len(self.audio_buffer) > 0:
            # Force final transcription
            incomplete = self.transcript_buffer.complete()
            if incomplete:
                result = self._format_output(incomplete)
        
        # Clear buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset += len(self.audio_buffer) / SAMPLE_RATE
        
        return result


class WebSocketConnection:
    """Manages WebSocket connection for streaming audio."""
    
    def __init__(self, websocket, processor):
        self.websocket = websocket
        self.processor = processor
        self.last_sent_text = ""
        
    async def send_transcription(self, result):
        """Send transcription result without duplicates."""
        if not result:
            return
            
        # Avoid sending duplicate text
        if result['text'] == self.last_sent_text:
            return
            
        self.last_sent_text = result['text']
        await self.websocket.send(json.dumps(result))
        
    async def process_audio_stream(self):
        """Main processing loop for streaming audio."""
        try:
            while True:
                # Receive audio chunk
                message = await self.websocket.receive()
                
                if message['type'] == 'websocket.receive':
                    if 'bytes' in message:
                        # Process audio chunk
                        audio_bytes = message['bytes']
                        
                        # Convert bytes to numpy array
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Insert and process
                        self.processor.insert_audio_chunk(audio_array)
                        result = self.processor.process_iter()
                        await self.send_transcription(result)
                        
                    elif 'text' in message:
                        # Handle control messages
                        control = json.loads(message['text'])
                        if control.get('action') == 'finalize':
                            final_result = self.processor.finish()
                            await self.send_transcription(final_result)
                            break
                            
        except Exception as e:
            print(f"Error in streaming: {e}")
            await self.websocket.send(json.dumps({'error': str(e)}))


# Modal image setup
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi[standard]",
        "numpy",
        "librosa",
        "huggingface_hub[hf_transfer]==0.26.2",
        "torch",
        "ctranslate2",
        "faster_whisper",
        "transformers",
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
    concurrency_limit=10,
)
class StreamingWhisperService:
    """Real-time streaming Whisper service with WebSocket support."""
    
    model_id = MODEL_ID
    
    @modal.enter()
    def enter(self):
        """Load model and initialize."""
        print(f"Loading Whisper model: {self.model_id}")
        
        # Load or convert model
        model_dir = MODEL_MOUNT_DIR / MODEL_DOWNLOAD_DIR
        model_path = maybe_download_and_convert_model(model_dir, self.model_id)
        
        # Initialize FasterWhisper model
        self.whisper_model = WhisperModel(
            model_path, 
            device="cuda", 
            compute_type="float16",
            cpu_threads=4,
            num_workers=1
        )
        print("Model loaded successfully")
        
        # Initialize processor
        self.processor = None
        
    def get_processor(self):
        """Get or create streaming processor for this connection."""
        if self.processor is None:
            self.processor = StreamingASRProcessor(
                self.whisper_model,
                min_chunk_size=MIN_CHUNK_SIZE,
                buffer_trimming_sec=BUFFER_TRIMMING_SEC
            )
        return self.processor
    
    @modal.fastapi_endpoint(docs=True, method="POST")
    def transcribe_file(self, 
                       wav: bytes = File(..., description="WAV audio file (16kHz mono)"),
                       language: str = Form(default=None, description="Optional language code"),
                       word_timestamps: bool = Form(default=False, description="Include word-level timestamps")):
        """Simple file-based transcription endpoint (non-streaming)."""
        import librosa
        
        # Load audio
        audio_array, _ = librosa.load(io.BytesIO(wav), sr=SAMPLE_RATE)
        
        # Process entire file
        segments, _ = self.whisper_model.transcribe(
            audio_array,
            beam_size=BEAM_SIZE,
            language=language,
            task="transcribe",
            vad_filter=True,
            word_timestamps=word_timestamps,
        )
        
        # Format results
        transcription = ""
        all_segments = []
        all_words = []
        
        for segment in segments:
            transcription += segment.text + " "
            all_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'confidence': np.exp(segment.avg_logprob)
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
            'language_detected': None  # Add if needed
        }
    
    @modal.asgi_app()
    def streaming_endpoint(self):
        """WebSocket endpoint for real-time streaming."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        
        web_app = FastAPI()
        
        @web_app.get("/")
        async def get():
            return HTMLResponse("""
            <html>
                <body>
                    <h1>Whisper Streaming Service</h1>
                    <p>WebSocket endpoint available at: ws://[your-url]/ws</p>
                    <p>Send audio chunks as binary data (16kHz PCM int16)</p>
                </body>
            </html>
            """)
        
        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket connection accepted")
            
            # Create new processor for this connection
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
                print(f"Error in WebSocket: {e}")
            finally:
                # Send final transcription if any
                final = processor.finish()
                if final:
                    await websocket.send(json.dumps(final))
                await websocket.close()
                
        return web_app


# Optional: Client example script
@app.local_entrypoint()
def test_streaming():
    """Test the streaming service with a sample file."""
    import requests
    import numpy as np
    import librosa
    import websockets
    import asyncio
    import sys
    
    print("=" * 60)
    print("Testing Streaming Whisper Service")
    print("=" * 60)
    
    # Test file upload endpoint
    url = "https://" + app.app_id + "--streaming-whisper-service-transcribe-file.modal.run"
    print(f"\n1. Testing file upload endpoint at: {url}")
    
    # You would need to have a test audio file
    # For now, we'll just print instructions
    print("\nTo test file upload:")
    print(f'curl -X POST "{url}" -F "wav=@your_audio.wav"')
    
    print("\nTo test WebSocket streaming:")
    print("ws_url = wss://" + app.app_id + "--streaming-whisper-service-streaming-endpoint.modal.run/ws")
    print("\nUse a WebSocket client to connect and send audio chunks")
    print("\nExample Python WebSocket client:")
    print("""
import asyncio
import websockets
import numpy as np
import soundfile as sf

async def stream_audio():
    uri = "wss://your-endpoint/ws"
    async with websockets.connect(uri) as websocket:
        # Load and stream audio in chunks
        audio, sr = sf.read('test.wav')
        chunk_size = 16000  # 1 second chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            # Convert to int16 bytes
            bytes_data = (chunk * 32768).astype(np.int16).tobytes()
            await websocket.send(bytes_data)
            response = await websocket.recv()
            print(response)
        # Signal end of stream
        await websocket.send('{"action": "finalize"}')

asyncio.run(stream_audio())
    """)
