# 30sAI_asr

30s AI is an application that transforms unstructured speech into structured text in real time and sends the transcription to a customer service or support agent over WebSocket. It was developed by Tinyefuza Joe, Nabaccwa.M.Jema and Adam Katongole for the ASR hackathon Uganda 2026.

## What is in the repo

- `services/modal_streaming_whisper.py` - real-time Whisper ASR service with `/ws`, `/transcribe`, and `/health`
- `services/gpt2_service.py` - next-word prediction service
- `services/tts_service.py` - text-to-speech service
- `services/feedback_service.py` - stores corrections for later training
- `client/index.html` - main browser UI
- `client/browser_client.html` - minimal browser WebSocket client
- `client/python_websocket_client.py` - CLI streaming client
- `static/index.html` - static demo UI

## Prerequisites

- Modal CLI installed and authenticated
- Hugging Face secret configured for Whisper if required by your Modal account
- Python 3.10+ for local client testing

## Deploy the endpoints

Run these from the repository root:

```bash
modal deploy services/modal_streaming_whisper.py
modal deploy services/gpt2_service.py
modal deploy services/tts_service.py
modal deploy services/feedback_service.py
```

Or deploy everything with the helper script:

```bash
./scripts/deploy_all.sh
```

If this is your first time using Modal on the machine, run:

```bash
modal setup
```

## Run locally

The Whisper service has a local entrypoint that prints the deployed URLs for the file upload and WebSocket endpoints:

```bash
modal run services/modal_streaming_whisper.py
```

To open the browser UI locally, serve the repository over HTTP:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/client/index.html` in your browser and paste the deployed service URLs into the Settings screen.

## Test the services

### 1. Test file transcription

After running `modal run services/modal_streaming_whisper.py`, copy the printed file upload URL and send a WAV file:

```bash
curl -X POST "https://YOUR-MODAL-URL/transcribe" -F "wav=@your_audio.wav"
```

### 2. Test streaming transcription

Use the Python WebSocket client against the deployed WebSocket URL:

```bash
python3 client/python_websocket_client.py --audio your_audio.wav --url wss://YOUR-MODAL-URL/ws
```

### 3. Test in the browser

1. Open the main UI at `http://localhost:8000/client/index.html`.
2. Paste the Whisper WebSocket URL, GPT-2 endpoint, TTS endpoint, and Feedback endpoint into Settings.
3. Save the settings and start recording.
4. Confirm partial transcription updates, final transcripts, predictions, and optional correction submission.

### 4. Health check

Call the health endpoint after deployment:

```bash
curl https://YOUR-MODAL-URL/health
```

## Notes

- The main ASR service is the only required backend for basic transcription.
- GPT-2, TTS, and Feedback are optional but enable the full assistant flow.
- The exact Modal URLs are printed by `modal run services/modal_streaming_whisper.py` and by the Modal dashboard after deployment.
- For the app Whisper setting, use the **streaming endpoint** host with `/ws` and WebSocket scheme. Example:
  - `wss://<streaming-endpoint>.modal.run/ws`
  - If you paste `https://...`, the client now auto-converts it to `wss://.../ws`.
