"""
GPT-2 Next Word Prediction Service — Streaming
Streams tokens as they are generated using TextIteratorStreamer + SSE,
so the UI can show word-by-word completions with minimal latency.
"""

import modal
import torch
import re
import os

app = modal.App("gpt2-service-v2")

models_volume = modal.Volume.from_name("aac-models", create_if_missing=True)
MODELS_PATH = "/mnt/aac-models"
GPT2_PATH   = "/mnt/aac-models/gpt2"

gpt2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.35.0",
        "accelerate==0.24.1",
        "peft==0.7.0",
        "numpy<2",  # torch 2.1.0 was built against NumPy 1.x; 2.x breaks it
        "fastapi[standard]",
    )
)

try:
    hf_secret = modal.Secret.from_name("huggingface")
except Exception:
    hf_secret = None

USE_GPU = os.environ.get("GPT2_USE_GPU", "0") == "1"

CORS_ORIGINS = [
    "https://30sai.netlify.app",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

FALLBACK_PHRASES = [
    "thank you", "I need help", "please", "yes", "no",
    "my account", "PIN reset", "balance", "transfer",
    "connect to agent", "repeat that",
]


def clean_text(text: str) -> str:
    """Remove disfluency markers and normalise."""
    return re.sub(r"\[(REP|PROLONG|PARTIAL|FILLER|BLOCK)\]\s*", "", text).strip()


gpt2_cls_kwargs = {
    "memory": 2048,
    "image": gpt2_image,
    "scaledown_window": 600,
    "secrets": [hf_secret] if hf_secret else [],
    "volumes": {MODELS_PATH: models_volume},
}
if USE_GPU:
    gpt2_cls_kwargs["gpu"] = "L4"


@app.cls(**gpt2_cls_kwargs)
class GPT2Service:
    """GPT-2 next-word prediction service with token streaming."""

    # ------------------------------------------------------------------
    @modal.enter()
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.ready  = False
        self.device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
        print(f"GPT2 device: {self.device}")

        def _load(path: str):
            tok = AutoTokenizer.from_pretrained(path)
            tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(path).eval().to(self.device)
            return tok, mdl

        # Volume first
        if os.path.exists(GPT2_PATH) and os.listdir(GPT2_PATH):
            try:
                print(f"Loading GPT-2 from volume: {GPT2_PATH}")
                self.tokenizer, self.model = _load(GPT2_PATH)
                self.ready = True
                print("GPT-2 loaded from volume")
                return
            except Exception as e:
                print(f"Volume load failed: {e}")

        # HF fallback
        try:
            print("Loading base GPT-2 from HuggingFace")
            self.tokenizer, self.model = _load("gpt2")
            self.ready = True
            print("GPT-2 base loaded")

            # Save to volume so future cold starts skip the 548 MB download
            try:
                print(f"Saving model to volume: {GPT2_PATH}")
                os.makedirs(GPT2_PATH, exist_ok=True)
                self.tokenizer.save_pretrained(GPT2_PATH)
                self.model.save_pretrained(GPT2_PATH)
                models_volume.commit()
                print("Model saved to volume — next cold start will load locally")
            except Exception as save_e:
                print(f"Volume save skipped: {save_e}")
        except Exception as e:
            print(f"Failed to load GPT-2: {e}")

    # ------------------------------------------------------------------
    # Shared generation helper — yields raw token strings one at a time
    # ------------------------------------------------------------------
    def _token_stream(self, text: str, n: int = 4):
        """
        Generator that yields decoded tokens as the model produces them.
        Uses TextIteratorStreamer so generation runs in a background thread
        while this thread streams the output.
        """
        from transformers import TextIteratorStreamer
        import threading

        if not self.ready or not text or len(text.strip()) < 2:
            return

        text = clean_text(text)
        enc  = self.tokenizer(text, return_tensors="pt")
        dev  = next(self.model.parameters()).device
        input_ids      = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,        # don't re-emit the input
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,       # enough for 2-word completions × n sequences
            do_sample=True,
            temperature=0.75,
            top_k=50,
            top_p=0.92,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
            # Note: num_return_sequences > 1 is incompatible with streaming;
            # we run n separate passes below instead.
        )

        def _generate():
            try:
                self.model.generate(**gen_kwargs)
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print(f"Device mismatch, retrying on CPU: {e}")
                    self.model = self.model.to("cpu")
                    self.device = torch.device("cpu")
                    cpu_enc = self.tokenizer(text, return_tensors="pt")
                    self.model.generate(
                        input_ids=cpu_enc["input_ids"],
                        attention_mask=cpu_enc.get("attention_mask"),
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=0.75,
                        top_k=50,
                        top_p=0.92,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                    )
                else:
                    raise

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        yield from streamer
        thread.join()

    # ------------------------------------------------------------------
    # Non-streaming batch prediction (Modal method, for internal callers)
    # ------------------------------------------------------------------
    @modal.method()
    def predict(self, text: str, n: int = 4) -> list:
        """
        Return n next-word predictions in one shot.
        Kept for backward-compat with other Modal functions.
        """
        if not self.ready or not text or len(text.strip()) < 2:
            return []

        text = clean_text(text)
        seen, predictions = set(), []

        for _ in range(n):
            collected = ""
            try:
                for token in self._token_stream(text, n=1):
                    collected += token
            except Exception as e:
                print(f"Prediction error: {e}")
                continue

            words = collected.strip().split()[:2]
            pred  = " ".join(words).strip(" .,!?;:'\"")
            if pred and pred.lower() not in seen:
                seen.add(pred.lower())
                predictions.append(pred)

        # Fill up with fallbacks if needed
        for fb in FALLBACK_PHRASES:
            if len(predictions) >= n:
                break
            if fb not in seen:
                predictions.append(fb)
                seen.add(fb)

        return predictions[:n]

    # ------------------------------------------------------------------
    # Streaming ASGI app
    # ------------------------------------------------------------------
    @modal.asgi_app()
    def web(self):
        """
        FastAPI app with two prediction endpoints:

        GET  /predict-stream?text=...&n=4
             — Server-Sent Events stream.
               Each SSE event is one of:
                 data: {"type":"token","token":"hello"}     ← raw token
                 data: {"type":"prediction","text":"hello world"}  ← completed 2-word phrase
                 data: {"type":"done","predictions":["hello world","..."]}
               The browser subscribes with EventSource and updates the
               suggestion bar word-by-word.

        POST /predict
             — Returns JSON list of n predictions (non-streaming, legacy).
        """
        import asyncio
        import json as _json
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse, JSONResponse

        web_app = FastAPI(title="GPT-2 Streaming Service", version="2.0.0")
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # ── SSE streaming endpoint ────────────────────────────────────
        @web_app.get("/predict-stream")
        async def predict_stream(text: str = "", n: int = 4):
            """
            Stream predictions as Server-Sent Events.

            The client code:
              const es = new EventSource(`/predict-stream?text=${encodeURIComponent(transcript)}&n=4`);
              es.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'token')      updateSuggestionBar(msg.token);
                if (msg.type === 'prediction') addSuggestion(msg.text);
                if (msg.type === 'done')       es.close();
              };
            """
            async def _event_generator():
                if not text.strip():
                    yield f"data: {_json.dumps({'type':'done','predictions':[]})}\n\n"
                    return

                seen, predictions = set(), []

                for pass_idx in range(n):
                    collected = ""
                    # Run each generation pass in a thread so we don't block the event loop
                    loop = asyncio.get_running_loop()

                    def _run_stream(t=text):
                        return list(self._token_stream(t, n=1))

                    tokens = await loop.run_in_executor(None, _run_stream)

                    for token in tokens:
                        collected += token
                        yield f"data: {_json.dumps({'type':'token','token':token,'pass':pass_idx})}\n\n"
                        await asyncio.sleep(0)  # yield control to the event loop

                    words = collected.strip().split()[:2]
                    pred  = " ".join(words).strip(" .,!?;:'\"")
                    if pred and pred.lower() not in seen:
                        seen.add(pred.lower())
                        predictions.append(pred)
                        yield f"data: {_json.dumps({'type':'prediction','text':pred})}\n\n"

                # Fill fallbacks
                for fb in FALLBACK_PHRASES:
                    if len(predictions) >= n:
                        break
                    if fb not in seen:
                        predictions.append(fb)

                yield f"data: {_json.dumps({'type':'done','predictions':predictions[:n]})}\n\n"

            return StreamingResponse(
                _event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # disable nginx buffering
                },
            )

        # ── Legacy batch endpoint ─────────────────────────────────────
        @web_app.post("/predict")
        async def predict_batch(request: Request):
            """Return JSON list of predictions (non-streaming, backward-compat)."""
            body = await request.json()
            text = body.get("text", "")
            n    = int(body.get("n", 4))
            return self.predict.local(text, n)

        # Kept for the original query-param interface
        @web_app.get("/predict")
        async def predict_get(text: str = "", n: int = 4):
            return self.predict.local(text, n)

        # ── Health ────────────────────────────────────────────────────
        @web_app.get("/health")
        async def health():
            return {"status": "healthy", "ready": self.ready, "device": str(self.device)}

        return web_app
