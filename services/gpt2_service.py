"""
GPT-2 Next Word Prediction Service
Generates word predictions based on conversation context
"""

import modal
import torch
import re
from typing import List

# Create Modal app
app = modal.App("gpt2-service")

# Volume for models
models_volume = modal.Volume.from_name("aac-models", create_if_missing=True)
MODELS_PATH = "/mnt/aac-models"
GPT2_PATH = "/mnt/aac-models/gpt2"

# Image with dependencies
gpt2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.35.0",
        "accelerate==0.24.1",
        "peft==0.7.0",
        "numpy",
    )
)

# Try to get HuggingFace secret (optional - handle gracefully)
try:
    hf_secret = modal.Secret.from_name("huggingface")
except Exception:
    hf_secret = None


def clean_text(text: str) -> str:
    """Remove disfluency markers and normalize"""
    text = re.sub(r"\[(REP|PROLONG|PARTIAL|FILLER|BLOCK)\]\s*", "", text)
    return text.strip()


@app.cls(
    cpu=2,
    memory=2048,
    image=gpt2_image,
    scaledown_window=300,
    secrets=[hf_secret] if hf_secret else [],
    volumes={MODELS_PATH: models_volume},
)
class GPT2Service:
    """GPT-2 next-word prediction service"""
    
    @modal.enter()
    def load(self):
        """Load GPT-2 model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        
        self.ready = False
        
        def load_model(path: str):
            tokenizer = AutoTokenizer.from_pretrained(path)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(path)
            model.eval()
            return tokenizer, model
        
        # Try loading from volume
        if os.path.exists(GPT2_PATH) and os.listdir(GPT2_PATH):
            try:
                print(f"Loading GPT-2 from volume: {GPT2_PATH}")
                self.tokenizer, self.model = load_model(GPT2_PATH)
                self.ready = True
                print("GPT-2 loaded from volume")
                return
            except Exception as e:
                print(f"Failed to load from volume: {e}")
        
        # Fallback to base model
        try:
            print("Loading base GPT-2 from HuggingFace")
            self.tokenizer, self.model = load_model("gpt2")
            self.ready = True
            print("GPT-2 base loaded")
        except Exception as e:
            print(f"Failed to load GPT-2: {e}")
    
    @modal.method()
    def predict(self, text: str, n: int = 4) -> list:
        """
        Predict next words based on context
        
        Args:
            text: Current transcript text
            n: Number of predictions to return
            
        Returns:
            List of predicted phrases
        """
        if not self.ready or not text or len(text.strip()) < 2:
            return []
        
        text = clean_text(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                num_return_sequences=n,
                do_sample=True,
                temperature=0.75,
                top_k=50,
                top_p=0.92,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        seen = set()
        predictions = []
        
        for out in outputs:
            full = self.tokenizer.decode(out, skip_special_tokens=True)
            continuation = full[len(text):].strip()
            # Take first 1-2 words
            words = continuation.split()[:2]
            pred = " ".join(words).strip(" .,!?;:'\"")
            
            if pred and pred.lower() not in seen:
                seen.add(pred.lower())
                predictions.append(pred)
        
        # Add fallback predictions if needed
        fallbacks = [
            "thank you", "I need help", "please", "yes", "no", 
            "my account", "PIN reset", "balance", "transfer",
            "connect to agent", "repeat that"
        ]
        
        while len(predictions) < n:
            for fb in fallbacks:
                if fb not in seen:
                    predictions.append(fb)
                    seen.add(fb)
                    break
        
        return predictions[:n]
