"""
Feedback Collection Service
Stores user corrections for continual learning
"""

import modal
import os
import json
import base64
import time
from datetime import datetime

# Create Modal app
app = modal.App("feedback-service")

# Volume for storing feedback
feedback_volume = modal.Volume.from_name("aac-models", create_if_missing=True)
FEEDBACK_PATH = "/mnt/aac-models/feedback"

feedback_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.24.3", "fastapi[standard]")
)


@app.cls(
    cpu=1,
    memory=256,
    image=feedback_image,
    scaledown_window=300,
    volumes={"/mnt/aac-models": feedback_volume},
)
class FeedbackService:
    """Store user corrections for model improvement"""
    
    @modal.enter()
    def setup(self):
        """Create feedback directory if needed"""
        os.makedirs(FEEDBACK_PATH, exist_ok=True)
    
    @modal.method()
    def save_correction(self, audio_b64: str, original: str, corrected: str) -> dict:
        """
        Save a user correction for later fine-tuning
        
        Args:
            audio_b64: Base64 encoded PCM audio
            original: Original transcription
            corrected: User-corrected transcription
            
        Returns:
            dict with status and total pairs
        """
        if not audio_b64 or not corrected:
            return {"status": "error", "message": "Missing data"}
        
        ts = int(time.time() * 1000)
        base = os.path.join(FEEDBACK_PATH, str(ts))
        
        # Save PCM audio
        pcm = base64.b64decode(audio_b64)
        with open(f"{base}.pcm", "wb") as f:
            f.write(pcm)
        
        # Save metadata
        with open(f"{base}.json", "w") as f:
            json.dump({
                "timestamp": ts,
                "datetime": datetime.now().isoformat(),
                "original": original,
                "corrected": corrected,
                "samples": len(pcm) // 2,
                "duration_s": len(pcm) / 2 / 16000,
            }, f, indent=2)
        
        feedback_volume.commit()
        
        # Count total pairs
        pairs = len([x for x in os.listdir(FEEDBACK_PATH) if x.endswith(".json")])
        
        return {
            "status": "saved",
            "total_pairs": pairs,
            "message": f"Correction saved. {pairs} pairs accumulated."
        }
    
    @modal.method()
    def get_stats(self) -> dict:
        """Get feedback statistics"""
        if not os.path.exists(FEEDBACK_PATH):
            return {"total_pairs": 0}
        
        json_files = [x for x in os.listdir(FEEDBACK_PATH) if x.endswith(".json")]
        return {"total_pairs": len(json_files)}

    @modal.fastapi_endpoint(method="POST")
    def save_correction_web(self, audio_b64: str, original: str, corrected: str) -> dict:
        """HTTP endpoint for saving corrections (called from frontend)"""
        return self.save_correction.local(audio_b64, original, corrected)

    @modal.fastapi_endpoint(method="GET")
    def get_stats_web(self) -> dict:
        """HTTP endpoint for feedback statistics"""
        return self.get_stats.local()