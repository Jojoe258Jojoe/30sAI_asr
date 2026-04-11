"""
Text-to-Speech Service
Converts text to speech using gTTS
"""

import modal
import io
import base64

# Create Modal app
app = modal.App("tts-service")

# Image with dependencies
tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gTTS==2.5.3", "fastapi[standard]")
)


@app.cls(
    cpu=1,
    memory=512,
    image=tts_image,
    scaledown_window=300,
)
class TTSService:
    """Text-to-speech service using gTTS"""
    
    @modal.method()
    def speak(self, text: str) -> dict:
        """
        Convert text to speech and return base64 MP3
        
        Args:
            text: Text to convert to speech
            
        Returns:
            dict with audio_b64 and duration_ms
        """
        from gtts import gTTS
        
        if not text or not text.strip():
            return {"audio_b64": "", "duration_ms": 0}
        
        buf = io.BytesIO()
        tts = gTTS(text=text.strip(), lang="en", slow=False)
        tts.write_to_fp(buf)
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        
        # Rough estimate: ~40ms per character
        duration_ms = int(len(text) * 40)
        
        return {
            "audio_b64": audio_b64,
            "duration_ms": duration_ms
        }

    @modal.fastapi_endpoint(method="POST")
    def speak_web(self, text: str) -> dict:
        """HTTP endpoint for text-to-speech (called from frontend)"""
        return self.speak.local(text)