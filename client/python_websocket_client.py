#!/usr/bin/env python3
import asyncio
import websockets
import numpy as np
import soundfile as sf
import json
import argparse

async def stream_audio(audio_file=None, url="ws://localhost:8000/ws", chunk_seconds=1.0):
    """Stream audio to Whisper service"""
    
    async with websockets.connect(url) as ws:
        print(f"Connected to {url}")
        
        if audio_file:
            # Stream from file
            audio, sr = sf.read(audio_file)
            if sr != 16000:
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
            
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            audio_int16 = (audio * 32767).astype(np.int16)
            chunk_size = int(chunk_seconds * 16000)
            
            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                await ws.send(chunk.tobytes())
                
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    result = json.loads(response)
                    if result.get('text'):
                        print(f"\r{result['text']}", end='')
                except:
                    pass
            
            await ws.send(json.dumps({'action': 'finalize'}))
            final = await ws.recv()
            result = json.loads(final)
            print(f"\n\nFinal: {result.get('text', '')}")
        
        else:
            # Stream from microphone
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=int(16000 * chunk_seconds))
            
            print("Recording... Press Ctrl+C to stop")
            try:
                while True:
                    data = stream.read(int(16000 * chunk_seconds), exception_on_overflow=False)
                    await ws.send(data)
                    
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        result = json.loads(response)
                        if result.get('text'):
                            print(f"\r{result['text']}", end='')
                    except:
                        pass
            except KeyboardInterrupt:
                print("\n\nStopping...")
                await ws.send(json.dumps({'action': 'finalize'}))
                final = await ws.recv()
                result = json.loads(final)
                print(f"Final: {result.get('text', '')}")
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Audio file to stream")
    parser.add_argument("--url", default="ws://localhost:8000/ws", help="WebSocket URL")
    args = parser.parse_args()
    
    asyncio.run(stream_audio(args.audio, args.url))
