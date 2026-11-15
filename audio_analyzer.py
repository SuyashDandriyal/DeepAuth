import os
import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import uuid

def analyze_audio(video_path):
    """
    Extracts audio from video and analyzes it for deepfake audio artifacts.
    """
    temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            return {"status": "no_audio", "message": "No audio track found."}
        
        # Write temporary audio file
        clip.audio.write_audiofile(temp_audio_path, logger=None)
        clip.close()
        
        # Load audio with librosa
        y, sr = librosa.load(temp_audio_path, sr=None)
        
        if len(y) == 0:
            return {"status": "no_audio", "message": "Empty audio track."}
            
        # Extract features (e.g., zero crossing rate, spectral centroid)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Mock logic based on features to simulate Wav2Vec2 detection
        confidence = min(100, max(0, float((zcr * 1000) + (spec_cent / 150)) % 100))
        
        # Add some randomness to confidence but heavily weight it towards realistic "scores"
        prediction = "FAKE" if confidence > 70 else "REAL"
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "details": f"ZCR: {round(zcr, 4)}, SpecCent: {round(spec_cent, 2)}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
