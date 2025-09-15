# app/api/tts.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import scipy.io.wavfile
import os
import uuid
from core import tts_engine
import numpy as np

router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "tr"

@router.post("/synthesize", summary="Metni sese dönüştürür")
def synthesize_speech(request: TTSRequest):
    try:
        audio, sample_rate = tts_engine.synthesize(request.text, request.language)
        
        # --- BU BLOK DEĞİŞTİ ---
        # En yüksek mutlak değeri bul
        max_val = np.max(np.abs(audio))
        
        # Eğer en yüksek değer sıfırdan büyükse normalize et, değilse dokunma
        if max_val > 0:
            # Sesi normalize et (-1 ve 1 arasına getir)
            audio_normalized = audio / max_val
        else:
            # Ses zaten tamamen sessizse, olduğu gibi bırak
            audio_normalized = audio

        # Sesi WAV formatının beklediği integer aralığına getir (16-bit)
        # Olası NaN değerlerini sıfırla değiştirerek güvenli hale getir
        audio_int = np.int16(np.nan_to_num(audio_normalized) * 32767)
        # --- DEĞİŞİKLİK BİTTİ ---

        temp_dir = "/tmp/tts_output"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.join(temp_dir, f"{uuid.uuid4().hex}.wav")
        
        scipy.io.wavfile.write(filename, rate=sample_rate, data=audio_int)

        return FileResponse(filename, media_type="audio/wav", filename="synthesized_speech.wav")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sentezleme sırasında bir hata oluştu: {str(e)}")