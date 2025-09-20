# sentiric-tts-vibe-service/app/api/tts.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import scipy.io.wavfile
import os
import uuid
import numpy as np
import structlog

# DEĞİŞİKLİK: Standart import yolları
from app.core import tts_engine

router = APIRouter()
log = structlog.get_logger(__name__)

class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "tr"

@router.post("/synthesize", summary="Metni sese dönüştürür")
def synthesize_speech(request: TTSRequest):
    log.info("Sentezleme isteği alındı.", text=request.text)
    try:
        audio, sample_rate = tts_engine.synthesize(request.text, request.language)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio / max_val
        else:
            audio_normalized = audio

        audio_int = np.int16(np.nan_to_num(audio_normalized) * 32767)

        temp_dir = "/tmp/tts_output"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.join(temp_dir, f"{uuid.uuid4().hex}.wav")
        
        scipy.io.wavfile.write(filename, rate=sample_rate, data=audio_int)
        
        log.info("Ses başarıyla sentezlendi.", output_file=filename)
        return FileResponse(filename, media_type="audio/wav", filename="synthesized_speech.wav")
        
    except Exception as e:
        log.error("Sentezleme sırasında hata oluştu.", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sentezleme sırasında bir hata oluştu: {str(e)}")