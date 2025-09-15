from fastapi import FastAPI
from contextlib import asynccontextmanager
# --- DEĞİŞİKLİK BURADA ---
from core import tts_engine
from api import tts
# --- DEĞİŞİKLİK BİTTİ ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başladığında modeli yükle
    tts_engine.load_model()
    yield
    # Uygulama kapandığında modeli temizle
    tts_engine.unload_model()

app = FastAPI(
    title="VibeVoice TTS Service (Simple Imports)",
    version="6.0",
    lifespan=lifespan
)

# API endpoint'lerini (yollarını) dahil et
app.include_router(tts.router)

# Statik dosyalar (frontend) için bir yol oluştur
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Ana sayfa (/) isteği geldiğinde index.html'i göster
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(static_dir, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)