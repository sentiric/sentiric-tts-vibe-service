# sentiric-tts-vibe-service/app/main.py
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import uuid
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

# DEĞİŞİKLİK: Standart import yolları kullanılıyor
from app.core.config import settings
from app.core.logging import setup_logging
from app.core import tts_engine
from app.api import tts

SERVICE_NAME = "tts-vibe-service"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loglamayı ve modeli başlat
    setup_logging(log_level=settings.LOG_LEVEL, env=settings.ENV)
    log = structlog.get_logger().bind(service=SERVICE_NAME)
    
    log.info(
        "Uygulama başlatılıyor...",
        project=settings.PROJECT_NAME,
        version=settings.SERVICE_VERSION,
        commit=settings.GIT_COMMIT,
        build_date=settings.BUILD_DATE,
    )

    tts_engine.load_model()
    yield
    # Uygulama kapandığında modeli temizle
    log.info("Uygulama kapatılıyor...")
    tts_engine.unload_model()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

# Middleware'i ekleyelim
@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    log = structlog.get_logger(__name__)
    clear_contextvars()
    
    # Health check'leri log gürültüsünden ayıklayalım
    if request.url.path in ["/health", "/healthz"]:
        return await call_next(request)
        
    trace_id = request.headers.get("X-Trace-ID") or f"tts-vibe-trace-{uuid.uuid4()}"
    bind_contextvars(trace_id=trace_id)

    log.info("İstek alındı", http_method=request.method, http_path=request.url.path)
    response = await call_next(request)
    log.info("İstek tamamlandı", http_status_code=response.status_code)
    return response

# API endpoint'lerini (yollarını) dahil et
app.include_router(tts.router, prefix=settings.API_V1_STR)

# Statik dosyalar (frontend) için bir yol oluştur
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Ana sayfa (/) isteği geldiğinde index.html'i göster
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root():
    with open(os.path.join(static_dir, "index.html")) as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Standart Health Check Endpoint'i
@app.get("/health", tags=["Health"])
async def health_check():
    is_ready = tts_engine.is_model_loaded()
    status_code = 200 if is_ready else 503
    return Response(
        status_code=status_code,
        content=f'{{"status": "ok" if is_ready else "loading_model", "model_ready": {str(is_ready).lower()}}}',
        media_type="application/json"
    )

# Docker için basit Health Check
@app.get("/healthz", include_in_schema=False)
async def healthz():
    return Response(status_code=200)

# Dockerfile'daki CMD komutunu da güncelleyelim
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "14050"]