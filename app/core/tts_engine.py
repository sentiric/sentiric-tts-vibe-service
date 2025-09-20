# sentiric-tts-vibe-service/app/core/tts_engine.py
import torch
import os
import structlog
# --- DEĞİŞİKLİK BURADA ---
from app.models.modeling_vibevoice import VibeVoiceModel
from app.models.tokenization_vibevoice import VibeVoiceTokenizer
# --- DEĞİŞİKLİK BİTTİ ---


# Global değişkenler
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
log = structlog.get_logger(__name__) # Logger'ı burada alalım

def load_model():
    """Modeli ve tokenizer'ı belleğe yükler."""
    global model, tokenizer
    if model is None or tokenizer is None:
        log.info("Uygulama başlıyor, VibeVoice modeli YEREL KAYNAKTAN yükleniyor...")
        try:
            model_id = "microsoft/VibeVoice-1.5B"
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            
            log.info(f"'{model_id}' ağırlıkları YEREL 'VibeVoiceModel' sınıfı ile yükleniyor...")
            model = VibeVoiceModel.from_pretrained(model_id).to(device)
            
            log.info(f"Tokenizer'ı YEREL dizinden ('{models_dir}') yüklüyor...")
            tokenizer = VibeVoiceTokenizer.from_pretrained(models_dir)
            
            log.info(f"Model {device} üzerine başarıyla yüklendi!")
        except Exception as e:
            log.critical("Model yüklenirken kritik bir hata oluştu.", error=str(e), exc_info=True)
            raise RuntimeError(f"Model yüklenirken kritik bir hata oluştu: {e}")
    else:
        log.info("Model zaten yüklü.")

def unload_model():
    """Modeli ve tokenizer'ı bellekten temizler."""
    global model, tokenizer
    model = None
    tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Model bellekten temizlendi.")

def is_model_loaded() -> bool:
    """Modelin yüklenip yüklenmediğini kontrol eder."""
    return model is not None and tokenizer is not None

def synthesize(text: str, language: str):
    """Verilen metni sese dönüştürür."""
    if not is_model_loaded():
        raise Exception("Model is not loaded.")
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, tokenizer=tokenizer, language=language)
        
    audio = output.waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate
    
    return audio, sample_rate