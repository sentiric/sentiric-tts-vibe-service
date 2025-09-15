import torch
import os
# --- DEĞİŞİKLİK BURADA ---
# "app." öneklerini buradan da kaldırıyoruz ki her şey tutarlı olsun.
from models.modeling_vibevoice import VibeVoiceModel
from models.tokenization_vibevoice import VibeVoiceTokenizer
# --- DEĞİŞİKLİK BİTTİ ---

# Global değişkenler
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Modeli ve tokenizer'ı belleğe yükler."""
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Uygulama başlıyor, model ve tokenizer YEREL KAYNAKTAN yükleniyor...")
        try:
            model_id = "microsoft/VibeVoice-1.5B"
            # Projemizin içindeki 'app/models' klasörünün yolunu buluyoruz
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            
            print(f"'{model_id}' ağırlıkları YEREL 'VibeVoiceModel' sınıfı ile yükleniyor...")
            model = VibeVoiceModel.from_pretrained(model_id).to(device)
            
            print(f"Tokenizer'ı YEREL dizinden ('{models_dir}') yüklüyor...")
            # Tokenizer'a dosyaların nerede olduğunu doğrudan söylüyoruz.
            tokenizer = VibeVoiceTokenizer.from_pretrained(models_dir)
            
            print(f"Model {device} üzerine başarıyla yüklendi!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Model yüklenirken kritik bir hata oluştu: {e}")
    else:
        print("Model zaten yüklü.")

def unload_model():
    """Modeli ve tokenizer'ı bellekten temizler."""
    global model, tokenizer
    model = None
    tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model bellekten temizlendi.")

def synthesize(text: str, language: str):
    """Verilen metni sese dönüştürür."""
    if not model or not tokenizer:
        raise Exception("Model is not loaded.")
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, tokenizer=tokenizer, language=language)
        
    audio = output.waveform[0].cpu().numpy()
    sample_rate = model.config.sampling_rate
    
    return audio, sample_rate