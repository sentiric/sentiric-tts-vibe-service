# sentiric-tts-vibe-service/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Uygulamanın tüm yapılandırmasını ortam değişkenlerinden ve .env dosyasından
    yöneten merkezi sınıf.
    """
    # Proje Meta Verileri
    PROJECT_NAME: str = "Sentiric Vibe-TTS Service"
    API_V1_STR: str = "/api/v1"

    # Ortam Ayarları
    ENV: str = Field("production", validation_alias="ENV")
    LOG_LEVEL: str = Field("INFO", validation_alias="LOG_LEVEL")
    HTTP_PORT: int = Field(14050, validation_alias="TTS_VIBE_SERVICE_HTTP_PORT")

    # Gözlemlenebilirlik (CI/CD tarafından doldurulur)
    SERVICE_VERSION: str = Field("0.0.0", validation_alias="SERVICE_VERSION")
    GIT_COMMIT: str = Field("unknown", validation_alias="GIT_COMMIT")
    BUILD_DATE: str = Field("unknown", validation_alias="BUILD_DATE")
    
    # Pydantic'e .env dosyasını okumasını ve büyük/küçük harf duyarsız olmasını söyler
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        extra='ignore', 
        case_sensitive=False
    )

settings = Settings()