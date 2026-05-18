import os
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _parse_cors_origins(raw: str) -> List[str]:
    if not raw.strip():
        return ["http://localhost:3000"]
    return [o.strip() for o in raw.split(",") if o.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    MODEL_PATH: str = "models/web_deployment_models.pkl"
    FACE_PCA_PATH: str = "models/face_pca.pkl"
    FINGERPRINT_PCA_PATH: str = "models/fingerprint_pca.pkl"
    IRIS_PCA_PATH: str = "models/iris_pca.pkl"

    FACE_WEIGHT: float = 0.207
    FINGERPRINT_WEIGHT: float = 0.365
    IRIS_WEIGHT: float = 0.428

    IDENTIFICATION_THRESHOLD: float = 0.85
    VERIFICATION_THRESHOLD: float = 0.85

    # Layer 1 — image quality
    MIN_IMAGE_SIDE: int = 96
    MIN_LAPLACIAN_VARIANCE: float = 45.0
    MIN_CONTRAST: float = 14.0
    MIN_BRIGHTNESS: float = 35.0
    MAX_BRIGHTNESS: float = 225.0
    MIN_IMAGE_QUALITY_SCORE: float = 0.38

    # Layer 3 — liveness (face)
    MIN_HF_TEXTURE_RATIO: float = 0.18
    MIN_COLOR_VARIANCE: float = 0.08
    MIN_LIVENESS_SCORE: float = 0.32

    # Layer 2 — embedding collapse (tuned for near-uniform / centroid-like vectors)
    MIN_EMBEDDING_PEAKINESS: float = 2.0
    MAX_EMBEDDING_ENTROPY_RATIO: float = 0.985
    MAX_EFFECTIVE_DIM_RATIO: float = 0.92
    MIN_EMBEDDING_CONFIDENCE: float = 0.22

    # Layer 4 — dynamic thresholds (raise bar when quality is low)
    QUALITY_THRESHOLD_PENALTY: float = 0.12
    MIN_QUALITY_FOR_FULL_THRESHOLD: float = 0.72

    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-secret-key-change-this")
    JWT_ALGORITHM: str = "HS256"

    API_V1_PREFIX: str = "/api/v1"

    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,https://your-nextjs-app.com",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return _parse_cors_origins(self.CORS_ORIGINS)


settings = Settings()
