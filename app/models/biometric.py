from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BiometricImages(BaseModel):
    """Base64-encoded images (optionally data URLs)."""

    face_image: Optional[str] = None
    fingerprint_image: Optional[str] = None
    iris_image: Optional[str] = None


class IdentifyRequest(BiometricImages):
    pass


class IdentifyResponse(BaseModel):
    success: bool
    user_id: Optional[int] = None
    confidence: float = 0.0
    scores: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class VerifyRequest(BiometricImages):
    user_id: str = Field(..., min_length=1, description="Prisma user id (cuid string)")


class VerifyResponse(BaseModel):
    verified: bool
    confidence: float = 0.0
    scores: Dict[str, float] = Field(default_factory=dict)
    message: str = ""
    quality: float = 0.0
    quality_by_modality: Dict[str, float] = Field(default_factory=dict)
    checks: Dict[str, Any] = Field(default_factory=dict)
    threshold_used: Optional[float] = None


class EnrollRequest(BiometricImages):
    user_id: Optional[str] = None
    display_name: Optional[str] = None


class EnrollResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    message: str = ""
    embeddings: Dict[str, List[float]] = Field(default_factory=dict)


class GalleryRecord(BaseModel):
    user_id: int
    display_name: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
