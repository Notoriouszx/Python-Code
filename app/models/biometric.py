from typing import Dict, List, Optional

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
    user_id: int = Field(..., ge=1, description="1-based user id matching gallery labels")


class VerifyResponse(BaseModel):
    verified: bool
    confidence: float = 0.0
    scores: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class EnrollRequest(BiometricImages):
    user_id: Optional[int] = None
    display_name: Optional[str] = None


class EnrollResponse(BaseModel):
    success: bool
    user_id: Optional[int] = None
    message: str = ""


class GalleryRecord(BaseModel):
    user_id: int
    display_name: Optional[str] = None
    modalities: List[str] = Field(default_factory=list)
