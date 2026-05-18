# app/routes/verify.py
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.biometric import VerifyRequest, VerifyResponse
from app.services.verification import VerificationService

router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest) -> VerifyResponse:
    """
    Verify a user's biometrics against stored templates
    """
    # Create verification service instance
    verification_service = VerificationService()
    
    # Convert request to dict for the service
    request_dict = {
        "user_id": request.user_id,
        "face_image": request.face_image,
        "fingerprint_image": request.fingerprint_image,
        "iris_image": request.iris_image,
    }
    
    # Call verification service
    result = await verification_service.verify(request_dict)
    
    if result.get("error"):
        return VerifyResponse(
            verified=False,
            confidence=0.0,
            scores={},
            message=result["error"],
        )
    
    return VerifyResponse(
        verified=result.get("verified", False),
        confidence=result.get("confidence", 0.0),
        scores=result.get("scores", {}),
        message=result.get("message", "Verification completed"),
    )
