from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.biometric import VerifyRequest, VerifyResponse
from app.models.inference import get_inference_model
from app.services.extraction import extract_embeddings_from_request

router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest) -> VerifyResponse:
    model = get_inference_model()
    embeddings = await extract_embeddings_from_request(
        request.face_image,
        request.fingerprint_image,
        request.iris_image,
    )
    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid biometric images provided")

    result = model.verify(embeddings, request.user_id)
    if result.get("error"):
        return VerifyResponse(
            verified=False,
            confidence=0.0,
            scores={},
            message=str(result["error"]),
        )

    scores = {k: float(v) for k, v in result.get("scores", {}).items()}
    verified = bool(result.get("verified"))
    conf = float(result.get("confidence", 0.0))
    msg = "Verified" if verified else f"Below threshold ({settings.VERIFICATION_THRESHOLD})"
    return VerifyResponse(verified=verified, confidence=conf, scores=scores, message=msg)
