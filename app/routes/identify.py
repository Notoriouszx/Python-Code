from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.biometric import IdentifyRequest, IdentifyResponse
from app.models.inference import get_inference_model
from app.services.extraction import extract_embeddings_from_request

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponse)
async def identify(request: IdentifyRequest) -> IdentifyResponse:
    model = get_inference_model()
    embeddings = await extract_embeddings_from_request(
        request.face_image,
        request.fingerprint_image,
        request.iris_image,
    )
    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid biometric images provided")

    result = model.identify(embeddings)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=str(result["error"]))

    conf = float(result["confidence"])
    scores = {k: float(v) for k, v in result.get("scores", {}).items()}

    if conf >= settings.IDENTIFICATION_THRESHOLD:
        return IdentifyResponse(
            success=True,
            user_id=int(result["user_id"]) if result.get("user_id") is not None else None,
            confidence=conf,
            scores=scores,
            message="User identified successfully",
        )
    return IdentifyResponse(
        success=False,
        user_id=None,
        confidence=conf,
        scores=scores,
        message=f"Low confidence: {conf:.2f}",
    )
