# app/routes/enroll.py
from fastapi import APIRouter, HTTPException
import numpy as np
import uuid

from app.models.biometric import EnrollRequest, EnrollResponse
from app.services import database as db
from app.services.extraction import extract_embeddings_with_reports
from app.models.inference import get_inference_model

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest) -> EnrollResponse:
    """
    Enroll a user's biometric templates (store embeddings in Prisma biometric_sample).
    """
    if not db.is_configured():
        return EnrollResponse(
            success=False,
            user_id=request.user_id,
            message="DATABASE_URL is not set; configure Postgres to record enrollments.",
        )

    await db.init_schema()

    try:
        get_inference_model()

        if not request.user_id or not str(request.user_id).strip():
            return EnrollResponse(
                success=False,
                user_id=None,
                message="user_id is required for enrollment.",
            )

        uid = str(request.user_id).strip()

        embeddings, reports, reject_errors = await extract_embeddings_with_reports(
            request.face_image,
            request.fingerprint_image,
            request.iris_image,
        )

        if reject_errors:
            return EnrollResponse(
                success=False,
                user_id=uid,
                message="; ".join(reject_errors),
            )

        if not embeddings:
            return EnrollResponse(
                success=False,
                user_id=uid,
                message="No valid biometric images provided. Please provide at least one image.",
            )

        embeddings_json: dict[str, list[float]] = {}
        for modality, embedding in embeddings.items():
            embedding_list = (
                embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
            )
            embeddings_json[modality] = embedding_list
            quality_score = reports.get(modality).combined_quality if modality in reports else 0.0
            await db.save_biometric_template(
                uid, modality, np.array(embedding_list), quality=quality_score
            )

        return EnrollResponse(
            success=True,
            user_id=uid,
            message=f"Enrolled {len(embeddings)} biometric modality(ies). Modalities: {list(embeddings.keys())}",
            embeddings=embeddings_json,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/enroll/status/{user_id}")
async def get_enrollment_status(user_id: str):
    """Check which biometric modalities a user has enrolled."""
    if not db.is_configured():
        return {"error": "Database not configured"}

    await db.init_schema()
    templates = await db.get_biometric_templates(user_id)
    modalities = [
        {"modality": m, "enrolled": True}
        for m in sorted(templates.keys())
    ]

    return {
        "user_id": user_id,
        "enrolled_modalities": modalities,
        "has_biometric_data": len(modalities) > 0,
    }
