# app/routes/enroll.py
from fastapi import APIRouter, HTTPException
import numpy as np
from app.models.biometric import EnrollRequest, EnrollResponse
from app.services import database as db
from app.services.extraction import extract_embeddings_from_request
from app.models.inference import get_inference_model

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest) -> EnrollResponse:
    """
    Enroll a user's biometric templates (store embeddings in database)
    """
    if not db.is_configured():
        return EnrollResponse(
            success=False,
            user_id=request.user_id,
            message="DATABASE_URL is not set; configure Postgres to record enrollments.",
        )

    await db.init_schema()
    
    try:
        # Get or create user
        if request.user_id is not None:
            exists = await db.user_exists(request.user_id)
            if not exists:
                return EnrollResponse(
                    success=False,
                    user_id=request.user_id,
                    message="Provided user_id does not exist; omit user_id to create a new user.",
                )
            uid = request.user_id
        else:
            uid = await db.insert_user(request.display_name)
        
        # Extract embeddings from provided images
        model = get_inference_model()
        embeddings = await extract_embeddings_from_request(
            request.face_image,
            request.fingerprint_image,
            request.iris_image,
        )
        
        if not embeddings:
            return EnrollResponse(
                success=False,
                user_id=uid,
                message="No valid biometric images provided. Please provide at least one image.",
            )
        
        # Store embeddings in database
        async with db.pool.acquire() as conn:
            for modality, embedding in embeddings.items():
                # Convert numpy array to list for JSON storage
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                
                # Check if user already has this modality enrolled
                existing = await conn.fetchrow(
                    """
                    SELECT id FROM biometric_sample 
                    WHERE user_id = $1 AND modality = $2
                    """,
                    uid, modality
                )
                
                if existing:
                    # Update existing template
                    await conn.execute(
                        """
                        UPDATE biometric_sample 
                        SET embedding = $3, updated_at = NOW()
                        WHERE user_id = $1 AND modality = $2
                        """,
                        uid, modality, embedding_list
                    )
                else:
                    # Insert new template
                    await conn.execute(
                        """
                        INSERT INTO biometric_sample (user_id, modality, embedding, created_at)
                        VALUES ($1, $2, $3, NOW())
                        """,
                        uid, modality, embedding_list
                    )
        
        return EnrollResponse(
            success=True,
            user_id=uid,
            message=f"Enrolled {len(embeddings)} biometric modality(ies) successfully. Modalities: {list(embeddings.keys())}",
        )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/enroll/status/{user_id}")
async def get_enrollment_status(user_id: int):
    """Check which biometric modalities a user has enrolled"""
    if not db.is_configured():
        return {"error": "Database not configured"}
    
    await db.init_schema()
    
    async with db.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT modality, created_at, updated_at 
            FROM biometric_sample 
            WHERE user_id = $1
            """,
            user_id
        )
        
        modalities = [dict(row) for row in rows]
        
        return {
            "user_id": user_id,
            "enrolled_modalities": modalities,
            "has_biometric_data": len(modalities) > 0
        }
