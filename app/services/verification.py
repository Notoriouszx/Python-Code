# app/services/verification.py
from typing import Dict, Any
import numpy as np
from app.services import database as db
from app.config import settings


class VerificationService:
    async def verify(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify user by comparing provided biometrics against stored templates.
        Templates are read from Prisma `biometric_sample` (userId, modality, embedding).
        """
        user_id = str(request.get("user_id", "")).strip()
        face_image = request.get("face_image")
        fingerprint_image = request.get("fingerprint_image")
        iris_image = request.get("iris_image")

        if not user_id:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "error": "user_id is required",
            }

        templates = await db.get_biometric_templates(user_id)
        if not templates:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "error": f"No biometric templates found for user {user_id}. Please enroll first.",
            }

        from app.services.extraction import extract_embeddings_from_request

        request_embeddings = await extract_embeddings_from_request(
            face_image, fingerprint_image, iris_image
        )

        if not request_embeddings:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "error": "No valid biometric images provided for verification",
            }

        scores: Dict[str, float] = {}
        total_confidence = 0.0

        for modality, request_embedding in request_embeddings.items():
            if modality in templates:
                stored_embedding = templates[modality]
                similarity = self._calculate_similarity(request_embedding, stored_embedding)
                scores[modality] = similarity
                total_confidence += similarity

        overall_confidence = 0.0
        verified = False
        if scores:
            overall_confidence = total_confidence / len(scores)
            verified = overall_confidence >= settings.VERIFICATION_THRESHOLD

        await db.log_verification_attempt(user_id, verified, overall_confidence, scores)

        return {
            "verified": verified,
            "confidence": overall_confidence,
            "scores": scores,
            "message": "Verification successful" if verified else "Verification failed",
        }

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float((cosine_sim + 1) / 2)
