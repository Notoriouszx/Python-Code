# app/services/verification.py
from typing import Dict, Any, Optional
import numpy as np
from app.services import database as db
from app.models.inference import get_inference_model
from app.config import settings
import json

class VerificationService:
    async def verify(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify user by comparing provided biometrics against stored templates
        """
        user_id = request.get("user_id")
        face_image = request.get("face_image")
        fingerprint_image = request.get("fingerprint_image")
        iris_image = request.get("iris_image")
        
        # Get stored templates from database
        async with db.pool.acquire() as conn:
            stored_templates = await conn.fetch(
                """
                SELECT modality, embedding 
                FROM biometric_sample 
                WHERE user_id = $1
                """,
                user_id
            )
        
        if not stored_templates:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "error": f"No biometric templates found for user {user_id}. Please enroll first."
            }
        
        # Convert stored templates to dict
        templates = {row['modality']: np.array(row['embedding']) for row in stored_templates}
        
        # Extract embeddings from request
        from app.services.extraction import extract_embeddings_from_request
        request_embeddings = await extract_embeddings_from_request(
            face_image, fingerprint_image, iris_image
        )
        
        if not request_embeddings:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "error": "No valid biometric images provided for verification"
            }
        
        # Compare each modality
        scores = {}
        verified = False
        total_confidence = 0.0
        
        for modality, request_embedding in request_embeddings.items():
            if modality in templates:
                # Calculate similarity score (cosine similarity or Euclidean distance)
                stored_embedding = templates[modality]
                similarity = self._calculate_similarity(request_embedding, stored_embedding)
                scores[modality] = similarity
                total_confidence += similarity
        
        # Calculate overall confidence (average)
        if scores:
            overall_confidence = total_confidence / len(scores)
            # Use threshold from config
            from app.config import settings
            verified = overall_confidence >= settings.VERIFICATION_THRESHOLD
        
        # Log verification attempt
        await self._log_attempt(user_id, verified, overall_confidence, scores)
        
        return {
            "verified": verified,
            "confidence": overall_confidence,
            "scores": scores,
            "message": "Verification successful" if verified else "Verification failed"
        }
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        # Convert from [-1, 1] to [0, 1]
        return float((cosine_sim + 1) / 2)
    
    async def _log_attempt(self, user_id: int, verified: bool, confidence: float, scores: Dict):
        """Log verification attempt to database"""
        try:
            async with db.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO biometric_attempt (user_id, verified, confidence, scores, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    """,
                    user_id, verified, confidence, scores
                )
        except Exception as e:
            print(f"Failed to log attempt: {e}")
