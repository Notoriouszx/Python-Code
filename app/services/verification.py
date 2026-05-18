# app/services/verification.py
from typing import Any, Dict, Optional

import numpy as np

from app.config import settings
from app.models.quality import ModalityExtractionResult
from app.services import database as db
from app.services.extraction import extract_embeddings_with_reports


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
            return self._fail("user_id is required")

        templates = await db.get_biometric_templates(user_id)
        if not templates:
            return self._fail(
                f"No biometric templates found for user {user_id}. Please enroll first."
            )

        request_embeddings, reports, reject_errors = await extract_embeddings_with_reports(
            face_image, fingerprint_image, iris_image
        )

        quality_summary = self._summarize_quality(reports)

        if reject_errors:
            return {
                "verified": False,
                "confidence": 0.0,
                "scores": {},
                "quality": quality_summary.get("overall", 0.0),
                "quality_by_modality": quality_summary.get("by_modality", {}),
                "checks": quality_summary.get("checks", {}),
                "error": "; ".join(reject_errors),
                "message": "Biometric quality checks failed",
            }

        if not request_embeddings:
            return self._fail(
                "No valid biometric images provided for verification",
                quality=quality_summary,
            )

        scores: Dict[str, float] = {}
        total_confidence = 0.0
        effective_thresholds: Dict[str, float] = {}

        for modality, request_embedding in request_embeddings.items():
            if modality not in templates:
                continue
            stored_embedding = templates[modality]
            similarity = self._calculate_similarity(request_embedding, stored_embedding)
            scores[modality] = similarity
            total_confidence += similarity

            modality_quality = quality_summary.get("by_modality", {}).get(modality, 0.5)
            effective_thresholds[modality] = self._dynamic_threshold(modality_quality)

        overall_confidence = 0.0
        verified = False
        if scores:
            overall_confidence = total_confidence / len(scores)
            min_quality = min(
                quality_summary.get("by_modality", {}).get(m, 0.5) for m in scores
            )
            threshold = self._dynamic_threshold(min_quality)
            verified = all(
                scores[m] >= effective_thresholds.get(m, threshold) for m in scores
            ) and overall_confidence >= threshold

        await db.log_verification_attempt(user_id, verified, overall_confidence, scores)

        return {
            "verified": verified,
            "confidence": overall_confidence,
            "scores": scores,
            "quality": quality_summary.get("overall", 0.0),
            "quality_by_modality": quality_summary.get("by_modality", {}),
            "checks": quality_summary.get("checks", {}),
            "threshold_used": (
                self._dynamic_threshold(
                    min(quality_summary.get("by_modality", {}).values(), default=0.5)
                )
                if scores
                else settings.VERIFICATION_THRESHOLD
            ),
            "message": "Verification successful" if verified else "Verification failed",
        }

    def _dynamic_threshold(self, quality_score: float) -> float:
        """Raise verification bar when capture/embedding quality is low."""
        base = settings.VERIFICATION_THRESHOLD
        if quality_score >= settings.MIN_QUALITY_FOR_FULL_THRESHOLD:
            return base
        gap = settings.MIN_QUALITY_FOR_FULL_THRESHOLD - quality_score
        penalty = gap * settings.QUALITY_THRESHOLD_PENALTY
        return float(min(0.99, base + penalty))

    def _summarize_quality(
        self, reports: Dict[str, ModalityExtractionResult]
    ) -> Dict[str, Any]:
        by_modality: Dict[str, float] = {}
        checks: Dict[str, Any] = {}
        for modality, report in reports.items():
            by_modality[modality] = report.combined_quality
            checks[modality] = {
                "rejected": report.rejected,
                "reject_reason": report.reject_reason,
                "image": report.image_quality.to_dict() if report.image_quality else None,
                "embedding": (
                    report.embedding_quality.to_dict() if report.embedding_quality else None
                ),
            }
        overall = float(sum(by_modality.values()) / len(by_modality)) if by_modality else 0.0
        return {"overall": overall, "by_modality": by_modality, "checks": checks}

    def _fail(
        self, message: str, quality: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        q = quality or {}
        return {
            "verified": False,
            "confidence": 0.0,
            "scores": {},
            "quality": q.get("overall", 0.0),
            "quality_by_modality": q.get("by_modality", {}),
            "checks": q.get("checks", {}),
            "error": message,
            "message": message,
        }

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float((cosine_sim + 1) / 2)
