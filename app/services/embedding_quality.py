"""Layer 2: detect collapsed / low-information embeddings after extraction."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from app.config import settings
from app.models.quality import EmbeddingQualityReport


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class EmbeddingQualityChecker:
    """Detect generic centroid-like embeddings that inflate similarity scores."""

    def assess(self, embedding: np.ndarray, modality: str) -> EmbeddingQualityReport:
        vec = np.asarray(embedding, dtype=np.float64).ravel()
        n = vec.size
        if n == 0:
            return EmbeddingQualityReport(
                modality=modality,
                passed=False,
                confidence=0.0,
                peakiness=0.0,
                entropy_ratio=0.0,
                effective_dim_ratio=0.0,
                reject_reason="Empty embedding vector",
            )

        abs_vals = np.abs(vec)
        abs_sum = float(np.sum(abs_vals)) + 1e-12
        probs = abs_vals / abs_sum

        peakiness = float(np.max(abs_vals) / (float(np.mean(abs_vals)) + 1e-12))

        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = math.log(n) if n > 1 else 1.0
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0

        # Participation ratio: low => energy concentrated in few dimensions (good).
        # Very high with flat spectrum => collapsed generic vector.
        pr = (abs_sum**2) / (float(np.sum(abs_vals**2)) + 1e-12)
        effective_dim_ratio = pr / n

        peak_score = _clamp01(
            (peakiness - settings.MIN_EMBEDDING_PEAKINESS)
            / max(settings.MIN_EMBEDDING_PEAKINESS, 1e-8)
        )
        entropy_penalty = _clamp01(
            (entropy_ratio - settings.MAX_EMBEDDING_ENTROPY_RATIO)
            / max(1.0 - settings.MAX_EMBEDDING_ENTROPY_RATIO, 1e-8)
        )
        entropy_score = 1.0 - entropy_penalty

        eff_penalty = _clamp01(
            (effective_dim_ratio - settings.MAX_EFFECTIVE_DIM_RATIO)
            / max(1.0 - settings.MAX_EFFECTIVE_DIM_RATIO, 1e-8)
        )
        eff_score = 1.0 - eff_penalty

        confidence = _clamp01(0.45 * peak_score + 0.35 * entropy_score + 0.20 * eff_score)

        passed, reason = self._evaluate(
            peakiness=peakiness,
            entropy_ratio=entropy_ratio,
            effective_dim_ratio=effective_dim_ratio,
            confidence=confidence,
        )

        return EmbeddingQualityReport(
            modality=modality,
            passed=passed,
            confidence=confidence,
            peakiness=peakiness,
            entropy_ratio=entropy_ratio,
            effective_dim_ratio=effective_dim_ratio,
            reject_reason=reason,
        )

    def _evaluate(
        self,
        *,
        peakiness: float,
        entropy_ratio: float,
        effective_dim_ratio: float,
        confidence: float,
    ) -> Tuple[bool, str | None]:
        # Centroid-like / collapsed vectors: flat spectrum AND low peakiness together.
        collapsed = (
            peakiness < settings.MIN_EMBEDDING_PEAKINESS
            and entropy_ratio > settings.MAX_EMBEDDING_ENTROPY_RATIO
        )
        if collapsed:
            return False, "Collapsed embedding detected (generic face vector)"

        uniform_spectrum = (
            entropy_ratio > settings.MAX_EMBEDDING_ENTROPY_RATIO
            and effective_dim_ratio > settings.MAX_EFFECTIVE_DIM_RATIO
            and peakiness < settings.MIN_EMBEDDING_PEAKINESS * 1.25
        )
        if uniform_spectrum:
            return False, "Embedding lacks discriminative structure"

        if confidence < settings.MIN_EMBEDDING_CONFIDENCE:
            return False, "Embedding confidence below threshold"

        return True, None
