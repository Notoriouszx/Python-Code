from typing import Dict, Iterable, Tuple

import numpy as np

from app.config import settings


def default_weights() -> Dict[str, float]:
    return {
        "face": settings.FACE_WEIGHT,
        "fingerprint": settings.FINGERPRINT_WEIGHT,
        "iris": settings.IRIS_WEIGHT,
    }


def fuse_modal_scores(
    per_modality_scores: Dict[str, np.ndarray],
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    """
    Fuse per-modality score vectors (same length = gallery size) using nonnegative weights.
    Weights are renormalized over modalities present in `per_modality_scores`.
    """
    if not per_modality_scores:
        return np.array([])
    w = weights or default_weights()
    n = next(iter(per_modality_scores.values())).shape[0]
    fused = np.zeros(n, dtype=np.float64)
    wsum = 0.0
    for m, vec in per_modality_scores.items():
        if m not in w:
            continue
        fused += float(w[m]) * np.asarray(vec, dtype=np.float64)
        wsum += float(w[m])
    if wsum > 0:
        fused /= wsum
    return fused


def fuse_scalar_scores(
    scores: Dict[str, float],
    weights: Dict[str, float] | None = None,
) -> float:
    w = weights or default_weights()
    num = 0.0
    den = 0.0
    for m, s in scores.items():
        if m not in w:
            continue
        num += float(w[m]) * float(s)
        den += float(w[m])
    return float(num / den) if den > 0 else 0.0


def normalize_weights(pairs: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    items = [(k, float(v)) for k, v in pairs if float(v) > 0]
    s = sum(v for _, v in items) or 1.0
    return {k: v / s for k, v in items}
