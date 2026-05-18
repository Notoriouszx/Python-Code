"""End-to-end guarded extraction: quality → embedding → collapse detection."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.models.quality import ModalityExtractionResult
from app.services.biometric_quality import BiometricQualityChecker
from app.services.embedding_quality import EmbeddingQualityChecker
from app.utils.image_processing import (
    _decode_base64_image,
    _embedding_dim_for_modality,
    _content_seed,
)


_image_checker = BiometricQualityChecker()
_embedding_checker = EmbeddingQualityChecker()


def _raw_embedding_from_bgr(bgr: np.ndarray, modality: str, image_bytes: bytes) -> np.ndarray:
    """Produce normalized feature vector (same algorithm as legacy image_processing)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize((224, 224))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    dim = _embedding_dim_for_modality(modality)
    feats = cv2.resize(gray, (32, 32)).astype(np.float64).ravel()

    if feats.size < dim:
        pad = np.zeros(dim - feats.size, dtype=np.float64)
        feats = np.concatenate([feats, pad])
    elif feats.size > dim:
        feats = feats[:dim]

    rng = np.random.default_rng(_content_seed(image_bytes, modality))
    mix = rng.standard_normal(dim) * 0.02
    embedding = feats / (np.linalg.norm(feats) + 1e-8) + mix
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding.astype(np.float64)


async def extract_modality_guarded(
    base64_image: str, modality: str
) -> ModalityExtractionResult:
    try:
        image, image_bytes = _decode_base64_image(base64_image)
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        img_report = _image_checker.assess(bgr, modality)
        if not img_report.passed:
            return ModalityExtractionResult(
                image_quality=img_report,
                rejected=True,
                reject_reason=img_report.reject_reason or "Image quality check failed",
            )

        embedding = _raw_embedding_from_bgr(bgr, modality, image_bytes)
        emb_report = _embedding_checker.assess(embedding, modality)
        if not emb_report.passed:
            return ModalityExtractionResult(
                embedding=None,
                image_quality=img_report,
                embedding_quality=emb_report,
                rejected=True,
                reject_reason=emb_report.reject_reason or "Embedding quality check failed",
            )

        return ModalityExtractionResult(
            embedding=embedding,
            image_quality=img_report,
            embedding_quality=emb_report,
            rejected=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error in guarded extraction ({modality}): {exc}")
        return ModalityExtractionResult(
            rejected=True,
            reject_reason=f"Extraction error: {exc}",
        )


async def extract_embeddings_guarded(
    face_image: Optional[str],
    fingerprint_image: Optional[str],
    iris_image: Optional[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, ModalityExtractionResult], list[str]]:
    """
    Returns (accepted embeddings, per-modality reports, rejection messages).
    """
    pairs = [
        ("face", face_image),
        ("fingerprint", fingerprint_image),
        ("iris", iris_image),
    ]
    embeddings: Dict[str, np.ndarray] = {}
    reports: Dict[str, ModalityExtractionResult] = {}
    errors: list[str] = []

    for modality, payload in pairs:
        if not payload:
            continue
        result = await extract_modality_guarded(payload, modality)
        reports[modality] = result
        if result.rejected or result.embedding is None:
            label = modality.capitalize()
            errors.append(f"{label}: {result.reject_reason or 'rejected'}")
        else:
            embeddings[modality] = result.embedding

    return embeddings, reports, errors
