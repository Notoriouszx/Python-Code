from typing import Dict, Optional, Tuple

import numpy as np

from app.models.quality import ModalityExtractionResult
from app.services.biometric_pipeline import extract_embeddings_guarded
from app.utils.image_processing import extract_embedding_from_image


async def extract_embeddings_from_request(
    face_image: Optional[str],
    fingerprint_image: Optional[str],
    iris_image: Optional[str],
    *,
    use_guards: bool = True,
) -> Dict[str, np.ndarray]:
    embeddings, _, _ = await extract_embeddings_with_reports(
        face_image, fingerprint_image, iris_image, use_guards=use_guards
    )
    return embeddings


async def extract_embeddings_with_reports(
    face_image: Optional[str],
    fingerprint_image: Optional[str],
    iris_image: Optional[str],
    *,
    use_guards: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, ModalityExtractionResult], list[str]]:
    if use_guards:
        return await extract_embeddings_guarded(face_image, fingerprint_image, iris_image)

    out: Dict[str, np.ndarray] = {}
    reports: Dict[str, ModalityExtractionResult] = {}
    errors: list[str] = []

    for modality, payload in [
        ("face", face_image),
        ("fingerprint", fingerprint_image),
        ("iris", iris_image),
    ]:
        if not payload:
            continue
        emb = await extract_embedding_from_image(payload, modality)
        if emb is None:
            errors.append(f"{modality.capitalize()}: extraction failed")
            reports[modality] = ModalityExtractionResult(
                rejected=True, reject_reason="extraction failed"
            )
        else:
            out[modality] = emb
            reports[modality] = ModalityExtractionResult(embedding=emb, rejected=False)

    return out, reports, errors
