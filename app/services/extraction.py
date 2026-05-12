from typing import Dict, Optional

import numpy as np

from app.utils.image_processing import extract_embedding_from_image


async def extract_embeddings_from_request(
    face_image: Optional[str],
    fingerprint_image: Optional[str],
    iris_image: Optional[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if face_image:
        emb = await extract_embedding_from_image(face_image, "face")
        if emb is not None:
            out["face"] = emb
    if fingerprint_image:
        emb = await extract_embedding_from_image(fingerprint_image, "fingerprint")
        if emb is not None:
            out["fingerprint"] = emb
    if iris_image:
        emb = await extract_embedding_from_image(iris_image, "iris")
        if emb is not None:
            out["iris"] = emb
    return out
