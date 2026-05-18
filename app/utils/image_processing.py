import base64
import hashlib
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config import settings


def _decode_base64_image(base64_image: str) -> Tuple[Image.Image, bytes]:
    raw = base64_image
    if "," in raw:
        raw = raw.split(",", 1)[1]
    image_bytes = base64.b64decode(raw)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), image_bytes


def _content_seed(image_bytes: bytes, modality: str) -> int:
    digest = hashlib.sha256(image_bytes + modality.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**32)


def _embedding_dim_for_modality(modality: str) -> int:
    """Match PCA input size when a standalone PCA pickle exists; else default CNN size."""
    path_map = {
        "face": settings.FACE_PCA_PATH,
        "fingerprint": settings.FINGERPRINT_PCA_PATH,
        "iris": settings.IRIS_PCA_PATH,
    }
    import os
    import pickle

    p = path_map.get(modality, "")
    if p and os.path.isfile(p):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if hasattr(obj, "mean_"):
            return int(np.asarray(obj.mean_).ravel().shape[0])
        if isinstance(obj, dict) and "mean" in obj:
            return int(np.asarray(obj["mean"]).ravel().shape[0])
    return 512


async def extract_embedding_from_image(
    base64_image: str, modality: str
) -> Optional[np.ndarray]:
    """
    Decode image and produce a normalized feature vector.

    Replace the random projection with your trained backbone (face / finger / iris).
    Output length must match the PCA `mean` length used in `BiometricInference` for that modality,
    or already match gallery dimension if you skip PCA.
    """
    try:
        image, image_bytes = _decode_base64_image(base64_image)
        image = image.resize((224, 224))
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        dim = _embedding_dim_for_modality(modality)
        feats = cv2.resize(gray, (32, 32)).astype(np.float64).ravel()

        if feats.size < dim:
            pad = np.zeros(dim - feats.size, dtype=np.float64)
            feats = np.concatenate([feats, pad])
        elif feats.size > dim:
            feats = feats[:dim]

        # Image-dependent noise (modality-only seed made unrelated images match).
        rng = np.random.default_rng(_content_seed(image_bytes, modality))
        mix = rng.standard_normal(dim) * 0.02
        embedding = feats / (np.linalg.norm(feats) + 1e-8) + mix
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding.astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        print(f"Error extracting embedding ({modality}): {exc}")
        return None
