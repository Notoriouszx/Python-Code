import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.services.biometric_quality import BiometricQualityChecker
from app.services.embedding_quality import EmbeddingQualityChecker


def _b64_rgb_image(size: tuple[int, int], *, noise: bool = True) -> str:
    arr = np.random.randint(40, 200, (*size, 3), dtype=np.uint8) if noise else np.full((*size, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_tiny_flat_image_rejected_by_quality_checker():
    flat = np.full((32, 32, 3), 120, dtype=np.uint8)
    report = BiometricQualityChecker().assess(flat, "face")
    assert not report.passed
    assert report.reject_reason


def test_textured_face_image_passes_quality_checker():
    textured = np.random.randint(30, 220, (224, 224, 3), dtype=np.uint8)
    report = BiometricQualityChecker().assess(textured, "face")
    assert report.passed
    assert report.overall_score > 0.35
    assert report.liveness_score > 0.2


def test_collapsed_embedding_rejected():
    n = 512
    uniform = np.ones(n, dtype=np.float64) / np.sqrt(n)
    report = EmbeddingQualityChecker().assess(uniform, "face")
    assert not report.passed
    assert report.peakiness < 1.1
    assert report.entropy_ratio > 0.99


def test_structured_embedding_passes():
    vec = np.zeros(512, dtype=np.float64)
    vec[0] = 3.0
    vec[17] = 2.0
    vec[91] = -1.5
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    report = EmbeddingQualityChecker().assess(vec, "face")
    assert report.passed
    assert report.confidence > 0.35
