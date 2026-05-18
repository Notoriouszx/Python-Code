import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _tiny_png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _textured_png_b64(size: tuple[int, int] = (224, 224)) -> str:
    import numpy as np

    arr = np.random.randint(35, 210, (*size, 3), dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "6.1"
    assert "face" in body["modalities"]


def test_identify_accepts_payload(client: TestClient):
    b64 = _textured_png_b64()
    r = client.post(
        "/api/v1/identify",
        json={
            "face_image": b64,
            "fingerprint_image": b64,
            "iris_image": b64,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "success" in data
    assert "confidence" in data


def test_identify_rejects_empty(client: TestClient):
    r = client.post("/api/v1/identify", json={})
    assert r.status_code == 400


def test_identify_rejects_low_quality(client: TestClient):
    b64 = _tiny_png_b64()
    r = client.post("/api/v1/identify", json={"face_image": b64})
    assert r.status_code == 400


def test_verify_payload(client: TestClient):
    b64 = _textured_png_b64()
    r = client.post(
        "/api/v1/verify",
        json={"user_id": "test-user", "face_image": b64},
    )
    assert r.status_code == 200
    body = r.json()
    assert "verified" in body
    assert "quality" in body
    assert "checks" in body
