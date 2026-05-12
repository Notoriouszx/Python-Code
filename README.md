# Multimodal Biometric API (FastAPI)

FastAPI service for face, fingerprint, and iris workflows: **identify**, **verify**, and **enroll** (metadata in PostgreSQL when configured).

## Requirements

- **Python 3.10+** recommended. The repo root `requirements.txt` uses flexible versions so **Python 3.14** (and similar) can install prebuilt wheels. **Docker** uses Python 3.10 with pinned PyTorch (see `requirements-docker-ml.txt`).
- Optional: **PostgreSQL** for enrollment records (`DATABASE_URL`).

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: DATABASE_URL, JWT_SECRET, CORS_ORIGINS
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Identify: `POST http://localhost:8000/api/v1/identify` with JSON `{ "face_image": "<base64>", ... }`
- Verify: `POST http://localhost:8000/api/v1/verify` with `{ "user_id": 1, "face_image": "<base64>" }`
- Enroll: `POST http://localhost:8000/api/v1/enroll` (requires `DATABASE_URL` to persist rows)

## Models

Place `web_deployment_models.pkl` under `models/` (see `models/README.md`). That file is **gitignored**. If it is missing, the service starts with a **synthetic gallery** for local development only.

Replace the logic in `app/utils/image_processing.py` with your trained backbones; today it emits deterministic-length vectors compatible with the bundled PCA layout.

## Docker & Compose

```bash
docker compose up --build
```

Compose runs Postgres and the API, mounts `./models`, and sets `DATABASE_URL`.

## Render

Use the included `Dockerfile`. Upload model artifacts to the image or mount storage so `models/web_deployment_models.pkl` exists at runtime.

## Tests

```powershell
pytest tests\test_api.py -v
```

## Layout

```
app/
  main.py              # FastAPI app + lifespan
  config.py            # Settings
  models/              # Pydantic + BiometricInference
  routes/              # identify, verify, enroll
  services/            # extraction, fusion, database
  utils/               # image_processing, normalization
models/                # Runtime pickles (ignored by git)
tests/
```

## Security

Do not commit `.env`. Copy from `.env.example` and set strong `JWT_SECRET` and database credentials.
