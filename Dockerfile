FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV and scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-docker-ml.txt requirements-docker-sync.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-docker-ml.txt \
    && pip install --no-cache-dir -r requirements-docker-sync.txt \
    && pip install --no-cache-dir "gdown>=5.0.0"

COPY . .

# Download after COPY so nothing overwrites the artifact; .dockerignore excludes
# models/*.pkl from the build context (avoids truncated OneDrive/local copies).
RUN mkdir -p models \
    && gdown "1shq9S4nmUcGznqJuQdIJqq_4YM20-UZh" \
        -O models/web_deployment_models.pkl \
    && python -c "\
import pickle; \
from pathlib import Path; \
p = Path('models/web_deployment_models.pkl'); \
n = p.stat().st_size; \
assert n > 50_000, f'model file too small ({n} B), download likely failed'; \
with p.open('rb') as f: \
    d = pickle.load(f); \
assert 'gallery_data' in d and 'pca_models' in d, 'unexpected pickle layout'; \
print('models/web_deployment_models.pkl OK', n, 'bytes') \
"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
