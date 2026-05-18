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
RUN mkdir -p models

RUN wget -O models/web_deployment_models.pkl \
    "https://raw.githubusercontent.com/Notoriouszx/Python-Code/dev/models/web_deployment_models.pkl"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
