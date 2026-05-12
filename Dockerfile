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


# Add to Dockerfile
RUN wget -O models/web_deployment_models.pkl "https://drive.google.com/drive/folders/1vUFHWfAb0kypL-mk0di8TVeCCXHMQRsd/web_deployment_models.pkl

COPY requirements.txt requirements-docker-ml.txt requirements-docker-sync.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-docker-ml.txt \
    && pip install --no-cache-dir -r requirements-docker-sync.txt

COPY . .

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
