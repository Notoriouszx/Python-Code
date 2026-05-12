from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.models.inference import get_inference_model
from app.routes import enroll, identify, verify
from app.services import database as db


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Biometric Service...")
    get_inference_model()
    print("Models loaded successfully")
    await db.init_schema()
    yield
    await db.close_pool()


app = FastAPI(
    title="Biometric Authentication Service",
    description="Multimodal biometric identification (Face + Fingerprint + Iris)",
    version="6.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(identify.router, prefix=settings.API_V1_PREFIX, tags=["identification"])
app.include_router(verify.router, prefix=settings.API_V1_PREFIX, tags=["verification"])
app.include_router(enroll.router, prefix=settings.API_V1_PREFIX, tags=["enrollment"])


@app.get("/")
async def root():
    return {
        "service": "Multimodal Biometric Authentication",
        "version": "6.1",
        "status": "running",
        "modalities": ["face", "fingerprint", "iris"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
