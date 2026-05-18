from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.models.inference import get_inference_model
from app.routes import enroll, identify, verify
from app.services import database as db
from app.models.biometric import VerifyRequest # Import this for conversion


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

# ===== ADD THIS ENDPOINT FOR FRONTEND COMPATIBILITY =====
@app.post("/api/verify")
async def frontend_verify(request_data: dict):
    """
    Compatibility endpoint for Next.js frontend.
    Converts frontend format to your existing VerifyRequest format.
    
    Frontend sends: { user_id: "...", face_image: "...", fingerprint_image: "...", iris_image: "..." }
    Your backend expects: VerifyRequest with same field names actually!
    """
    try:
        user_id = request_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        user_id_str = str(user_id).strip()
        if not user_id_str:
            raise HTTPException(status_code=400, detail="user_id must be non-empty")

        verify_request = VerifyRequest(
            user_id=user_id_str,
            face_image=request_data.get("face_image"),
            fingerprint_image=request_data.get("fingerprint_image"),
            iris_image=request_data.get("iris_image"),
        )
        
        # Call your existing verification logic
        # Import the verify function from your routes
        from app.routes.verify import verify as verify_function
        
        # Call your existing verify function
        result = await verify_function(verify_request)
        
        # Return in format your frontend expects
        return {
            "success": True,
            "verified": result.verified,
            "confidence": result.confidence,
            "scores": result.scores,
            "message": result.message,
            "quality": result.quality,
            "quality_by_modality": result.quality_by_modality,
            "checks": result.checks,
            "threshold_used": result.threshold_used,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in frontend verify endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Also add versioned endpoint to be safe
@app.post("/api/v1/verify")
async def frontend_verify_v1(request_data: dict):
    """Versioned compatibility endpoint"""
    return await frontend_verify(request_data)

# ===== END OF ADDED ENDPOINTS =====


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
