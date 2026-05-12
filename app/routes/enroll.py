from fastapi import APIRouter, HTTPException

from app.models.biometric import EnrollRequest, EnrollResponse
from app.services import database as db

router = APIRouter()


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest) -> EnrollResponse:
    """
    Persist enrollment metadata in PostgreSQL when `DATABASE_URL` is set.

    Updating the on-disk `web_deployment_models.pkl` gallery must be done by your
    training/export pipeline; this endpoint does not mutate the pickle file.
    """
    if not db.is_configured():
        return EnrollResponse(
            success=False,
            user_id=request.user_id,
            message="DATABASE_URL is not set; configure Postgres to record enrollments.",
        )

    await db.init_schema()
    try:
        if request.user_id is not None:
            exists = await db.user_exists(request.user_id)
            if not exists:
                return EnrollResponse(
                    success=False,
                    user_id=request.user_id,
                    message="Provided user_id does not exist; omit user_id to create a new user.",
                )
            uid = request.user_id
        else:
            uid = await db.insert_user(request.display_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EnrollResponse(
        success=True,
        user_id=uid,
        message="Enrollment recorded. Re-export gallery pickle to activate matching for new users.",
    )
