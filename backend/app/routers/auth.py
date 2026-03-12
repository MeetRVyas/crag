from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.database import get_db
from app.redis_client import  get_redis
from app.services.auth_service import Auth_Service
from app.middleware.auth import get_current_session
from app.models.auth import APIKeyPayload, TokenResponse, MessageResponse

router = APIRouter(prefix = "/auth", tags = ["Authentication"])

def get_auth_service(redis = Depends(get_redis)) :
    return Auth_Service(redis)

@router.get("/login")
async def login(request : Request, auth_service : Auth_Service = Depends(get_auth_service)) :
    redirect_uri = request.url_for("auth_callback")
    return await auth_service.get_login_url(request, redirect_uri)

@router.get("/callback", name = "auth_callback", response_model = TokenResponse)
async def auth_callback(
    request : Request,
    auth_service : Auth_Service = Depends(get_auth_service),
    db = Depends(get_db)
) :
    try :
        token = await auth_service.handle_callback(request, db)
        return {
            "access_token" : token,
            "token_type" : "bearer"
        }
    except Exception as e :
        raise HTTPException(status_code = 400, detail = str(e))

@router.post("/set_keys", response_model = MessageResponse)
async def set_api_keys(
    keys : APIKeyPayload,
    session_id : str = Depends(get_current_session),
    auth_service : Auth_Service = Depends(get_auth_service)
) :
    keys_dict = {k : v for k, v in keys.model_dump().items() if v is not None}

    if not keys_dict :
        raise HTTPException(status_code = 400, detail = "No valid API keys provided")

    try :
        await auth_service.store_api_keys(session_id, keys_dict)
        return {"message" : "Keys encrypted and stored"}
    except ValueError as e :
        raise HTTPException(status_code = 400, detail = str(e))

@router.post("/logout", response_model = MessageResponse)
async def logout(
    session_id : str = Depends(get_current_session),
    auth_service : Auth_Service = Depends(get_auth_service)
) :
    await auth_service.logout(session_id)
    return {"message" : "Logout successful"}