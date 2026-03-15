from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from jose import jwt, JWTError

from app.config import settings
from app.redis_client import get_redis

security = HTTPBearer()

async def get_current_session(
        credentials : HTTPAuthorizationCredentials = Depends(security),
        redis = Depends(get_redis)
        ) :
    token = credentials.credentials

    try :
        payload = jwt.decode(
            token,
            key = settings.JWT_SECRET_KEY,
            algorithms = [settings.JWT_ALGORITHM],
            issuer = "crag_system"
        )

        session_id = payload.get("session_id")

        if not await redis.exists(f"session:{session_id}") :
            raise HTTPException(status_code = 401, detail = "Session Invalid")
        
        return session_id
    except JWTError as e :
        raise HTTPException(status_code = 401, detail = "Token Invalid")