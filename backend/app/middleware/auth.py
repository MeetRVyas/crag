from fastapi import Request, HTTPException, Depends, Query
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

        if not await redis.exists(f"session:{session_id}"):
            raise HTTPException(status_code = 401, detail = "Session Invalid")
        
        return session_id
    except JWTError as e :
        raise HTTPException(status_code = 401, detail = "Token Invalid")


async def get_session_from_query(
        token: str = Query(..., description="JWT passed as ?token= for SSE clients"),
        redis=Depends(get_redis),
) -> str:
    """
    Identical auth logic to get_current_session, but reads the JWT from a
    query parameter instead of the Authorization header.  Required for
    EventSource, which cannot set custom headers.
    """
    try:
        payload = jwt.decode(
            token,
            key=settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            issuer="crag_system",
        )
        session_id = payload.get("session_id")
        if not await redis.exists(f"session:{session_id}"):
            raise HTTPException(status_code=401, detail="Session Invalid")
        return session_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token Invalid")