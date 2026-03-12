import json
import uuid
from datetime import datetime, timedelta, timezone

from jose import jwt
from cryptography.fernet import Fernet
from authlib.integrations.starlette_client import OAuth

from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.models.user import User

class Auth_Service :
    def __init__(self, redis_client) :
        self.redis = redis_client

        self.cipher = Fernet(settings.ENCRYPTION_KEY)

        self.oauth = OAuth()
        self.oauth.register(
            name = "google",
            client_id = settings.GOOGLE_CLIENT_ID,
            client_secret = settings.GOOGLE_CLIENT_SECRET,
            server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs = {"scope" : "openid email profile"}
        )
    
    async def get_login_url(self, request : Request, redirect_uri : str) :
        return await self.oauth.google.authorize_redirect(request, redirect_uri)
    
    async def handle_callback(self, request : Request, db : Session) :
        # Authorize user
        token = await self.oauth.google.authorize_access_token(request)
        
        user_info = token.get("userinfo")

         # Guard against missing data
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")
        
        email = user_info["email"].lower().strip()
        google_id = user_info["sub"]

        # Signin / Signup user
        user = db.query(User).filter(User.google_id == google_id).first()

        if not user :
            user = User(
                email = email,
                google_id = google_id,
                username = user_info.get("name", "").strip()
            )
            db.add(user)
        else :
            user.last_login = datetime.now(timezone.utc)
        
        try :
            db.commit()
            db.refresh(user)
        except IntegrityError :
            db.rollback()
            user = db.query(User).filter(User.google_id == google_id).first()

        # Session
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id" : user.id,
            "created_at" : datetime.now(timezone.utc).isoformat(),
            "api_keys" : {} # User provides the keys later
        }

        # Session to redis
        self.redis.setex(
            f"session:{session_id}",

            # Redis setex expects TTL in seconds or integer
            # but Redis client handles conversion
            timedelta(hours = settings.JWT_EXPIRY_HOURS),
            json.dumps(session_data)
        )

        # JWT data that will get encoded
        jwt_payload = {
            "sub" : str(user.id),
            "session_id" : session_id,
            "iss" : "crag_system",
            "exp" : datetime.now(timezone.utc) + timedelta(hours = settings.JWT_EXPIRY_HOURS)
        }

        # Access token the user uses everytime for requests
        access_token = jwt.encode(
            jwt_payload,
            key = settings.JWT_SECRET_KEY,
            algorithm = settings.JWT_ALGORITHM,
        )

        return access_token
    
    async def store_api_keys(self, session_id : str, api_keys : dict) :
        raw_session = self.redis.get(f"session:{session_id}")

        if not raw_session :
            raise ValueError("Session Invalid")
        
        session_data = json.loads(raw_session)

        encrypted_keys = session_data.get("api_keys", {})
        for provider, key in api_keys.items() :
            if key :
                encrypted_keys[provider.lower()] = self.cipher.encrypt(key.encode()).decode()
        session_data["api_keys"] = encrypted_keys

        ttl = self.redis.ttl(f"session:{session_id}")
        if ttl <= 0 :
            raise HTTPException(status_code = 401, detail = "Session Invalid")
        
        self.redis.setex(
            f"session:{session_id}",
            ttl,
            json.dumps(session_data)
        )
    
    async def get_api_key(self, session_id : str, provider : str) -> str :
        raw_session = self.redis.get(f"session:{session_id}")

        if not raw_session :
            return None
        
        session_data = json.loads(raw_session)

        encrypted_key = session_data.get("api_keys", {}).get(provider.lower(), "")

        if encrypted_key :
            try :
                return self.cipher.decrypt(encrypted_key.encode()).decode()
            except Exception :
                return None
        
        return None
    
    async def logout(self, session_id : str) :
        await self.redis.delete(f"session:{session_id}")