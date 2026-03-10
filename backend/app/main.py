from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.database import Base, engine
from app.config import settings
from app.routers import auth, documents, crag

# Create Database Tables
Base.metadata.create_all(bind = engine)

# App that Uvicorn runs
app = FastAPI(title = settings.APP_NAME)

# Required for OAuth flow
app.add_middleware(
    SessionMiddleware,
    secret_key = settings.JWT_SECRET_KEY
)

# Required for requests fronted <-> backend
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"], # Change later
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(crag.router)

@app.get("/")
def health_check() :
    return {
        "status" : "running",
        "system" : "CRAG System"
    }