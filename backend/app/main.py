import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.database import Base, engine
from app.config import settings
from app.routers import auth, documents, crag, ollama
from app.services.ollama_service import get_ollama_service

# Create Database Tables
Base.metadata.create_all(bind = engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    print("Preloading default Ollama models...")
    try:
        get_ollama_service().preload_defaults()
    except Exception as e:
        print(f"Ollama preload failed (non-fatal): {e}")
    yield


async def _wait_for_ollama(timeout: int = 120, interval: int = 3) -> bool:
    """Poll Ollama's health endpoint until ready or timed out."""
    url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    deadline = asyncio.get_event_loop().time() + timeout

    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url, timeout=3.0)
                if r.status_code == 200:
                    print("Ollama is ready.")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            print("Waiting for Ollama to be ready...")
            await asyncio.sleep(interval)

    print(f"Ollama did not become ready within {timeout}s.")
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    ready = await _wait_for_ollama(timeout=120, interval=3)
    if ready:
        print("Preloading default Ollama models...")
        try:
            # ollama client is sync — offload to thread so we don't block the event loop
            await asyncio.get_event_loop().run_in_executor(
                None, get_ollama_service().preload_defaults
            )
        except Exception as e:
            print(f"Ollama preload failed (non-fatal): {e}")
    else:
        print("Skipping model preload — Ollama not reachable.")

    yield

# App that Uvicorn runs
app = FastAPI(title = settings.APP_NAME, lifespan = lifespan)

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
app.include_router(ollama.router)

@app.get("/")
def health_check() :
    return {
        "status" : "running",
        "system" : "CRAG System"
    }

@app.get("/models")
def list_allowed_models():
    """Expose allowed model lists to the frontend."""
    return {
        "llm": sorted(settings.OLLAMA_ALLOWED_LLM_MODELS),
        "embedding": sorted(settings.OLLAMA_ALLOWED_EMBEDDING_MODELS),
        "defaults": {
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
        },
    }