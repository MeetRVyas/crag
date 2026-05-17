from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/providers", tags=["Providers"])

@router.get("/llm")
def llm_provider():
    return {
        "provider": settings.PROVIDER,
        "model": settings.LLM_MODEL,
    }

@router.get("/embedding")
def embedding_provider():
    return {
        "provider": settings.EMBEDDING_PROVIDER,
        "model": settings.EMBEDDING_MODEL,
    }