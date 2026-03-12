import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.middleware.auth import get_current_session
from app.services.ollama_service import get_ollama_service
from app.models.ollama import ModelStatus, ModelListResponse, PullRequest, PullResponse, HealthResponse, DefaultsPullResponse

router = APIRouter(prefix="/ollama", tags=["Ollama"])


def _build_model_list(allowed: list[str], default: str) -> ModelListResponse:
    svc = get_ollama_service()
    installed = svc.list_installed()
    return ModelListResponse(
        models=[
            ModelStatus(
                name=m,
                installed=m in installed or f"{m}:latest" in installed,
                is_default=(m == default),
            )
            for m in allowed
        ]
    )


# ---------------------------
# Health
# ---------------------------

@router.get("/health", response_model=HealthResponse)
async def ollama_health(
    session_id: str = Depends(get_current_session),
):
    """Check whether Ollama is reachable and list all installed models."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    svc = get_ollama_service()
    installed = svc.list_installed()

    if installed is None:
        # list_installed returns empty set on error — check connectivity separately
        raise HTTPException(503, "Ollama is unreachable.")

    return HealthResponse(
        status="ok",
        installed_models=sorted(installed),
    )


# ---------------------------
# LLM Models
# ---------------------------

@router.get("/models/llm", response_model=ModelListResponse)
def list_llm_models(
    session_id: str = Depends(get_current_session),
):
    """List allowed LLM models with their install status."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    return _build_model_list(
        allowed=settings.OLLAMA_ALLOWED_LLM_MODELS,
        default=settings.LLM_MODEL,
    )


@router.post("/models/llm/pull", response_model=PullResponse)
async def pull_llm_model(
    req: PullRequest,
    session_id: str = Depends(get_current_session),
):
    """
    Validate and pull an LLM model.
    Rejects models not on the allowlist (Case C).
    No-ops if already installed (Case A).
    Pulls if allowed but missing (Case B).
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    if req.model not in settings.OLLAMA_ALLOWED_LLM_MODELS:
        raise HTTPException(
            400,
            f"Model '{req.model}' is not allowed. "
            f"Allowed: {sorted(settings.OLLAMA_ALLOWED_LLM_MODELS)}",
        )

    svc = get_ollama_service()

    if svc.is_installed(req.model):
        return PullResponse(model=req.model, message="Already installed.")

    # Pull is sync and slow — offload to thread
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: svc.pull(req.model)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Pull failed: {e}")

    return PullResponse(model=req.model, message="Pulled successfully.")


# ---------------------------
# Embedding Models
# ---------------------------

@router.get("/models/embedding", response_model=ModelListResponse)
def list_embedding_models(
    session_id: str = Depends(get_current_session),
):
    """List allowed embedding models with their install status."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    return _build_model_list(
        allowed=settings.OLLAMA_ALLOWED_EMBEDDING_MODELS,
        default=settings.EMBEDDING_MODEL,
    )


@router.post("/models/embedding/pull", response_model=PullResponse)
async def pull_embedding_model(
    req: PullRequest,
    session_id: str = Depends(get_current_session),
):
    """
    Validate and pull an embedding model.
    Same allowlist logic as LLM pull.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    if req.model not in settings.OLLAMA_ALLOWED_EMBEDDING_MODELS:
        raise HTTPException(
            400,
            f"Model '{req.model}' is not allowed. "
            f"Allowed: {sorted(settings.OLLAMA_ALLOWED_EMBEDDING_MODELS)}",
        )

    svc = get_ollama_service()

    if svc.is_installed(req.model):
        return PullResponse(model=req.model, message="Already installed.")

    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: svc.pull(req.model)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Pull failed: {e}")

    return PullResponse(model=req.model, message="Pulled successfully.")


# ---------------------------
# Default Models (1 LLM and 1 Embedding model)
# ---------------------------

@router.get("/models/defaults", response_model=ModelListResponse)
def list_default_models(
    session_id: str = Depends(get_current_session),
):
    """List default LLM and Embedding models with their install status."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    svc = get_ollama_service()
    installed = svc.list_installed()
    return ModelListResponse(
        models=[
            ModelStatus(
                name = m,
                installed = m in installed or f"{m}:latest" in installed,
                is_default = True,
            )
            for m in [settings.LLM_MODEL, settings.EMBEDDING_MODEL]
        ]
    )


@router.post("/models/defaults/pull", response_model=DefaultsPullResponse)
async def pull_default_model(
    session_id: str = Depends(get_current_session),
):
    """Validate and pull an default LLM and Embedding models."""
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    svc = get_ollama_service()

    responses = []

    for model in [settings.LLM_MODEL, settings.EMBEDDING_MODEL] :
        if svc.is_installed(model) :
            responses.append(PullResponse(model=model, message="Already installed."))

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: svc.pull(model)
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(503, f"Pull failed: {e}")

        responses.append(PullResponse(model=model, message="Pulled successfully."))
    return DefaultsPullResponse(responses = responses)