from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.services.auth_service import Auth_Service
from app.services.document_service import DocumentService
from app.services.crag_service import CRAG_Service
from app.middleware.auth import get_current_session
from app.redis_client import get_redis
from app.models.crag import CRAGRequest, CRAGResponse

router = APIRouter(prefix="/crag", tags=["CRAG"])

# Providers that run locally and need no API key fetched from the session
_LOCAL_LLM_PROVIDERS = {"ollama", "huggingface_local"}
_LOCAL_EMBEDDING_PROVIDERS = {"ollama", "huggingface"}

@router.post("/chat", response_model = CRAGResponse)
async def chat(
    req: CRAGRequest,
    session_id: dict = Depends(get_current_session),
    redis_client = Depends(get_redis),
) -> CRAGResponse :
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    # 1. Get API Keys from Redis (via Auth_Service)
    auth_service = Auth_Service(redis_client)
    api_keys : dict[str, str] = {}
    
    # Local providers (ollama, huggingface_local) need no key.
    # All cloud providers (groq, anthropic, huggingface_api, google) do.
    if req.llm_provider not in _LOCAL_LLM_PROVIDERS:
        try:
            key = await auth_service.get_api_key(session_id, req.llm_provider)
            if not key:
                raise Exception
            api_keys[req.llm_provider] = key
        except:
            raise HTTPException(400, f"{req.llm_provider} llm_provider selected but no API key found in session.")
    
    # Local providers (ollama, huggingface) need no key.
    # groq/anthropic are not valid embedding providers — the factory falls back
    # to Ollama automatically, so they also need no key here.
    # Only google requires a key for embeddings.
    if req.embedding_provider not in _LOCAL_EMBEDDING_PROVIDERS:
        try:
            key = await auth_service.get_api_key(session_id, req.embedding_provider)
            if not key:
                raise Exception
            api_keys[req.embedding_provider] = key
        except:
            raise HTTPException(400, f"{req.embedding_provider} embedding_provider selected but no API key found in session.")
    
    try:
        tavily_key = await auth_service.get_api_key(session_id, "tavily")
        if tavily_key:
            api_keys["tavily"] = tavily_key
    except:
        pass # Tavily is optional-ish (will warn in logs)

    # 2. Initialize Document Service to get Retriever
    doc_service = DocumentService(session_id)
    retriever = doc_service.get_retriever(
        model = req.embedding_model,
        provider = req.embedding_provider,
        api_key = api_keys.get(req.embedding_provider)
    )
    
    if not retriever:
        raise HTTPException(
            400,
            "No index found. Please upload documents first."
        )

    # 3. Initialize CRAG Service
    try:
        crag = CRAG_Service(
            session_id = session_id,
            retriever = retriever,
            provider = req.llm_provider,
            model_name = req.llm_model or settings.LLM_MODEL,
            api_keys = api_keys
        )
        
        # 4. Run Pipeline
        result = crag.run(req.question)
        
        return {
            "answer": result["answer"],
            "verdict": result["verdict"],
            "web_search_used": bool(result.get("web_docs", [])),
        }
    
    except HTTPException :
        raise
    except Exception as e:
        raise HTTPException(500, f"CRAG Pipeline failed: {e}")