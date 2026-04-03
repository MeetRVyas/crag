import asyncio
import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

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

# Answer cache TTL: 24 hours
_CACHE_TTL = 86_400

# SSE poll timeout: 60 seconds
_SSE_TIMEOUT = 60

# SSE poll interval: 200 ms
_SSE_POLL_INTERVAL = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(filepath: str) -> str:
    """SHA-256 of the file's raw bytes."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _compute_cache_key(session_id: str, question: str, doc_filenames: list[str], llm_provider : str, model : str) -> str:
    """
    Deterministic Redis cache key scoped to a session.

    The hash encodes the semantic content (question + document set), while the
    session_id prefix makes per-session invalidation via SCAN cheap and safe.

    Key format: cache:answer:{session_id}:{sha256_hex}
    """

    normalised_question = question.lower().strip()
    sorted_hashes = "|".join(sorted(_hash_file(filepath) for filepath in doc_filenames))
    raw = f"{normalised_question}|{sorted_hashes}"
    sha = hashlib.sha256(raw.encode()).hexdigest()
    
#     semantic = question.lower().strip() + "|" + ",".join(sorted(doc_filenames))
#     sha = hashlib.sha256(semantic.encode()).hexdigest()
    return f"cache:answer:{session_id}:{llm_provider}|{model}:{sha}"


# ---------------------------------------------------------------------------
# GET /crag/status  —  real-time pipeline progress via Server-Sent Events
# ---------------------------------------------------------------------------

@router.get("/status")
async def pipeline_status(
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    """
    Stream pipeline progress events to the client using Server-Sent Events.

    The frontend opens this connection *before* submitting POST /crag/chat.
    The endpoint polls the Redis list `pipeline:status:{session_id}` and
    forwards each event as it arrives.  It exits on:
      - a completion marker  ({"complete": true, ...})
      - a 60-second hard timeout

    Each event payload is one of:
      {"step": str, "message": str, "timestamp": str}
      {"complete": true, "verdict": str, "timestamp": str}
    """
    async def event_generator():
        key = f"pipeline:status:{session_id}"
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _SSE_TIMEOUT
        index = 0  # next unread position in the Redis list

        try:
            while loop.time() < deadline:
                # Read all events that have arrived since we last checked
                events = await redis_client.lrange(key, index, -1)

                if events:
                    for event_json in events:
                        index += 1
                        # SSE wire format: "data: <json>\n\n"
                        yield f"data: {event_json}\n\n"

                        # Stop streaming once we see the completion marker
                        try:
                            if json.loads(event_json).get("complete"):
                                return
                        except (json.JSONDecodeError, AttributeError):
                            pass
                else:
                    # No new events yet — yield a keep-alive comment so the
                    # connection doesn't time out on proxies, then wait
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(_SSE_POLL_INTERVAL)

        finally:
            # Clean up the status list regardless of how we exit
            await redis_client.delete(key)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx output buffering
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /crag/chat  —  run the CRAG pipeline (unchanged contract, adds caching)
# ---------------------------------------------------------------------------

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
        raise HTTPException(400, "No index found. Please upload documents first.")

    # 3. Answer cache check
    #    Key encodes: session_id (for invalidation scoping) + question + doc set.
    doc_filenames = [d["filename"] for d in doc_service.list_documents()]
    cache_key = _compute_cache_key(session_id, req.question, doc_filenames)

    cached_raw = await redis_client.get(cache_key)
    if cached_raw:
        try:
            cached_data = json.loads(cached_raw)
            return CRAGResponse(**cached_data, cached=True)
        except Exception:
            # Corrupted cache entry — treat as a miss and re-run the pipeline
            await redis_client.delete(cache_key)

    # 4. Initialize CRAG Service
    #    Pass the running event loop so the sync pipeline nodes can schedule
    #    Redis status writes back onto it via run_coroutine_threadsafe.
    try:
        loop = asyncio.get_event_loop()

        crag = CRAG_Service(
            session_id = session_id,
            retriever = retriever,
            provider = req.llm_provider,
            model_name = req.llm_model or settings.LLM_MODEL,
            api_keys = api_keys,
            redis = redis_client,
            loop = loop,
        )

        # 5. Run the synchronous pipeline in a thread pool so the event loop
        #    remains free to serve the concurrent SSE /status request.
        result = await loop.run_in_executor(None, crag.run, req.question)

        # 6. Cache the result for 24 hours
        payload = {
            "answer": result["answer"],
            "verdict": result["verdict"],
            "web_search_used": bool(result.get("web_docs", [])),
        }
        await redis_client.setex(cache_key, _CACHE_TTL, json.dumps(payload))

        return CRAGResponse(**payload, cached=False)
    
    except HTTPException :
        raise
    except Exception as e:
        raise HTTPException(500, f"CRAG Pipeline failed: {e}")