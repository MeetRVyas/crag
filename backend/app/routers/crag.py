import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config import settings
from app.services.auth_service import Auth_Service
from app.services.document_service import DocumentService
from app.services.crag_service import CRAG_Service
from app.services.snapshot_service import (
    get_current_snapshot_id,
    resolve_snapshot_order,
    delete_snapshot_keys,
    delete_all_snapshot_keys,
)
from app.routers.documents import load_registry  # Redis file registry — single source of truth
from app.middleware.auth import get_current_session
from app.redis_client import get_redis
from app.models.crag import (
    CRAGRequest,
    CRAGResponse,
    SnapshotCacheResponse,
    CacheEntry,
)

router = APIRouter(prefix="/crag", tags=["CRAG"])


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

async def _get_doc_content_hashes(
    session_id: str,
    filenames: list[str],
    redis_client,
) -> list[str]:
    """
    Fetch pre-computed per-file content hashes from Redis.

    Stored at upload time under  doc:hash:{session_id}:{filename}.
    Falls back to hashing the filename string if the key is missing
    (e.g. files uploaded before this feature was deployed).
    """
    hashes = []
    for filename in filenames:
        stored = await redis_client.get(f"doc:hash:{session_id}:{filename}")
        hashes.append(
            stored if stored else hashlib.sha256(filename.encode()).hexdigest()
        )
    return hashes


def _compute_cache_key(
    session_id: str,
    snapshot_id: str,
    question: str,
    doc_content_hashes: list[str],
    llm_provider: str,
    llm_model: str,
    upper_threshold: float,
    lower_threshold: float,
) -> str:
    """
    Build the two-level hierarchical cache key.

    Level 1 (segments 2–6): snapshot + LLM identity + thresholds
    Level 2 (segment 7):    content hash of question + document set

    SCAN patterns:
        Session-scoped:   cache:answer:{session_id}:*
        Snapshot-scoped:  cache:answer:{session_id}:{snapshot_id}:*
    """
    safe_provider = llm_provider.lower().replace(":", "-")
    safe_model    = llm_model.lower().replace(":", "-").replace("/", "-")

    inner_raw = question.lower().strip() + "|" + "|".join(sorted(doc_content_hashes))
    inner_sha  = hashlib.sha256(inner_raw.encode()).hexdigest()

    return (
        f"cache:answer:{session_id}:{snapshot_id}"
        f":{safe_provider}:{safe_model}"
        f":{upper_threshold}:{lower_threshold}"
        f":{inner_sha}"
    )


async def _invalidate_snapshot_cache(
    session_id: str,
    snapshot_ids: list[str],
    redis_client,
) -> None:
    """Delete all cache entries that belong to the given snapshot IDs."""
    for sid in snapshot_ids:
        pattern = f"cache:answer:{session_id}:{sid}:*"
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                await redis_client.delete(*keys)
            if cursor == 0:
                break


# ---------------------------------------------------------------------------
# GET /crag/status  —  SSE pipeline progress stream
# ---------------------------------------------------------------------------

@router.get("/status")
async def pipeline_status(
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    """
    Stream pipeline progress events to the client via Server-Sent Events.

    Open this connection *before* POST /crag/chat.  The endpoint polls
    ``pipeline:status:{session_id}`` and forwards each event as it arrives.
    It exits on a completion marker or after a 60-second hard timeout.

    Event shapes:
        {"step": str, "message": str, "timestamp": str}
        {"complete": true, "verdict": str, "timestamp": str}
    """
    async def event_generator():
        key      = f"pipeline:status:{session_id}"
        loop     = asyncio.get_event_loop()
        deadline = loop.time() + settings._SSE_TIMEOUT
        index    = 0

        try:
            while loop.time() < deadline:
                events = await redis_client.lrange(key, index, -1)
                if events:
                    for event_json in events:
                        index += 1
                        yield f"data: {event_json}\n\n"
                        try:
                            if json.loads(event_json).get("complete"):
                                return
                        except (json.JSONDecodeError, AttributeError):
                            pass
                else:
                    yield ": keep-alive\n\n"
                    await asyncio.sleep(settings._SSE_POLL_INTERVAL)
        finally:
            await redis_client.delete(key)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# POST /crag/chat  —  pipeline with snapshot-aware cache
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=CRAGResponse)
async def chat(
    req: CRAGRequest,
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
) -> CRAGResponse:
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    # --- 1. Collect API keys ------------------------------------------------
    auth_service = Auth_Service(redis_client)
    api_keys: dict[str, str] = {}

    if req.llm_provider not in settings._LOCAL_LLM_PROVIDERS:
        try:
            key = await auth_service.get_api_key(session_id, req.llm_provider)
            if not key:
                raise ValueError("missing key")
            api_keys[req.llm_provider] = key
        except Exception:
            raise HTTPException(
                400,
                f"'{req.llm_provider}' selected as llm_provider but no API key found in session.",
            )

    if req.embedding_provider not in settings._LOCAL_EMBEDDING_PROVIDERS:
        try:
            key = await auth_service.get_api_key(session_id, req.embedding_provider)
            if not key:
                raise ValueError("missing key")
            api_keys[req.embedding_provider] = key
        except Exception:
            raise HTTPException(
                400,
                f"'{req.embedding_provider}' selected as embedding_provider but no API key found in session.",
            )

    try:
        tavily_key = await auth_service.get_api_key(session_id, "tavily")
        if tavily_key:
            api_keys["tavily"] = tavily_key
    except Exception:
        pass  # Tavily is optional — pipeline degrades gracefully without it

    # --- 2. Build retriever -------------------------------------------------
    doc_service = DocumentService(session_id)
    retriever = doc_service.get_retriever(
        model=req.embedding_model,
        provider=req.embedding_provider,
        api_key=api_keys.get(req.embedding_provider),
    )
    if not retriever:
        raise HTTPException(400, "No index found. Please upload and process documents first.")

    # --- 3. Prepare cache key components ------------------------------------
    #
    # Use the Redis file registry (the single source of truth) rather than
    # listing the upload directory from disk.  This guarantees that the
    # filenames used to build the cache key match exactly what was recorded
    # at upload time.
    registry      = await load_registry(session_id, redis_client)
    doc_filenames = [entry["filename"] for entry in registry]

    doc_content_hashes = await _get_doc_content_hashes(
        session_id, doc_filenames, redis_client
    )
    llm_model = req.llm_model or settings.LLM_MODEL
    upper     = CRAG_Service.UPPER_THRESHOLD
    lower     = CRAG_Service.LOWER_THRESHOLD

    # --- 4. Multi-snapshot cache lookup ------------------------------------
    current_snapshot_id = await get_current_snapshot_id(session_id, redis_client)

    search_order: list[str]
    if req.snapshot_ids:
        search_order = await resolve_snapshot_order(
            session_id, req.snapshot_ids, redis_client
        )
    else:
        search_order = [current_snapshot_id] if current_snapshot_id else []

    for snap_id in search_order:
        cache_key = _compute_cache_key(
            session_id=session_id,
            snapshot_id=snap_id,
            question=req.question,
            doc_content_hashes=doc_content_hashes,
            llm_provider=req.llm_provider,
            llm_model=llm_model,
            upper_threshold=upper,
            lower_threshold=lower,
        )
        cached_raw = await redis_client.get(cache_key)
        if cached_raw:
            try:
                cached_data = json.loads(cached_raw)
                return CRAGResponse(**cached_data, cached=True, snapshot_id=snap_id)
            except Exception:
                # Corrupted entry — delete and keep searching
                await redis_client.delete(cache_key)

    # --- 5. Live pipeline run -----------------------------------------------
    try:
        loop = asyncio.get_event_loop()

        crag = CRAG_Service(
            session_id=session_id,
            retriever=retriever,
            provider=req.llm_provider,
            model_name=llm_model,
            api_keys=api_keys,
            redis=redis_client,
            loop=loop,
        )

        result = await loop.run_in_executor(None, crag.run, req.question)

        # --- 6. Cache under the current snapshot ----------------------------
        payload = {
            "question":        req.question,
            "answer":          result["answer"],
            "verdict":         result["verdict"],
            "web_search_used": bool(result.get("web_docs", [])),
            "created_at":      datetime.now(timezone.utc).isoformat(),
        }

        if current_snapshot_id:
            live_cache_key = _compute_cache_key(
                session_id=session_id,
                snapshot_id=current_snapshot_id,
                question=req.question,
                doc_content_hashes=doc_content_hashes,
                llm_provider=req.llm_provider,
                llm_model=llm_model,
                upper_threshold=upper,
                lower_threshold=lower,
            )
            await redis_client.setex(live_cache_key, settings._CACHE_TTL, json.dumps(payload))

        return CRAGResponse(
            answer=payload["answer"],
            verdict=payload["verdict"],
            web_search_used=payload["web_search_used"],
            cached=False,
            snapshot_id=current_snapshot_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"CRAG Pipeline failed: {e}")


# ---------------------------------------------------------------------------
# GET /crag/cache  —  list cached answers per snapshot
# ---------------------------------------------------------------------------

@router.get("/cache", response_model=SnapshotCacheResponse)
async def list_cache(
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
    snapshot_ids: Optional[list[str]] = Query(
        default=None,
        description=(
            "Snapshot IDs to inspect. Omit to return all cached entries "
            "across every snapshot in this session."
        ),
    ),
):
    """
    List cached answers for the requested snapshot(s).

    Each entry exposes the question, answer, verdict, and the snapshot it
    belongs to so the frontend can group answers by document-set version.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    patterns = (
        [f"cache:answer:{session_id}:{sid}:*" for sid in snapshot_ids]
        if snapshot_ids
        else [f"cache:answer:{session_id}:*"]
    )

    entries: list[CacheEntry] = []

    for pattern in patterns:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                raw = await redis_client.get(key)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    # Key: cache:answer:{session_id}:{snapshot_id}:...
                    parts   = key.split(":")
                    snap_id = parts[3] if len(parts) > 3 else "unknown"
                    entries.append(
                        CacheEntry(
                            question=data.get("question", ""),
                            answer=data.get("answer", ""),
                            verdict=data.get("verdict", ""),
                            web_search_used=data.get("web_search_used", False),
                            snapshot_id=snap_id,
                            created_at=data.get("created_at", ""),
                        )
                    )
                except Exception:
                    continue
            if cursor == 0:
                break

    return SnapshotCacheResponse(entries=entries)


# ---------------------------------------------------------------------------
# DELETE /crag/cache/snapshots  —  delete cache for selected snapshots
# ---------------------------------------------------------------------------

@router.delete("/cache/snapshots")
async def delete_snapshot_cache(
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
    snapshot_ids: list[str] = Query(
        description=(
            "One or more snapshot IDs whose cached answers should be deleted. "
            "Pass snapshot_ids=all to wipe the entire session answer cache."
        ),
    ),
):
    """
    Delete cached answers for the specified snapshot(s).

    Passing ``snapshot_ids=all`` wipes the entire session answer cache and
    removes all snapshot metadata so they no longer appear in
    GET /documents/snapshots.

    This does NOT affect the live FAISS index or uploaded files.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    if "all" in snapshot_ids:
        pattern = f"cache:answer:{session_id}:*"
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                await redis_client.delete(*keys)
            if cursor == 0:
                break
        await delete_all_snapshot_keys(session_id, redis_client)
        return {"deleted": "all", "session_id": session_id}

    await _invalidate_snapshot_cache(session_id, snapshot_ids, redis_client)
    await delete_snapshot_keys(session_id, snapshot_ids, redis_client)
    return {"deleted": snapshot_ids, "session_id": session_id}
