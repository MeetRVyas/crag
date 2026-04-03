import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Optional

from app.models.documents import RouterProcessRequest, QueryRequest
from app.middleware.auth import get_current_session
from app.redis_client import get_redis
from app.services.document_service import DocumentService
from app.services.auth_service import Auth_Service
from app.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Cache invalidation helper
# ---------------------------------------------------------------------------

async def _invalidate_session_cache(session_id: str, redis_client) -> None:
    """
    Delete every cached CRAG answer that belongs to this session.

    Cache keys follow the pattern:  cache:answer:{session_id}:{sha256}
    We scan for that prefix and bulk-delete any matches.  At this scale
    (1-5 docs per user) the scan will rarely return more than a handful
    of keys, so this is perfectly efficient.
    """
    pattern = f"cache:answer:{session_id}:*"
    cursor = 0
    while True:
        cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
        if keys:
            await redis_client.delete(*keys)
        if cursor == 0:
            break


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    service = DocumentService(session_id)

    try:
        filename = service.save_upload(file)

        # A new document changes the document set, so any cached answers
        # computed against the old set are now stale.
        await _invalidate_session_cache(session_id, redis_client)

        return {
            "filename": filename,
            "status": "uploaded",
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")
    
    finally :
        file.file.close()


# ---------------------------------------------------------------------------
# Process / Index
# ---------------------------------------------------------------------------

@router.post("/process")
async def process_documents(
    req: RouterProcessRequest,
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)

    api_key: Optional[str] = None

    # If using Google embeddings, retrieve API key from session storage
    # TODO : Use API Key from session storage
    # TODO : Implement other providers
    if req.provider == "google":
        api_key = await Auth_Service(redis_client).get_api_key(session_id, req.provider)

    try:
        result = service.process_documents(
            provider = req.provider,
            api_key = api_key,
            model = req.embedding_model or settings.EMBEDDING_MODEL
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Document processing failed")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{filename}")
async def delete_document(
    filename: str,
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)

    try:
        service.delete_document(filename)

        # Removing a document invalidates all cached answers for this session —
        # the document set has changed so previous answers may no longer be valid.
        await _invalidate_session_cache(session_id, redis_client)

        return {"filename": filename, "status": "deleted"}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="File deletion failed")


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("/")
async def list_documents(
    session_id: str = Depends(get_current_session),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)
    return {"documents": service.list_documents()}


# ---------------------------------------------------------------------------
# Query (Temporary Retrieval Test)
# ---------------------------------------------------------------------------

@router.post("/query")
async def query_documents(
    req: QueryRequest,
    session_id: str = Depends(get_current_session),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)

    try:
        retriever = service.get_retriever(
            model = req.model,
            provider = req.provider,
        )
        if not retriever:
            raise HTTPException(
                status_code=400,
                detail="Index not found. Process documents first."
            )

        docs = retriever.invoke(req.query)

        # TODO : Implement a model {page_content : str, metadata : dict}
        # TODO : Implement a model for response of this function {count : int, results : Model}

        def _clean_metadata(meta: dict) -> dict:
            cleaned = {}
            for k, v in meta.items():
                # Skip internal/sensitive fields
                if k in ("id", "source"):
                    continue
                # Normalize numpy scalars
                if hasattr(v, "item"):
                    v = v.item()
                cleaned[k] = v
            # Expose only the filename, not the full path
            if "source" in meta:
                cleaned["filename"] = os.path.basename(meta["source"])
            return cleaned

        return {
            "count" : len(docs),
            "results": [
                {
                    "content": d.page_content,
                    "metadata": _clean_metadata(d.metadata)
                }
                for d in docs
            ]
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Query failed")