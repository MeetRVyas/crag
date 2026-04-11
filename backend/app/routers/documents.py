import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from app.models.documents import RouterProcessRequest, QueryRequest
from app.models.crag import SnapshotListResponse, Snapshot, SnapshotFile
from app.middleware.auth import get_current_session
from app.redis_client import get_redis
from app.services.document_service import DocumentService
from app.services.auth_service import Auth_Service
from app.services.snapshot_service import create_snapshot, list_snapshots
from app.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# File content hash helpers
# ---------------------------------------------------------------------------

def _hash_file(filepath: str) -> str:
    """SHA-256 of the file's raw bytes, streamed in 64 KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def _store_doc_hash(
    session_id: str, filename: str, filepath: str, redis_client
) -> None:
    """
    Compute and persist the file's content hash at upload time.

    Key:  doc:hash:{session_id}:{filename}

    The CRAG cache-key function retrieves this cheaply (O(1) Redis GET) on
    every subsequent query instead of re-reading the file from disk.
    """
    content_hash = _hash_file(filepath)
    await redis_client.setex(
        f"doc:hash:{session_id}:{filename}",
        settings._DOC_HASH_TTL,
        content_hash,
    )


async def _delete_doc_hash(session_id: str, filename: str, redis_client) -> None:
    """Remove the stored content hash when the document is deleted."""
    await redis_client.delete(f"doc:hash:{session_id}:{filename}")


# ---------------------------------------------------------------------------
# File registry helpers
# ---------------------------------------------------------------------------

async def load_registry(session_id: str, redis_client) -> list[dict]:
    """Return the current file registry for this session (may be empty)."""
    raw = await redis_client.get(f"files:{session_id}")
    return json.loads(raw) if raw else []


async def _save_registry(session_id: str, files: list[dict], redis_client) -> None:
    await redis_client.setex(
        f"files:{session_id}",
        settings._REGISTRY_TTL,
        json.dumps(files),
    )


async def _add_to_registry(
    session_id: str, filename: str, redis_client
) -> list[dict]:
    """Append a file entry (de-duplicating by filename) and persist."""
    files = await load_registry(session_id, redis_client)
    files = [f for f in files if f["filename"] != filename]
    files.append(
        {"filename": filename, "uploaded_at": datetime.now(timezone.utc).isoformat()}
    )
    await _save_registry(session_id, files, redis_client)
    return files


async def _remove_from_registry(
    session_id: str, filename: str, redis_client
) -> list[dict]:
    """Remove a file entry and persist."""
    files = await load_registry(session_id, redis_client)
    files = [f for f in files if f["filename"] != filename]
    await _save_registry(session_id, files, redis_client)
    return files


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

        # 1. Persist content hash for O(1) cache-key lookups later
        filepath = os.path.join(service.upload_dir, filename)
        await _store_doc_hash(session_id, filename, filepath, redis_client)

        # 2. Update the file registry
        updated_files = await _add_to_registry(session_id, filename, redis_client)

        # 3. Snapshot the new document set.
        #    Existing cache entries are deliberately NOT deleted — they remain
        #    accessible by their snapshot IDs.
        snapshot_id = await create_snapshot(session_id, updated_files, redis_client)

        return {
            "filename":    filename,
            "status":      "uploaded",
            "snapshot_id": snapshot_id,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")

    finally:
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

    if req.provider == "google":
        api_key = await Auth_Service(redis_client).get_api_key(session_id, req.provider)

    try:
        result = service.process_documents(
            provider=req.provider,
            api_key=api_key,
            model=req.embedding_model or settings.EMBEDDING_MODEL,
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

        # 1. Drop the content hash — the file no longer exists on disk
        await _delete_doc_hash(session_id, filename, redis_client)

        # 2. Update the registry
        updated_files = await _remove_from_registry(session_id, filename, redis_client)

        # 3. Snapshot the reduced document set.
        #    Cache entries for the previous snapshot remain intact.
        snapshot_id = await create_snapshot(session_id, updated_files, redis_client)

        return {
            "filename":    filename,
            "status":      "deleted",
            "snapshot_id": snapshot_id,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="File deletion failed")


# ---------------------------------------------------------------------------
# List current documents
# ---------------------------------------------------------------------------

@router.get("/")
async def list_documents(
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    files = await load_registry(session_id, redis_client)
    return {"documents": files}


# ---------------------------------------------------------------------------
# List snapshots
# ---------------------------------------------------------------------------

@router.get("/snapshots", response_model=SnapshotListResponse)
async def list_document_snapshots(
    session_id: str = Depends(get_current_session),
    redis_client=Depends(get_redis),
):
    """
    Return all document-set snapshots for this session, ordered oldest first.

    Each snapshot represents the state of the document set at a specific
    moment.  The frontend uses this to render a timeline the user can select
    cached answers from.
    """
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    raw_snapshots = await list_snapshots(session_id, redis_client)
    return SnapshotListResponse(
        snapshots=[
            Snapshot(
                id=s["id"],
                created_at=s["created_at"],
                files=[
                    SnapshotFile(filename=f["filename"], uploaded_at=f["uploaded_at"])
                    for f in s["files"]
                ],
            )
            for s in raw_snapshots
        ]
    )


# ---------------------------------------------------------------------------
# Query (retrieval test endpoint)
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
        retriever = service.get_retriever(model=req.model, provider=req.provider)
        if not retriever:
            raise HTTPException(
                status_code=400, detail="Index not found. Process documents first."
            )

        docs = retriever.invoke(req.query)

        def _clean_metadata(meta: dict) -> dict:
            cleaned = {}
            for k, v in meta.items():
                if k in ("id", "source"):
                    continue
                if hasattr(v, "item"):
                    v = v.item()
                cleaned[k] = v
            if "source" in meta:
                cleaned["filename"] = os.path.basename(meta["source"])
            return cleaned

        return {
            "count":   len(docs),
            "results": [
                {"content": d.page_content, "metadata": _clean_metadata(d.metadata)}
                for d in docs
            ],
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Query failed")
