"""
documents.py
------------
Handles PDF upload, indexing, deletion, and document listing.

Snapshot lifecycle
------------------
Every upload or delete creates a new snapshot — an immutable record of the
exact file set present at that moment.  We no longer wipe cache entries on
document changes; instead, users can optionally query cached answers from
any previous snapshot.

The file registry (the per-session list of files with upload timestamps)
lives in Redis under  files:{session_id}  as a JSON string.  It is the
source of truth for snapshot creation and for the file content-hash lookups
that the cache-key function uses.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Optional

from app.models.documents import RouterProcessRequest, QueryRequest
from app.models.crag import SnapshotListResponse, Snapshot, SnapshotFile
from app.middleware.auth import get_current_session
from app.redis_client import get_redis
from app.services.document_service import DocumentService
from app.services.auth_service import Auth_Service
from app.services.snapshot_service import (
    create_snapshot,
    list_snapshots,
)
from app.config import settings

router = APIRouter(prefix="/documents", tags=["documents"])

# TTL for file-content hashes and the file registry (24 h — same as session)
_DOC_HASH_TTL   = 86_400
_REGISTRY_TTL   = 86_400


# ---------------------------------------------------------------------------
# File content hash helpers
# ---------------------------------------------------------------------------

def _hash_file(filepath: str) -> str:
    """SHA-256 of the file's raw bytes, read in 64 KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def _store_doc_hash(session_id: str, filename: str, filepath: str, redis_client) -> None:
    """
    Compute and persist the file's content hash.

    Key:  doc:hash:{session_id}:{filename}
    Called once at upload time — the CRAG cache-key function retrieves it
    cheaply on every subsequent query without re-reading the file.
    """
    content_hash = _hash_file(filepath)
    await redis_client.setex(
        f"doc:hash:{session_id}:{filename}",
        _DOC_HASH_TTL,
        content_hash,
    )


async def _delete_doc_hash(session_id: str, filename: str, redis_client) -> None:
    """Remove the stored content hash when the document is deleted."""
    await redis_client.delete(f"doc:hash:{session_id}:{filename}")


# ---------------------------------------------------------------------------
# File registry helpers
#
# The registry is a JSON list of {filename, uploaded_at} dicts stored in
# Redis.  It is the single source of truth for what files are currently
# present in the session, and it is snapshotted on every change.
# ---------------------------------------------------------------------------

_REGISTRY_KEY = "files:{session_id}"


async def _load_registry(session_id: str, redis_client) -> list[dict]:
    raw = await redis_client.get(f"files:{session_id}")
    return json.loads(raw) if raw else []


async def _save_registry(session_id: str, files: list[dict], redis_client) -> None:
    await redis_client.setex(
        f"files:{session_id}",
        _REGISTRY_TTL,
        json.dumps(files),
    )


async def _add_to_registry(session_id: str, filename: str, redis_client) -> list[dict]:
    """Append a file entry and persist.  Returns the updated registry."""
    files = await _load_registry(session_id, redis_client)
    # Avoid duplicates if the same filename is re-uploaded
    files = [f for f in files if f["filename"] != filename]
    files.append({
        "filename":    filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    await _save_registry(session_id, files, redis_client)
    return files


async def _remove_from_registry(session_id: str, filename: str, redis_client) -> list[dict]:
    """Remove a file entry and persist.  Returns the updated registry."""
    files = await _load_registry(session_id, redis_client)
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

        # 1. Store content hash so cache-key lookups are O(1) Redis GETs
        filepath = os.path.join(service.upload_dir, filename)
        await _store_doc_hash(session_id, filename, filepath, redis_client)

        # 2. Update the file registry
        updated_files = await _add_to_registry(session_id, filename, redis_client)

        # 3. Snapshot the new document set — we intentionally do NOT delete
        #    existing cache entries; they remain accessible via their snapshot IDs.
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

        # 1. Remove content hash — file no longer exists
        await _delete_doc_hash(session_id, filename, redis_client)

        # 2. Update registry
        updated_files = await _remove_from_registry(session_id, filename, redis_client)

        # 3. Snapshot the reduced document set — cache entries for the previous
        #    snapshot remain intact and are still retrievable on request.
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

    files = await _load_registry(session_id, redis_client)
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
    moment.  The frontend uses this to render the timeline of sections the
    user can choose to retrieve cached answers from.
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
                    SnapshotFile(
                        filename=f["filename"],
                        uploaded_at=f["uploaded_at"],
                    )
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
        retriever = service.get_retriever(
            model=req.model,
            provider=req.provider,
        )
        if not retriever:
            raise HTTPException(
                status_code=400,
                detail="Index not found. Process documents first.",
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
                {
                    "content":  d.page_content,
                    "metadata": _clean_metadata(d.metadata),
                }
                for d in docs
            ],
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Query failed")
