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


# ---------------------------
# Upload
# ---------------------------

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: dict = Depends(get_current_session),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    service = DocumentService(session_id)

    try:
        return {
            "filename": service.save_upload(file),
            "status": "uploaded"
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="File upload failed")
    
    finally :
        file.file.close()

# ---------------------------
# Process / Index
# ---------------------------

@router.post("/process")
async def process_documents(
    req: RouterProcessRequest,
    session_id: dict = Depends(get_current_session),
    redis_client: dict = Depends(get_redis),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)

    api_key: Optional[str] = None

    # If using Google embeddings, retrieve API key from session storage
    # TODO : Use API Key from session storage
    # TODO : Implement other providers
    if req.provider == "google":
        api_key = Auth_Service(redis_client).get_api_key(session_id, req.provider)

    try:
        result = service.process_documents(
            provider=req.provider,
            api_key=api_key,
            model = req.embedding_model or settings.EMBEDDING_MODEL
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Document processing failed")


# ---------------------------
# Query (Temporary Retrieval Test)
# ---------------------------

@router.post("/query")
async def query_documents(
    req: QueryRequest,
    session_id: dict = Depends(get_current_session),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)

    try:
        retriever = service.get_retriever(
            model = req.model,
            provider=req.provider
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