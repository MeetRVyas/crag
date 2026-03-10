import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional

from app.models.documents import RouterProcessRequest, QueryRequest
from app.middleware.auth import get_current_session
from app.services.document_service import DocumentService

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
):
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    service = DocumentService(session_id)

    api_key: Optional[str] = None

    # If using Google embeddings, retrieve API key from session storage
    # TODO : Use API Key from session storage
    # TODO : Implement other providers
    if req.provider == "google":
        # Replace with secure retrieval from Redis or encrypted store
        raise HTTPException(status_code=400, detail="Google provider not configured")

    try:
        result = service.process_documents(
            provider=req.provider,
            api_key=api_key
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
        return {
            "count" : len(docs),
            "results": [
                {
                    "content": d.page_content,
                    "metadata": d.metadata
                }
                for d in docs
            ]
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Query failed")