from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.services.crag_service import CRAG_Service
from app.services.document_service import DocumentService
from app.middleware.auth import get_current_session
from app.database import get_db
from app.redis_client import redis_client
from app.services.auth_service import Auth_Service
from app.models.crag import CRAGRequest, CRAGResponse

router = APIRouter(prefix="/crag", tags=["CRAG"])

@router.post("/chat", response_model = CRAGResponse)
async def chat(
    req: CRAGRequest,
    session_id: dict = Depends(get_current_session),
    db = Depends(get_db)
) -> CRAGResponse :
    # 1. Get API Keys from Redis (via Auth_Service)
    auth_service = Auth_Service(redis_client, db)
    api_keys = {}
    
    # We need to decrypt keys to pass them to CRAG_Service
    # Note: In a real app, be careful passing raw keys. 
    # Here CRAGService lives in memory for the request duration only.
    if req.provider != "ollama":
        try:
            api_keys[req.provider] = auth_service.get_api_key(session_id, req.provider)
        except:
            raise HTTPException(400, f"{req.provider} provider selected but no API key found in session.")
            
    try:
        tavily_key = auth_service.get_api_key(session_id, "tavily")
        if tavily_key:
            api_keys["tavily"] = tavily_key
    except:
        pass # Tavily is optional-ish (will warn in logs)

    # 2. Initialize Document Service to get Retriever
    doc_service = DocumentService(session_id)
    retriever = doc_service.get_retriever(provider=req.provider, api_key=api_keys.get(req.provider))
    
    if not retriever:
        raise HTTPException(400, "No index found. Please upload documents first.")

    # 3. Initialize CRAG Service
    # Set default models if not provided
    model_name = req.model
    if not model_name:
        model_name = settings.LLM_MODEL

    try:
        crag = CRAG_Service(
            session_id=session_id,
            retriever=retriever,
            llm_provider=req.provider,
            model_name=model_name,
            api_keys=api_keys
        )
        
        # 4. Run Pipeline
        result = crag.run(req.question)
        
        return {
            "answer": result["answer"],
            "verdict": result["verdict"],
            "web_search_used": len(result.get("web_docs", [])) > 0,
        }
        
    except Exception as e:
        raise HTTPException(500, f"CRAG Pipeline failed: {str(e)}")