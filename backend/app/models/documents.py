from pydantic import BaseModel, Field
from app.config import settings

class ProcessResult(BaseModel):
    status: str
    chunks: int
    documents: int
    parent_chunks: int
    corrupted_docs: int

class RouterProcessRequest(BaseModel):
    provider: str = Field(default=settings.EMBEDDING_PROVIDER, pattern="^(ollama|google)$")
    embedding_model: str = Field(default=settings.EMBEDDING_MODEL)


class QueryRequest(BaseModel):
    model: str = Field(default=settings.EMBEDDING_MODEL)
    query: str = Field(..., min_length=3, max_length=2000)
    provider: str = Field(default=settings.EMBEDDING_PROVIDER)