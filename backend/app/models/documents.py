from pydantic import BaseModel, Field

class ProcessResult(BaseModel):
    status: str
    chunks: int
    documents: int
    parent_chunks: int
    corrupted_docs: int

class RouterProcessRequest(BaseModel):
    provider: str = Field(default="ollama", pattern="^(ollama|google)$")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    provider: str = Field(default="ollama", pattern="^(ollama|google)$")
