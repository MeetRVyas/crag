from pydantic import BaseModel, Field
from typing import Optional
from app.config import settings

class Score(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reason: str = Field(description="Brief justification for the score")

class KeepOrDrop(BaseModel):
    keep: bool = Field(description="True if sentence is directly relevant")

class CRAGRequest(BaseModel):
    question: str = Field(description="Original user query", min_length=3, max_length=2000)
    embedding_provider: str = Field(description="Embedding provider, for example, ollama, google, huggingface, etc", default = settings.EMBEDDING_PROVIDER)
    llm_provider: str = Field(description="LLM provider, for example, ollama, google, huggingface, etc", default = settings.PROVIDER)
    embedding_model: Optional[str] = Field(description="Embedding model name, for example, nomic-embed-text, embeddinggemma:300m, etc", default = settings.EMBEDDING_MODEL)
    llm_model: Optional[str] = Field(description="LLM model name, for example, phi3:mini, gemini-1.5-flash, etc", default = settings.LLM_MODEL)

class CRAGResponse(BaseModel) :
    answer : str = Field(description="Answer given by LLM")
    verdict : str = Field(description="Whether the documents retrieved were AMBIGUOUS, CORRECT or INCORRECT")
    web_search_used : bool = Field(description="Whether web search was used or not. Used if the verdict was AMBIGUOUS or INCORRECT")
    cached: bool = Field(description="Whether this answer was served from the Redis cache", default=False)