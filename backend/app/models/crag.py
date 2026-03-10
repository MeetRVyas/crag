from pydantic import BaseModel, Field
from typing import Optional

class Score(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reason: str = Field(description="Brief justification for the score")

class KeepOrDrop(BaseModel):
    keep: bool = Field(description="True if sentence is directly relevant")

class WebQuery(BaseModel):
    query: str = Field(description="Optimized web search query")

class CRAGRequest(BaseModel):
    question: str
    provider: str = "ollama"       # "ollama" or "google"
    model: Optional[str] = None    # e.g., "phi3:medium", "gemini-1.5-flash"

class CRAGResponse(BaseModel) :
    answer : str
    verdict : str
    web_search_use : bool