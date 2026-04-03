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
    embedding_provider: str = Field(
        description="Embedding provider, e.g. ollama, google, huggingface",
        default=settings.EMBEDDING_PROVIDER,
    )
    llm_provider: str = Field(
        description="LLM provider, e.g. ollama, google, huggingface",
        default=settings.PROVIDER,
    )
    embedding_model: Optional[str] = Field(
        description="Embedding model name, e.g. nomic-embed-text",
        default=settings.EMBEDDING_MODEL,
    )
    llm_model: Optional[str] = Field(
        description="LLM model name, e.g. phi3:mini, gemini-1.5-flash",
        default=settings.LLM_MODEL,
    )
    # When provided, the chat endpoint looks for a cached answer inside these
    # snapshots before running the live pipeline.  Most-recent snapshot wins
    # when the same question is cached under multiple snapshot IDs.
    # When omitted (the default), only the current active snapshot is checked.
    snapshot_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of snapshot IDs to search for a cached answer. "
            "If the question is cached under multiple IDs the most recently "
            "created snapshot is used.  Omit to use the current snapshot only."
        ),
    )


class CRAGResponse(BaseModel):
    answer: str = Field(description="Answer given by LLM")
    verdict: str = Field(description="CORRECT, AMBIGUOUS, or INCORRECT")
    web_search_used: bool = Field(
        description="True when Tavily web search was triggered"
    )
    cached: bool = Field(
        description="True when this answer was served from the Redis cache",
        default=False,
    )
    snapshot_id: Optional[str] = Field(
        default=None,
        description=(
            "The snapshot ID under which this answer was cached or stored. "
            "None when the answer was not cached."
        ),
    )


# ---------------------------------------------------------------------------
# Snapshot models (used by /documents/snapshots and /crag/cache endpoints)
# ---------------------------------------------------------------------------

class SnapshotFile(BaseModel):
    filename: str
    uploaded_at: str  # ISO-8601


class Snapshot(BaseModel):
    id: str
    created_at: str   # ISO-8601
    files: list[SnapshotFile]


class SnapshotListResponse(BaseModel):
    snapshots: list[Snapshot]


class CacheEntry(BaseModel):
    question: str
    answer: str
    verdict: str
    web_search_used: bool
    snapshot_id: str
    created_at: str  # ISO-8601 stored alongside the answer payload


class SnapshotCacheResponse(BaseModel):
    """Response for GET /crag/cache — all cached answers for the requested snapshots."""
    entries: list[CacheEntry]
