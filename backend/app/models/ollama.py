from pydantic import BaseModel, Field
from typing import List

class ModelStatus(BaseModel):
    name: str = Field(description = "Name of a valid model (provided by Ollama)")
    installed: bool = Field(description = "If model is available (installed) in Ollama service or not")
    is_default: bool = Field(description = "If the model is a default LLM/Embedding model")


class ModelListResponse(BaseModel):
    models: list[ModelStatus] = Field(description = "List of ModeStatus for allowed LLM/Embedding models")


class PullRequest(BaseModel):
    model: str = Field(description = "Name of a valid model to be pulled (downloaded)")


class PullResponse(BaseModel):
    model: str = Field(description = "Name of model pulled")
    message: str = Field(description = "Status of model pull (Success/Already installed)")


class DefaultsPullResponse(BaseModel):
    responses: List[PullResponse] = Field(description = "Response after using the default pull method.\
                                        Pulls the default LLM and Embedding models.", min_length = 2, max_length = 2)


class HealthResponse(BaseModel):
    status: str =Field(pattern = "ok|unreachable")
    installed_models: list[str] = Field(description = "A list of models available in Ollama service")