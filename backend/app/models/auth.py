from pydantic import BaseModel, Field

class APIKeyPayload(BaseModel) :
    google : str | None = Field(None, min_length = 10)
    tavily : str | None = Field(None, min_length = 10)
    ollama : str | None = Field(None, min_length = 10)

class TokenResponse(BaseModel) :
    access_token : str
    token_type : str

class MessageResponse(BaseModel) :
    message : str