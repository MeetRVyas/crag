from pydantic import BaseModel, Field, field_validator

_API_KEY_PROVIDERS = {"groq", "anthropic", "google", "huggingface", "tavily"}

class APIKeyPayload(BaseModel) :
    keys: dict[str, str] = Field(
        description="Map of provider name to API key",
        examples=[{"groq": "gsk_...", "tavily": "tvly-..."}]
    )

    @field_validator("keys")
    @classmethod
    def validate_keys(cls, v: dict) -> dict:
        unknown = set(v) - _API_KEY_PROVIDERS
        if unknown:
            raise ValueError(
                f"Unknown provider(s): {unknown}. "
                f"Valid providers: {_API_KEY_PROVIDERS}"
            )
        short = [k for k, val in v.items() if len(val) < 10]
        if short:
            raise ValueError(f"API key too short for: {short}")
        if not v:
            raise ValueError("At least one key must be provided")
        return {k.lower(): val for k, val in v.items()}

class TokenResponse(BaseModel) :
    access_token : str
    token_type : str

class MessageResponse(BaseModel) :
    message : str