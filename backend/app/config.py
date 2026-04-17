from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "CRAG System"
    
    DATABASE_URL: str
    
    # 2nd redis instead of local because we are using docker
    # so redis is the docker container name
    REDIS_URL: str
    
    JWT_SECRET_KEY: str      # We will generate this
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    OLLAMA_BASE_URL : str

    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str
    
    ENCRYPTION_KEY: str      # We will generate this

    EMBEDDING_PROVIDER : str
    PROVIDER : str

    EMBEDDING_MODEL : str
    LLM_MODEL : str

    HF_TOKEN : str

    EMBEDDING_FALLBACK : str = "ollama"

    MAX_LLM_RETRIES_ON_API_LIMITS : int = 10

    OLLAMA_ALLOWED_EMBEDDING_MODELS : list[str]
    OLLAMA_ALLOWED_LLM_MODELS : list[str]

    _LOCAL_LLM_PROVIDERS : set[str] = {"ollama", "huggingface_local"}
    _LOCAL_EMBEDDING_PROVIDERS : set[str] = {"ollama", "huggingface"}

    _CACHE_TTL : int = 86_400  # 24 hours
    _SSE_TIMEOUT : int = 60      # seconds
    _SSE_POLL_INTERVAL : int = 0.2     # seconds
    _SNAPSHOT_TTL : int = 86_400  # 24 hours

    _DOC_HASH_TTL : int = 86_400  # 24 h — matches session TTL
    _REGISTRY_TTL : int = 86_400

    # TODO : Implement topic inputfrom user
    TOPIC : str = "general"

    @field_validator("LLM_MODEL")
    @classmethod
    def validate_default_llm(cls, v, info):
        if info.data.get("PROVIDER", "") == "ollama" :
            allowed = info.data.get("OLLAMA_ALLOWED_LLM_MODELS", [])
            # Only validate if allowlist is already resolved (ordering not guaranteed in v2)
            if allowed and v not in allowed:
                raise ValueError(
                    f"LLM_MODEL '{v}' must be in OLLAMA_ALLOWED_LLM_MODELS: {allowed}"
                )
        return v

    @field_validator("EMBEDDING_MODEL")
    @classmethod
    def validate_default_embedding(cls, v, info):
        if info.data.get("EMBEDDING_PROVIDER", "") == "ollama" :
            allowed = info.data.get("OLLAMA_ALLOWED_EMBEDDING_MODELS", [])
            # Only validate if allowlist is already resolved (ordering not guaranteed in v2)
            if allowed and v not in allowed:
                raise ValueError(
                    f"EMBEDDING_MODEL '{v}' must be in OLLAMA_ALLOWED_EMBEDDING_MODELS: {allowed}"
                )
        return v

    class Config:
        env_file = ".env"

settings = Settings()