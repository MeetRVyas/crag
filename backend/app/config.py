from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "CRAG System"
    
    DATABASE_URL: str = "sqlite:///./data/crag.db"
    
    # 2nd redis instead of local because we are using docker
    # so redis is the docker container name
    REDIS_URL: str = "redis://redis:6379/0"
    
    JWT_SECRET_KEY: str      # We will generate this
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str
    
    ENCRYPTION_KEY: str      # We will generate this

    EMBEDDING_MODEL : str = "nomic-embed-text"
    LLM_MODEL : str = "qwen2.5:1.5b" # Small but works fine

    # TODO : Implement topic inputfrom user
    TOPIC : str = "general"

    class Config:
        env_file = ".env"

settings = Settings()