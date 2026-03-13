from redis.asyncio import from_url
from app.config import settings

redis_client = from_url(
    settings.REDIS_URL,
    decode_responses = True
)

def get_redis() :
    return redis_client