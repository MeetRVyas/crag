from pathlib import Path
import shutil
import asyncio
# import docker
# from docker.errors import NotFound

from app.redis_client import get_redis
from app.config import settings

CONTAINER_PREFIX = "ollama-user-"

async def cleanup_orphaned_sessions(interval_seconds: int = 3600):
    """Periodically remove session dirs whose Redis session key no longer exists."""
    from app.redis_client import redis_client

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            base = Path(settings.UPLOAD_DIR)
            if not base.exists():
                continue

            for session_dir in base.iterdir():
                if not session_dir.is_dir():
                    continue
                session_id = session_dir.name
                exists = await redis_client.exists(f"session:{session_id}")
                if not exists:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    print(f"Cleaned up orphaned session dir: {session_id}")
        except Exception as e:
            print(f"Session cleanup error (non-fatal): {e}")


async def cleanup_idle_containers(interval: int = 300):
    """
    Runs every `interval` seconds.
    Stops any Ollama container whose Redis TTL has expired
    (i.e. user hasn't made a request within OLLAMA_IDLE_TIMEOUT).
    """
    pass
    # client = docker.from_env()

    # while True:
    #     await asyncio.sleep(interval)
    #     try:
    #         redis = await get_redis()
    #         running = client.containers.list(
    #             filters={"name": CONTAINER_PREFIX}
    #         )

    #         for container in running:
    #             user_id = container.name.replace(CONTAINER_PREFIX, "")
    #             key_exists = await redis.exists(f"ollama:container:{user_id}")

    #             if not key_exists:
    #                 print(f"Idle timeout — stopping container: {container.name}")
    #                 container.stop()
    #                 container.remove()

    #     except Exception as e:
    #         print(f"Cleanup task error: {e}")