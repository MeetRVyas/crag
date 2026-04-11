import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from app.config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _snapshot_list_key(session_id: str) -> str:
    return f"snapshots:{session_id}"


def _snapshot_meta_key(session_id: str, snapshot_id: str) -> str:
    return f"snapshot:{session_id}:{snapshot_id}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_snapshot(session_id: str, files: list[dict], redis_client) -> str:
    """
    Record the current document set as a new snapshot.

    Parameters
    ----------
    files
        List of dicts with at least ``filename`` and ``uploaded_at`` (ISO-8601).
        Pass the full file registry from _load_registry().

    Returns the new snapshot_id.
    """
    snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    meta = {
        "id":         snapshot_id,
        "created_at": now,
        "files":      json.dumps(files),
    }

    meta_key = _snapshot_meta_key(session_id, snapshot_id)
    list_key = _snapshot_list_key(session_id)

    pipe = redis_client.pipeline()
    pipe.hset(meta_key, mapping=meta)
    pipe.expire(meta_key, settings._SNAPSHOT_TTL)
    pipe.rpush(list_key, snapshot_id)
    pipe.expire(list_key, settings._SNAPSHOT_TTL)
    await pipe.execute()

    return snapshot_id


async def get_snapshot(session_id: str, snapshot_id: str, redis_client) -> Optional[dict]:
    """Return the metadata dict for one snapshot, or None if not found."""
    raw = await redis_client.hgetall(_snapshot_meta_key(session_id, snapshot_id))
    if not raw:
        return None
    return {
        "id":         raw["id"],
        "created_at": raw["created_at"],
        "files":      json.loads(raw["files"]),
    }


async def list_snapshots(session_id: str, redis_client) -> list[dict]:
    """Return all snapshots for this session, ordered oldest → newest."""
    ids = await redis_client.lrange(_snapshot_list_key(session_id), 0, -1)
    snapshots = []
    for sid in ids:
        snap = await get_snapshot(session_id, sid, redis_client)
        if snap:
            snapshots.append(snap)
    return snapshots


async def get_current_snapshot_id(session_id: str, redis_client) -> Optional[str]:
    """Return the most recent snapshot ID, or None if none exist yet."""
    return await redis_client.lindex(_snapshot_list_key(session_id), -1)


async def resolve_snapshot_order(
    session_id: str,
    snapshot_ids: list[str],
    redis_client,
) -> list[str]:
    """
    Given a list of snapshot IDs from the caller, return them sorted newest-first.

    Unknown (deleted) IDs are silently dropped.  Used to implement the
    "most recent wins" rule during multi-snapshot cache lookup.
    """
    all_ids = await redis_client.lrange(_snapshot_list_key(session_id), 0, -1)
    position = {sid: i for i, sid in enumerate(all_ids)}
    valid = [sid for sid in snapshot_ids if sid in position]
    valid.sort(key=lambda sid: position[sid], reverse=True)
    return valid


async def delete_snapshot_keys(
    session_id: str,
    snapshot_ids: list[str],
    redis_client,
) -> None:
    """
    Delete metadata and remove from the ordered list for each snapshot ID.

    Cache entries tagged to these snapshots must be deleted separately via
    the cache invalidation helper in routers/crag.py.
    """
    list_key = _snapshot_list_key(session_id)
    for sid in snapshot_ids:
        await redis_client.lrem(list_key, 0, sid)
        await redis_client.delete(_snapshot_meta_key(session_id, sid))


async def delete_all_snapshot_keys(session_id: str, redis_client) -> None:
    """
    Wipe every snapshot for this session.  Called at logout / session expiry.
    """
    all_ids = await redis_client.lrange(_snapshot_list_key(session_id), 0, -1)
    for sid in all_ids:
        await redis_client.delete(_snapshot_meta_key(session_id, sid))
    await redis_client.delete(_snapshot_list_key(session_id))
