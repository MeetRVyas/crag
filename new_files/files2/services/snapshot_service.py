"""
snapshot_service.py
-------------------
Manages document-set snapshots in Redis.

A snapshot is an immutable record of the exact set of files present in a
session at a specific point in time.  A new snapshot is created whenever
the document set changes (upload or delete).  Cache entries are tagged to
the snapshot they were computed under so that users can optionally retrieve
answers from previous document-set versions.

Redis layout
------------
snapshots:{session_id}
    Redis List of snapshot IDs, ordered oldest → newest (RPUSH appends).
    Lets callers retrieve the chronological snapshot history cheaply.

snapshot:{session_id}:{snapshot_id}
    Redis Hash with fields:
        id          – same as snapshot_id
        created_at  – ISO-8601 UTC timestamp
        files       – JSON array of {filename, uploaded_at} objects

All keys share the session TTL: they are wiped automatically when the
session expires, and they are bulk-deleted on explicit logout.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

# TTL for all snapshot keys.  Should match or exceed the session TTL.
_SNAPSHOT_TTL = 86_400  # 24 hours


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
        List of dicts, each with at least ``filename`` and ``uploaded_at``
        (ISO-8601).  Pass the full file registry from _get_file_registry().

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
    list_key  = _snapshot_list_key(session_id)

    # Store metadata hash and append to the ordered list atomically (pipeline)
    pipe = redis_client.pipeline()
    pipe.hset(meta_key, mapping=meta)
    pipe.expire(meta_key, _SNAPSHOT_TTL)
    pipe.rpush(list_key, snapshot_id)
    pipe.expire(list_key, _SNAPSHOT_TTL)
    await pipe.execute()

    return snapshot_id


async def get_snapshot(session_id: str, snapshot_id: str, redis_client) -> Optional[dict]:
    """
    Return the raw metadata dict for one snapshot, or None if not found.
    """
    raw = await redis_client.hgetall(_snapshot_meta_key(session_id, snapshot_id))
    if not raw:
        return None
    return {
        "id":         raw["id"],
        "created_at": raw["created_at"],
        "files":      json.loads(raw["files"]),
    }


async def list_snapshots(session_id: str, redis_client) -> list[dict]:
    """
    Return all snapshots for this session, ordered oldest → newest.
    Each entry is the same dict shape as get_snapshot().
    """
    ids = await redis_client.lrange(_snapshot_list_key(session_id), 0, -1)
    snapshots = []
    for sid in ids:
        snap = await get_snapshot(session_id, sid, redis_client)
        if snap:
            snapshots.append(snap)
    return snapshots


async def get_current_snapshot_id(session_id: str, redis_client) -> Optional[str]:
    """
    Return the most recent snapshot ID, or None if none exist yet.
    """
    sid = await redis_client.lindex(_snapshot_list_key(session_id), -1)
    return sid  # returns None when list is empty


async def resolve_snapshot_order(
    session_id: str,
    snapshot_ids: list[str],
    redis_client,
) -> list[str]:
    """
    Given a list of snapshot IDs supplied by the caller, return them sorted
    newest-first using their position in the global ordered list.

    Unknown IDs (deleted snapshots) are silently dropped.

    This is used to implement the "most recent wins" rule: when the same
    question is cached under multiple snapshot IDs, the caller iterates this
    list and returns the first cache hit it finds.
    """
    all_ids = await redis_client.lrange(_snapshot_list_key(session_id), 0, -1)
    # Build position index: snapshot_id → index (0 = oldest)
    position = {sid: i for i, sid in enumerate(all_ids)}

    # Filter to only valid IDs, sort descending by position (newest first)
    valid = [sid for sid in snapshot_ids if sid in position]
    valid.sort(key=lambda sid: position[sid], reverse=True)
    return valid


async def delete_snapshot_keys(
    session_id: str,
    snapshot_ids: list[str],
    redis_client,
) -> None:
    """
    Delete the metadata hash for each given snapshot ID and remove it from
    the ordered list.  Cache entries tagged to these snapshots must be
    deleted separately via the cache invalidation helper in crag.py.
    """
    list_key = _snapshot_list_key(session_id)
    for sid in snapshot_ids:
        # Remove from the ordered list (LREM removes all occurrences)
        await redis_client.lrem(list_key, 0, sid)
        # Delete metadata hash
        await redis_client.delete(_snapshot_meta_key(session_id, sid))


async def delete_all_snapshot_keys(session_id: str, redis_client) -> None:
    """
    Wipe every snapshot for this session.  Called at session end / logout.
    No need to also delete the ordered list — it expires with the session.
    """
    all_ids = await redis_client.lrange(_snapshot_list_key(session_id), 0, -1)
    for sid in all_ids:
        await redis_client.delete(_snapshot_meta_key(session_id, sid))
    await redis_client.delete(_snapshot_list_key(session_id))
