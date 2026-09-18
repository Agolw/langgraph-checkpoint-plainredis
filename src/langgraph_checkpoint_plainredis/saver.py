"""Async LangGraph checkpoint saver backed by **plain Redis** (no modules required).

Why this exists
---------------
The official ``langgraph-checkpoint-redis`` requires the **RedisJSON** and
**RediSearch** modules (bundled with Redis 8.0+, or shipped by the Redis Stack
image). Redis deployments that cannot load modules -- internal/legacy servers,
managed instances exposing only the core command set, Redis 5/6/7 -- cannot use
it at all.

This saver only uses core Redis data structures (strings, hashes, sorted sets),
so it runs on Redis 5.0+ with **no modules**.

Key layout
----------
All keys start with ``{prefix}:`` (``prefix`` defaults to ``lg``)::

    {prefix}:cp:{thread_id}:{checkpoint_ns}:{checkpoint_id}   checkpoint payload (string)
    {prefix}:blob:{thread_id}:{ns}:{channel}:{version}        channel value       (string)
    {prefix}:wr:{thread_id}:{ns}:{checkpoint_id}              pending writes      (hash)
    {prefix}:idx:{thread_id}:{ns}                             timeline            (sorted set)
    {prefix}:threads                                          (thread, ns) registry (hash)
    {prefix}:seq                                              monotonic counter   (string)

Channel values are stored **per (channel, version)** as separate blobs, exactly
like the in-memory/sqlite savers do, so unchanged channels are not duplicated
across checkpoints. Blobs are re-hydrated on read; a channel whose blob is
missing or marked ``empty`` is simply absent from ``channel_values``.

Semantics
---------
* ``alist`` yields checkpoints newest-first by **write recency** (the timeline
  score). LangGraph's default checkpoint ids are monotonic UUID6 strings, so
  this matches "descending by checkpoint id" in practice.
* ``before`` is exclusive: it yields checkpoints older than the given one.
* ``ttl`` (seconds) is applied to checkpoint/blob/timeline/write keys on every
  write. Set it to bound memory on long-lived Redis instances; LangGraph will
  then raise its normal "no checkpoint" errors for expired threads.
* Async only: the synchronous ``get_tuple`` / ``put`` / ``put_writes`` / ``list``
  methods raise ``NotImplementedError``. Use ``ainvoke`` / ``astream`` (or the
  equivalent ``aget_state*`` calls).
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

import redis.asyncio as aioredis
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

__all__ = ["AsyncRedisSaver", "DEFAULT_PREFIX"]

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "lg"

# Redis glob metacharacters that must be escaped before they are used in SCAN patterns
_GLOB_SPECIAL = "*?[]\\"
# Tag used for channels that were written as "no value" (mirrors InMemorySaver's ("empty", b""))
_EMPTY_TAG = "empty"


def _escape_glob(value: str) -> str:
    """Escape glob metacharacters so e.g. a thread id like ``a*[b]`` cannot widen a pattern."""
    out: list[str] = []
    for ch in value:
        if ch in _GLOB_SPECIAL:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _pack_typed(tag: str, data: bytes) -> bytes:
    """Pack a ``serde.dumps_typed`` pair into a single binary-safe Redis string.

    Redis strings are binary safe, but a tiny JSON wrapper keeps the payload
    inspectable with ``redis-cli`` at the cost of base64 expansion.
    """
    return json.dumps(
        {"t": tag, "d": base64.b64encode(data).decode("ascii")}
    ).encode("utf-8")


def _unpack_typed(raw: bytes) -> tuple[str, bytes]:
    """Inverse of :func:`_pack_typed`."""
    obj = json.loads(raw)
    return obj["t"], base64.b64decode(obj["d"])


def _write_sort_key(field: str) -> tuple[str, int]:
    """Sort key for write fields (``"{task_id}:{inner_idx}"``) -> (task_id, idx)."""
    task_id, _, idx = field.rpartition(":")
    try:
        return task_id, int(idx)
    except ValueError:
        return field, 0


class AsyncRedisSaver(BaseCheckpointSaver):
    """LangGraph checkpointer on plain Redis (Redis 5.0+, no RedisJSON/RediSearch).

    Args:
        client: Ready-made ``redis.asyncio.Redis`` instance. When given, every
            other connection argument is ignored.
        url: Redis URL, e.g. ``redis://:password@127.0.0.1:6379/0``.
        host/port/password/db: connection parts used when neither ``client``
            nor ``url`` is given.
        protocol: RESP protocol version. Defaults to ``2`` on purpose: RESP3
            (``HELLO``) is unavailable before Redis 6, and RESP2 is enough here.
        ttl: optional key TTL in seconds (``None`` = keys live until deleted).
        prefix: key namespace, defaults to ``"lg"`` (see module docstring).
        serde: custom LangGraph serializer; defaults to the base
            ``JsonPlusSerializer``.
        socket_timeout / socket_connect_timeout: passed to the Redis client.

    Example:
        >>> saver = AsyncRedisSaver(url="redis://127.0.0.1:6379/0", ttl=7 * 24 * 3600)
        >>> graph = builder.compile(checkpointer=saver)
        >>> await graph.ainvoke(state, {"configurable": {"thread_id": "t-1"}})
    """

    def __init__(
        self,
        client: Optional[aioredis.Redis] = None,
        *,
        url: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        protocol: int = 2,
        ttl: Optional[int] = None,
        prefix: str = DEFAULT_PREFIX,
        serde: Any = None,
        socket_timeout: Optional[float] = 5.0,
        socket_connect_timeout: Optional[float] = 5.0,
    ) -> None:
        super().__init__(serde=serde)
        if client is not None:
            self.client = client
        elif url:
            self.client = aioredis.from_url(
                url,
                protocol=protocol,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
            )
        else:
            self.client = aioredis.Redis(
                host=host,
                port=port,
                password=password or None,
                db=db,
                protocol=protocol,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=False,
            )
        self.ttl = int(ttl) if ttl else None
        self.prefix = (prefix or DEFAULT_PREFIX).rstrip(":")
        if not self.prefix:
            raise ValueError("prefix must be a non-empty string")

    # ------------------------------------------------------------------ keys

    def _key_checkpoint(self, thread_id: str, ns: str, cid: str) -> str:
        return f"{self.prefix}:cp:{thread_id}:{ns}:{cid}"

    def _key_blob(self, thread_id: str, ns: str, channel: str, version: Any) -> str:
        return f"{self.prefix}:blob:{thread_id}:{ns}:{channel}:{version}"

    def _key_writes(self, thread_id: str, ns: str, cid: str) -> str:
        return f"{self.prefix}:wr:{thread_id}:{ns}:{cid}"

    def _key_index(self, thread_id: str, ns: str) -> str:
        return f"{self.prefix}:idx:{thread_id}:{ns}"

    def _key_threads(self) -> str:
        return f"{self.prefix}:threads"

    def _key_seq(self) -> str:
        return f"{self.prefix}:seq"

    # ----------------------------------------------------------- read helpers

    async def _load_tuple(
        self, thread_id: str, ns: str, cid: str
    ) -> Optional[CheckpointTuple]:
        raw = await self.client.get(self._key_checkpoint(thread_id, ns, cid))
        if raw is None:
            return None
        payload = self.serde.loads_typed(_unpack_typed(raw))
        stored: Checkpoint = payload["checkpoint"]

        # Re-hydrate channel values from per-(channel, version) blobs
        versions: dict[str, Any] = stored.get("channel_versions") or {}
        channel_values: dict[str, Any] = {}
        if versions:
            blob_keys = [
                self._key_blob(thread_id, ns, ch, ver) for ch, ver in versions.items()
            ]
            blobs = await self.client.mget(blob_keys)
            for (ch, _ver), blob in zip(versions.items(), blobs):
                if blob is None:
                    continue
                tag, data = _unpack_typed(blob)
                if tag == _EMPTY_TAG:
                    continue
                channel_values[ch] = self.serde.loads_typed((tag, data))

        checkpoint: Checkpoint = {**stored, "channel_values": channel_values}

        writes = await self.client.hgetall(self._key_writes(thread_id, ns, cid))
        pending_writes: list[tuple[str, str, Any]] = []
        for field, val in sorted(writes.items(), key=lambda kv: _write_sort_key(kv[0].decode())):
            # LangGraph expects plain 3-tuples; the serializer turns tuples into lists,
            # so rebuild the tuple explicitly (tolerates a stored task_path as 4th item).
            item = self.serde.loads_typed(_unpack_typed(val))
            pending_writes.append((item[0], item[1], item[2]))

        parent_id = payload.get("parent_id")
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cid,
                }
            },
            checkpoint=checkpoint,
            metadata=payload["metadata"],
            pending_writes=pending_writes,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
        )

    async def _iter_registry(self, thread_id: Optional[str] = None) -> list[tuple[str, str]]:
        """Return ``[(thread_id, checkpoint_ns), ...]`` from the registry hash.

        Parsing and filtering only - **no per-entry round trips**. A registry entry
        whose timeline has expired is detected by the caller while it already walks
        the timeline (:meth:`alist`, :meth:`adelete_thread`), which is where the dead
        entry gets removed. That keeps the registry self-healing at zero extra cost,
        which matters for ``alist(None)`` on a fleet with thousands of threads.
        """
        registry = await self.client.hgetall(self._key_threads())
        targets: list[tuple[str, str]] = []
        stale: list[bytes] = []
        for field, value in registry.items():
            try:
                info = json.loads(value)
            except (ValueError, TypeError):
                stale.append(field)  # unparsable entry: nothing to check against
                continue
            if thread_id is not None and info.get("thread") != thread_id:
                continue
            targets.append((info["thread"], info["ns"]))
        if stale:
            await self.client.hdel(self._key_threads(), *stale)
        return targets

    # ------------------------------------------------------------ async reads

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Load a checkpoint tuple; without ``checkpoint_id`` the latest one is returned."""
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)
        if cid is None:
            latest = await self.client.zrevrange(self._key_index(thread_id, ns), 0, 0)
            if not latest:
                return None
            cid = latest[0].decode() if isinstance(latest[0], bytes) else str(latest[0])
        return await self._load_tuple(thread_id, ns, cid)

    async def alist(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints newest-first.

        ``config=None`` walks every (thread, ns) known to the registry.
        """
        if limit is not None and limit <= 0:
            return

        if config is None:
            targets = await self._iter_registry()
        else:
            thread_id = config["configurable"]["thread_id"]
            ns = config["configurable"].get("checkpoint_ns")
            targets = (
                [(thread_id, ns)]
                if ns is not None
                else await self._iter_registry(thread_id)
            )

        cid_filter = get_checkpoint_id(config) if config else None
        before_cid = get_checkpoint_id(before) if before else None
        remaining = limit

        for thread_id, ns in targets:
            idx_key = self._key_index(thread_id, ns)
            entries = await self.client.zrevrange(idx_key, 0, -1, withscores=True)
            if not entries:
                # Timeline expired (TTL) or was deleted -> drop the dead registry entry.
                # This is where the registry heals itself, and it costs no extra round
                # trip because the timeline had to be fetched anyway.
                await self.client.hdel(self._key_threads(), idx_key)
                continue
            before_score = None
            if before_cid is not None:
                before_score = await self.client.zscore(idx_key, before_cid)
            for member, score in entries:
                cid = member.decode() if isinstance(member, bytes) else str(member)
                if cid_filter and cid != cid_filter:
                    continue
                if before_cid is not None:
                    if before_score is not None:
                        if float(score) >= float(before_score):
                            continue
                    elif not cid < before_cid:
                        continue
                tup = await self._load_tuple(thread_id, ns, cid)
                if tup is None:
                    continue
                if filter and any(tup.metadata.get(k) != v for k, v in filter.items()):
                    continue
                yield tup
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

    # ----------------------------------------------------------- async writes

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint plus the blobs for every channel in ``new_versions``."""
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid: str = checkpoint["id"]
        values: dict[str, Any] = checkpoint.get("channel_values") or {}
        stored = {k: v for k, v in checkpoint.items() if k != "channel_values"}

        payload = _pack_typed(
            *self.serde.dumps_typed(
                {
                    "checkpoint": stored,
                    "metadata": get_checkpoint_metadata(config, metadata),
                    "parent_id": config["configurable"].get("checkpoint_id"),
                }
            )
        )

        idx_key = self._key_index(thread_id, ns)
        seq = await self.client.incr(self._key_seq())
        pipe = self.client.pipeline()
        pipe.set(self._key_checkpoint(thread_id, ns, cid), payload)
        for channel, version in (new_versions or {}).items():
            blob_key = self._key_blob(thread_id, ns, channel, version)
            if channel in values:
                pipe.set(blob_key, _pack_typed(*self.serde.dumps_typed(values[channel])))
            else:
                # channel was cleared / has no value in this step (mirrors InMemorySaver)
                pipe.set(blob_key, _pack_typed(_EMPTY_TAG, b""))
            if self.ttl:
                pipe.expire(blob_key, self.ttl)
        pipe.zadd(idx_key, {cid: seq})
        pipe.hset(
            self._key_threads(),
            idx_key,
            json.dumps({"thread": thread_id, "ns": ns}, ensure_ascii=False),
        )
        if self.ttl:
            pipe.expire(self._key_checkpoint(thread_id, ns, cid), self.ttl)
            pipe.expire(idx_key, self.ttl)
        await pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": cid,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],  # items are (channel, value), same contract as the base class
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending writes of a task.

        Writes are keyed by ``task_id`` + channel index so that re-running the
        same task does not duplicate regular channels, while special channels
        (``__interrupt__`` etc.) overwrite - same rule as ``InMemorySaver``.
        ``task_path`` is accepted for API compatibility and not stored.
        """
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid: str = config["configurable"]["checkpoint_id"]
        wr_key = self._key_writes(thread_id, ns, cid)

        existing: set[str] = set()
        if await self.client.exists(wr_key):
            existing = {f.decode() for f in await self.client.hkeys(wr_key)}

        pipe = self.client.pipeline()
        for idx, (channel, value) in enumerate(writes):
            inner_idx = WRITES_IDX_MAP.get(channel, idx)
            field = f"{task_id}:{inner_idx}"
            if inner_idx >= 0 and field in existing:
                continue
            pipe.hset(
                wr_key,
                field,
                _pack_typed(*self.serde.dumps_typed((task_id, channel, value))),
            )
        if self.ttl:
            pipe.expire(wr_key, self.ttl)
        await pipe.execute()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob, write, timeline and registry entry of a thread.

        Thread ids may contain ``:`` (LangGraph users routinely build them as
        ``f"{user_id}:{session_id}"``), so deletion is driven by the registry and
        by exact key names instead of glob patterns.
        """
        targets = await self._iter_registry(thread_id)
        if not targets:
            await self._delete_by_pattern(thread_id)
            return

        for ns in {ns for _thread, ns in targets}:
            idx_key = self._key_index(thread_id, ns)
            cids = [
                m.decode() if isinstance(m, bytes) else str(m)
                for m in await self.client.zrange(idx_key, 0, -1)
            ]
            pipe = self.client.pipeline()
            for cid in cids:
                cp_key = self._key_checkpoint(thread_id, ns, cid)
                raw = await self.client.get(cp_key)
                if raw is not None:
                    payload = self.serde.loads_typed(_unpack_typed(raw))
                    versions = (payload.get("checkpoint") or {}).get("channel_versions") or {}
                    for channel, version in versions.items():
                        pipe.delete(self._key_blob(thread_id, ns, channel, version))
                pipe.delete(cp_key, self._key_writes(thread_id, ns, cid))
            pipe.delete(idx_key)
            pipe.hdel(self._key_threads(), idx_key)
            await pipe.execute()

    async def _delete_by_pattern(self, thread_id: str) -> None:
        """Best-effort cleanup for keys with no registry entry (e.g. written by an older version).

        Timeline keys are matched by an escaped pattern and then treated as exact
        keys. Before touching one, the registry is consulted again: if it maps that
        timeline to a *different* thread - which happens when one thread id is a
        prefix of another (``"a"`` vs ``"a:b"``) - the key is skipped. Only when the
        registry is missing entirely does the prefix overlap remain a risk, which is
        logged as a warning.
        """
        pattern = f"{self.prefix}:idx:{_escape_glob(thread_id)}:*"
        logger.warning(
            "no registry entry for thread %r; falling back to pattern scan %s", thread_id, pattern
        )
        async for key in self.client.scan_iter(match=pattern, count=200):
            idx_key = key.decode()
            registered = await self.client.hget(self._key_threads(), idx_key)
            if registered is not None:
                try:
                    if json.loads(registered).get("thread") != thread_id:
                        logger.warning(
                            "skipping timeline %s: registry says it belongs to another thread",
                            idx_key,
                        )
                        continue
                except (ValueError, TypeError):
                    pass
            ns = idx_key[len(f"{self.prefix}:idx:{thread_id}:") :]
            cids = [
                m.decode() if isinstance(m, bytes) else str(m)
                for m in await self.client.zrange(idx_key, 0, -1)
            ]
            pipe = self.client.pipeline()
            for cid in cids:
                cp_key = self._key_checkpoint(thread_id, ns, cid)
                raw = await self.client.get(cp_key)
                if raw is not None:
                    payload = self.serde.loads_typed(_unpack_typed(raw))
                    versions = (payload.get("checkpoint") or {}).get("channel_versions") or {}
                    for channel, version in versions.items():
                        pipe.delete(self._key_blob(thread_id, ns, channel, version))
                pipe.delete(cp_key, self._key_writes(thread_id, ns, cid))
            pipe.delete(idx_key)
            pipe.hdel(self._key_threads(), idx_key)
            await pipe.execute()

    # ------------------------------------------------------- sync API stubs
    # This saver is async-only (the synchronous Redis client is deliberately not
    # pulled in). The stubs below turn the base class' bare NotImplementedError
    # into an actionable message.

    def _sync_unsupported(self, method: str) -> None:
        raise NotImplementedError(
            f"AsyncRedisSaver only implements the async API, so synchronous {method}() "
            "is unavailable. Use graph.ainvoke()/astream() (or aget_state*), or a "
            "synchronous saver such as langgraph-checkpoint-sqlite."
        )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:  # noqa: D102
        self._sync_unsupported("get_tuple")

    def put(  # noqa: D102
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._sync_unsupported("put")

    def put_writes(  # noqa: D102
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._sync_unsupported("put_writes")

    def list(  # noqa: D102
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        self._sync_unsupported("list")

    # ----------------------------------------------------------------- misc

    async def aclose(self) -> None:
        """Close the underlying Redis connections (call it on shutdown)."""
        close = getattr(self.client, "aclose", None) or getattr(self.client, "close")
        try:
            await close()
        except Exception:  # noqa: BLE001 - closing must never raise into callers
            logger.warning("error while closing the Redis client", exc_info=True)
