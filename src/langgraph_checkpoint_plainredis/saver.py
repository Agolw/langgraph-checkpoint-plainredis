"""LangGraph checkpoint savers backed by **plain Redis** (no modules required).

Why this exists
---------------
The official ``langgraph-checkpoint-redis`` requires the **RedisJSON** and
**RediSearch** modules (bundled with Redis 8.0+, or shipped by the Redis Stack
image). Redis deployments that cannot load modules -- internal/legacy servers,
managed instances exposing only the core command set, Redis 5/6/7 -- cannot use
it at all.

These savers only use core Redis data structures (strings, hashes, sorted sets),
so they run on Redis 5.0+ with **no modules**.

* :class:`AsyncRedisSaver` -- async API (``graph.ainvoke()`` / ``astream()``),
  built on ``redis.asyncio``.
* :class:`RedisSaver` -- sync API (``graph.invoke()`` / ``stream()``), built on
  the blocking ``redis`` client. Pick whichever matches your call style; each
  one raises a pointer to the other if you use the wrong one.

Key layout
----------
All keys start with ``{prefix}:`` (``prefix`` defaults to ``lg``)::

    {prefix}:cp:{thread_id}:{ns}:{checkpoint_id}   checkpoint payload (string)
    {prefix}:blob:{thread_id}:{ns}:{channel}:{version}   channel value  (string)
    {prefix}:wr:{thread_id}:{ns}:{checkpoint_id}   pending writes      (hash)
    {prefix}:idx:{thread_id}:{ns}                  timeline            (sorted set)
    {prefix}:threads                               (thread, ns) registry (hash)
    {prefix}:seq                                   monotonic counter   (string)

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
* Sync and async are separate classes: both implement the same storage
  semantics against the same key space (and are tested against each other and
  against ``InMemorySaver``).
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

import redis
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

__all__ = ["AsyncRedisSaver", "RedisSaver", "DEFAULT_PREFIX"]

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "lg"

# Redis glob metacharacters that must be escaped before they are used in SCAN patterns
_GLOB_SPECIAL = "*?[]\\"
# Tag used for channels that were written as "no value" (mirrors InMemorySaver's ("empty", b""))
_EMPTY_TAG = "empty"

# Sentinel returned by _decode_blob for "no value" / "blob gone"
_MISSING = object()


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


def _as_str(value: Any) -> str:
    """Redis returns bytes; tests/other clients may hand us str."""
    return value.decode() if isinstance(value, bytes) else str(value)


class _RedisCheckpointSaver(BaseCheckpointSaver):
    """Storage layout, codecs and pure planning helpers shared by both savers.

    Nothing in here performs I/O: the async and sync subclasses execute the plans
    with their own client. That keeps the two implementations from drifting apart.
    """

    def __init__(
        self,
        *,
        ttl: Optional[int] = None,
        prefix: str = DEFAULT_PREFIX,
        serde: Any = None,
    ) -> None:
        super().__init__(serde=serde)
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

    # ------------------------------------------------- codecs (no I/O at all)

    def _payload_for_checkpoint(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> bytes:
        """Serialize a checkpoint without its channel values (those go to blobs)."""
        stored = {k: v for k, v in checkpoint.items() if k != "channel_values"}
        return _pack_typed(
            *self.serde.dumps_typed(
                {
                    "checkpoint": stored,
                    "metadata": get_checkpoint_metadata(config, metadata),
                    "parent_id": config["configurable"].get("checkpoint_id"),
                }
            )
        )

    def _parse_checkpoint_payload(
        self, raw: bytes
    ) -> tuple[Checkpoint, CheckpointMetadata, Optional[str], dict[str, Any]]:
        payload = self.serde.loads_typed(_unpack_typed(raw))
        stored: Checkpoint = payload["checkpoint"]
        versions: dict[str, Any] = stored.get("channel_versions") or {}
        return stored, payload["metadata"], payload.get("parent_id"), versions

    def _plan_blobs(
        self,
        thread_id: str,
        ns: str,
        values: dict[str, Any],
        new_versions: Optional[ChannelVersions],
    ) -> list[tuple[str, bytes]]:
        """Keys+payloads for every channel in ``new_versions`` (shared across checkpoints)."""
        planned: list[tuple[str, bytes]] = []
        for channel, version in (new_versions or {}).items():
            key = self._key_blob(thread_id, ns, channel, version)
            if channel in values:
                planned.append((key, _pack_typed(*self.serde.dumps_typed(values[channel]))))
            else:
                # channel was cleared / has no value in this step (mirrors InMemorySaver)
                planned.append((key, _pack_typed(_EMPTY_TAG, b"")))
        return planned

    def _blob_keys(self, thread_id: str, ns: str, versions: dict[str, Any]) -> list[str]:
        return [
            self._key_blob(thread_id, ns, channel, version)
            for channel, version in versions.items()
        ]

    def _decode_blob(self, raw: Optional[bytes]) -> Any:
        """Decode one blob; returns ``_MISSING`` for absent/empty blobs."""
        if raw is None:
            return _MISSING
        tag, data = _unpack_typed(raw)
        if tag == _EMPTY_TAG:
            return _MISSING
        return self.serde.loads_typed((tag, data))

    def _parse_pending_writes(
        self, thread_id: str, ns: str, cid: str, raw_writes: dict[bytes, bytes]
    ) -> list[tuple[str, str, Any]]:
        """Rebuild LangGraph's 3-tuple writes (the serializer would give us lists)."""
        pending: list[tuple[str, str, Any]] = []
        for field, val in sorted(raw_writes.items(), key=lambda kv: _write_sort_key(_as_str(kv[0]))):
            item = self.serde.loads_typed(_unpack_typed(val))
            pending.append((item[0], item[1], item[2]))
        return pending

    def _build_tuple(
        self,
        thread_id: str,
        ns: str,
        cid: str,
        stored: Checkpoint,
        metadata: CheckpointMetadata,
        parent_id: Optional[str],
        channel_values: dict[str, Any],
        pending_writes: list[tuple[str, str, Any]],
    ) -> CheckpointTuple:
        checkpoint: Checkpoint = {**stored, "channel_values": channel_values}
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cid,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
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

    def _encode_write(self, task_id: str, channel: str, value: Any) -> bytes:
        return _pack_typed(*self.serde.dumps_typed((task_id, channel, value)))

    @staticmethod
    def _registry_allows_delete(
        registered: Optional[bytes], idx_key: str, thread_id: str
    ) -> bool:
        """True when the registry does not attribute ``idx_key`` to a *different* thread."""
        if registered is None:
            return True
        try:
            other = json.loads(registered).get("thread")
        except (ValueError, TypeError):
            return True
        if other != thread_id:
            logger.warning(
                "skipping timeline %s: registry says it belongs to thread %r", idx_key, other
            )
            return False
        return True

    # ----------------------------------------------------------- pure planning

    def _plan_writes(
        self,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        existing: set[str],
    ) -> list[tuple[str, bytes]]:
        planned: list[tuple[str, bytes]] = []
        for idx, (channel, value) in enumerate(writes):
            inner_idx = WRITES_IDX_MAP.get(channel, idx)
            field = f"{task_id}:{inner_idx}"
            if inner_idx >= 0 and field in existing:
                continue
            planned.append((field, self._encode_write(task_id, channel, value)))
        return planned

    def _parse_registry(
        self, raw_registry: dict[bytes, bytes]
    ) -> tuple[list[tuple[str, str, str]], list[bytes]]:
        """Parse ``{prefix}:threads`` -> ([(thread, ns, idx_key)], unparsable fields).

        No I/O: a entry whose timeline expired is detected by the caller while it
        already walks that timeline (see the ``alist`` implementations).
        """
        targets: list[tuple[str, str, str]] = []
        stale: list[bytes] = []
        for field, value in raw_registry.items():
            try:
                info = json.loads(value)
            except (ValueError, TypeError):
                stale.append(field)
                continue
            targets.append((info["thread"], info["ns"], _as_str(field)))
        return targets, stale

    @staticmethod
    def _filter_cids(
        entries: Sequence[tuple[Any, Any]],
        cid_filter: Optional[str],
        before_cid: Optional[str],
        before_score: Optional[float],
    ) -> list[str]:
        """Apply ``checkpoint_id`` / ``before`` filtering to (member, score) pairs."""
        out: list[str] = []
        for member, score in entries:
            cid = _as_str(member)
            if cid_filter and cid != cid_filter:
                continue
            if before_cid is not None:
                if before_score is not None:
                    if float(score) >= float(before_score):
                        continue
                elif not cid < before_cid:
                    continue
            out.append(cid)
        return out

    def _keys_to_delete(
        self,
        thread_id: str,
        ns: str,
        cids: Sequence[str],
        payloads: Sequence[Optional[bytes]],
    ) -> list[str]:
        """Exact keys (checkpoint / blobs / writes / timeline) of one namespace."""
        keys: list[str] = []
        for cid, raw in zip(cids, payloads):
            keys.append(self._key_checkpoint(thread_id, ns, cid))
            keys.append(self._key_writes(thread_id, ns, cid))
            if raw is None:
                continue
            try:
                _stored, _meta, _parent, versions = self._parse_checkpoint_payload(raw)
            except Exception:  # noqa: BLE001 - a corrupt payload must not block deletion
                logger.warning("could not parse checkpoint %s while deleting", cid, exc_info=True)
                continue
            for channel, version in versions.items():
                keys.append(self._key_blob(thread_id, ns, channel, version))
        keys.append(self._key_index(thread_id, ns))
        return keys

    # ------------------------------------------------------------ error paths

    def _sync_unsupported(self, method: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} only implements the async API, so synchronous "
            f"{method}() is unavailable. Use AsyncRedisSaver (this package) with "
            "graph.ainvoke()/astream(), or switch to RedisSaver for the sync API."
        )

    def _async_unsupported(self, method: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} only implements the sync API, so {method}() is "
            "unavailable. Use RedisSaver (this package) with graph.invoke()/stream(), "
            "or switch to AsyncRedisSaver for the async API."
        )


class AsyncRedisSaver(_RedisCheckpointSaver):
    """LangGraph async checkpointer on plain Redis (Redis 5.0+, no modules).

    Args:
        client: ready-made ``redis.asyncio.Redis`` instance (other connection
            arguments are then ignored).
        url: Redis URL, e.g. ``redis://:password@127.0.0.1:6379/0``.
        host/port/password/db: used when neither ``client`` nor ``url`` is given.
        protocol: RESP protocol version; defaults to ``2`` on purpose (RESP3
            needs Redis 6+, and RESP2 is enough here).
        ttl: optional key TTL in seconds (``None`` = keys live until deleted).
        prefix: key namespace, defaults to ``"lg"``.
        serde: custom LangGraph serializer; defaults to ``JsonPlusSerializer``.
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
        super().__init__(ttl=ttl, prefix=prefix, serde=serde)
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
            cid = _as_str(latest[0])
        return await self._load_tuple(thread_id, ns, cid)

    async def _load_tuple(
        self, thread_id: str, ns: str, cid: str
    ) -> Optional[CheckpointTuple]:
        raw = await self.client.get(self._key_checkpoint(thread_id, ns, cid))
        if raw is None:
            return None
        stored, metadata, parent_id, versions = self._parse_checkpoint_payload(raw)

        channel_values: dict[str, Any] = {}
        if versions:
            keys = self._blob_keys(thread_id, ns, versions)
            blobs = await self.client.mget(keys)
            for (channel, _version), blob in zip(versions.items(), blobs):
                value = self._decode_blob(blob)
                if value is not _MISSING:
                    channel_values[channel] = value

        raw_writes = await self.client.hgetall(self._key_writes(thread_id, ns, cid))
        pending_writes = self._parse_pending_writes(thread_id, ns, cid, raw_writes)
        return self._build_tuple(
            thread_id, ns, cid, stored, metadata, parent_id, channel_values, pending_writes
        )

    async def alist(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints newest-first (``config=None`` walks every thread/namespace)."""
        if limit is not None and limit <= 0:
            return

        if config is None:
            targets = await self._async_targets()
        else:
            thread_id = config["configurable"]["thread_id"]
            ns = config["configurable"].get("checkpoint_ns")
            targets = (
                [(thread_id, ns, self._key_index(thread_id, ns))]
                if ns is not None
                else await self._async_targets(thread_id)
            )

        cid_filter = get_checkpoint_id(config) if config else None
        before_cid = get_checkpoint_id(before) if before else None
        remaining = limit

        for thread_id, ns, idx_key in targets:
            entries = await self.client.zrevrange(idx_key, 0, -1, withscores=True)
            if not entries:
                # timeline expired (TTL) or deleted -> drop the dead registry entry
                await self.client.hdel(self._key_threads(), idx_key)
                continue
            before_score = (
                await self.client.zscore(idx_key, before_cid) if before_cid is not None else None
            )
            for cid in self._filter_cids(entries, cid_filter, before_cid, before_score):
                tuple_ = await self._load_tuple(thread_id, ns, cid)
                if tuple_ is None:
                    continue
                if filter and any(tuple_.metadata.get(k) != v for k, v in filter.items()):
                    continue
                yield tuple_
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

    async def _async_targets(self, thread_id: Optional[str] = None) -> list[tuple[str, str, str]]:
        registry = await self.client.hgetall(self._key_threads())
        targets, stale = self._parse_registry(registry)
        if stale:
            await self.client.hdel(self._key_threads(), *stale)
        if thread_id is None:
            return targets
        return [t for t in targets if t[0] == thread_id]

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
        cp_key = self._key_checkpoint(thread_id, ns, cid)
        idx_key = self._key_index(thread_id, ns)
        payload = self._payload_for_checkpoint(config, checkpoint, metadata)
        blobs = self._plan_blobs(
            thread_id, ns, checkpoint.get("channel_values") or {}, new_versions
        )

        seq = await self.client.incr(self._key_seq())
        pipe = self.client.pipeline()
        pipe.set(cp_key, payload)
        for key, value in blobs:
            pipe.set(key, value)
            if self.ttl:
                pipe.expire(key, self.ttl)
        pipe.zadd(idx_key, {cid: seq})
        pipe.hset(
            self._key_threads(),
            idx_key,
            json.dumps({"thread": thread_id, "ns": ns}, ensure_ascii=False),
        )
        if self.ttl:
            pipe.expire(cp_key, self.ttl)
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
        writes: Sequence[tuple[str, Any]],  # items are (channel, value), same as the base class
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending writes of one task (idempotent per ``(task_id, channel)``).

        ``task_path`` is accepted for API compatibility and not stored.
        """
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid: str = config["configurable"]["checkpoint_id"]
        wr_key = self._key_writes(thread_id, ns, cid)

        existing: set[str] = set()
        if await self.client.exists(wr_key):
            existing = {_as_str(f) for f in await self.client.hkeys(wr_key)}

        pipe = self.client.pipeline()
        for field, value in self._plan_writes(writes, task_id, existing):
            pipe.hset(wr_key, field, value)
        if self.ttl:
            pipe.expire(wr_key, self.ttl)
        await pipe.execute()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob, write, timeline and registry entry of a thread.

        Thread ids may contain ``:`` (LangGraph users routinely build them as
        ``f"{user_id}:{session_id}"``), so deletion is registry-driven and uses exact
        key names instead of glob patterns.
        """
        targets = await self._async_targets(thread_id)
        if not targets:
            await self._async_delete_by_pattern(thread_id)
            return

        for ns in {ns for _thread, ns, _key in targets}:
            idx_key = self._key_index(thread_id, ns)
            cids = [_as_str(m) for m in await self.client.zrange(idx_key, 0, -1)]
            payloads = (
                await self.client.mget([self._key_checkpoint(thread_id, ns, c) for c in cids])
                if cids
                else []
            )
            keys = self._keys_to_delete(thread_id, ns, cids, payloads)
            pipe = self.client.pipeline()
            if keys:
                pipe.delete(*keys)
            pipe.hdel(self._key_threads(), idx_key)
            await pipe.execute()

    async def _async_delete_by_pattern(self, thread_id: str) -> None:
        """Best-effort cleanup when the registry has no entry for ``thread_id``.

        The timeline keys are matched by an escaped pattern and then treated as exact
        keys; the registry is consulted again for every match, so a longer thread
        (``"a"`` vs ``"a:b"``) is left alone whenever its own registry entry survives.
        Only a completely missing registry leaves the prefix overlap as a risk.
        """
        pattern = f"{self.prefix}:idx:{_escape_glob(thread_id)}:*"
        logger.warning(
            "no registry entry for thread %r; falling back to pattern scan %s", thread_id, pattern
        )
        async for key in self.client.scan_iter(match=pattern, count=200):
            idx_key = _as_str(key)
            registered = await self.client.hget(self._key_threads(), idx_key)
            if not self._registry_allows_delete(registered, idx_key, thread_id):
                continue
            ns = idx_key[len(f"{self.prefix}:idx:{thread_id}:") :]
            cids = [_as_str(m) for m in await self.client.zrange(idx_key, 0, -1)]
            payloads = (
                await self.client.mget([self._key_checkpoint(thread_id, ns, c) for c in cids])
                if cids
                else []
            )
            keys = self._keys_to_delete(thread_id, ns, cids, payloads)
            pipe = self.client.pipeline()
            if keys:
                pipe.delete(*keys)
            pipe.hdel(self._key_threads(), idx_key)
            await pipe.execute()

    # ------------------------------------------------------- sync API stubs

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

    def delete_thread(self, thread_id: str) -> None:  # noqa: D102
        self._sync_unsupported("delete_thread")

    # ----------------------------------------------------------------- misc

    async def aclose(self) -> None:
        """Close the underlying Redis connections (call it on shutdown)."""
        close = getattr(self.client, "aclose", None) or getattr(self.client, "close")
        try:
            await close()
        except Exception:  # noqa: BLE001 - closing must never raise into callers
            logger.warning("error while closing the Redis client", exc_info=True)


class RedisSaver(_RedisCheckpointSaver):
    """LangGraph **synchronous** checkpointer on plain Redis (Redis 5.0+, no modules).

    Same storage layout and semantics as :class:`AsyncRedisSaver`, but built on the
    blocking ``redis`` client, for graphs driven with ``graph.invoke()`` /
    ``graph.stream()``.

    Example:
        >>> saver = RedisSaver(url="redis://127.0.0.1:6379/0", ttl=7 * 24 * 3600)
        >>> graph = builder.compile(checkpointer=saver)
        >>> graph.invoke(state, {"configurable": {"thread_id": "t-1"}})
    """

    def __init__(
        self,
        client: Optional[redis.Redis] = None,
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
        super().__init__(ttl=ttl, prefix=prefix, serde=serde)
        if client is not None:
            self.client = client
        elif url:
            self.client = redis.from_url(
                url,
                protocol=protocol,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
            )
        else:
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password or None,
                db=db,
                protocol=protocol,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=False,
            )

    # ------------------------------------------------------------- sync reads

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Load a checkpoint tuple; without ``checkpoint_id`` the latest one is returned."""
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)
        if cid is None:
            latest = self.client.zrevrange(self._key_index(thread_id, ns), 0, 0)
            if not latest:
                return None
            cid = _as_str(latest[0])
        return self._load_tuple(thread_id, ns, cid)

    def _load_tuple(
        self, thread_id: str, ns: str, cid: str
    ) -> Optional[CheckpointTuple]:
        raw = self.client.get(self._key_checkpoint(thread_id, ns, cid))
        if raw is None:
            return None
        stored, metadata, parent_id, versions = self._parse_checkpoint_payload(raw)

        channel_values: dict[str, Any] = {}
        if versions:
            keys = self._blob_keys(thread_id, ns, versions)
            blobs = self.client.mget(keys)
            for (channel, _version), blob in zip(versions.items(), blobs):
                value = self._decode_blob(blob)
                if value is not _MISSING:
                    channel_values[channel] = value

        raw_writes = self.client.hgetall(self._key_writes(thread_id, ns, cid))
        pending_writes = self._parse_pending_writes(thread_id, ns, cid, raw_writes)
        return self._build_tuple(
            thread_id, ns, cid, stored, metadata, parent_id, channel_values, pending_writes
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints newest-first (``config=None`` walks every thread/namespace)."""
        if limit is not None and limit <= 0:
            return

        if config is None:
            targets = self._sync_targets()
        else:
            thread_id = config["configurable"]["thread_id"]
            ns = config["configurable"].get("checkpoint_ns")
            targets = (
                [(thread_id, ns, self._key_index(thread_id, ns))]
                if ns is not None
                else self._sync_targets(thread_id)
            )

        cid_filter = get_checkpoint_id(config) if config else None
        before_cid = get_checkpoint_id(before) if before else None
        remaining = limit

        for thread_id, ns, idx_key in targets:
            entries = self.client.zrevrange(idx_key, 0, -1, withscores=True)
            if not entries:
                self.client.hdel(self._key_threads(), idx_key)
                continue
            before_score = (
                self.client.zscore(idx_key, before_cid) if before_cid is not None else None
            )
            for cid in self._filter_cids(entries, cid_filter, before_cid, before_score):
                tuple_ = self._load_tuple(thread_id, ns, cid)
                if tuple_ is None:
                    continue
                if filter and any(tuple_.metadata.get(k) != v for k, v in filter.items()):
                    continue
                yield tuple_
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

    def _sync_targets(self, thread_id: Optional[str] = None) -> list[tuple[str, str, str]]:
        registry = self.client.hgetall(self._key_threads())
        targets, stale = self._parse_registry(registry)
        if stale:
            self.client.hdel(self._key_threads(), *stale)
        if thread_id is None:
            return targets
        return [t for t in targets if t[0] == thread_id]

    # ------------------------------------------------------------ sync writes

    def put(
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
        cp_key = self._key_checkpoint(thread_id, ns, cid)
        idx_key = self._key_index(thread_id, ns)
        payload = self._payload_for_checkpoint(config, checkpoint, metadata)
        blobs = self._plan_blobs(
            thread_id, ns, checkpoint.get("channel_values") or {}, new_versions
        )

        seq = self.client.incr(self._key_seq())
        pipe = self.client.pipeline()
        pipe.set(cp_key, payload)
        for key, value in blobs:
            pipe.set(key, value)
            if self.ttl:
                pipe.expire(key, self.ttl)
        pipe.zadd(idx_key, {cid: seq})
        pipe.hset(
            self._key_threads(),
            idx_key,
            json.dumps({"thread": thread_id, "ns": ns}, ensure_ascii=False),
        )
        if self.ttl:
            pipe.expire(cp_key, self.ttl)
            pipe.expire(idx_key, self.ttl)
        pipe.execute()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": cid,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],  # items are (channel, value), same as the base class
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending writes of one task (idempotent per ``(task_id, channel)``)."""
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        cid: str = config["configurable"]["checkpoint_id"]
        wr_key = self._key_writes(thread_id, ns, cid)

        existing: set[str] = set()
        if self.client.exists(wr_key):
            existing = {_as_str(f) for f in self.client.hkeys(wr_key)}

        pipe = self.client.pipeline()
        for field, value in self._plan_writes(writes, task_id, existing):
            pipe.hset(wr_key, field, value)
        if self.ttl:
            pipe.expire(wr_key, self.ttl)
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob, write, timeline and registry entry of a thread."""
        targets = self._sync_targets(thread_id)
        if not targets:
            self._delete_by_pattern(thread_id)
            return

        for ns in {ns for _thread, ns, _key in targets}:
            idx_key = self._key_index(thread_id, ns)
            cids = [_as_str(m) for m in self.client.zrange(idx_key, 0, -1)]
            payloads = (
                self.client.mget([self._key_checkpoint(thread_id, ns, c) for c in cids])
                if cids
                else []
            )
            keys = self._keys_to_delete(thread_id, ns, cids, payloads)
            pipe = self.client.pipeline()
            if keys:
                pipe.delete(*keys)
            pipe.hdel(self._key_threads(), idx_key)
            pipe.execute()

    def _delete_by_pattern(self, thread_id: str) -> None:
        """Best-effort cleanup when the registry has no entry (mirrors the async version)."""
        pattern = f"{self.prefix}:idx:{_escape_glob(thread_id)}:*"
        logger.warning(
            "no registry entry for thread %r; falling back to pattern scan %s", thread_id, pattern
        )
        for key in self.client.scan_iter(match=pattern, count=200):
            idx_key = _as_str(key)
            registered = self.client.hget(self._key_threads(), idx_key)
            if not self._registry_allows_delete(registered, idx_key, thread_id):
                continue
            ns = idx_key[len(f"{self.prefix}:idx:{thread_id}:") :]
            cids = [_as_str(m) for m in self.client.zrange(idx_key, 0, -1)]
            payloads = (
                self.client.mget([self._key_checkpoint(thread_id, ns, c) for c in cids])
                if cids
                else []
            )
            keys = self._keys_to_delete(thread_id, ns, cids, payloads)
            pipe = self.client.pipeline()
            if keys:
                pipe.delete(*keys)
            pipe.hdel(self._key_threads(), idx_key)
            pipe.execute()

    # -------------------------------------------------------- async API stubs

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:  # noqa: D102
        self._async_unsupported("aget_tuple")

    async def aput(  # noqa: D102
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._async_unsupported("aput")

    async def aput_writes(  # noqa: D102
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._async_unsupported("aput_writes")

    async def alist(  # type: ignore[override]  # noqa: D102
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        self._async_unsupported("alist")
        # The bare `yield` is load-bearing, do not "clean it up": `alist` must return an
        # AsyncIterator, so callers use `async for`. Only an async generator raises here at
        # iteration time; without the yield the call returns a coroutine and `async for`
        # fails first with "requires __aiter__", hiding the class pointer below.
        yield  # pragma: no cover - makes this an async generator

    async def adelete_thread(self, thread_id: str) -> None:  # noqa: D102
        self._async_unsupported("adelete_thread")

    # ----------------------------------------------------------------- misc

    def close(self) -> None:
        """Close the underlying Redis connections (call it on shutdown)."""
        try:
            self.client.close()
        except Exception:  # noqa: BLE001 - closing must never raise into callers
            logger.warning("error while closing the Redis client", exc_info=True)
