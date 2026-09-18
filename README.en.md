# langgraph-checkpoint-plainredis

> 中文版（主）: [README.md](README.md)

[![CI](https://github.com/Agolw/langgraph-checkpoint-plainredis/actions/workflows/ci.yml/badge.svg)](https://github.com/Agolw/langgraph-checkpoint-plainredis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langgraph-checkpoint-plainredis.svg)](https://pypi.org/project/langgraph-checkpoint-plainredis/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](#compatibility)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**LangGraph checkpoint savers on plain Redis (sync + async) — no RedisJSON, no RediSearch, works on Redis 5.0+.**

Persist LangGraph agent state (threads, checkpoints, pending writes, time travel)
in a Redis that only offers the core command set: no modules, no Redis Stack, no
Redis 8 requirement.

```python
from langgraph_checkpoint_plainredis import AsyncRedisSaver   # for graph.ainvoke()
from langgraph_checkpoint_plainredis import RedisSaver        # for graph.invoke()

saver = AsyncRedisSaver(url="redis://127.0.0.1:6379/0", ttl=7 * 24 * 3600)
graph = builder.compile(checkpointer=saver)

await graph.ainvoke(state, {"configurable": {"thread_id": "user-42:session-7"}})
```

---

## Why not `langgraph-checkpoint-redis`?

The official Redis checkpointer is a great piece of software, but it **requires
Redis modules** — the project states it plainly:

> **IMPORTANT:** This library requires Redis with the following modules:
> **RedisJSON** (for storing and manipulating JSON data) and **RediSearch**
> (for search and indexing). Redis 8.0+ includes both by default; for older
> versions you need Redis Stack or to install the modules separately.

That rules it out for a large class of real deployments:

* internal / legacy servers pinned to Redis 5, 6 or 7;
* managed Redis instances where `MODULE LOAD` is not available;
* slim self-hosted containers that intentionally ship without modules.

This package was written for exactly that situation and uses **only core Redis
data structures** (strings, hashes, sorted sets).

| | `langgraph-checkpoint-plainredis` | `langgraph-checkpoint-redis` |
|:--|:--|:--|
| Redis modules | **none** | RedisJSON + RediSearch |
| Minimum Redis | **5.0** (RESP2) | Redis Stack, or Redis 8.0+ |
| Python dependencies | `redis`, `langgraph-checkpoint` | `redis`, `redisvl`, `orjson`, `langgraph-checkpoint` |
| Storage backend | core data structures | JSON documents + search indexes |
| Sync API | `RedisSaver` | both |
| Client-side search | registry hash + timeline zset | RediSearch indexes |

If your Redis is 8.0+ (or Redis Stack), use the official package — it is more
featureful. This one exists for the environments that cannot run it.

## Install

```bash
pip install langgraph-checkpoint-plainredis
```

Requirements: Python 3.10+, Redis 5.0+, `langgraph-checkpoint>=4.1,<5`.

## Quickstart

```python
import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph_checkpoint_plainredis import AsyncRedisSaver


class State(TypedDict):
    messages: Annotated[list, add_messages]


def reply(state: State) -> State:
    return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}


async def main() -> None:
    # ttl is optional: seconds after which a thread's keys expire
    saver = AsyncRedisSaver(url="redis://127.0.0.1:6379/0", ttl=7 * 24 * 3600)
    try:
        graph = StateGraph(State).add_node("reply", reply).add_edge(START, "reply").compile(
            checkpointer=saver
        )
        config = {"configurable": {"thread_id": "user-42:session-7"}}

        await graph.ainvoke({"messages": [HumanMessage("hello")]}, config)
        await graph.ainvoke({"messages": [HumanMessage("again")]}, config)

        state = await graph.aget_state(config)
        for message in state.values["messages"]:
            print(message.type, "|", message.content)

        # time travel: replay the recorded history
        async for snapshot in graph.aget_state_history(config):
            print(snapshot.config["configurable"]["checkpoint_id"], snapshot.next)
    finally:
        await saver.aclose()  # close the Redis connections


asyncio.run(main())
```

Reuse an existing connection pool by passing a client (every other connection
argument is then ignored):

```python
import redis.asyncio as aioredis
from langgraph_checkpoint_plainredis import AsyncRedisSaver

client = aioredis.Redis(host="redis.internal", port=6379, db=2, protocol=2)
saver = AsyncRedisSaver(client=client, prefix="myapp:agent")
```

## What gets stored

All keys live under `{prefix}:` (`prefix` defaults to `lg`):

| Key | Type | Content |
|:--|:--|:--|
| `{prefix}:cp:{thread_id}:{ns}:{checkpoint_id}` | string | checkpoint payload (parent id + metadata) |
| `{prefix}:blob:{thread_id}:{ns}:{channel}:{version}` | string | one channel value |
| `{prefix}:wr:{thread_id}:{ns}:{checkpoint_id}` | hash | pending writes of that checkpoint |
| `{prefix}:idx:{thread_id}:{ns}` | sorted set | timeline (member = checkpoint id, score = monotonic counter) |
| `{prefix}:threads` | hash | `(thread_id, checkpoint_ns)` registry, used by listing |
| `{prefix}:seq` | string | monotonic counter for timeline scores |

`{prefix}:threads` and `{prefix}:seq` are namespace-level keys: `adelete_thread()`
removes a thread's checkpoints, blobs, writes, timeline and registry entry, so drop
the namespace if you tear a whole application down:

```bash
# --scan instead of KEYS (KEYS blocks large databases)
redis-cli --scan --pattern "lg:*" | xargs -r -n 500 redis-cli DEL
```

```python
# or from Python, reusing the client this package owns
keys = [key async for key in client.scan_iter(match=f"{prefix}:*", count=500)]
if keys:
    await client.delete(*keys)
```

> Those two keys deliberately have **no TTL**: `seq` orders the timeline (expiring it
> would shuffle new scores below old ones) and an expiring `threads` would make idle
> threads disappear from listings.

Channel values are kept **per (channel, version)** and shared between
checkpoints, exactly like the in-memory and SQLite savers: appending to a
message list does not re-serialize the whole history for other untouched
channels. Timeline scores make "latest checkpoint" and "checkpoints before X"
`O(log n)` sorted-set operations instead of a scan.

## API

Two classes implement the same contract - one async, one sync:

| Capability | `AsyncRedisSaver` | `RedisSaver` | Notes |
|:--|:--|:--|:--|
| read one checkpoint | `aget_tuple(config)` | `get_tuple(config)` | latest unless `checkpoint_id` is pinned |
| list checkpoints | `alist(...)` | `list(...)` | `config=None` walks every thread/namespace |
| write a checkpoint | `aput(...)` | `put(...)` | checkpoint + its channel blobs |
| write pending writes | `aput_writes(...)` | `put_writes(...)` | idempotent per `(task_id, channel)` |
| delete a thread | `adelete_thread(id)` | `delete_thread(id)` | exact keys: checkpoints/blobs/writes/timeline/registry |
| close | `aclose()` | `close()` | call it on shutdown |

The unused side (e.g. `RedisSaver.aget_tuple`) raises `NotImplementedError` and names
the class you should use instead.

> ⚠️ The class name matches the **official package** and the **official docs' DIY
example** (all three call it `AsyncRedisSaver`), but the **import path differs**: here it is
`from langgraph_checkpoint_plainredis import AsyncRedisSaver`, the official one lives under
`langgraph.checkpoint.redis` — do not mix them up when both are installed.

## Semantics and guarantees

* **Ordering** — `alist` yields newest-first by *write recency* (timeline score).
  LangGraph's default checkpoint ids are monotonic UUID6 strings, so this
  matches the reference savers' "descending by checkpoint id" for real graphs.
* **`before`** — exclusive, and resolved through the timeline score, so it keeps
  working even when ids are not lexicographically sortable.
* **`filter`** — matched against checkpoint metadata, same rule as the reference
  implementations.
* **Writes** — regular channels are written once per `(task_id, channel)`;
  special channels (`__interrupt__`, `__error__`, `__scheduled__`, `__resume__`)
  overwrite, mirroring `InMemorySaver`.
* **Cleared channels** — a channel listed in `new_versions` but absent from
  `channel_values` is stored as an "empty" blob and omitted on read.
* **`ttl`** — applied to checkpoint, blob, timeline and write keys on every
  write, so an active thread keeps refreshing while abandoned ones disappear.
* **Deletion is exact** — thread ids routinely contain `:` (e.g.
  `f"{user_id}:{session_id}"`) and may contain glob metacharacters; deletion uses
  the registry plus exact key names, never a bare pattern. Deleting thread `a`
  does not touch thread `a:b`.
* **Self-healing registry** — entries whose timeline expired are dropped lazily
  while listing. That check rides along with the timeline fetch listing performs
  anyway, so `alist(None)` issues **no** extra per-entry `EXISTS` round trip.
* **Fallback deletion caveat** — `adelete_thread()` normally deletes exact keys driven
  by the registry. The pattern fallback runs only when the registry holds no entry for
  that thread, and it re-checks the registry for every matched timeline (skipping keys
  owned by another thread). With the registry missing entirely, a prefix overlap
  (`a` vs `a:b`) can still over-delete — do not prune `{prefix}:threads` by hand.

## Limitations and roadmap

* **Two separate classes** (sync / async), each implementing only its own side of the
  interface; using the wrong one points you at the other class.
* Listing is driven by a registry hash rather than server-side indexes, so
  `alist(None)` cost grows linearly with the number of *threads* (not with the
  size of the data). Per-thread listing is `O(log n)`.
* Blobs are re-hydrated with one `MGET` per checkpoint; a checkpoint with many
  channels costs one round trip, not one per channel.
* Payloads use a small JSON+base64 wrapper around the LangGraph serializer so
  keys stay inspectable with `redis-cli`. Storing raw bytes would be ~25%
  smaller — on the roadmap if anyone needs it.
* Not implemented: sync API, cross-thread `search`-like queries.
* `adelete_thread()` reads one checkpoint per stored checkpoint (to derive its blob
  keys). The cost is bounded by the number of checkpoints in *one* thread, not by the
  fleet size, and deletion is rare — not optimised for now.

## Compatibility

* **Redis**: 5.0 → 8.x, with or without modules. RESP2 is the default
  (`protocol=2`); set `protocol=3` if your server is 6+ and you prefer RESP3.
  CI runs the whole suite against `redis:5`, `redis:7` and `redis:8` containers.
* **Python**: 3.10, 3.11, 3.12, 3.13.
* **LangGraph**: `langgraph-checkpoint >= 4.1, < 5`. The checkpoint surface is
  still evolving upstream; the pin keeps upgrades honest.

## Security

Checkpoints are serialized with the LangGraph serializer (by default
`JsonPlusSerializer`), which can fall back to pickle-based encoding for exotic
payloads. Treat the Redis instance as **trusted storage**: anyone able to write
into the key namespace could otherwise craft a payload that gets deserialized by
your process. Pass a stricter serializer through `serde=` when that is not
acceptable, and use a dedicated `db`/`prefix` per application.

## Development

A real Redis is required (that is the point of the package). One-liners:

```bash
# 1) a Redis with no modules at all
docker run --rm -d -p 6379:6379 --name plainredis-test redis:5

# 2) environment + install (conda or venv both work)
conda create -y -n PlainRedis python=3.11 && conda activate PlainRedis
pip install -e ".[dev]"          # dev extra brings pytest / pytest-asyncio / langgraph

# 3) run the suite (either form)
pytest -v --redis-url=redis://127.0.0.1:6379/15
PLAINREDIS_TEST_URL=redis://127.0.0.1:6379/15 pytest -v

# 4) run the example
python examples/basic.py         # async (AsyncRedisSaver + ainvoke)
python examples/basic_sync.py    # sync  (RedisSaver + invoke), same key layout
# PLAINREDIS_URL overrides the default redis://127.0.0.1:6379/0 for both
```

The suite defaults to database 15 and a random key prefix per test, then removes
its own keys, so it is safe to point at a shared development instance.

## License

MIT — see [LICENSE](LICENSE).
