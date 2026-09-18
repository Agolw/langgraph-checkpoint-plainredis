"""Edge cases: TTL, write idempotency, deletion, key hygiene, concurrency."""

from __future__ import annotations

import asyncio

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph_checkpoint_plainredis import AsyncRedisSaver

from .conftest import purge


def _checkpoint(cid: str, values: dict | None = None, versions: dict | None = None) -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = cid
    checkpoint["channel_values"] = values if values is not None else {"messages": [cid]}
    checkpoint["channel_versions"] = versions if versions is not None else {"messages": "1"}
    return checkpoint


def _config(thread: str = "edge", ns: str = "") -> dict:
    return {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}


async def _keys(saver: AsyncRedisSaver, pattern: str = "*") -> list[str]:
    return [key.decode() async for key in saver.client.scan_iter(match=pattern, count=500)]


async def test_ttl_expires_checkpoints(saver, make_saver, prefix):
    ttl_saver = make_saver(ttl=1)
    config = await ttl_saver.aput(_config(), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    assert await ttl_saver.aget_tuple(config) is not None

    await asyncio.sleep(1.5)

    assert await ttl_saver.aget_tuple(config) is None
    assert [t async for t in ttl_saver.alist(_config())] == []
    assert await _keys(ttl_saver, f"{prefix}:cp:*") == []


async def test_ttl_is_refreshed_by_new_writes(saver, make_saver, prefix):
    ttl_saver = make_saver(ttl=2)
    await ttl_saver.aput(_config(), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    await asyncio.sleep(1.2)
    config = await ttl_saver.aput(_config(), _checkpoint("cp-2"), {"source": "loop"}, {"messages": "2"})
    await asyncio.sleep(1.2)

    # cp-2 was written later, so it (and the timeline) must still be alive
    assert (await ttl_saver.aget_tuple(config)).checkpoint["id"] == "cp-2"


async def test_regular_writes_are_idempotent_special_writes_overwrite(saver):
    config = _config(thread="writes")
    config = await saver.aput(config, _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})

    await saver.aput_writes(config, [("messages", "first")], task_id="task-1")
    await saver.aput_writes(config, [("messages", "second")], task_id="task-1")

    # special channels (WRITES_IDX_MAP, negative index) overwrite instead
    await saver.aput_writes(config, [("__interrupt__", "i-1")], task_id="task-2")
    await saver.aput_writes(config, [("__interrupt__", "i-2")], task_id="task-2")

    writes = (await saver.aget_tuple(config)).pending_writes
    assert ("task-1", "messages", "first") in writes
    assert ("task-1", "messages", "second") not in writes
    assert writes.count(("task-2", "__interrupt__", "i-2")) == 1
    assert ("task-2", "__interrupt__", "i-1") not in writes


async def test_delete_thread_is_exact_for_colon_thread_ids(saver, prefix):
    """LangGraph users build thread ids like ``user:session`` - delete must stay scoped."""
    for thread in ("a", "a:b"):
        config = _config(thread=thread)
        config = await saver.aput(config, _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
        await saver.aput_writes(config, [("messages", "w")], task_id="task")

    await saver.adelete_thread("a")

    assert await saver.aget_tuple(_config(thread="a")) is None
    assert [t async for t in saver.alist(_config(thread="a"))] == []
    survivor = await saver.aget_tuple(_config(thread="a:b"))
    assert survivor is not None and survivor.checkpoint["channel_values"] == {"messages": ["cp-1"]}

    # exact key check: thread a is gone, thread a:b survived (a plain glob would have
    # matched both, which is why deletion does not use one)
    keys = await _keys(saver, f"{prefix}:cp:*")
    assert f"{prefix}:cp:a::cp-1" not in keys
    assert f"{prefix}:cp:a:b::cp-1" in keys


async def test_delete_thread_removes_blobs_timeline_and_registry(saver, prefix):
    for cid in ("cp-1", "cp-2"):
        config = _config(thread="full")
        config = await saver.aput(
            config,
            _checkpoint(cid, {"messages": [cid], "scratchpad": cid}, {"messages": "1", "scratchpad": "1"}),
            {"source": "loop"},
            {"messages": "1", "scratchpad": "1"},
        )
        await saver.aput_writes(config, [("messages", "w")], task_id="task")

    assert await _keys(saver, f"{prefix}:blob:*")
    await saver.adelete_thread("full")

    assert await _keys(saver, f"{prefix}:cp:*") == []
    assert await _keys(saver, f"{prefix}:blob:*") == []
    assert await _keys(saver, f"{prefix}:wr:*") == []
    assert await _keys(saver, f"{prefix}:idx:*") == []
    assert await saver.client.hlen(f"{prefix}:threads") == 0


async def test_delete_falls_back_to_pattern_when_registry_is_gone(saver, prefix):
    config = await saver.aput(_config(thread="orphan"), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    assert (await saver.aget_tuple(config)) is not None

    await saver.client.hdel(f"{prefix}:threads", f"{prefix}:idx:orphan:")
    await saver.adelete_thread("orphan")

    assert await _keys(saver, f"{prefix}:cp:*") == []
    assert await _keys(saver, f"{prefix}:idx:*") == []


async def test_delete_fallback_skips_other_threads_timeline(saver, prefix):
    """A lost registry entry must not make the pattern fallback delete a longer thread."""
    for thread in ("a", "a:b"):
        await saver.aput(_config(thread=thread), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})

    # partially damaged registry: only thread "a" loses its entry, "a:b" keeps its own
    await saver.client.hdel(f"{prefix}:threads", f"{prefix}:idx:a:")

    await saver.adelete_thread("a")

    assert await saver.aget_tuple(_config(thread="a")) is None
    assert await _keys(saver, f"{prefix}:idx:a:") == []
    # the longer thread keeps both its data and its timeline (the old code deleted the idx key)
    survivor = await saver.aget_tuple(_config(thread="a:b"))
    assert survivor is not None
    assert [t.checkpoint["id"] async for t in saver.alist(_config(thread="a:b"))] == ["cp-1"]


async def test_listing_and_deleting_need_no_existence_checks(saver):
    """Self-healing rides along with the timeline fetch - no EXISTS per registry entry."""
    for thread in ("t1", "t2", "t3"):
        await saver.aput(_config(thread=thread), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})

    async def _forbidden(*args, **kwargs):
        raise AssertionError("alist()/adelete_thread() must not issue EXISTS")

    saver.client.exists = _forbidden
    try:
        listed = [
            t.config["configurable"]["thread_id"] async for t in saver.alist(None)
        ]
        assert sorted(listed) == ["t1", "t2", "t3"]
        await saver.adelete_thread("t1")
    finally:
        saver.client.__dict__.pop("exists", None)

    assert [t.checkpoint["id"] async for t in saver.alist(_config(thread="t2"))] == ["cp-1"]


async def test_registry_heals_itself_when_timeline_disappears(saver, prefix):
    await saver.aput(_config(thread="heal"), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    assert await saver.client.hlen(f"{prefix}:threads") == 1

    await saver.client.delete(f"{prefix}:idx:heal:")

    assert [t async for t in saver.alist(None)] == []
    assert await saver.client.hlen(f"{prefix}:threads") == 0


async def test_unknown_thread_is_empty_not_an_error(saver):
    assert await saver.aget_tuple(_config(thread="nope")) is None
    assert [t async for t in saver.alist(_config(thread="nope"))] == []
    assert [t async for t in saver.alist(None)] == []


async def test_unicode_and_large_payloads_round_trip(saver):
    big = "😀" * 50_000
    payload = {
        "messages": [{"role": "user", "content": big, "nested": {"键": ["值", 1, 2.5, None, True]}}]
    }
    config = await saver.aput(
        _config(thread="unicode"),
        _checkpoint("cp-big", {"messages": payload}, {"messages": "1"}),
        {"source": "loop", "note": "中文备注"},
        {"messages": "1"},
    )

    tuple_ = await saver.aget_tuple(config)
    assert tuple_.checkpoint["channel_values"] == {"messages": payload}
    assert tuple_.metadata["note"] == "中文备注"


async def test_concurrent_writes_do_not_lose_checkpoints(saver):
    config = _config(thread="concurrent")

    await asyncio.gather(
        *[
            saver.aput(config, _checkpoint(f"cp-{i:03d}"), {"source": "loop"}, {"messages": "1"})
            for i in range(20)
        ]
    )

    ids = [t.checkpoint["id"] async for t in saver.alist(config)]
    assert len(ids) == len(set(ids)) == 20


async def test_concurrent_writes_to_same_task_do_not_duplicate(saver):
    config = _config(thread="concurrent-writes")
    config = await saver.aput(config, _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})

    await asyncio.gather(
        *[saver.aput_writes(config, [("messages", f"v{i}")], task_id="same-task") for i in range(10)]
    )

    writes = (await saver.aget_tuple(config)).pending_writes
    assert len(writes) == 1


async def test_channel_blobs_are_shared_between_checkpoints(saver, prefix):
    """Unchanged channels must not be duplicated per checkpoint."""
    for cid in ("cp-1", "cp-2", "cp-3"):
        await saver.aput(
            _config(thread="blobs"),
            _checkpoint(cid),
            {"source": "loop"},
            {"messages": "1"},  # same version every time -> one blob
        )

    assert len(await _keys(saver, f"{prefix}:blob:*")) == 1


async def test_glob_metacharacters_in_thread_id(saver):
    thread = "weird*?[t]:x"
    config = await saver.aput(_config(thread=thread), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    assert (await saver.aget_tuple(config)).checkpoint["id"] == "cp-1"

    await saver.adelete_thread(thread)
    assert await saver.aget_tuple(_config(thread=thread)) is None


async def test_custom_prefix_is_used_verbatim(saver, make_saver, redis_url):
    custom = make_saver(prefix="my-app:checkpoints")
    try:
        config = await custom.aput(_config(thread="p"), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
        assert await custom.aget_tuple(config) is not None

        keys = [k.decode() async for k in custom.client.scan_iter(match="my-app:checkpoints:cp:*", count=100)]
        assert keys
        assert all(not k.startswith("lg:") for k in keys)
    finally:
        # the fixture only purges the random test prefix, so clean this namespace up here
        keys = [k async for k in custom.client.scan_iter(match="my-app:checkpoints:*", count=500)]
        if keys:
            await custom.client.delete(*keys)


async def test_empty_prefix_rejected(saver):
    with pytest.raises(ValueError, match="prefix"):
        AsyncRedisSaver(client=saver.client, prefix=":")


async def test_sync_api_raises_actionable_error(saver):
    config = _config(thread="sync")
    with pytest.raises(NotImplementedError, match="async"):
        saver.get_tuple(config)
    with pytest.raises(NotImplementedError, match="async"):
        saver.put(config, _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    with pytest.raises(NotImplementedError, match="async"):
        saver.put_writes(config, [("messages", "v")], task_id="t")
    with pytest.raises(NotImplementedError, match="async"):
        saver.list(config)


async def test_namespace_scoped_delete(saver, prefix):
    await saver.aput(_config(thread="nsy", ns=""), _checkpoint("cp-top"), {"source": "loop"}, {"messages": "1"})
    await saver.aput(_config(thread="nsy", ns="sub"), _checkpoint("cp-sub"), {"source": "loop"}, {"messages": "1"})

    await saver.adelete_thread("nsy")

    assert await _keys(saver, f"{prefix}:cp:*") == []
    assert await _keys(saver, f"{prefix}:idx:*") == []
    assert await saver.client.hlen(f"{prefix}:threads") == 0


async def test_purge_helper_keeps_namespace_clean(saver, prefix):
    await saver.aput(_config(thread="cleanup"), _checkpoint("cp-1"), {"source": "loop"}, {"messages": "1"})
    removed = await purge(saver, prefix)
    assert removed > 0
    assert await _keys(saver, f"{prefix}:*") == []
