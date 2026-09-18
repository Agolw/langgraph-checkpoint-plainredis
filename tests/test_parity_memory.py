"""Behaviour parity with the reference in-memory saver.

Both savers are driven through the identical call sequence and the resulting
``CheckpointTuple``s are compared, so any drift from LangGraph's reference
semantics fails loudly.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

THREAD = "parity"
NS = ""

_CHECKPOINT_IDS = ["cp-0001", "cp-0002", "cp-0003"]


def _make_checkpoint(cid: str, values: dict[str, Any], versions: dict[str, Any]) -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = cid
    checkpoint["channel_values"] = dict(values)
    checkpoint["channel_versions"] = dict(versions)
    return checkpoint


def _config(cid: str | None = None, thread: str = THREAD, ns: str = NS) -> dict:
    config: dict = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
    if cid:
        config["configurable"]["checkpoint_id"] = cid
    return config


async def _seed(saver, thread: str = THREAD) -> list[dict]:
    """Write three chained checkpoints (each one is the parent of the next), with a write."""
    configs = []
    config = _config(thread=thread)
    for step, cid in enumerate(_CHECKPOINT_IDS, start=1):
        checkpoint = _make_checkpoint(
            cid, {"messages": [f"m{step}"]}, {"messages": f"{step}"}
        )
        # passing the previous returned config makes the previous checkpoint the parent
        config = await saver.aput(
            config,
            checkpoint,
            {"source": "loop", "step": step},
            {"messages": f"{step}"},
        )
        await saver.aput_writes(config, [("messages", f"w{step}")], task_id=f"task-{step}")
        configs.append(config)
    return configs


def _comparable(tup) -> dict:
    return {
        "checkpoint_id": tup.checkpoint["id"],
        "channel_values": tup.checkpoint["channel_values"],
        "channel_versions": tup.checkpoint["channel_versions"],
        "metadata": dict(tup.metadata),
        "parent_id": (tup.parent_config or {}).get("configurable", {}).get("checkpoint_id"),
        "pending_writes": sorted(
            (tup.pending_writes or []), key=lambda w: (w[0], str(w[1]))
        ),
    }


@pytest.fixture
def memory_saver() -> InMemorySaver:
    return InMemorySaver()


async def test_get_latest_checkpoint_matches_memory(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)

    ours = await saver.aget_tuple(_config())
    theirs = await memory_saver.aget_tuple(_config())

    assert _comparable(ours) == _comparable(theirs)
    assert ours.checkpoint["channel_values"] == {"messages": ["m3"]}


async def test_get_by_id_and_parent_chain_match_memory(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)

    for cid in _CHECKPOINT_IDS:
        ours = await saver.aget_tuple(_config(cid=cid))
        theirs = await memory_saver.aget_tuple(_config(cid=cid))
        assert _comparable(ours) == _comparable(theirs), cid

    # walking parent links must reproduce the whole timeline
    walked = []
    config = _config(cid=_CHECKPOINT_IDS[-1])
    while config is not None:
        tup = await saver.aget_tuple(config)
        walked.append(tup.checkpoint["id"])
        config = tup.parent_config
    assert walked == list(reversed(_CHECKPOINT_IDS))

    # the same walk on the in-memory saver must agree
    theirs = []
    config = _config(cid=_CHECKPOINT_IDS[-1])
    while config is not None:
        tup = await memory_saver.aget_tuple(config)
        theirs.append(tup.checkpoint["id"])
        config = tup.parent_config
    assert walked == theirs


async def test_missing_checkpoint_returns_none(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)
    for candidate in (saver, memory_saver):
        assert await candidate.aget_tuple(_config(cid="does-not-exist")) is None
        assert await candidate.aget_tuple(_config(thread="unknown-thread")) is None


async def test_list_order_limit_and_before_match_memory(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)

    async def ids(s, **kwargs):
        return [tup.checkpoint["id"] async for tup in s.alist(_config(), **kwargs)]

    async def ids_memory(s, **kwargs):
        return [tup.checkpoint["id"] for tup in s.list(_config(), **kwargs)]

    assert await ids(saver) == await ids_memory(memory_saver) == list(reversed(_CHECKPOINT_IDS))
    assert await ids(saver, limit=2) == await ids_memory(memory_saver, limit=2) == _CHECKPOINT_IDS[::-1][:2]
    assert await ids(saver, limit=0) == []
    assert await ids(saver, before=_config(cid="cp-0002")) == await ids_memory(
        memory_saver, before=_config(cid="cp-0002")
    ) == ["cp-0001"]
    assert await ids(saver, before=_config(cid="cp-0001")) == []


async def test_list_filter_matches_memory(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)

    ours = [tup.checkpoint["id"] async for tup in saver.alist(_config(), filter={"step": 2})]
    theirs = [tup.checkpoint["id"] for tup in memory_saver.list(_config(), filter={"step": 2})]

    assert ours == theirs == ["cp-0002"]


async def test_list_pinned_to_one_checkpoint_matches_memory(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)

    ours = [tup.checkpoint["id"] async for tup in saver.alist(_config(cid="cp-0002"))]
    theirs = [tup.checkpoint["id"] for tup in memory_saver.list(_config(cid="cp-0002"))]

    assert ours == theirs == ["cp-0002"]


async def test_pending_writes_match_memory(saver, memory_saver):
    await _seed(saver)
    await _seed(memory_saver)

    for cid, step in zip(_CHECKPOINT_IDS, (1, 2, 3)):
        ours = await saver.aget_tuple(_config(cid=cid))
        theirs = await memory_saver.aget_tuple(_config(cid=cid))
        assert ours.pending_writes == theirs.pending_writes == [(f"task-{step}", "messages", f"w{step}")]


async def test_cleared_channel_omitted_like_memory(saver, memory_saver):
    """A channel in ``new_versions`` but absent from ``channel_values`` stays empty on read."""
    for candidate in (saver, memory_saver):
        config = await candidate.aput(
            _config(),
            _make_checkpoint("cp-cleared", {}, {"messages": "1", "scratchpad": "1"}),
            {"source": "loop", "step": 1},
            {"messages": "1", "scratchpad": "1"},
        )
    ours = await saver.aget_tuple(config)
    theirs = await memory_saver.aget_tuple(config)

    assert ours.checkpoint["channel_values"] == theirs.checkpoint["channel_values"] == {}
    assert _comparable(ours)["channel_versions"] == _comparable(theirs)["channel_versions"]


async def test_list_without_config_walks_every_thread(saver, memory_saver):
    await _seed(saver, thread="t-a")
    await _seed(saver, thread="t-b")
    await _seed(memory_saver, thread="t-a")
    await _seed(memory_saver, thread="t-b")

    ours = {(t.config["configurable"]["thread_id"], t.checkpoint["id"]) async for t in saver.alist(None)}
    theirs = {(t.config["configurable"]["thread_id"], t.checkpoint["id"]) for t in memory_saver.list(None)}

    assert ours == theirs
    assert len(ours) == 6


async def test_namespaces_are_isolated_like_memory(saver, memory_saver):
    for candidate in (saver, memory_saver):
        await candidate.aput(
            _config(ns="subgraph:node"),
            _make_checkpoint("cp-sub", {"messages": ["sub"]}, {"messages": "1"}),
            {"source": "loop"},
            {"messages": "1"},
        )
        await _seed(candidate)

    ours_sub = await saver.aget_tuple(_config(ns="subgraph:node"))
    theirs_sub = await memory_saver.aget_tuple(_config(ns="subgraph:node"))
    assert _comparable(ours_sub) == _comparable(theirs_sub)
    assert ours_sub.checkpoint["channel_values"] == {"messages": ["sub"]}

    # top-level namespace must not see the subgraph checkpoints
    ours_top = [t.checkpoint["id"] async for t in saver.alist(_config())]
    theirs_top = [t.checkpoint["id"] for t in memory_saver.list(_config())]
    assert ours_top == theirs_top == list(reversed(_CHECKPOINT_IDS))

    # without checkpoint_ns in the config, every namespace of the thread is listed
    ours_all = [t.checkpoint["id"] async for t in saver.alist({"configurable": {"thread_id": THREAD}})]
    theirs_all = [t.checkpoint["id"] for t in memory_saver.list({"configurable": {"thread_id": THREAD}})]
    assert set(ours_all) == set(theirs_all) == {*_CHECKPOINT_IDS, "cp-sub"}
