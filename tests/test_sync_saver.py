"""The blocking ``RedisSaver``: parity, a sync graph end-to-end and interoperability.

Sync and async savers share one storage layout, so these tests also pin that
contract: whatever one class writes, the other class must read back identically.
"""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_checkpoint_plainredis import AsyncRedisSaver, RedisSaver

from .conftest import purge_sync
from .test_parity_memory import _CHECKPOINT_IDS, _comparable, _config, _make_checkpoint

pytest.importorskip("langgraph.graph", reason="graph e2e tests need the langgraph package")

THREAD = "sync-thread"


def _cfg(cid: str | None = None, thread: str = THREAD) -> dict:
    """Config pinned to *this* module's thread (the imported helper defaults elsewhere)."""
    return _config(cid=cid, thread=thread)


def _cp(cid: str, values: dict | None = None, versions: dict | None = None) -> dict:
    return _make_checkpoint(
        cid,
        values if values is not None else {"messages": [cid]},
        versions if versions is not None else {"messages": "1"},
    )


def _seed_sync(saver, thread: str = THREAD) -> list[dict]:
    """Same sequence as the async parity helper, using the blocking API."""
    configs = []
    config = _cfg(thread=thread)
    for step, cid in enumerate(_CHECKPOINT_IDS, start=1):
        config = saver.put(
            config, _cp(cid, {"messages": [f"m{step}"]}, {"messages": f"{step}"}),
            {"source": "loop", "step": step}, {"messages": f"{step}"},
        )
        saver.put_writes(config, [("messages", f"w{step}")], task_id=f"task-{step}")
        configs.append(config)
    return configs


# --------------------------------------------------------------- parity


def test_sync_matches_memory_saver(sync_saver):
    memory = InMemorySaver()
    _seed_sync(sync_saver)
    _seed_sync(memory)

    for cid in _CHECKPOINT_IDS:
        ours = sync_saver.get_tuple(_cfg(cid=cid))
        theirs = memory.get_tuple(_cfg(cid=cid))
        assert _comparable(ours) == _comparable(theirs), cid

    # parent chain walks the whole timeline (both implementations agree)
    walked, config = [], _cfg(cid=_CHECKPOINT_IDS[-1])
    while config is not None:
        tup = sync_saver.get_tuple(config)
        walked.append(tup.checkpoint["id"])
        config = tup.parent_config
    assert walked == list(reversed(_CHECKPOINT_IDS))

    theirs_walk, config = [], _cfg(cid=_CHECKPOINT_IDS[-1])
    while config is not None:
        tup = memory.get_tuple(config)
        theirs_walk.append(tup.checkpoint["id"])
        config = tup.parent_config
    assert walked == theirs_walk


def test_sync_list_order_limit_before_filter(sync_saver):
    memory = InMemorySaver()
    _seed_sync(sync_saver)
    _seed_sync(memory)

    def ids(saver, **kwargs):
        return [t.checkpoint["id"] for t in saver.list(_cfg(), **kwargs)]

    assert ids(sync_saver) == ids(memory) == list(reversed(_CHECKPOINT_IDS))
    assert ids(sync_saver, limit=2) == ids(memory, limit=2)
    assert ids(sync_saver, before=_cfg(cid="cp-0002")) == ["cp-0001"]
    assert ids(sync_saver, filter={"step": 3}) == ids(memory, filter={"step": 3}) == ["cp-0003"]


def test_sync_list_without_config_walks_every_thread(sync_saver):
    _seed_sync(sync_saver, thread="s-a")
    _seed_sync(sync_saver, thread="s-b")

    seen = {
        (t.config["configurable"]["thread_id"], t.checkpoint["id"]) for t in sync_saver.list(None)
    }
    assert len(seen) == 6


async def test_sync_and_async_share_one_layout(sync_saver, make_saver):
    """What the blocking saver writes, the async saver must read (and vice versa)."""
    _seed_sync(sync_saver, thread="shared")
    async_saver: AsyncRedisSaver = make_saver()

    tup = await async_saver.aget_tuple(_cfg(cid="cp-0002", thread="shared"))
    assert tup is not None and tup.checkpoint["channel_values"] == {"messages": ["m2"]}
    assert tup.pending_writes == [("task-2", "messages", "w2")]

    config = await async_saver.aput(
        _cfg(thread="shared"),
        _cp("cp-async", {"messages": ["from-async"]}, {"messages": "9"}),
        {"source": "loop", "step": 9},
        {"messages": "9"},
    )
    await async_saver.aput_writes(config, [("messages", "w-async")], task_id="task-async")

    # the blocking saver sees what the async saver just wrote
    latest = sync_saver.get_tuple(_cfg(thread="shared"))
    assert latest.checkpoint["id"] == "cp-async"
    assert latest.checkpoint["channel_values"] == {"messages": ["from-async"]}
    assert latest.pending_writes == [("task-async", "messages", "w-async")]


# ------------------------------------------------------------------ edges


def test_sync_ttl_expires(make_sync_saver, prefix):
    saver = make_sync_saver(ttl=1)
    config = saver.put(_cfg(thread="ttl"), _cp("cp-1"), {"source": "loop"}, {"messages": "1"})
    assert saver.get_tuple(config) is not None

    time.sleep(1.5)

    assert saver.get_tuple(config) is None
    assert list(saver.list(_cfg(thread="ttl"))) == []
    # only the namespace-level counter survives - it deliberately has no TTL
    remaining = [k.decode() for k in saver.client.scan_iter(match=f"{prefix}:*", count=100)]
    assert remaining == [f"{prefix}:seq"]


def test_sync_delete_thread_is_exact(sync_saver, prefix):
    for thread in ("d", "d:e"):
        config = sync_saver.put(_cfg(thread=thread), _cp("cp-1"), {"source": "loop"}, {"messages": "1"})
        sync_saver.put_writes(config, [("messages", "w")], task_id="task")

    sync_saver.delete_thread("d")

    assert sync_saver.get_tuple(_cfg(thread="d")) is None
    keys = list(sync_saver.client.scan_iter(match=f"{prefix}:cp:*", count=100))
    assert f"{prefix}:cp:d::cp-1".encode() not in keys
    assert f"{prefix}:cp:d:e::cp-1".encode() in keys
    assert sync_saver.get_tuple(_cfg(thread="d:e")) is not None


def test_sync_registry_heals_itself(sync_saver, prefix):
    sync_saver.put(_cfg(thread="heal"), _cp("cp-1"), {"source": "loop"}, {"messages": "1"})
    assert sync_saver.client.hlen(f"{prefix}:threads") == 1

    sync_saver.client.delete(f"{prefix}:idx:heal:")

    assert list(sync_saver.list(None)) == []
    assert sync_saver.client.hlen(f"{prefix}:threads") == 0


# ------------------------------------------------------------- wrong API


async def test_wrong_api_raises_pointing_to_the_other_class(sync_saver, saver):
    with pytest.raises(NotImplementedError, match="RedisSaver"):
        saver.get_tuple(_cfg())
    with pytest.raises(NotImplementedError, match="RedisSaver"):
        saver.put(_cfg(), empty_checkpoint(), {"source": "loop"}, {})
    with pytest.raises(NotImplementedError, match="AsyncRedisSaver"):
        await sync_saver.aget_tuple(_cfg())
    with pytest.raises(NotImplementedError, match="AsyncRedisSaver"):
        await sync_saver.aput(_cfg(), empty_checkpoint(), {"source": "loop"}, {})


# ------------------------------------------------------------- sync graph


async def test_wrong_api_on_alist_keeps_the_guidance(sync_saver, saver):
    """`async for` over the wrong class must surface our message, not a bare TypeError.

    This pins the bare `yield` inside ``RedisSaver.alist``: without it the call returns a
    coroutine and `async for` dies with "requires __aiter__" before our class pointer is
    ever raised.
    """
    with pytest.raises(NotImplementedError, match="AsyncRedisSaver"):
        async for _tuple in sync_saver.alist({"configurable": {"thread_id": "x"}}):
            del _tuple
    with pytest.raises(NotImplementedError, match="RedisSaver"):
        for _tuple in saver.list({"configurable": {"thread_id": "x"}}):
            del _tuple


def _build_graph(saver):
    def echo(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}

    return (
        StateGraph(MessagesState)
        .add_node("echo", echo)
        .add_edge(START, "echo")
        .add_edge("echo", END)
        .compile(checkpointer=saver)
    )


def test_sync_graph_invoke_persists_and_resumes(sync_saver, make_sync_saver):
    graph = _build_graph(sync_saver)
    config = {"configurable": {"thread_id": THREAD}}

    graph.invoke({"messages": [HumanMessage("first")]}, config)
    graph.invoke({"messages": [HumanMessage("second")]}, config)

    state = graph.get_state(config)
    assert [m.content for m in state.values["messages"]] == [
        "first",
        "echo: first",
        "second",
        "echo: second",
    ]

    history = list(graph.get_state_history(config))
    assert len(history) >= 4
    assert history[0].next == ()

    # a second saver = process restart
    restarted = _build_graph(make_sync_saver())
    assert [m.content for m in restarted.get_state(config).values["messages"]] == [
        "first",
        "echo: first",
        "second",
        "echo: second",
    ]


async def test_sync_graph_and_async_graph_can_share_a_thread(sync_saver, make_saver):
    """A thread written by a sync graph is readable by an async one (mixed deployments)."""
    sync_graph = _build_graph(sync_saver)
    config = {"configurable": {"thread_id": "mixed"}}
    sync_graph.invoke({"messages": [HumanMessage("from sync")]}, config)

    async_graph = _build_graph(make_saver())
    state = await async_graph.aget_state(config)
    assert [m.content for m in state.values["messages"]] == ["from sync", "echo: from sync"]
