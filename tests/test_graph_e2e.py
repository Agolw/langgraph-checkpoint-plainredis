"""End-to-end tests: a real LangGraph ``StateGraph`` compiled with the saver.

These cover what unit-level parity tests cannot: the graph calling
``aput`` / ``aput_writes`` / ``aget_tuple`` / ``alist`` on its own, state surviving
a fresh saver instance (process restart) and history being replayable.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# The end-to-end tests drive a real graph, so they need the full ``langgraph`` package
# (part of the ``dev`` extra). Keep the rest of the suite running without it.
pytest.importorskip("langgraph.graph", reason="graph e2e tests need the langgraph package")

from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

THREAD = "graph-thread"


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


async def _contents(graph, config) -> list[str]:
    state = await graph.aget_state(config)
    return [message.content for message in state.values["messages"]]


async def test_state_persists_across_turns(saver):
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": THREAD}}

    await graph.ainvoke({"messages": [HumanMessage("first")]}, config)
    assert await _contents(graph, config) == ["first", "echo: first"]

    await graph.ainvoke({"messages": [HumanMessage("second")]}, config)
    assert await _contents(graph, config) == [
        "first",
        "echo: first",
        "second",
        "echo: second",
    ]


async def test_state_survives_new_saver_instance(saver, make_saver):
    """A second saver on the same prefix is what a process restart looks like."""
    config = {"configurable": {"thread_id": THREAD}}
    graph = _build_graph(saver)
    await graph.ainvoke({"messages": [HumanMessage("hello")]}, config)
    expected = await _contents(graph, config)

    restarted = _build_graph(make_saver())
    assert await _contents(restarted, config) == expected


async def _snapshots(graph, config) -> list[tuple[list[str], tuple]]:
    """(message contents, next) of every recorded snapshot, newest first."""
    out = []
    async for snapshot in graph.aget_state_history(config):
        out.append(
            ([m.content for m in snapshot.values.get("messages", [])], snapshot.next)
        )
    return out


async def test_history_is_replayable(saver):
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": THREAD}}
    await graph.ainvoke({"messages": [HumanMessage("first")]}, config)
    await graph.ainvoke({"messages": [HumanMessage("second")]}, config)

    history = await _snapshots(graph, config)
    contents = [entry[0] for entry in history]

    # newest snapshot holds the whole conversation and the graph is at rest there
    assert contents[0] == ["first", "echo: first", "second", "echo: second"]
    assert history[0][1] == ()
    # the state as of the end of turn 1 is still in the history
    assert ["first", "echo: first"] in contents
    # the oldest snapshot is the very beginning of the thread
    assert contents[-1] == []
    assert len(history) >= 4


async def test_threads_do_not_leak_into_each_other(saver):
    graph = _build_graph(saver)
    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    await graph.ainvoke({"messages": [HumanMessage("from a")]}, config_a)
    await graph.ainvoke({"messages": [HumanMessage("from b")]}, config_b)

    assert await _contents(graph, config_a) == ["from a", "echo: from a"]
    assert await _contents(graph, config_b) == ["from b", "echo: from b"]


async def test_time_travel_to_earlier_checkpoint(saver):
    """Re-running from the checkpoint that ended turn 1 forks from that state."""
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": THREAD}}
    await graph.ainvoke({"messages": [HumanMessage("first")]}, config)
    await graph.ainvoke({"messages": [HumanMessage("second")]}, config)

    target_config = None
    async for snapshot in graph.aget_state_history(config):
        contents = [m.content for m in snapshot.values.get("messages", [])]
        if contents == ["first", "echo: first"] and snapshot.next == ():
            target_config = snapshot.config
            break
    assert target_config is not None, "turn-1 checkpoint not found in history"

    forked = await graph.ainvoke(
        {"messages": [HumanMessage("rewritten")]}, target_config
    )
    assert [message.content for message in forked["messages"]] == [
        "first",
        "echo: first",
        "rewritten",
        "echo: rewritten",
    ]


async def test_delta_channel_history_uses_async_base_implementation(saver):
    """The saver inherits ``aget_delta_channel_history``; it must not fall back to sync."""
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": THREAD}}
    await graph.ainvoke({"messages": [HumanMessage("first")]}, config)
    await graph.ainvoke({"messages": [HumanMessage("second")]}, config)

    latest = await saver.aget_tuple(config)
    history = await saver.aget_delta_channel_history(
        config=latest.config, channels=["messages"]
    )
    assert set(history) == {"messages"}
    assert isinstance(history["messages"], dict)
