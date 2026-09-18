"""Minimal **synchronous** example: persist a LangGraph agent thread in plain Redis.

The async twin of this file is ``examples/basic.py`` (``AsyncRedisSaver`` +
``await graph.ainvoke``); both use the same key layout, so either class can read
what the other wrote.

Run it against any Redis 5.0+ (a bare ``redis:5`` container is enough - no
modules, no Redis Stack)::

    docker run --rm -d -p 6379:6379 --name plainredis-example redis:5
    python examples/basic_sync.py
"""

from __future__ import annotations

import os

import redis
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_checkpoint_plainredis import RedisSaver

REDIS_URL = os.environ.get("PLAINREDIS_URL", "redis://127.0.0.1:6379/0")
PREFIX = "example:agent-sync"
THREAD_ID = "user-42:session-8"


def reply(state: MessagesState) -> dict:
    """Trivial agent node: echo the last human message."""
    return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}


def build_graph(saver: RedisSaver):
    return (
        StateGraph(MessagesState)
        .add_node("reply", reply)
        .add_edge(START, "reply")
        .add_edge("reply", END)
        .compile(checkpointer=saver)
    )


def main() -> None:
    # Reuse one blocking Redis client for the whole process
    client = redis.Redis.from_url(REDIS_URL, protocol=2, decode_responses=False)
    saver = RedisSaver(client=client, prefix=PREFIX, ttl=7 * 24 * 3600)
    graph = build_graph(saver)

    thread = {"configurable": {"thread_id": THREAD_ID}}

    graph.invoke({"messages": [HumanMessage("hello")]}, thread)
    graph.invoke({"messages": [HumanMessage("second turn")]}, thread)

    state = graph.get_state(thread)
    print("current state:")
    for message in state.values["messages"]:
        print(f"  {message.type:9} | {message.content}")

    print("\nhistory (newest first):")
    for snapshot in graph.get_state_history(thread):  # a plain generator in sync mode
        contents = [m.content for m in snapshot.values.get("messages", [])]
        print(f"  next={str(snapshot.next):14} messages={contents}")

    # A second saver on the same prefix sees the same thread (what a restart looks like)
    other = build_graph(RedisSaver(client=client, prefix=PREFIX))
    resumed = other.get_state(thread)
    print(f"\nresumed by another saver instance: {len(resumed.values['messages'])} messages")

    # Cleanup: drop the thread's data, then the namespace counters
    saver.delete_thread(THREAD_ID)
    leftovers = list(client.scan_iter(match=f"{PREFIX}:*", count=200))
    if leftovers:
        client.delete(*leftovers)
    saver.close()


if __name__ == "__main__":
    main()
