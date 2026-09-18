"""Minimal end-to-end example: persist a LangGraph agent thread in plain Redis.

Run it against any Redis 5.0+ (a bare ``redis:5`` container is enough - no
modules, no Redis Stack)::

    docker run --rm -d -p 6379:6379 --name plainredis-example redis:5
    python examples/basic.py
"""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph_checkpoint_plainredis import AsyncRedisSaver

REDIS_URL = os.environ.get("PLAINREDIS_URL", "redis://127.0.0.1:6379/0")


def reply(state: MessagesState) -> dict:
    """Trivial agent node: echo the last human message."""
    return {"messages": [AIMessage(content=f"echo: {state['messages'][-1].content}")]}


def build_graph(saver: AsyncRedisSaver):
    return (
        StateGraph(MessagesState)
        .add_node("reply", reply)
        .add_edge(START, "reply")
        .add_edge("reply", END)
        .compile(checkpointer=saver)
    )


async def main() -> None:
    # Reuse one Redis client for the whole process (a saver per graph is fine too)
    import redis.asyncio as aioredis

    client = aioredis.Redis.from_url(REDIS_URL, protocol=2, decode_responses=False)
    saver = AsyncRedisSaver(client=client, prefix="example:agent", ttl=7 * 24 * 3600)
    graph = build_graph(saver)

    thread = {"configurable": {"thread_id": "user-42:session-7"}}

    await graph.ainvoke({"messages": [HumanMessage("hello")]}, thread)
    await graph.ainvoke({"messages": [HumanMessage("second turn")]}, thread)

    state = await graph.aget_state(thread)
    print("current state:")
    for message in state.values["messages"]:
        print(f"  {message.type:9} | {message.content}")

    print("\nhistory (newest first):")
    async for snapshot in graph.aget_state_history(thread):
        contents = [m.content for m in snapshot.values.get("messages", [])]
        print(f"  next={str(snapshot.next):14} messages={contents}")

    # A second graph instance on the same storage sees the same thread
    other = build_graph(AsyncRedisSaver(client=client, prefix="example:agent"))
    resumed = await other.aget_state(thread)
    print(f"\nresumed by another saver instance: {len(resumed.values['messages'])} messages")

    # Cleanup: drop the thread's data, then the namespace counters. ``adelete_thread``
    # removes the thread's checkpoints/blob/writes/timeline, while ``{prefix}:seq`` and
    # ``{prefix}:threads`` are namespace-level keys - remove them when tearing down.
    await saver.adelete_thread("user-42:session-7")
    leftovers = [k async for k in client.scan_iter(match="example:agent:*", count=200)]
    if leftovers:
        await client.delete(*leftovers)
    await saver.aclose()


if __name__ == "__main__":
    asyncio.run(main())
