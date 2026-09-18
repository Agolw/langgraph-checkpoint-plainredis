# langgraph-checkpoint-plainredis

[![CI](https://github.com/Agolw/langgraph-checkpoint-plainredis/actions/workflows/ci.yml/badge.svg)](https://github.com/Agolw/langgraph-checkpoint-plainredis/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langgraph-checkpoint-plainredis.svg)](https://pypi.org/project/langgraph-checkpoint-plainredis/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](#兼容性)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> English: [README.en.md](README.en.md)

**基于「裸 Redis」的 LangGraph 异步 checkpointer —— 不需要 RedisJSON、不需要 RediSearch，Redis 5.0+ 即可运行。**

用它可以把 LangGraph Agent 的状态（多会话 thread、checkpoint、pending writes、时间旅行）持久化到
一个**只提供核心命令集**的 Redis 里：不装模块、不上 Redis Stack、不要求 Redis 8。

```python
from langgraph_checkpoint_plainredis import AsyncRedisSaver

saver = AsyncRedisSaver(url="redis://127.0.0.1:6379/0", ttl=7 * 24 * 3600)
graph = builder.compile(checkpointer=saver)

await graph.ainvoke(state, {"configurable": {"thread_id": "user-42:session-7"}})
```

---

## 为什么不用官方 `langgraph-checkpoint-redis`？

官方 Redis checkpointer 本身写得很好，但它**强制要求 Redis 模块**——官方 README 原文：

> **IMPORTANT:** This library requires Redis with the following modules:
> **RedisJSON**（存取 JSON 数据）和 **RediSearch**（检索与索引）。
> Redis 8.0+ 默认自带；低版本需要 Redis Stack，或自行安装模块。

这就排除了一大类真实环境：

* 内网/遗留服务器，Redis 被钉在 5、6、7；
* 托管 Redis 实例不允许 `MODULE LOAD`；
* 刻意不带模块的精简自建容器。

本包就是为这类环境写的：**只用核心数据结构**（string / hash / zset）。

| | `langgraph-checkpoint-plainredis` | `langgraph-checkpoint-redis` |
|:--|:--|:--|
| Redis 模块 | **无需任何模块** | 需要 RedisJSON + RediSearch |
| 最低 Redis | **5.0**（RESP2） | Redis Stack，或 Redis 8.0+ |
| Python 依赖 | `redis`、`langgraph-checkpoint` | 还要 `redisvl`、`orjson` |
| 存储方式 | 核心数据结构 | JSON 文档 + 搜索索引 |
| 同步 API | 未实现（仅异步） | 两者都有 |
| 客户端侧检索 | 注册表 hash + 时间线 zset | RediSearch 索引 |

> 如果你的 Redis 是 8.0+ 或 Redis Stack，**请优先用官方包**（功能更全）；
> 本包的存在意义，是让跑不了官方包的环境也能用上持久化。

## 安装

```bash
pip install langgraph-checkpoint-plainredis
```

要求：Python 3.10+、Redis 5.0+、`langgraph-checkpoint>=4.1,<5`。

## 快速开始

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
    # ttl 可选：线程的 key 存活秒数
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

        # 时间旅行：回放整条历史
        async for snapshot in graph.aget_state_history(config):
            print(snapshot.config["configurable"]["checkpoint_id"], snapshot.next)
    finally:
        await saver.aclose()  # 关闭 Redis 连接


asyncio.run(main())
```

复用已有连接池：直接传客户端（其余连接参数会被忽略）：

```python
import redis.asyncio as aioredis
from langgraph_checkpoint_plainredis import AsyncRedisSaver

client = aioredis.Redis(host="redis.internal", port=6379, db=2, protocol=2)
saver = AsyncRedisSaver(client=client, prefix="myapp:agent")
```

> 只有**异步**接口：请使用 `ainvoke` / `astream`（以及 `aget_state*`）。
> 调用同步的 `get_tuple / put / put_writes / list` 会抛出带说明的 `NotImplementedError`。

## 数据是怎么存的

所有 key 都在 `{prefix}:` 下（`prefix` 默认 `lg`）：

| Key | 类型 | 内容 |
|:--|:--|:--|
| `{prefix}:cp:{thread_id}:{ns}:{checkpoint_id}` | string | checkpoint 载荷（父节点 id + metadata） |
| `{prefix}:blob:{thread_id}:{ns}:{channel}:{version}` | string | 单个 channel 的值 |
| `{prefix}:wr:{thread_id}:{ns}:{checkpoint_id}` | hash | 该 checkpoint 的 pending writes |
| `{prefix}:idx:{thread_id}:{ns}` | zset | 时间线（member = checkpoint id，score = 单调计数器） |
| `{prefix}:threads` | hash | `(thread_id, checkpoint_ns)` 注册表，列表查询用 |
| `{prefix}:seq` | string | 时间线 score 用的单调计数器 |

channel 值按 **(channel, version)** 分开存、**多个 checkpoint 共享**，和 InMemory / SQLite saver 的做法一致：
往消息列表里追加内容，不会把其它未变更 channel 的整份值重复序列化。时间线用 zset 存 score，
因此"取最新 checkpoint"和"取某个 checkpoint 之前的"都是 `O(log n)` 的有序集操作，而不是扫描。

`{prefix}:threads` 与 `{prefix}:seq` 是**命名空间级**的 key：`adelete_thread()` 会清掉某个线程的
checkpoint / blob / writes / 时间线 / 注册表项，若要整个应用下线，直接删命名空间：

```bash
# 用 --scan 而不是 KEYS（KEYS 在大库上会阻塞）
redis-cli --scan --pattern "lg:*" | xargs -r -n 500 redis-cli DEL
```

```python
# 或在 Python 里（本包的连接就能用）
keys = [key async for key in client.scan_iter(match=f"{prefix}:*", count=500)]
if keys:
    await client.delete(*keys)
```

> 这两个 key **故意不设 TTL**：`seq` 是时间线排序的依据，过期归零会让新旧存档的 score 错乱；
> `threads` 过期则会让没有新写入的会话从列表里消失（数据还在，但列不出来）。

## API

`AsyncRedisSaver` 实现 `BaseCheckpointSaver` 的异步接口：

| 方法 | 说明 |
|:--|:--|
| `aget_tuple(config)` | 取最新 checkpoint，或 `checkpoint_id` 指定的那一条 |
| `alist(config, *, filter, before, limit)` | 新→旧列出；`config=None` 时遍历全部 thread/namespace |
| `aput(config, checkpoint, metadata, new_versions)` | 写入 checkpoint 与对应 channel blob |
| `aput_writes(config, writes, task_id, task_path)` | 按 `(task_id, channel)` 幂等，与 `InMemorySaver` 一致 |
| `adelete_thread(thread_id)` | 删除该线程的 checkpoint、blob、writes、时间线与注册表项 |
| `aclose()` | 关闭 Redis 连接 |

同步方法（`get_tuple` / `put` / `put_writes` / `list`）抛出 `NotImplementedError` 并给出替代方案——
请用 `ainvoke` / `astream`（或 `aget_state*`）。

## 语义与保证

* **顺序** —— `alist` 按**写入时间**倒序（时间线 score）。LangGraph 默认 checkpoint id 是单调的 UUID6，
  因此这与参考实现的"按 checkpoint id 倒序"在真实图里等价。
* **`before`** —— 排他（只返回更早的），通过时间线 score 定位；即使 id 不能按字典序比较也正确。
* **`filter`** —— 对 checkpoint metadata 过滤，与参考实现同规则。
* **写入幂等** —— 常规 channel 每个 `(task_id, channel)` 只写一次；特殊 channel
  （`__interrupt__` / `__error__` / `__scheduled__` / `__resume__`）覆盖，与 `InMemorySaver` 相同。
* **被清空的 channel** —— 出现在 `new_versions` 但不在 `channel_values` 里的 channel，
  会写成 "empty" blob，回读时从 `channel_values` 中省略。
* **`ttl`** —— 每次写入都会刷新 checkpoint / blob / 时间线 / writes 四类 key 的过期时间，
  活跃线程不断续期，被抛弃的线程自然消失。
* **删除是精确的** —— thread id 常含 `:`（例如 `f"{user_id}:{session_id}"`），甚至可能含 glob 元字符；
  删除走注册表 + 精确 key，**不用裸模式匹配**。删除线程 `a` 不会碰到线程 `a:b`。
* **注册表自愈** —— 时间线已过期（TTL）的条目会在列表查询时被惰性清理；这一步搭在“本来就要读时间线”的
  那次查询上，因此 `alist(None)` **不会**为每个条目产生额外的 `EXISTS` 往返。
* **兜底删除的前缀风险** —— `adelete_thread()` 正常走注册表精确删除；仅当注册表里没有该线程的条目时
  才回退到模式扫描。回退时会再用注册表核对命中的时间线归属（能对上就跳过别的线程），
  但**若注册表整体缺失**，`a` 与 `a:b` 这类前缀重叠仍可能误删 —— 别手动删 `{prefix}:threads` 里的条目。

## 已知限制与 Roadmap

* **仅异步**（如上）。
* 列表查询靠注册表 hash，而不是服务端索引：`alist(None)` 的代价随**线程数量**线性增长（而非随数据量）。
  按单个线程查询是 `O(log n)`。
* 回读一个 checkpoint 用一次 `MGET` 取全部 blob（不是每个 channel 一次往返）。
* 载荷外面套了一层 JSON+base64，方便用 `redis-cli` 直接查看；直接存原始字节能再省约 25%，有需要再加。
* 未实现：同步 API、类似 `search` 的跨线程查询。
* `adelete_thread()` 每个存档一次 `GET`（为了算出它用了哪些 blob）；代价受**单线程存档数**约束
  （不是全库线程数），且删档是低频操作，暂不优化。

## 兼容性

* **Redis**：5.0 → 8.x，有无模块均可。默认 RESP2（`protocol=2`）；服务端 6+ 想用 RESP3 可传 `protocol=3`。
  CI 会针对 `redis:5`、`redis:7`、`redis:8` 三种容器跑完整测试。
* **Python**：3.10 / 3.11 / 3.12 / 3.13。
* **LangGraph**：`langgraph-checkpoint >= 4.1, < 5`。上游 checkpoint 接口仍在演进，锁大版本让升级可控。

## 安全

checkpoint 使用 LangGraph 的序列化器（默认 `JsonPlusSerializer`），对特殊载荷可能回落到 pickle 编码。
请把 Redis 当作**可信存储**：能写入该 key 空间的人，就可能构造出被你的进程反序列化的载荷。
如果这不接受，可通过 `serde=` 传入更严格的序列化器，并为每个应用分配独立的 `db` / `prefix`。

## 开发与测试

测试需要**真实 Redis**（这正是本包的意义所在）：

```bash
# 1) 准备一个完全没有模块的 Redis（redis:5 即可）
docker run --rm -d -p 6379:6379 --name plainredis-test redis:5

# 2) 建环境并安装（conda 或 venv 都行）
conda create -y -n PlainRedis python=3.11 && conda activate PlainRedis
pip install -e ".[dev]"          # dev 里含 pytest / pytest-asyncio / langgraph

# 3) 跑测试（二选一）
pytest -v --redis-url=redis://127.0.0.1:6379/15
PLAINREDIS_TEST_URL=redis://127.0.0.1:6379/15 pytest -v

# 4) 跑示例
python examples/basic.py         # 可用 PLAINREDIS_URL 覆盖默认 redis://127.0.0.1:6379/0
```

测试默认使用 **db 15** + 每个用例随机 key 前缀，跑完自行清理，因此可以安全地指向共享的开发实例。
其中 `tests/test_parity_memory.py` 会把同一串操作分别打到本包与官方 `InMemorySaver` 上并逐字段比对，
任何语义漂移都会立刻失败。

## License

MIT —— 见 [LICENSE](LICENSE)。
