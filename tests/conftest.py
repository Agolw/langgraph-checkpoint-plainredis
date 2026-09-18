"""Shared fixtures.

Tests talk to a **real Redis** (that is the whole point of the package) in a
throwaway namespace: every test picks a random key prefix and the fixtures delete
``{prefix}:*`` afterwards, so a shared dev instance is never polluted.

Point ``PLAINREDIS_TEST_URL`` at your server, or pass ``--redis-url`` to pytest
(default is database 15 of ``127.0.0.1:6379`` so an existing database 0 is left alone).
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from langgraph_checkpoint_plainredis import AsyncRedisSaver

DEFAULT_TEST_URL = "redis://127.0.0.1:6379/15"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--redis-url",
        action="store",
        default=None,
        help="Redis URL used by the test suite (overrides PLAINREDIS_TEST_URL).",
    )


@pytest.fixture(scope="session")
def redis_url(pytestconfig: pytest.Config) -> str:
    return (
        pytestconfig.getoption("--redis-url")
        or os.environ.get("PLAINREDIS_TEST_URL")
        or DEFAULT_TEST_URL
    )


@pytest.fixture
def prefix() -> str:
    return f"lgtst{uuid.uuid4().hex[:12]}"


async def purge(saver: AsyncRedisSaver, prefix: str) -> int:
    """Delete every key of the test namespace."""
    keys = [key async for key in saver.client.scan_iter(match=f"{prefix}:*", count=500)]
    if keys:
        await saver.client.delete(*keys)
    return len(keys)


@pytest_asyncio.fixture
async def saver(redis_url: str, prefix: str):
    instance = AsyncRedisSaver(url=redis_url, prefix=prefix)
    try:
        yield instance
    finally:
        await purge(instance, prefix)
        await instance.aclose()


@pytest_asyncio.fixture
async def make_saver(redis_url: str, prefix: str):
    """Factory for extra savers sharing one namespace (e.g. simulate a restart)."""
    created: list[AsyncRedisSaver] = []

    def _factory(**kwargs) -> AsyncRedisSaver:
        instance = AsyncRedisSaver(
            url=redis_url, prefix=kwargs.pop("prefix", prefix), **kwargs
        )
        created.append(instance)
        return instance

    try:
        yield _factory
    finally:
        for instance in created:
            await purge(instance, prefix)
            await instance.aclose()
