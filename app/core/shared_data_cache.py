# app/core/shared_data_cache.py
# MCP 跨服务 shared_data 缓存层
# 职责：以 audit_task_id 为键，将 AgentContext.shared_data 存储到 Redis
# 供 MCP Server 在工具执行前读取、执行后写回，替代进程内内存传递

from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis
from loguru import logger

from config.settings import settings

_KEY_PREFIX = "shared_data"

_redis: Optional[aioredis.Redis] = None


async def _get_redis() -> Optional[aioredis.Redis]:
    global _redis
    if _redis is None:
        try:
            _redis = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                max_connections=10,
                socket_connect_timeout=2,
            )
            logger.debug(f"[shared_data_cache] Redis 连接建立 url={settings.REDIS_URL}")
        except Exception as e:
            logger.warning(f"[shared_data_cache] Redis 连接失败，shared_data 将不持久化: {e}")
    return _redis


def _key(audit_task_id: str) -> str:
    return f"{_KEY_PREFIX}:{audit_task_id}"


async def get_shared_data(audit_task_id: str) -> dict:
    """读取指定任务的 shared_data，Redis 不可用或 key 不存在时返回空 dict"""
    r = await _get_redis()
    if r is None:
        return {}
    try:
        raw = await r.get(_key(audit_task_id))
        return json.loads(raw) if raw else {}
    except Exception as e:
        logger.debug(f"[shared_data_cache] get 失败 task={audit_task_id}: {e}")
        return {}


async def set_shared_data(
    audit_task_id: str,
    data: dict,
    ttl: int = 0,
) -> bool:
    """写入（覆盖）指定任务的 shared_data，ttl=0 时读取 settings.SHARED_DATA_TTL"""
    r = await _get_redis()
    if r is None:
        return False
    expire = ttl or settings.SHARED_DATA_TTL
    try:
        raw = json.dumps(data, ensure_ascii=False, default=str)
        await r.set(_key(audit_task_id), raw, ex=expire)
        return True
    except Exception as e:
        logger.debug(f"[shared_data_cache] set 失败 task={audit_task_id}: {e}")
        return False


async def merge_shared_data(
    audit_task_id: str,
    updates: dict,
    ttl: int = 0,
) -> bool:
    """读取现有 shared_data，将 updates 合并后写回（节点完成后局部更新）"""
    existing = await get_shared_data(audit_task_id)
    existing.update(updates)
    return await set_shared_data(audit_task_id, existing, ttl)


async def delete_shared_data(audit_task_id: str) -> bool:
    """任务完成后清理，TTL 到期也会自动清理，此函数用于提前释放"""
    r = await _get_redis()
    if r is None:
        return False
    try:
        await r.delete(_key(audit_task_id))
        return True
    except Exception as e:
        logger.debug(f"[shared_data_cache] delete 失败 task={audit_task_id}: {e}")
        return False