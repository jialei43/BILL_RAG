# app/graph/__init__.py
# LangGraph 图层模块初始化
# 职责：
#   1. 创建全局 AsyncPostgresSaver 实例（检查点持久化，替代内存状态）
#   2. 提供 get_checkpointer() 供 builder.py 构建图时注入
#   3. 提供 setup_checkpointer() / teardown_checkpointer() 供 lifespan 调用
#
# 多容器持久化原理：
#   - 每个 LangGraph 节点执行后，框架自动将完整 state 序列化为 JSON
#   - AsyncPostgresSaver 将 JSON 写入 PostgreSQL 的 checkpoints 表
#   - 其他容器副本通过 thread_id（= audit_task_id）读取相同状态
#   - 任意副本崩溃后，重新分配到其他副本时可从 checkpoints 表恢复

from __future__ import annotations

from typing import Optional                               # 类型注解

from loguru import logger                                # 日志
from config.settings import settings                    # 全局配置（含 DATABASE_URL）

# psycopg3 异步连接池（AsyncPostgresSaver 的底层连接管理）
from psycopg_pool import AsyncConnectionPool
# LangGraph PostgreSQL 检查点持久化实现（异步版本）
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# ── 全局实例（延迟初始化，应用启动时调用 setup_checkpointer()）────────────────
_checkpointer: Optional[AsyncPostgresSaver] = None      # None 表示尚未初始化
_pool: Optional[AsyncConnectionPool] = None             # 连接池引用，关闭时需要显式 close


async def setup_checkpointer() -> AsyncPostgresSaver:
    """
    初始化 AsyncConnectionPool + AsyncPostgresSaver 并创建检查点表结构

    应在 FastAPI 的 lifespan 启动事件中调用（app/main.py）。
    AsyncPostgresSaver.setup() 会自动创建 checkpoints 等表（幂等）。

    Returns:
        AsyncPostgresSaver：已初始化的检查点存储实例
    """
    global _checkpointer, _pool                          # 修改模块级全局变量

    # 将 asyncpg DSN（postgresql+asyncpg://...）转换为 psycopg3 DSN（postgresql://...）
    # AsyncPostgresSaver 使用 psycopg3，不兼容 asyncpg 的 DSN 前缀
    dsn = settings.DATABASE_URL.replace(
        "postgresql+asyncpg://",                         # SQLAlchemy asyncpg 专用前缀
        "postgresql://",                                  # psycopg3 标准前缀
    )

    # 创建 psycopg3 异步连接池
    # autocommit=True：LangGraph 内部自己管理事务，不依赖连接级 autocommit
    # prepare_threshold=0：禁用 prepared statement 缓存，避免多进程 worker 之间的状态冲突
    _pool = AsyncConnectionPool(
        conninfo=dsn,
        max_size=10,                                     # 最多 10 个并发连接（按需调整）
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,                                      # 先不开连接，await open() 后再建立
    )
    await _pool.open()                                   # 建立连接池（异步初始化连接）

    # 将连接池注入 AsyncPostgresSaver（内部所有操作都从池中借用连接）
    _checkpointer = AsyncPostgresSaver(_pool)

    # 执行建表操作（创建 checkpoints / checkpoint_blobs / checkpoint_writes 三张表）
    # 这些表是 LangGraph 持久化状态所必需的，与业务表完全分离
    await _checkpointer.setup()
    logger.info("[graph] LangGraph PostgresSaver 初始化成功，检查点表已就绪")

    return _checkpointer


async def teardown_checkpointer() -> None:
    """
    关闭连接池，释放数据库连接资源

    应在 FastAPI 的 lifespan 关闭事件中调用（app/main.py yield 之后）。
    """
    global _pool
    if _pool is not None:
        await _pool.close()                              # 等待所有借出连接归还后关闭
        logger.info("[graph] LangGraph 连接池已关闭")
        _pool = None


def get_checkpointer() -> Optional[AsyncPostgresSaver]:
    """
    获取已初始化的 PostgresSaver 实例

    Returns:
        已初始化的 AsyncPostgresSaver 实例，或 None（应用启动前不应调用）
    """
    return _checkpointer                                 # 返回全局单例（可能为 None）
