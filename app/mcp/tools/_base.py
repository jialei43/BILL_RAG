# app/mcp/tools/_base.py
# MCP 工具层的共享工具函数
# 职责：
#   1. 提供 build_minimal_context()：为 MCP 工具构造最小化的 AgentContext
#   2. 提供 get_db_session()：异步上下文管理器，为 MCP 工具获取数据库会话
#   3. 提供 tool_result()：统一工具返回格式（{success, data, error}）
#   4. 提供 run_agent_tool()：通用 Agent 调用包装，捕获异常返回标准格式
#
# 设计说明：
#   MCP 工具调用不走 FastAPI 的依赖注入（Depends），
#   因此需要手动创建数据库会话和 AgentContext，
#   避免重复代码，统一在本文件中维护这些底层工具。

from __future__ import annotations

import uuid                                           # 生成临时 audit_task_id（工具单独调用时）
from contextlib import asynccontextmanager            # 异步上下文管理器装饰器
from typing import Any, Optional                     # 类型注解

from loguru import logger                             # 结构化日志
from sqlalchemy.ext.asyncio import AsyncSession       # 异步数据库会话类型

from app.agents.base_agent import AgentContext        # Agent 执行上下文数据类
from app.core.database import AsyncSessionLocal       # 会话工厂（每次调用创建独立会话）


# ──────────────────────────────────────────────────────────────────────────────
# 数据库会话管理
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def get_db_session():
    """
    异步上下文管理器：为 MCP 工具提供独立的数据库会话

    使用方式：
        async with get_db_session() as db:
            result = await agent.run(ctx, db, ...)
            await db.commit()

    设计原则：
        - 每次 MCP 工具调用都创建新会话（隔离，避免跨请求共享状态）
        - 发生异常时自动回滚（保证数据库一致性）
        - finally 块保证会话始终被关闭（归还连接到连接池）
    """
    session: AsyncSession = AsyncSessionLocal()       # 从连接池取出一个新的数据库会话
    try:
        yield session                                  # 将会话交给调用方使用
    except Exception as e:
        await session.rollback()                       # 出现任何异常：回滚未提交的变更
        logger.warning(f"[mcp_base] DB 事务回滚: {type(e).__name__}: {e}")
        raise                                          # 继续向上抛出，让 tool_result 捕获
    finally:
        await session.close()                          # 无论成功失败都关闭会话，释放连接


# ──────────────────────────────────────────────────────────────────────────────
# AgentContext 构建
# ──────────────────────────────────────────────────────────────────────────────

def build_minimal_context(
    audit_task_id: Optional[str] = None,              # 审核任务 ID，为空则生成临时 UUID
    tenant_id: str = "mcp_caller",                    # 租户 ID，MCP 直接调用时使用默认值
    shared_data: Optional[dict] = None,               # 初始共享数据（可注入预填充要素）
) -> AgentContext:
    """
    构造最小化的 AgentContext，供 MCP 工具在没有 HTTP 请求上下文时使用

    Args:
        audit_task_id: 任务 UUID；为 None 时自动生成（适合单独调用某个工具的场景）
        tenant_id:     租户标识；MCP 外部调用场景下使用 "mcp_caller"
        shared_data:   初始共享数据字典；用于传递前序 Agent 的结果

    Returns:
        AgentContext 实例，可直接传入任何 Agent 的 run() 方法
    """
    return AgentContext(
        audit_task_id=audit_task_id or str(uuid.uuid4()),  # 无 task_id 时生成临时 UUID
        tenant_id=tenant_id,                               # 租户隔离标识
        shared_data=shared_data or {},                     # 共享数据初始化为空字典
    )


# ──────────────────────────────────────────────────────────────────────────────
# 标准化工具返回格式
# ──────────────────────────────────────────────────────────────────────────────

def tool_result(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
) -> dict:
    """
    构造统一的 MCP 工具返回格式

    MCP 工具必须返回可 JSON 序列化的 dict。
    统一格式方便 LangGraph 节点和外部 LLM 客户端判断工具执行结果。

    Returns:
        {
          "success": bool,        # True=成功，False=失败
          "data": Any,            # 成功时的输出数据（可为 dict/list/str）
          "error": str|None,      # 失败时的错误描述
          "error_code": str|None  # 失败时的错误码（便于程序化处理）
        }
    """
    return {
        "success": success,       # 执行是否成功
        "data": data,             # 成功时的业务数据
        "error": error,           # 错误描述（失败时填写）
        "error_code": error_code, # 错误码（失败时填写，如 "E_PARSE_NO_FILE"）
    }


# ──────────────────────────────────────────────────────────────────────────────
# 通用 Agent 调用包装
# ──────────────────────────────────────────────────────────────────────────────

async def run_agent_tool(
    agent_instance,                    # Agent 实例（如 DocumentParserAgent()）
    ctx: AgentContext,                 # 执行上下文
    commit: bool = True,               # 是否在执行后提交事务（默认 True）
    **kwargs                           # 透传给 Agent.run() 的业务参数
) -> dict:
    """
    通用 Agent 工具执行包装：创建 DB 会话 → 调用 Agent.execute() → 返回标准格式

    Args:
        agent_instance: 已实例化的 Agent 对象
        ctx:            执行上下文（含 audit_task_id 和 shared_data）
        commit:         True=执行成功后提交事务；False=只读场景不提交
        **kwargs:       透传给 Agent.run() 的额外参数（如 file_path、contract_text）

    Returns:
        tool_result() 格式的 dict
    """
    agent_name = getattr(agent_instance, "agent_name", "unknown")  # 获取 Agent 名称用于日志

    try:
        async with get_db_session() as db:                          # 创建独立数据库会话
            result = await agent_instance.execute(ctx, db, **kwargs)  # 调用 Agent 执行（含计时+日志）

            if result.success and commit:
                await db.commit()                                   # 执行成功时提交数据库变更

            if not result.success:
                # Agent 执行失败：返回失败格式（不抛异常，保持 MCP 工具健壮性）
                logger.warning(
                    f"[mcp_tool:{agent_name}] 执行失败 "
                    f"error_code={result.error_code} msg={result.error_msg}"
                )
                return tool_result(
                    success=False,
                    error=result.error_msg,
                    error_code=result.error_code,
                )

            # 执行成功：返回 Agent 的 data 字段
            return tool_result(
                success=True,
                data=result.data,                                   # Agent 的业务输出数据
            )

    except Exception as e:
        # 系统级异常（数据库连接失败、网络超时等）
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"[mcp_tool:{agent_name}] 系统级异常: {error_msg}")
        return tool_result(
            success=False,
            error=error_msg,
            error_code="MCP_SYSTEM_ERROR",                          # 系统级错误码
        )
