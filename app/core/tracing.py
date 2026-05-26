# app/core/tracing.py
# 全链路追踪 ID 管理
# 职责：
#   1. 提供 ContextVar 存储每个协程独立的 trace_id（多容器并发安全）
#   2. 提供 set_trace_id() / get_trace_id() 工具函数
#   3. 提供 FastAPI 中间件，每个 HTTP 请求入口自动生成并绑定 trace_id
#
# 设计说明：
#   - ContextVar 是 Python 3.7+ 的标准库，协程切换时自动隔离上下文
#   - 每个并发请求都有独立的 trace_id，不会互相污染（对比 threading.local 更安全）
#   - trace_id 写入响应头（X-Trace-ID），方便客户端/API 网关日志关联
#   - 多容器部署时，外部传入同一 trace_id 可聚合所有容器的日志

from __future__ import annotations

import uuid                                          # 生成唯一追踪 ID
from contextvars import ContextVar                   # 协程安全的上下文变量

from fastapi import Request                          # FastAPI 请求对象（中间件使用）
from fastapi.responses import Response               # FastAPI 响应对象（中间件使用）


# ── 模块级 ContextVar（全局单例，每个协程独立存储自己的值）──────────────────────
# default="" 表示未设置 trace_id 时返回空字符串（而非抛 LookupError）
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """
    获取当前协程的 trace_id

    多容器部署说明：
      每个容器副本处理不同请求，各自的协程有独立的 trace_id。
      在相同容器内，同一请求的所有协程（节点函数）共享同一 trace_id。

    Returns:
        当前协程绑定的 trace_id，未设置时返回空字符串
    """
    return _trace_id_var.get()                       # 从当前协程的上下文中读取


def set_trace_id(trace_id: str = None) -> str:
    """
    设置当前协程的 trace_id（每个请求入口调用一次）

    Args:
        trace_id: 指定的 trace_id（如 API 网关或客户端传入）；
                  为 None 时自动生成 UUID 前 8 位（短而唯一，适合日志展示）

    Returns:
        实际设置的 trace_id（用于写入响应头或传递到 LangGraph 状态）
    """
    tid = trace_id or str(uuid.uuid4())[:8]          # UUID 前 8 位：日志中不占太多空间
    _trace_id_var.set(tid)                           # 绑定到当前协程的上下文
    return tid


async def trace_id_middleware(request: Request, call_next) -> Response:
    """
    FastAPI HTTP 中间件：为每个请求自动绑定 trace_id

    执行流程：
      1. 从请求头 X-Trace-ID 读取外部传入的 trace_id（支持 API 网关透传）
      2. 如果没有外部 trace_id，自动生成一个新的 UUID（前 8 位）
      3. 将 trace_id 写入当前协程的 ContextVar（后续所有日志/LangGraph 节点可读取）
      4. 将 trace_id 写入响应头（X-Trace-ID），方便客户端/Nginx 日志关联

    Args:
        request:   FastAPI 请求对象（含请求头）
        call_next: 下一个中间件或路由处理器

    Returns:
        FastAPI 响应对象（含 X-Trace-ID 响应头）
    """
    # 优先使用外部传入的 trace_id（如 API 网关、上游服务的请求链路 ID）
    incoming_tid = request.headers.get("X-Trace-ID")  # 从请求头读取（None 表示未传入）
    tid = set_trace_id(incoming_tid)                   # 设置或生成 trace_id

    # 执行后续中间件和路由处理器
    response = await call_next(request)

    # 将 trace_id 写回响应头（前端/API 网关可用于关联请求日志）
    response.headers["X-Trace-ID"] = tid              # 返回实际使用的 trace_id

    return response
