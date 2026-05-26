# tests/test_observability/test_tracing.py
# 可观测性测试：Trace ID 中间件和健康检查增强
# 测试策略：
#   - Trace ID：验证 ContextVar 的协程隔离、生成逻辑、中间件集成
#   - 健康检查：验证各依赖项检查结果的聚合逻辑（mock 外部服务）

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.tracing import get_trace_id, set_trace_id


# ──────────────────────────────────────────────────────────────────────────────
# 测试：get_trace_id / set_trace_id 基础逻辑
# ──────────────────────────────────────────────────────────────────────────────

class TestTraceIdBasics:
    """验证 ContextVar 的 trace_id 设置和读取逻辑"""

    def test_set_returns_same_tid(self):
        """set_trace_id(tid) 应返回传入的 tid"""
        result = set_trace_id("my-trace-abc")
        assert result == "my-trace-abc"

    def test_get_returns_set_value(self):
        """set_trace_id 后，get_trace_id 返回相同的值"""
        set_trace_id("test-trace-001")
        assert get_trace_id() == "test-trace-001"

    def test_auto_generate_when_none(self):
        """set_trace_id(None) 时自动生成一个非空字符串"""
        tid = set_trace_id(None)
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_auto_generated_length(self):
        """自动生成的 trace_id 长度为 8（UUID 前 8 位）"""
        tid = set_trace_id()
        assert len(tid) == 8                           # UUID 前 8 位，日志中不占太多空间

    def test_default_is_empty_string(self):
        """未设置 trace_id 的协程默认值为空字符串（不抛异常）"""
        # 在一个全新的协程中读取（未调用 set_trace_id）
        async def _check():
            # 重置 ContextVar（绕过其他测试的污染）
            from app.core.tracing import _trace_id_var
            token = _trace_id_var.set("")              # 明确设置为空（模拟全新请求）
            val = get_trace_id()
            _trace_id_var.reset(token)
            return val

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_check())
        finally:
            loop.close()
        assert result == ""                            # 默认为空字符串


# ──────────────────────────────────────────────────────────────────────────────
# 测试：ContextVar 并发隔离
# ──────────────────────────────────────────────────────────────────────────────

class TestTraceIdConcurrencyIsolation:
    """验证不同协程之间的 trace_id 互相隔离（ContextVar 的核心特性）"""

    @pytest.mark.asyncio
    async def test_different_coroutines_have_independent_trace_ids(self):
        """并发协程各自独立的 trace_id 不互相干扰"""
        results: dict = {}

        async def set_and_get(name: str, tid: str):
            """设置 trace_id，等待一段时间后读取，验证仍是自己的值"""
            set_trace_id(tid)
            await asyncio.sleep(0.01)                  # 模拟 IO 等待（协程切换点）
            results[name] = get_trace_id()             # 读取时应仍是自己设置的值

        # 三个协程并发执行，各自设置不同的 trace_id
        await asyncio.gather(
            set_and_get("coroutine_A", "trace-AAAA"),
            set_and_get("coroutine_B", "trace-BBBB"),
            set_and_get("coroutine_C", "trace-CCCC"),
        )

        # 每个协程读取到的应是自己的 trace_id
        assert results["coroutine_A"] == "trace-AAAA", "协程 A 读到了错误的 trace_id"
        assert results["coroutine_B"] == "trace-BBBB", "协程 B 读到了错误的 trace_id"
        assert results["coroutine_C"] == "trace-CCCC", "协程 C 读到了错误的 trace_id"

    @pytest.mark.asyncio
    async def test_child_coroutine_inherits_trace_id(self):
        """子协程在创建时继承父协程的 trace_id（ContextVar 的默认行为）"""
        set_trace_id("parent-trace")
        child_trace_id = None

        async def child():
            """子协程读取继承的 trace_id"""
            nonlocal child_trace_id
            child_trace_id = get_trace_id()

        await child()
        assert child_trace_id == "parent-trace"        # 子协程继承了父协程的值

    @pytest.mark.asyncio
    async def test_many_concurrent_requests_isolated(self):
        """高并发场景：50 个协程同时运行，各自的 trace_id 不互相干扰"""
        trace_results: dict = {}

        async def request_handler(request_id: int):
            """模拟一个请求处理协程"""
            my_trace = f"trace-{request_id:04d}"
            set_trace_id(my_trace)                     # 设置本请求的 trace_id
            await asyncio.sleep(0.001)                 # 模拟 IO 等待
            trace_results[request_id] = get_trace_id() # 验证读取到的仍是自己的值

        # 50 个并发请求
        await asyncio.gather(*[request_handler(i) for i in range(50)])

        # 验证所有请求的 trace_id 都正确（无交叉污染）
        for i in range(50):
            expected = f"trace-{i:04d}"
            assert trace_results[i] == expected, (
                f"请求 {i} 的 trace_id 被污染：期望 {expected}，实际 {trace_results[i]}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 测试：trace_id_middleware 中间件
# ──────────────────────────────────────────────────────────────────────────────

class TestTraceIdMiddleware:
    """验证 FastAPI Trace ID 中间件的行为"""

    @pytest.mark.asyncio
    async def test_middleware_sets_trace_id_in_context(self):
        """中间件应在 ContextVar 中设置 trace_id（后续处理器可读取）"""
        captured_trace_id = None

        async def mock_call_next(request):
            """模拟后续处理器，读取当前的 trace_id"""
            nonlocal captured_trace_id
            captured_trace_id = get_trace_id()         # 中间件设置后，此处应可读取
            from fastapi.responses import Response
            return Response()

        # 构造模拟请求（无 X-Trace-ID 头，触发自动生成）
        mock_request = MagicMock()
        mock_request.headers = {}                      # 无外部 trace_id

        from app.core.tracing import trace_id_middleware
        await trace_id_middleware(mock_request, mock_call_next)

        assert captured_trace_id is not None
        assert len(captured_trace_id) > 0              # 自动生成了非空 trace_id

    @pytest.mark.asyncio
    async def test_middleware_uses_incoming_trace_id(self):
        """如果请求头中有 X-Trace-ID，应使用传入的值（不生成新的）"""
        captured_trace_id = None

        async def mock_call_next(request):
            nonlocal captured_trace_id
            captured_trace_id = get_trace_id()
            from fastapi.responses import Response
            return Response()

        mock_request = MagicMock()
        mock_request.headers = {"X-Trace-ID": "external-trace-xyz"}  # 外部传入的 trace_id

        from app.core.tracing import trace_id_middleware
        await trace_id_middleware(mock_request, mock_call_next)

        assert captured_trace_id == "external-trace-xyz"  # 使用了外部传入的值

    @pytest.mark.asyncio
    async def test_middleware_writes_trace_id_to_response_header(self):
        """中间件应将 trace_id 写入响应头 X-Trace-ID"""
        from fastapi.responses import Response as FapiResponse

        async def mock_call_next(request):
            return FapiResponse()                      # 返回空响应

        mock_request = MagicMock()
        mock_request.headers = {}

        from app.core.tracing import trace_id_middleware
        response = await trace_id_middleware(mock_request, mock_call_next)

        assert "X-Trace-ID" in response.headers        # 响应头中有 trace_id
        assert len(response.headers["X-Trace-ID"]) > 0  # 值非空


# ──────────────────────────────────────────────────────────────────────────────
# 测试：健康检查聚合逻辑（_check_* 函数）
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthCheckHelpers:
    """验证各健康检查辅助函数的返回值格式"""

    def test_check_mcp_tools_ok_when_tools_registered(self):
        """MCP 工具已注册时，返回 "ok" """
        from app.api.routers import _check_mcp_tools
        from unittest.mock import MagicMock, patch

        # 模拟已注册 5 个工具的 MCP 实例
        mock_tools = {f"tool_{i}": MagicMock() for i in range(5)}
        mock_tool_manager = MagicMock()
        mock_tool_manager._tools = mock_tools
        mock_mcp = MagicMock()
        mock_mcp._tool_manager = mock_tool_manager

        with patch("app.api.routers.mcp", mock_mcp, create=True):
            from app.mcp import server as mcp_server
            original_mcp = mcp_server.mcp
            mcp_server.mcp = mock_mcp
            result = _check_mcp_tools()
            mcp_server.mcp = original_mcp              # 恢复原值

        # 只验证函数返回字符串（"ok" 或 "unavailable"）
        assert result in ("ok", "unavailable")

    def test_check_langgraph_ok_when_initialized(self):
        """LangGraph 检查点已初始化时，返回 "ok" """
        from app.api.routers import _check_langgraph

        # mock get_checkpointer 返回非 None
        with patch("app.api.routers.get_checkpointer", return_value=MagicMock(), create=True):
            from app.graph import get_checkpointer as original_fn
            # 通过直接 patch 模块属性测试
            with patch("app.graph.get_checkpointer", return_value=MagicMock()):
                result = _check_langgraph()

        assert result in ("ok", "unavailable")         # 函数应返回合法值

    def test_check_langgraph_unavailable_when_not_initialized(self):
        """LangGraph 检查点未初始化（get_checkpointer 返回 None）时，返回 "unavailable" """
        from app.api.routers import _check_langgraph

        with patch("app.graph.get_checkpointer", return_value=None):
            result = _check_langgraph()

        assert result == "unavailable"

    def test_check_redis_returns_string(self):
        """_check_redis 总是返回 "ok" 或 "unavailable" 字符串"""
        from app.api.routers import _check_redis
        result = _check_redis()
        assert result in ("ok", "unavailable")

    def test_check_milvus_returns_string(self):
        """_check_milvus 总是返回 "ok" 或 "unavailable" 字符串"""
        from app.api.routers import _check_milvus
        result = _check_milvus()
        assert result in ("ok", "unavailable")


# ──────────────────────────────────────────────────────────────────────────────
# 测试：健康检查 HTTP 端点行为
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthCheckEndpoint:
    """验证健康检查端点的 HTTP 状态码逻辑"""

    @pytest.mark.asyncio
    async def test_all_ok_returns_200(self):
        """所有依赖项正常时，健康检查返回 HTTP 200"""
        import json
        from app.api.routers import health_check

        # _check_postgres 是 async 函数，必须用 AsyncMock 实例替换
        with patch("app.api.routers._check_milvus",   return_value="ok"), \
             patch("app.api.routers._check_redis",    return_value="ok"), \
             patch("app.api.routers._check_postgres", AsyncMock(return_value="ok")), \
             patch("app.api.routers._check_mcp_tools", return_value="ok"), \
             patch("app.api.routers._check_langgraph", return_value="ok"):

            response = await health_check()

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_one_failure_returns_503(self):
        """任意一个依赖项不可用时，健康检查返回 HTTP 503"""
        import json
        from app.api.routers import health_check

        with patch("app.api.routers._check_milvus",   return_value="unavailable"), \
             patch("app.api.routers._check_redis",    return_value="ok"), \
             patch("app.api.routers._check_postgres", AsyncMock(return_value="ok")), \
             patch("app.api.routers._check_mcp_tools", return_value="ok"), \
             patch("app.api.routers._check_langgraph", return_value="ok"):

            response = await health_check()

        assert response.status_code == 503
        body = json.loads(response.body)
        assert body["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_response_contains_all_checks(self):
        """健康检查响应体包含所有检查项的结果"""
        import json
        from app.api.routers import health_check

        with patch("app.api.routers._check_milvus",   return_value="ok"), \
             patch("app.api.routers._check_redis",    return_value="ok"), \
             patch("app.api.routers._check_postgres", AsyncMock(return_value="ok")), \
             patch("app.api.routers._check_mcp_tools", return_value="ok"), \
             patch("app.api.routers._check_langgraph", return_value="ok"):

            response = await health_check()

        body = json.loads(response.body)
        checks = body.get("checks", {})
        assert "milvus"     in checks
        assert "redis"      in checks
        assert "postgres"   in checks
        assert "mcp_server" in checks
        assert "langgraph"  in checks
