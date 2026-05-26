# tests/test_mcp/test_mcp_tools.py
# MCP 工具层测试套件
# 测试目标：
#   1. MCP Server 能正确注册全部 11 个工具（不依赖外部服务）
#   2. 每个工具的参数 schema 包含必要字段
#   3. 工具调用在 Mock 环境下返回标准 dict 格式
#   4. 工具调用失败时返回 {success: false, error: ...} 而非抛异常
#
# 运行方式：
#   python -m pytest tests/test_mcp/test_mcp_tools.py -v

import asyncio                                            # 异步测试运行
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch   # Mock 工具：避免调用真实 Agent 和数据库

# 将项目根目录加入 Python 路径（确保 app.* 模块可被导入）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：MCP Server 工具注册验证
# ──────────────────────────────────────────────────────────────────────────────

class TestMCPServerRegistration(unittest.TestCase):
    """验证 MCP Server 能正确注册所有 11 个工具"""

    @classmethod
    def setUpClass(cls):
        """测试类级别初始化：导入 MCP 实例（触发所有工具注册）"""
        # 导入 mcp 实例（会触发 server.py 中的所有 import，注册所有工具）
        from app.mcp.server import mcp
        cls.mcp = mcp                                     # 保存 mcp 实例供各测试方法使用

    def test_mcp_instance_created(self):
        """验证 FastMCP 实例创建成功"""
        self.assertIsNotNone(self.mcp)                    # mcp 实例不应为 None
        self.assertEqual(self.mcp.name, "bill-audit")     # 实例名称应为 "bill-audit"

    def test_all_11_tools_registered(self):
        """验证全部 11 个 Agent 工具都已注册到 MCP Server"""
        # 获取已注册的工具名称集合
        tools = run_async(self.mcp.list_tools())          # list_tools() 是异步方法
        tool_names = {t.name for t in tools}              # 提取工具名称集合

        # 期望注册的 11 个工具名称（与 @mcp.tool() 装饰的函数名一致）
        expected_tools = {
            "parse_bill_document",          # 文档解析工具
            "extract_bill_elements",        # 要素抽取工具
            "check_compliance",             # 合规检索工具
            "analyze_endorsement_chain",    # 背书链分析工具
            "detect_fraud",                 # 欺诈检测工具
            "review_contract",              # 合同审核工具
            "assess_risk",                  # 风险评估工具
            "generate_report",              # 报告生成工具
            "check_bill_issuance",          # 出票预检工具
            "track_bill_flow",              # 流转追踪工具
            "schedule_batch_audit",         # 批量调度工具
        }

        for tool_name in expected_tools:
            self.assertIn(                                 # 逐一验证工具是否已注册
                tool_name,
                tool_names,
                f"工具 '{tool_name}' 未在 MCP Server 中注册",
            )

        self.assertEqual(                                  # 总数必须等于 11
            len(tool_names & expected_tools),
            11,
            f"期望 11 个工具，实际注册了 {len(tool_names & expected_tools)} 个",
        )

    def test_tools_have_descriptions(self):
        """验证所有工具都有非空的描述文字（LLM 需要描述来决定调用哪个工具）"""
        tools = run_async(self.mcp.list_tools())
        for tool in tools:
            self.assertTrue(                               # 工具描述不能为空
                tool.description and len(tool.description.strip()) > 0,
                f"工具 '{tool.name}' 的描述为空",
            )

    def test_tools_have_input_schema(self):
        """验证所有工具都有输入参数 Schema（MCP 协议要求）"""
        tools = run_async(self.mcp.list_tools())
        for tool in tools:
            self.assertIsNotNone(                          # inputSchema 不能为 None
                tool.inputSchema,
                f"工具 '{tool.name}' 缺少 inputSchema",
            )


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：工具参数 Schema 验证
# ──────────────────────────────────────────────────────────────────────────────

class TestMCPToolSchemas(unittest.TestCase):
    """验证各工具的 JSON Schema 包含必要的参数字段"""

    @classmethod
    def setUpClass(cls):
        """加载所有工具定义"""
        from app.mcp.server import mcp
        tools = run_async(mcp.list_tools())
        cls.tool_map = {t.name: t for t in tools}        # 构建 name → tool 映射，方便按名查找

    def _get_schema_properties(self, tool_name: str) -> set:
        """辅助方法：提取工具 inputSchema 中的参数名集合"""
        tool = self.tool_map.get(tool_name)
        self.assertIsNotNone(tool, f"工具 '{tool_name}' 不存在")
        schema = tool.inputSchema                         # inputSchema 是 JSON Schema dict
        props = schema.get("properties", {})             # properties 字段包含所有参数定义
        return set(props.keys())                          # 返回参数名集合

    def test_parse_bill_document_schema(self):
        """验证文档解析工具包含 file_path 必填参数"""
        props = self._get_schema_properties("parse_bill_document")
        self.assertIn("file_path", props, "parse_bill_document 缺少 file_path 参数")

    def test_extract_bill_elements_schema(self):
        """验证要素抽取工具包含 file_path 和 prefilled_element 参数"""
        props = self._get_schema_properties("extract_bill_elements")
        self.assertIn("file_path", props, "extract_bill_elements 缺少 file_path 参数")
        self.assertIn("prefilled_element", props, "extract_bill_elements 缺少 prefilled_element 参数")

    def test_check_compliance_schema(self):
        """验证合规检索工具包含 bill_element 必填参数"""
        props = self._get_schema_properties("check_compliance")
        self.assertIn("bill_element", props, "check_compliance 缺少 bill_element 参数")

    def test_assess_risk_schema(self):
        """验证风险评估工具包含四维输入参数"""
        props = self._get_schema_properties("assess_risk")
        self.assertIn("compliance_summary", props, "assess_risk 缺少 compliance_summary 参数")
        self.assertIn("endorsement_result", props, "assess_risk 缺少 endorsement_result 参数")
        self.assertIn("fraud_result", props, "assess_risk 缺少 fraud_result 参数")

    def test_schedule_batch_audit_schema(self):
        """验证批量调度工具包含 items 列表参数和 concurrency 控制参数"""
        props = self._get_schema_properties("schedule_batch_audit")
        self.assertIn("items", props, "schedule_batch_audit 缺少 items 参数")
        self.assertIn("concurrency", props, "schedule_batch_audit 缺少 concurrency 参数")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3：_base.py 基础工具函数测试
# ──────────────────────────────────────────────────────────────────────────────

class TestMCPBaseUtils(unittest.TestCase):
    """验证共享工具函数的正确性"""

    def test_build_minimal_context_with_task_id(self):
        """验证提供 audit_task_id 时 context 使用该 ID"""
        from app.mcp.tools._base import build_minimal_context
        ctx = build_minimal_context(audit_task_id="test-uuid-001", tenant_id="tenant-a")
        self.assertEqual(ctx.audit_task_id, "test-uuid-001")  # 使用传入的 ID
        self.assertEqual(ctx.tenant_id, "tenant-a")           # 使用传入的租户 ID

    def test_build_minimal_context_auto_uuid(self):
        """验证未提供 audit_task_id 时自动生成合法 UUID"""
        from app.mcp.tools._base import build_minimal_context
        import uuid
        ctx = build_minimal_context()
        # 验证自动生成的 ID 是合法 UUID 格式
        try:
            uuid.UUID(ctx.audit_task_id)                      # 能解析为 UUID 表示格式正确
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False
        self.assertTrue(is_valid_uuid, f"自动生成的 audit_task_id={ctx.audit_task_id} 不是合法 UUID")

    def test_build_minimal_context_shared_data(self):
        """验证传入 shared_data 时正确注入到 context"""
        from app.mcp.tools._base import build_minimal_context
        data = {"bill_element": {"ticket_number": "TEST001"}}
        ctx = build_minimal_context(shared_data=data)
        self.assertEqual(ctx.shared_data["bill_element"]["ticket_number"], "TEST001")

    def test_tool_result_success(self):
        """验证成功格式的 tool_result 结构"""
        from app.mcp.tools._base import tool_result
        result = tool_result(success=True, data={"count": 5})
        self.assertTrue(result["success"])                    # success 为 True
        self.assertEqual(result["data"]["count"], 5)          # data 正确传递
        self.assertIsNone(result["error"])                    # 成功时 error 为 None

    def test_tool_result_failure(self):
        """验证失败格式的 tool_result 结构"""
        from app.mcp.tools._base import tool_result
        result = tool_result(
            success=False,
            error="文件不存在",
            error_code="E_PARSE_FILE_NOT_FOUND",
        )
        self.assertFalse(result["success"])                   # success 为 False
        self.assertIsNone(result["data"])                     # 失败时 data 为 None
        self.assertEqual(result["error"], "文件不存在")        # 错误描述正确
        self.assertEqual(result["error_code"], "E_PARSE_FILE_NOT_FOUND")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 4：工具调用成功路径（Mock Agent）
# ──────────────────────────────────────────────────────────────────────────────

class TestMCPToolCallSuccess(unittest.TestCase):
    """验证工具调用在 Mock 环境下返回标准成功格式"""

    def test_run_agent_tool_success(self):
        """验证 run_agent_tool() 成功路径：Agent 返回成功 → 工具返回 {success: true}"""
        from app.mcp.tools._base import run_agent_tool, build_minimal_context
        from app.agents.base_agent import AgentResult

        # 创建一个总是返回成功的 Mock Agent 实例
        mock_agent = MagicMock()
        mock_agent.agent_name = "test_agent"
        mock_agent.execute = AsyncMock(
            return_value=AgentResult(                         # 模拟 Agent 成功返回
                agent_name="test_agent",
                success=True,
                data={"ticket_number": "BILL001"},
            )
        )

        ctx = build_minimal_context(audit_task_id="task-001")

        # Mock 数据库会话（不连接真实数据库）
        with patch("app.mcp.tools._base.AsyncSessionLocal") as mock_session_factory:
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session_factory.return_value = mock_session  # 会话工厂返回 Mock 会话

            result = run_async(run_agent_tool(mock_agent, ctx, commit=True))

        self.assertTrue(result["success"], f"期望 success=True，实际={result}")
        self.assertEqual(result["data"]["ticket_number"], "BILL001")   # 数据正确传递
        self.assertIsNone(result["error"])                              # 无错误信息

    def test_run_agent_tool_agent_failure(self):
        """验证 Agent 返回失败时，工具返回 {success: false} 而非抛异常"""
        from app.mcp.tools._base import run_agent_tool, build_minimal_context
        from app.agents.base_agent import AgentResult

        mock_agent = MagicMock()
        mock_agent.agent_name = "test_agent"
        mock_agent.execute = AsyncMock(
            return_value=AgentResult(                         # 模拟 Agent 失败返回
                agent_name="test_agent",
                success=False,
                error_code="E_TEST_FAIL",
                error_msg="测试失败原因",
            )
        )

        ctx = build_minimal_context()

        with patch("app.mcp.tools._base.AsyncSessionLocal") as mock_session_factory:
            mock_session = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session_factory.return_value = mock_session

            result = run_async(run_agent_tool(mock_agent, ctx, commit=True))

        self.assertFalse(result["success"], "Agent 失败时工具应返回 success=False")
        self.assertEqual(result["error_code"], "E_TEST_FAIL")          # 错误码正确传递
        self.assertIn("测试失败原因", result["error"])                   # 错误描述正确传递

    def test_run_agent_tool_system_exception(self):
        """验证 Agent 抛出系统异常时，工具捕获并返回 {success: false} 而非向上抛"""
        from app.mcp.tools._base import run_agent_tool, build_minimal_context

        mock_agent = MagicMock()
        mock_agent.agent_name = "test_agent"
        mock_agent.execute = AsyncMock(
            side_effect=ConnectionError("数据库连接失败")     # 模拟系统级异常
        )

        ctx = build_minimal_context()

        with patch("app.mcp.tools._base.AsyncSessionLocal") as mock_session_factory:
            mock_session = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session_factory.return_value = mock_session

            # 关键：不应该抛异常，而是返回失败格式
            result = run_async(run_agent_tool(mock_agent, ctx, commit=True))

        self.assertFalse(result["success"], "系统异常时工具应返回 success=False")
        self.assertEqual(result["error_code"], "MCP_SYSTEM_ERROR")    # 系统级错误码


# ──────────────────────────────────────────────────────────────────────────────
# 测试 5：get_mcp_asgi_app() 挂载接口
# ──────────────────────────────────────────────────────────────────────────────

class TestMCPAsgiApp(unittest.TestCase):
    """验证 MCP ASGI 应用可以正常获取（用于 FastAPI 挂载）"""

    def test_get_mcp_asgi_app_returns_callable(self):
        """验证 get_mcp_asgi_app() 返回可调用的 ASGI 应用"""
        from app.mcp.server import get_mcp_asgi_app
        asgi_app = get_mcp_asgi_app()
        self.assertIsNotNone(asgi_app, "get_mcp_asgi_app() 不应返回 None")
        self.assertTrue(callable(asgi_app), "ASGI 应用必须是可调用对象")


# ──────────────────────────────────────────────────────────────────────────────
# 测试入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
