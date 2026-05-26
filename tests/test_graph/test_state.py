# tests/test_graph/test_state.py
# LangGraph 状态定义测试套件
# 测试目标：
#   1. BillAuditState 字段定义正确（所有必要字段都存在）
#   2. make_initial_state() 正确初始化各字段
#   3. state_to_agent_context() 正确转换 shared_data
#   4. 状态可序列化为 JSON（PostgresSaver 持久化要求）
#   5. failed_nodes / skipped_nodes 字段更新不影响其他字段
#   6. PostgresSaver 初始化逻辑正确（Mock DB，不连真实数据库）
#
# 运行方式：
#   python -m pytest tests/test_graph/test_state.py -v

import asyncio
import json                                               # 验证 JSON 序列化
import sys
import os
import unittest
from unittest.mock import AsyncMock, patch              # Mock 工具

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：BillAuditState 字段定义验证
# ──────────────────────────────────────────────────────────────────────────────

class TestBillAuditStateDefinition(unittest.TestCase):
    """验证 BillAuditState 的字段定义完整且正确"""

    def test_state_is_typed_dict(self):
        """验证 BillAuditState 是 TypedDict 类型（LangGraph 状态基础要求）"""
        from app.graph.state import BillAuditState
        import typing
        # TypedDict 类的 __bases__ 包含 dict
        self.assertTrue(
            issubclass(BillAuditState, dict),
            "BillAuditState 必须继承自 dict（TypedDict 要求）",
        )

    def test_state_has_required_identity_fields(self):
        """验证状态包含任务标识字段（audit_task_id/tenant_id/task_type/trace_id）"""
        from app.graph.state import BillAuditState
        annotations = BillAuditState.__annotations__      # 获取字段注解
        required_fields = ["audit_task_id", "tenant_id", "task_type", "trace_id"]
        for field in required_fields:
            self.assertIn(field, annotations, f"BillAuditState 缺少字段: {field}")

    def test_state_has_all_agent_result_fields(self):
        """验证状态包含所有 Agent 结果字段"""
        from app.graph.state import BillAuditState
        annotations = BillAuditState.__annotations__
        agent_result_fields = [
            "parsed_doc",         # DocumentParserAgent 输出
            "bill_element",       # ElementExtractionAgent 输出
            "compliance_summary", # ComplianceRetrievalAgent 输出
            "endorsement_result", # EndorsementChainAgent 输出
            "fraud_result",       # FraudDetectionAgent 输出
            "contract_result",    # ContractReviewAgent 输出
            "risk_result",        # RiskAssessmentAgent 输出
            "report_result",      # ReportGenerationAgent 输出
            "issuance_result",    # BillIssuanceAgent 输出
            "flow_result",        # FlowTrackingAgent 输出
        ]
        for field in agent_result_fields:
            self.assertIn(field, annotations, f"BillAuditState 缺少 Agent 结果字段: {field}")

    def test_state_has_control_fields(self):
        """验证状态包含执行控制字段（failed_nodes/skipped_nodes/errors）"""
        from app.graph.state import BillAuditState
        annotations = BillAuditState.__annotations__
        control_fields = ["failed_nodes", "skipped_nodes", "errors"]
        for field in control_fields:
            self.assertIn(field, annotations, f"BillAuditState 缺少控制字段: {field}")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：make_initial_state() 初始化函数
# ──────────────────────────────────────────────────────────────────────────────

class TestMakeInitialState(unittest.TestCase):
    """验证 make_initial_state() 正确构造初始状态"""

    def setUp(self):
        """每个测试前准备基础参数"""
        from app.graph.state import make_initial_state
        self.make_initial_state = make_initial_state      # 缓存函数引用

    def test_basic_fields_populated(self):
        """验证基础字段（audit_task_id/tenant_id/task_type/trace_id）正确填入"""
        state = self.make_initial_state(
            audit_task_id="task-001",
            tenant_id="tenant-a",
            task_type="full_audit",
            trace_id="trace-abc",
        )
        self.assertEqual(state["audit_task_id"], "task-001")   # 任务 ID 正确
        self.assertEqual(state["tenant_id"], "tenant-a")       # 租户 ID 正确
        self.assertEqual(state["task_type"], "full_audit")     # 业务类型正确
        self.assertEqual(state["trace_id"], "trace-abc")       # 追踪 ID 正确

    def test_control_fields_initialized_empty(self):
        """验证控制字段初始化为空集合（不是 None）"""
        state = self.make_initial_state(
            audit_task_id="task-002",
            tenant_id="tenant-b",
            task_type="issuance_check",
            trace_id="trace-def",
        )
        self.assertEqual(state["failed_nodes"], [])    # 初始无失败节点
        self.assertEqual(state["skipped_nodes"], [])   # 初始无跳过节点
        self.assertEqual(state["errors"], {})           # 初始无错误

    def test_file_path_optional(self):
        """验证 file_path 可选，不传时为 None"""
        state = self.make_initial_state(
            audit_task_id="task-003",
            tenant_id="tenant-c",
            task_type="full_audit",
            trace_id="trace-ghi",
        )
        self.assertIsNone(state.get("file_path"))       # 不传 file_path 时应为 None

    def test_file_path_passed(self):
        """验证传入 file_path 时正确写入状态"""
        state = self.make_initial_state(
            audit_task_id="task-004",
            tenant_id="tenant-d",
            task_type="full_audit",
            trace_id="trace-jkl",
            file_path="/data/bills/test.pdf",           # 传入文件路径
        )
        self.assertEqual(state["file_path"], "/data/bills/test.pdf")

    def test_prefilled_element_injected(self):
        """验证预填充要素被注入到 bill_element 字段（跳过 OCR 快速路径）"""
        prefilled = {"ticket_number": "BILL-PRE-001", "amount_numeric": 100000}
        state = self.make_initial_state(
            audit_task_id="task-005",
            tenant_id="tenant-e",
            task_type="full_audit",
            trace_id="trace-mno",
            prefilled_element=prefilled,                # 注入预填充要素
        )
        self.assertIsNotNone(state.get("bill_element"), "预填充要素应写入 bill_element 字段")
        self.assertEqual(state["bill_element"]["ticket_number"], "BILL-PRE-001")

    def test_no_prefilled_element_no_bill_element(self):
        """验证未提供预填充要素时 bill_element 字段不存在（保持 None）"""
        state = self.make_initial_state(
            audit_task_id="task-006",
            tenant_id="tenant-f",
            task_type="full_audit",
            trace_id="trace-pqr",
        )
        # 未提供预填充要素时，bill_element 不应出现在 state 中（由 ElementExtractionAgent 填入）
        self.assertIsNone(state.get("bill_element"), "未预填充时 bill_element 应为 None")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3：state_to_agent_context() 转换函数
# ──────────────────────────────────────────────────────────────────────────────

class TestStateToAgentContext(unittest.TestCase):
    """验证 state_to_agent_context() 正确将 state 转换为 AgentContext"""

    def test_basic_fields_transferred(self):
        """验证 audit_task_id 和 tenant_id 正确从 state 传入 context"""
        from app.graph.state import state_to_agent_context, make_initial_state
        state = make_initial_state(
            audit_task_id="task-ctx-001",
            tenant_id="tenant-ctx",
            task_type="full_audit",
            trace_id="trace-ctx",
        )
        ctx = state_to_agent_context(state)
        self.assertEqual(ctx.audit_task_id, "task-ctx-001")  # 任务 ID 正确传入
        self.assertEqual(ctx.tenant_id, "tenant-ctx")         # 租户 ID 正确传入

    def test_agent_results_in_shared_data(self):
        """验证 state 中的 Agent 结果字段正确转入 ctx.shared_data"""
        from app.graph.state import state_to_agent_context
        state = {
            "audit_task_id": "task-ctx-002",
            "tenant_id":     "tenant-ctx",
            "task_type":     "full_audit",
            "trace_id":      "trace-ctx",
            "failed_nodes":  [],
            "skipped_nodes": [],
            "errors":        {},
            # 模拟前序 Agent 已填入的结果
            "bill_element":  {"ticket_number": "TEST-001", "amount_numeric": 50000},
            "compliance_summary": {"ticket_number": {"is_compliant": True}},
        }
        ctx = state_to_agent_context(state)
        # 验证 bill_element 和 compliance_summary 都传入了 shared_data
        self.assertIn("bill_element", ctx.shared_data, "bill_element 应在 shared_data 中")
        self.assertIn("compliance_summary", ctx.shared_data, "compliance_summary 应在 shared_data 中")
        self.assertEqual(ctx.shared_data["bill_element"]["ticket_number"], "TEST-001")

    def test_none_fields_excluded_from_shared_data(self):
        """验证 state 中为 None 的字段不会写入 shared_data"""
        from app.graph.state import state_to_agent_context
        state = {
            "audit_task_id":  "task-ctx-003",
            "tenant_id":      "tenant-ctx",
            "task_type":      "full_audit",
            "trace_id":       "trace-ctx",
            "failed_nodes":   [],
            "skipped_nodes":  [],
            "errors":         {},
            "bill_element":   None,          # 显式为 None 的字段不应写入 shared_data
            "parsed_doc":     None,
        }
        ctx = state_to_agent_context(state)
        self.assertNotIn("bill_element", ctx.shared_data, "None 字段不应写入 shared_data")
        self.assertNotIn("parsed_doc", ctx.shared_data, "None 字段不应写入 shared_data")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 4：JSON 序列化验证（PostgresSaver 持久化要求）
# ──────────────────────────────────────────────────────────────────────────────

class TestStateJsonSerializable(unittest.TestCase):
    """验证 BillAuditState 的所有字段都可以被 JSON 序列化（PostgresSaver 要求）"""

    def test_initial_state_json_serializable(self):
        """验证初始状态（仅基础字段）可以序列化为 JSON"""
        from app.graph.state import make_initial_state
        state = make_initial_state(
            audit_task_id="task-json-001",
            tenant_id="tenant-json",
            task_type="full_audit",
            trace_id="trace-json",
            file_path="/data/test.pdf",
        )
        try:
            json_str = json.dumps(state)                   # 尝试 JSON 序列化
            self.assertIsInstance(json_str, str)           # 序列化结果应为字符串
        except (TypeError, ValueError) as e:
            self.fail(f"初始状态无法 JSON 序列化: {e}")

    def test_full_state_json_serializable(self):
        """验证填充了所有 Agent 结果的完整状态可以序列化为 JSON"""
        from app.graph.state import make_initial_state
        state = make_initial_state(
            audit_task_id="task-json-002",
            tenant_id="tenant-json",
            task_type="full_audit",
            trace_id="trace-json",
        )
        # 模拟所有 Agent 执行完毕后的完整状态
        state["bill_element"] = {"ticket_number": "T001", "amount_numeric": 100000}
        state["compliance_summary"] = {"ticket_number": {"is_compliant": True, "score": 0.95}}
        state["risk_result"] = {"composite_score": 82.5, "risk_level": "LOW"}
        state["failed_nodes"] = []
        state["errors"] = {}

        try:
            json_str = json.dumps(state)                   # 完整状态序列化
            restored = json.loads(json_str)                # 反序列化验证
            self.assertEqual(                              # 验证关键字段恢复正确
                restored["bill_element"]["ticket_number"], "T001"
            )
        except (TypeError, ValueError) as e:
            self.fail(f"完整状态无法 JSON 序列化: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 5：控制字段更新不影响其他字段
# ──────────────────────────────────────────────────────────────────────────────

class TestStateControlFieldUpdate(unittest.TestCase):
    """验证执行控制字段（failed_nodes/errors）的更新语义"""

    def test_append_failed_node_preserves_other_fields(self):
        """验证向 failed_nodes 追加失败节点时，其他字段不受影响"""
        from app.graph.state import make_initial_state
        state = make_initial_state(
            audit_task_id="task-ctrl-001",
            tenant_id="tenant-ctrl",
            task_type="full_audit",
            trace_id="trace-ctrl",
            file_path="/data/test.pdf",
        )
        # 模拟 LangGraph 节点的增量更新（只更新 failed_nodes）
        update = {
            "failed_nodes": state["failed_nodes"] + ["document_parser"],  # 追加失败节点
            "errors": {**state["errors"], "document_parser": "文件不存在"},
        }
        state.update(update)                               # 合并增量更新到状态

        # 验证失败节点已追加
        self.assertIn("document_parser", state["failed_nodes"])
        self.assertEqual(state["errors"]["document_parser"], "文件不存在")

        # 验证其他字段未受影响
        self.assertEqual(state["audit_task_id"], "task-ctrl-001")  # 任务 ID 未变
        self.assertEqual(state["file_path"], "/data/test.pdf")     # 文件路径未变

    def test_skipped_nodes_independent_of_failed_nodes(self):
        """验证 skipped_nodes 和 failed_nodes 相互独立"""
        from app.graph.state import make_initial_state
        state = make_initial_state(
            audit_task_id="task-ctrl-002",
            tenant_id="tenant-ctrl",
            task_type="full_audit",
            trace_id="trace-ctrl",
        )
        state["failed_nodes"] = ["document_parser"]        # 一个失败节点
        state["skipped_nodes"] = ["element_extraction", "compliance_retrieval"]  # 两个跳过节点

        self.assertEqual(len(state["failed_nodes"]), 1)    # 失败节点只有 1 个
        self.assertEqual(len(state["skipped_nodes"]), 2)   # 跳过节点有 2 个
        self.assertNotIn("element_extraction", state["failed_nodes"])  # 跳过≠失败


# ──────────────────────────────────────────────────────────────────────────────
# 测试 6：get_checkpointer() 模块初始化
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointerInit(unittest.TestCase):
    """验证 get_checkpointer() 在未初始化时返回 None"""

    def test_get_checkpointer_returns_none_before_setup(self):
        """在调用 setup_checkpointer() 之前，get_checkpointer() 应返回 None"""
        import importlib
        import app.graph as graph_module                    # 导入 graph 包

        # 强制重置全局 _checkpointer 变量（模拟应用刚启动的状态）
        graph_module._checkpointer = None

        checkpointer = graph_module.get_checkpointer()
        self.assertIsNone(checkpointer, "未初始化时 get_checkpointer() 应返回 None")

    def test_setup_checkpointer_mock(self):
        """验证 setup_checkpointer() 正确创建连接池并调用 PostgresSaver.setup()"""
        import app.graph as graph_module
        from unittest.mock import MagicMock

        # Mock AsyncConnectionPool：open() 是协程，其余是普通方法
        mock_pool = AsyncMock()
        mock_pool.open = AsyncMock()
        mock_pool.close = AsyncMock()

        # Mock AsyncPostgresSaver 实例：setup() 是协程
        mock_saver = AsyncMock()
        mock_saver.setup = AsyncMock()

        with patch("app.graph.AsyncConnectionPool", return_value=mock_pool), \
             patch("app.graph.AsyncPostgresSaver", return_value=mock_saver) as mock_saver_cls:
            run_async(graph_module.setup_checkpointer())   # 执行初始化

        mock_pool.open.assert_called_once()                # 连接池必须被 open()
        mock_saver_cls.assert_called_once_with(mock_pool)  # 连接池被注入 AsyncPostgresSaver
        mock_saver.setup.assert_called_once()              # setup() 被调用一次（建表）
        self.assertIsNotNone(graph_module._checkpointer)   # 全局变量已被设置


# ──────────────────────────────────────────────────────────────────────────────
# 测试入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
