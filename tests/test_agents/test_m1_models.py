# tests/test_agents/test_m1_models.py
# M1 模块测试：验证 14 张新表字段结构、BaseAgent 基础行为、DAG 执行器逻辑
# 测试策略：
#   - ORM 测试：不连接真实数据库，通过反射检查字段名和类型（纯 Python 测试）
#   - BaseAgent 测试：使用 unittest.mock 模拟数据库会话，测试计时/状态上报/异常兜底
#   - DAG 执行器测试：使用 mock Agent，验证并行执行顺序和失败传播逻辑
# 运行方式：python -m pytest tests/test_agents/test_m1_models.py -v

import asyncio          # 运行 async 测试方法
import unittest         # 测试基础类
from unittest.mock import AsyncMock, MagicMock, patch  # Mock 工具
from typing import Any

# 将项目根目录加入 Python 路径
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import inspect as sa_inspect  # 通过反射获取 ORM 表结构（不连接数据库）
from app.models.agent_models import (
    AuditTask, BillElement, ComplianceCheck, EndorsementChain,
    ContractReview, RiskAssessment, AuditReport,
    FlowTrackingTask, FlowMessage, FraudDetection,
    BatchTask, BatchTaskItem, ErrorCodeMapping, BlacklistEntity,
    AuditTaskStatus, AuditTaskType, RiskLevel, ViolationLevel,
    FlowTrackingStatus, FlowMessageStatus, BatchTaskStatus, EntityType,
)
from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.agents.utils.dag_executor import DAGExecutor, DAGNode, DAGEdge, DAGPlan


# ──────────────────────────────────────────────────────────────────────────────
# 辅助工具：运行 async 测试方法的同步包装
# ──────────────────────────────────────────────────────────────────────────────

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：14 张表字段结构验证
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentModelStructure(unittest.TestCase):
    """
    验证 14 张 ORM 表的字段结构是否符合设计要求
    不连接数据库，通过 SQLAlchemy inspect 直接检查 ORM 元信息
    """

    def _get_column_names(self, model_class) -> set:
        """获取 ORM 模型的所有字段名"""
        mapper = sa_inspect(model_class)  # 获取 ORM mapper（包含所有字段信息）
        return {col.key for col in mapper.columns}  # 返回字段名集合

    def test_audit_task_required_fields(self):
        """验证 audit_tasks 表包含所有必要字段"""
        columns = self._get_column_names(AuditTask)
        required = {  # 根据设计文档的必要字段列表
            "id", "tenant_id", "document_id", "task_type", "status",
            "priority", "dag_plan", "progress_pct", "current_agent",
            "error_code", "error_msg", "started_at", "completed_at", "created_at",
        }
        missing = required - columns  # 找出缺失字段
        self.assertEqual(missing, set(), f"audit_tasks 缺少字段: {missing}")

    def test_bill_element_has_18_fields(self):
        """验证 bill_elements 表包含 18 个要素字段（15 基础 + 3 扩展）"""
        columns = self._get_column_names(BillElement)
        # 15 个基础字段
        base_fields = {
            "ticket_number", "ticket_type", "issue_date", "due_date",
            "amount_numeric", "amount_text", "currency", "drawer",
            "drawer_account", "drawer_bank", "acceptor", "payee",
            "drawee_bank", "endorsers", "maturity_days",
        }
        # 3 个扩展字段（新增）
        extra_fields = {"trade_purpose", "acceptance_clause", "special_remarks"}
        all_18 = base_fields | extra_fields

        missing = all_18 - columns
        self.assertEqual(missing, set(), f"bill_elements 缺少要素字段: {missing}")

    def test_compliance_check_required_fields(self):
        """验证 compliance_checks 表的关键字段"""
        columns = self._get_column_names(ComplianceCheck)
        required = {
            "id", "audit_task_id", "element_field", "is_compliant",
            "violation_level", "rag_query", "rag_answer", "regulation_ref",
        }
        self.assertEqual(required - columns, set())

    def test_endorsement_chain_required_fields(self):
        """验证 endorsement_chains 表的关键字段"""
        columns = self._get_column_names(EndorsementChain)
        required = {
            "id", "audit_task_id", "chain_graph", "is_continuous",
            "violation_codes", "has_cycle", "blank_endorsement_count",
        }
        self.assertEqual(required - columns, set())

    def test_risk_assessment_four_dimensions(self):
        """验证 risk_assessments 表包含四维评分字段"""
        columns = self._get_column_names(RiskAssessment)
        dimensions = {
            "compliance_score", "endorsement_score",
            "contract_score", "fraud_score", "composite_score", "risk_level",
        }
        self.assertEqual(dimensions - columns, set())

    def test_fraud_detection_five_dimensions(self):
        """验证 fraud_detections 表包含五维检测字段"""
        columns = self._get_column_names(FraudDetection)
        dimensions = {
            "seal_score", "seal_check_result",
            "duplicate_score", "duplicate_check_result",
            "tamper_score", "tamper_check_result",
            "network_score", "network_check_result",
            "blacklist_score", "blacklist_check_result",
            "overall_fraud_score",
        }
        self.assertEqual(dimensions - columns, set())

    def test_flow_tracking_task_fields(self):
        """验证 flow_tracking_tasks 表的状态机字段"""
        columns = self._get_column_names(FlowTrackingTask)
        required = {
            "id", "audit_task_id", "business_type", "total_steps",
            "completed_steps", "current_node", "status", "timeout_level",
        }
        self.assertEqual(required - columns, set())

    def test_flow_message_fields(self):
        """验证 flow_messages 表的报文节点字段"""
        columns = self._get_column_names(FlowMessage)
        required = {
            "id", "flow_task_id", "step_index", "node_name",
            "msg_type", "status", "arrived_at", "processing_ms", "is_anomaly",
        }
        self.assertEqual(required - columns, set())

    def test_batch_task_fields(self):
        """验证 batch_tasks 表的批量控制字段"""
        columns = self._get_column_names(BatchTask)
        required = {
            "id", "tenant_id", "task_type", "total_count",
            "completed_count", "failed_count", "concurrency", "status",
        }
        self.assertEqual(required - columns, set())

    def test_error_code_mapping_fields(self):
        """验证 error_code_mappings 表的字典字段"""
        columns = self._get_column_names(ErrorCodeMapping)
        required = {"id", "error_code", "category", "severity", "zh_title", "zh_desc"}
        self.assertEqual(required - columns, set())

    def test_blacklist_entity_fields(self):
        """验证 blacklist_entities 表的黑名单字段"""
        columns = self._get_column_names(BlacklistEntity)
        required = {
            "id", "entity_type", "entity_name", "entity_id",
            "source", "is_active", "is_global", "expired_at",
        }
        self.assertEqual(required - columns, set())

    def test_all_14_tables_registered(self):
        """验证所有 14 张表都已注册到 SQLAlchemy 元数据"""
        from app.models.db_models import Base  # 共用同一个 Base
        registered_tables = set(Base.metadata.tables.keys())

        expected_new_tables = {
            "audit_tasks", "bill_elements", "compliance_checks",
            "endorsement_chains", "contract_reviews", "risk_assessments",
            "audit_reports", "flow_tracking_tasks", "flow_messages",
            "fraud_detections", "batch_tasks", "batch_task_items",
            "error_code_mappings", "blacklist_entities",
        }

        missing = expected_new_tables - registered_tables
        self.assertEqual(missing, set(), f"以下表未注册到 metadata: {missing}")

    def test_enum_values(self):
        """验证枚举类型的值定义正确"""
        # 审核任务状态枚举
        self.assertEqual(AuditTaskStatus.PENDING.value, "pending")
        self.assertEqual(AuditTaskStatus.COMPLETED.value, "completed")
        # 风险等级枚举
        self.assertEqual(RiskLevel.LOW.value, "LOW")
        self.assertEqual(RiskLevel.CRITICAL.value, "CRITICAL")
        # 违规等级枚举
        self.assertEqual(ViolationLevel.SEVERE.value, "severe")
        # 流转状态枚举
        self.assertEqual(FlowTrackingStatus.COMPLETED.value, "completed")
        self.assertEqual(FlowMessageStatus.RESPONDED.value, "responded")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：BaseAgent 行为验证
# ──────────────────────────────────────────────────────────────────────────────

class MockSuccessAgent(BaseAgent):
    """测试用的成功 Agent（始终返回成功）"""
    agent_name = "mock_success_agent"

    async def run(self, ctx: AgentContext, db, **kwargs) -> AgentResult:
        return AgentResult(agent_name=self.agent_name, success=True, data={"result": "ok"})


class MockFailAgent(BaseAgent):
    """测试用的失败 Agent（始终返回失败）"""
    agent_name = "mock_fail_agent"

    async def run(self, ctx: AgentContext, db, **kwargs) -> AgentResult:
        return AgentResult(
            agent_name=self.agent_name, success=False,
            error_code="TEST_ERROR", error_msg="测试失败"
        )


class MockExceptionAgent(BaseAgent):
    """测试用的异常 Agent（抛出未处理异常）"""
    agent_name = "mock_exception_agent"

    async def run(self, ctx: AgentContext, db, **kwargs) -> AgentResult:
        raise ValueError("Agent 内部发生了未预期的错误")  # 模拟未处理异常


class TestBaseAgent(unittest.TestCase):
    """验证 BaseAgent.execute() 的计时、日志、异常兜底行为"""

    def setUp(self):
        """每个测试方法前初始化公共数据"""
        self.ctx = AgentContext(
            audit_task_id="test-task-001",
            tenant_id="test-tenant-001",
        )
        # 模拟数据库会话（不连接真实数据库）
        self.mock_db = AsyncMock()
        self.mock_db.execute = AsyncMock(return_value=None)  # 模拟 execute() 不报错

    def test_successful_agent_returns_success_result(self):
        """验证成功 Agent 返回 success=True 且 data 正确"""
        agent = MockSuccessAgent()
        result = run_async(agent.execute(self.ctx, self.mock_db))

        self.assertTrue(result.success)
        self.assertEqual(result.agent_name, "mock_success_agent")
        self.assertEqual(result.data, {"result": "ok"})

    def test_successful_agent_measures_elapsed_time(self):
        """验证 BaseAgent 自动填充 elapsed_ms 字段"""
        agent = MockSuccessAgent()
        result = run_async(agent.execute(self.ctx, self.mock_db))

        self.assertGreater(result.elapsed_ms, 0.0)   # 耗时应大于 0
        self.assertLess(result.elapsed_ms, 5000.0)   # 但不超过 5 秒（测试环境）

    def test_fail_agent_returns_failure_result(self):
        """验证失败 Agent 返回 success=False 且 error_code 正确"""
        agent = MockFailAgent()
        result = run_async(agent.execute(self.ctx, self.mock_db))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "TEST_ERROR")
        self.assertEqual(result.error_msg, "测试失败")

    def test_exception_agent_is_caught_by_base(self):
        """验证 BaseAgent 捕获子类未处理的异常，返回结构化失败结果而非崩溃"""
        agent = MockExceptionAgent()
        # 不应抛出 ValueError，应被 BaseAgent.execute() 捕获
        result = run_async(agent.execute(self.ctx, self.mock_db))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "SYS_AGENT_EXCEPTION")  # 系统级兜底错误码
        self.assertIn("ValueError", result.error_msg)               # 错误信息包含异常类型

    def test_exception_agent_elapsed_ms_filled(self):
        """验证即使 Agent 抛出异常，elapsed_ms 也被正确填充"""
        agent = MockExceptionAgent()
        result = run_async(agent.execute(self.ctx, self.mock_db))

        self.assertGreater(result.elapsed_ms, 0.0)  # 异常情况下耗时也应记录

    def test_agent_context_fields(self):
        """验证 AgentContext 的字段和默认值"""
        ctx = AgentContext(audit_task_id="task-001", tenant_id="tenant-001")
        self.assertEqual(ctx.audit_task_id, "task-001")
        self.assertEqual(ctx.tenant_id, "tenant-001")
        self.assertIsNone(ctx.bill_record_id)    # 默认为 None
        self.assertIsNone(ctx.document_id)       # 默认为 None
        self.assertEqual(ctx.shared_data, {})    # 默认为空字典

    def test_agent_result_fields(self):
        """验证 AgentResult 的字段和默认值"""
        result = AgentResult(agent_name="test", success=True)
        self.assertEqual(result.agent_name, "test")
        self.assertTrue(result.success)
        self.assertIsNone(result.data)           # 默认为 None
        self.assertIsNone(result.error_code)     # 默认为 None
        self.assertEqual(result.elapsed_ms, 0.0) # 默认为 0


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3：DAG 执行器逻辑验证
# ──────────────────────────────────────────────────────────────────────────────

class TestDAGExecutor(unittest.TestCase):
    """验证 DAGExecutor 的并行执行顺序、失败传播、超时处理"""

    def setUp(self):
        """每个测试方法前准备公共数据"""
        self.ctx = AgentContext(audit_task_id="dag-test-001", tenant_id="t-001")
        self.mock_db = AsyncMock()
        self.mock_db.execute = AsyncMock(return_value=None)

        # 执行顺序记录：验证 Agent 的实际调用顺序
        self.execution_order = []

    def _make_tracking_agent(self, name: str, success: bool = True) -> BaseAgent:
        """
        创建一个追踪执行顺序的 Mock Agent
        """
        execution_order = self.execution_order  # 捕获外部引用

        class TrackingAgent(BaseAgent):
            agent_name = name

            async def run(self, ctx, db, **kwargs) -> AgentResult:
                execution_order.append(name)  # 记录执行顺序
                return AgentResult(agent_name=name, success=success)

        return TrackingAgent()

    def test_single_node_executes(self):
        """验证单节点 DAG 正常执行"""
        registry = {"node_a": self._make_tracking_agent("node_a")}
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[DAGNode(node_id="node_a", agent_name="node_a")],
            edges=[],
            plan_name="single_node_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        self.assertIn("node_a", results)
        self.assertTrue(results["node_a"].success)
        self.assertIn("node_a", self.execution_order)

    def test_serial_execution_respects_dependencies(self):
        """验证串行依赖：B 必须在 A 完成后执行"""
        execution_order = self.execution_order

        class AAgent(BaseAgent):
            agent_name = "node_a"
            async def run(self, ctx, db, **kwargs):
                execution_order.append("a")
                return AgentResult(agent_name="node_a", success=True)

        class BAgent(BaseAgent):
            agent_name = "node_b"
            async def run(self, ctx, db, **kwargs):
                execution_order.append("b")
                return AgentResult(agent_name="node_b", success=True)

        registry = {"node_a": AAgent(), "node_b": BAgent()}
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[
                DAGNode(node_id="node_a", agent_name="node_a"),
                DAGNode(node_id="node_b", agent_name="node_b"),
            ],
            edges=[DAGEdge(from_node="node_a", to_node="node_b")],  # A → B
            plan_name="serial_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        # 验证 A 在 B 之前执行
        self.assertEqual(execution_order[0], "a")
        self.assertEqual(execution_order[1], "b")
        # 验证两个节点都成功
        self.assertTrue(results["node_a"].success)
        self.assertTrue(results["node_b"].success)

    def test_parallel_nodes_both_execute(self):
        """验证并行节点：A 和 B 没有依赖关系，应该都被执行"""
        registry = {
            "node_a": self._make_tracking_agent("node_a"),
            "node_b": self._make_tracking_agent("node_b"),
        }
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[
                DAGNode(node_id="node_a", agent_name="node_a"),
                DAGNode(node_id="node_b", agent_name="node_b"),
            ],
            edges=[],  # 无依赖关系，应并行执行
            plan_name="parallel_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        # 两个节点都应该执行且成功
        self.assertIn("node_a", results)
        self.assertIn("node_b", results)
        self.assertTrue(results["node_a"].success)
        self.assertTrue(results["node_b"].success)
        # 两个节点都应该在执行顺序中
        self.assertIn("node_a", self.execution_order)
        self.assertIn("node_b", self.execution_order)

    def test_critical_failure_skips_dependents(self):
        """验证关键节点失败时，其依赖节点被跳过"""
        registry = {
            "node_a": self._make_tracking_agent("node_a", success=False),  # A 失败
            "node_b": self._make_tracking_agent("node_b", success=True),   # B 依赖 A
        }
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[
                DAGNode(node_id="node_a", agent_name="node_a", is_critical=True),  # 关键节点
                DAGNode(node_id="node_b", agent_name="node_b"),
            ],
            edges=[DAGEdge(from_node="node_a", to_node="node_b")],  # B 依赖 A
            plan_name="failure_propagation_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        # A 失败
        self.assertFalse(results["node_a"].success)
        # B 被跳过（有结果但 success=False）
        self.assertIn("node_b", results)
        self.assertFalse(results["node_b"].success)
        self.assertEqual(results["node_b"].error_code, "AGENT_SKIPPED")
        # B 不应该实际执行
        self.assertNotIn("node_b", self.execution_order)

    def test_non_critical_failure_does_not_skip_dependents(self):
        """验证非关键节点失败时，依赖节点仍然执行"""
        registry = {
            "node_a": self._make_tracking_agent("node_a", success=False),  # A 失败（非关键）
            "node_b": self._make_tracking_agent("node_b", success=True),   # B 依赖 A
        }
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[
                DAGNode(node_id="node_a", agent_name="node_a", is_critical=False),  # 非关键节点
                DAGNode(node_id="node_b", agent_name="node_b"),
            ],
            edges=[DAGEdge(from_node="node_a", to_node="node_b")],
            plan_name="non_critical_failure_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        # A 失败
        self.assertFalse(results["node_a"].success)
        # B 仍然应该执行（因为 A 不是关键节点）
        self.assertTrue(results["node_b"].success)
        self.assertIn("node_b", self.execution_order)

    def test_unregistered_agent_returns_error(self):
        """验证请求未注册的 Agent 时返回错误结果而非崩溃"""
        registry = {}  # 空注册表
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[DAGNode(node_id="ghost", agent_name="nonexistent_agent")],
            edges=[],
            plan_name="missing_agent_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        self.assertIn("ghost", results)
        self.assertFalse(results["ghost"].success)
        self.assertEqual(results["ghost"].error_code, "AGENT_NOT_REGISTERED")

    def test_all_results_returned(self):
        """验证 DAG 执行结束后所有节点（包括失败和跳过的）都有结果返回"""
        registry = {
            "a": self._make_tracking_agent("a", success=False),
            "b": self._make_tracking_agent("b"),
            "c": self._make_tracking_agent("c"),
        }
        executor = DAGExecutor(registry)

        plan = DAGPlan(
            nodes=[
                DAGNode(node_id="a", agent_name="a", is_critical=True),
                DAGNode(node_id="b", agent_name="b"),
                DAGNode(node_id="c", agent_name="c"),
            ],
            edges=[
                DAGEdge(from_node="a", to_node="b"),
                DAGEdge(from_node="a", to_node="c"),
            ],
            plan_name="all_results_test",
        )

        results = run_async(executor.execute(plan, self.ctx, self.mock_db))

        # 所有 3 个节点都应该有结果
        self.assertEqual(len(results), 3)
        self.assertIn("a", results)
        self.assertIn("b", results)
        self.assertIn("c", results)


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)  # verbosity=2 显示每个测试方法名
