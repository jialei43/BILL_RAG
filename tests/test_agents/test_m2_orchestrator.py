# tests/test_agents/test_m2_orchestrator.py
# M2 模块测试：验证 OrchestratorAgent 的 LangGraph 执行流程
# 测试策略：
#   - 图节点信息：验证每种业务类型返回正确的节点列表
#   - 流式执行：mock build_audit_graph 和 graph.astream()，验证状态转变
#   - 失败传播：关键节点失败 → 整体任务标记为 FAILED
#   - shared_data 同步：LangGraph 最终状态正确写回 ctx.shared_data
#   - run_for_chat()：超时保护和 AuditTask 创建验证

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.base_agent import AgentContext
from app.agents.orchestrator_agent import OrchestratorAgent
from app.models.agent_models import AuditTaskType, AuditTaskStatus


# ──────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_db():
    """创建模拟数据库会话（支持 execute、add、flush 等常用操作）"""
    mock_db = AsyncMock()
    mock_db.execute  = AsyncMock(return_value=MagicMock())  # 模拟 ORM update 返回
    mock_db.add      = MagicMock()                          # 同步方法（AuditTask 插入）
    mock_db.flush    = AsyncMock()                          # 异步 flush（run_for_chat 使用）
    return mock_db


def make_ctx(task_id: str = "test-task-001", shared: dict = None) -> AgentContext:
    """创建测试用 AgentContext"""
    return AgentContext(
        audit_task_id=task_id,
        tenant_id="test-tenant",
        shared_data=shared or {},
    )


def make_mock_graph(events: list[dict]):
    """
    构建产生指定事件序列的 Mock LangGraph 图

    每个事件格式：{node_name: state_delta_dict}
    astream() 是异步生成器，每次 yield 一个事件（模拟 LangGraph 流式执行）
    """
    async def _astream(state, config=None):
        for event in events:
            yield event

    mock_graph = MagicMock()
    mock_graph.astream = _astream  # 替换为异步生成器函数
    return mock_graph


# 构造成功节点事件的辅助函数
def node_event(node_name: str, **result_fields) -> dict:
    """构造节点成功事件（包含空的 failed_nodes 和 errors）"""
    delta = {**result_fields}
    if "failed_nodes" not in delta:
        delta["failed_nodes"] = []
    if "errors" not in delta:
        delta["errors"] = {}
    return {node_name: delta}


def failed_node_event(node_name: str, failed_node_key: str, error_msg: str) -> dict:
    """构造节点失败事件"""
    return {
        node_name: {
            "failed_nodes": [failed_node_key],
            "errors": {failed_node_key: error_msg},
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# 测试：_get_graph_nodes 节点列表
# ──────────────────────────────────────────────────────────────────────────────

class TestGetGraphNodes:
    """验证每种业务类型返回正确的图节点列表"""

    def test_full_audit_has_contract_node(self):
        """全流程审核应包含 contract 节点"""
        nodes = OrchestratorAgent._get_graph_nodes("full_audit")
        assert "contract" in nodes

    def test_issuance_check_has_no_contract_node(self):
        """出票预检不应包含 contract 节点"""
        nodes = OrchestratorAgent._get_graph_nodes("issuance_check")
        assert "contract" not in nodes

    def test_endorsement_has_no_contract_node(self):
        """背书转让不应包含 contract 节点"""
        nodes = OrchestratorAgent._get_graph_nodes("endorsement")
        assert "contract" not in nodes

    def test_all_types_start_with_parse_extract(self):
        """所有业务类型都以 parse → extract 开头"""
        for task_type in ["full_audit", "issuance_check", "discount_apply",
                          "acceptance_prompt", "endorsement",
                          "payment_prompt", "pledge", "collection"]:
            nodes = OrchestratorAgent._get_graph_nodes(task_type)
            assert nodes[0] == "parse",   f"{task_type}: 第一个节点应为 parse"
            assert nodes[1] == "extract", f"{task_type}: 第二个节点应为 extract"

    def test_all_types_end_with_risk_report(self):
        """所有业务类型都以 risk → report 结束"""
        for task_type in ["full_audit", "issuance_check", "discount_apply",
                          "acceptance_prompt", "endorsement",
                          "payment_prompt", "pledge", "collection"]:
            nodes = OrchestratorAgent._get_graph_nodes(task_type)
            assert nodes[-1] == "report", f"{task_type}: 最后节点应为 report"
            assert nodes[-2] == "risk",   f"{task_type}: 倒数第二节点应为 risk"

    def test_full_audit_has_more_nodes_than_endorsement(self):
        """全流程审核节点数 ≥ 背书转让（全流程最完整）"""
        full = OrchestratorAgent._get_graph_nodes("full_audit")
        endorse = OrchestratorAgent._get_graph_nodes("endorsement")
        assert len(full) >= len(endorse)

    def test_unknown_type_falls_back_gracefully(self):
        """未知业务类型应降级返回有效节点列表"""
        nodes = OrchestratorAgent._get_graph_nodes("unknown_xyz")
        assert len(nodes) > 0                          # 降级后仍返回非空列表
        assert "parse" in nodes
        assert "report" in nodes


# ──────────────────────────────────────────────────────────────────────────────
# 测试：_sync_state_to_context
# ──────────────────────────────────────────────────────────────────────────────

class TestSyncStateToContext:
    """验证 LangGraph 最终状态正确写回 ctx.shared_data"""

    def test_result_fields_copied_to_shared_data(self):
        """状态中的 Agent 结果字段应复制到 shared_data"""
        orch = OrchestratorAgent()
        ctx = make_ctx()
        final_state = {
            "bill_element":       {"drawer": "公司A"},
            "compliance_summary": {"score": 95},
            "risk_result":        {"grade": "A"},
            "failed_nodes":       [],
            "skipped_nodes":      [],
            "errors":             {},
        }
        orch._sync_state_to_context(final_state, ctx)

        assert ctx.shared_data["bill_element"]       == {"drawer": "公司A"}
        assert ctx.shared_data["compliance_summary"] == {"score": 95}
        assert ctx.shared_data["risk_result"]        == {"grade": "A"}

    def test_none_fields_not_copied(self):
        """值为 None 的字段不应写入 shared_data"""
        orch = OrchestratorAgent()
        ctx = make_ctx()
        final_state = {
            "bill_element":  None,              # None 字段不应写入
            "risk_result":   {"grade": "B"},    # 有值的字段应写入
            "failed_nodes":  [],
            "skipped_nodes": [],
            "errors":        {},
        }
        orch._sync_state_to_context(final_state, ctx)

        assert "bill_element" not in ctx.shared_data    # None 字段未写入
        assert ctx.shared_data["risk_result"] == {"grade": "B"}

    def test_control_fields_always_synced(self):
        """控制字段（failed_nodes/skipped_nodes/errors）始终写入 shared_data"""
        orch = OrchestratorAgent()
        ctx = make_ctx()
        final_state = {
            "failed_nodes":  ["document_parser"],
            "skipped_nodes": ["compliance_retrieval"],
            "errors":        {"document_parser": "文件不存在"},
        }
        orch._sync_state_to_context(final_state, ctx)

        assert ctx.shared_data["failed_nodes"]  == ["document_parser"]
        assert ctx.shared_data["skipped_nodes"] == ["compliance_retrieval"]
        assert ctx.shared_data["errors"]["document_parser"] == "文件不存在"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：run() 完整编排流程
# ──────────────────────────────────────────────────────────────────────────────

class TestOrchestratorRun:
    """验证 OrchestratorAgent.run() 的完整流程"""

    @pytest.mark.asyncio
    async def test_all_success_returns_completed(self):
        """所有节点成功时，结果 success=True，data.status=COMPLETED"""
        events = [
            node_event("parse",    parsed_doc={}),
            node_event("extract",  bill_element={"drawer": "公司A"}),
            node_event("parallel", compliance_summary={}, endorsement_result={}, fraud_result={}),
            node_event("contract", contract_result={}),
            node_event("risk",     risk_result={"grade": "A"}),
            node_event("report",   report_result={"summary": "审核通过"}),
        ]
        mock_graph = make_mock_graph(events)

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            ctx    = make_ctx()
            db     = make_mock_db()
            result = await orch.run(ctx, db, task_type=AuditTaskType.FULL_AUDIT)

        assert result.success                                   # 编排成功
        assert result.data["status"] == AuditTaskStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_critical_failure_returns_failed(self):
        """关键节点（parse/extract/risk）失败时，result.success=False"""
        events = [
            # extract 关键节点失败（failed_nodes 包含 element_extraction）
            node_event("parse",   parsed_doc={}),
            failed_node_event("extract", "element_extraction", "OCR 失败"),
        ]
        mock_graph = make_mock_graph(events)

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            ctx    = make_ctx()
            db     = make_mock_db()
            result = await orch.run(ctx, db, task_type=AuditTaskType.FULL_AUDIT)

        assert not result.success                               # 编排失败
        assert result.data["status"] == AuditTaskStatus.FAILED.value
        assert "element_extraction" in result.data["failed_nodes"]

    @pytest.mark.asyncio
    async def test_non_critical_failure_still_completes(self):
        """非关键节点（fraud/contract等）失败时，如果关键节点均通过，result.success=True"""
        events = [
            node_event("parse",    parsed_doc={}),
            node_event("extract",  bill_element={}),
            # fraud_detection 失败（非关键节点），compliance 和 endorse 成功
            {"parallel": {
                "compliance_summary": {},
                "endorsement_result": {},
                "failed_nodes": ["fraud_detection"],    # fraud 失败（非关键）
                "errors": {"fraud_detection": "欺诈服务超时"},
            }},
            node_event("contract", contract_result={}),
            node_event("risk",     risk_result={}),
            node_event("report",   report_result={}),
        ]
        mock_graph = make_mock_graph(events)

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            ctx    = make_ctx()
            db     = make_mock_db()
            result = await orch.run(ctx, db, task_type=AuditTaskType.FULL_AUDIT)

        # fraud_detection 不是关键节点（不在 _get_critical_nodes 返回的列表中）
        assert result.success                                   # 整体仍然成功
        assert result.data["status"] == AuditTaskStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_state_synced_to_shared_data(self):
        """执行完成后，LangGraph 状态应同步到 ctx.shared_data"""
        events = [
            node_event("parse",    parsed_doc={"text": "票据"}),
            node_event("extract",  bill_element={"drawer": "公司A"}),
            node_event("parallel", compliance_summary={"score": 95}, endorsement_result={}, fraud_result={}),
            node_event("contract", contract_result={"match_score": 0.9}),
            node_event("risk",     risk_result={"grade": "A", "risk_level": "LOW"}),
            node_event("report",   report_result={"summary": "通过"}),
        ]
        mock_graph = make_mock_graph(events)

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            ctx    = make_ctx()
            db     = make_mock_db()
            await orch.run(ctx, db, task_type=AuditTaskType.FULL_AUDIT)

        # 验证关键结果字段已写入 shared_data
        assert ctx.shared_data.get("bill_element")  is not None
        assert ctx.shared_data.get("risk_result")   is not None
        assert ctx.shared_data.get("report_result") is not None

    @pytest.mark.asyncio
    async def test_build_audit_graph_called_with_correct_task_type(self):
        """验证 build_audit_graph 以正确的 task_type 调用"""
        events = [node_event("parse", parsed_doc={})]  # 最简单的执行流（parse 完就结束）
        mock_graph = make_mock_graph(events)

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph) as mock_build, \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch = OrchestratorAgent()
            ctx  = make_ctx()
            db   = make_mock_db()
            await orch.run(ctx, db, task_type=AuditTaskType.ISSUANCE_CHECK)

        # 验证调用了 build_audit_graph，且 task_type 为 "issuance_check"
        mock_build.assert_called_once()
        call_args = mock_build.call_args
        assert call_args[0][0] == "issuance_check"    # 第一个位置参数

    @pytest.mark.asyncio
    async def test_db_status_updated_running_and_completed(self):
        """编排过程中应依次更新状态为 RUNNING，再更新为 COMPLETED"""
        events = [
            node_event("parse",   parsed_doc={}),
            node_event("extract", bill_element={}),
            node_event("parallel", compliance_summary={}, endorsement_result={}, fraud_result={}),
            node_event("contract", contract_result={}),
            node_event("risk",    risk_result={}),
            node_event("report",  report_result={}),
        ]
        mock_graph = make_mock_graph(events)
        db = make_mock_db()

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch = OrchestratorAgent()
            ctx  = make_ctx()
            await orch.run(ctx, db, task_type=AuditTaskType.FULL_AUDIT)

        # 验证至少有 3 次 db.execute 调用（RUNNING + dag_plan + 进度更新 + COMPLETED）
        assert db.execute.call_count >= 3

    @pytest.mark.asyncio
    async def test_completed_nodes_tracked(self):
        """验证 result.data.completed_nodes 包含所有执行过的节点"""
        events = [
            node_event("parse",    parsed_doc={}),
            node_event("extract",  bill_element={}),
            node_event("parallel", compliance_summary={}, endorsement_result={}, fraud_result={}),
            node_event("contract", contract_result={}),
            node_event("risk",     risk_result={}),
            node_event("report",   report_result={}),
        ]
        mock_graph = make_mock_graph(events)

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            ctx    = make_ctx()
            db     = make_mock_db()
            result = await orch.run(ctx, db, task_type=AuditTaskType.FULL_AUDIT)

        completed = result.data["completed_nodes"]
        assert "parse"   in completed
        assert "extract" in completed
        assert "report"  in completed
        assert len(completed) == 6                              # 6 个节点全部完成


# ──────────────────────────────────────────────────────────────────────────────
# 测试：run_for_chat() 聊天场景入口
# ──────────────────────────────────────────────────────────────────────────────

class TestRunForChat:
    """验证 run_for_chat() 的超时保护、任务创建和报告生成"""

    @pytest.mark.asyncio
    async def test_success_returns_conclusion_dict(self):
        """成功执行时返回包含 task_id / conclusion / risk_level 的 dict"""
        events = [
            node_event("parse",    parsed_doc={}),
            node_event("extract",  bill_element={"ticket_number": "TEST-001"}),
            node_event("parallel", compliance_summary={}, endorsement_result={}, fraud_result={}),
            node_event("contract", contract_result={}),
            node_event("risk",     risk_result={"risk_level": "LOW", "composite_score": 90}),
            node_event("report",   report_result={}),
        ]
        mock_graph = make_mock_graph(events)
        db = make_mock_db()

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            result = await orch.run_for_chat(
                task_type_str="full_audit",
                tenant_id="test-tenant",
                db=db,
            )

        assert "task_id"     in result
        assert "conclusion"  in result
        assert "risk_level"  in result
        assert "issues"      in result
        assert "summary"     in result
        assert result["timed_out"] is False

    @pytest.mark.asyncio
    async def test_timeout_returns_pending_dict(self):
        """执行超时时返回包含 timed_out=True 的 dict（不抛异常）"""
        async def _astream(state, config=None):
            await asyncio.sleep(10)                 # 模拟超时（睡很长时间）
            yield node_event("parse", parsed_doc={}) # 永远不会到这里

        mock_graph = MagicMock()
        mock_graph.astream = _astream
        db = make_mock_db()

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch   = OrchestratorAgent()
            result = await orch.run_for_chat(
                task_type_str="full_audit",
                tenant_id="test-tenant",
                db=db,
                timeout_seconds=0.05,               # 极短超时（50ms）触发超时
            )

        assert result["timed_out"] is True
        assert "task_id" in result                  # 任务 ID 仍应返回
        assert result["conclusion"] == "审核中"

    @pytest.mark.asyncio
    async def test_invalid_task_type_falls_back_to_full_audit(self):
        """未知业务类型应降级为 full_audit（不抛异常）"""
        events = [node_event("parse", parsed_doc={})]
        mock_graph = make_mock_graph(events)
        db = make_mock_db()

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph) as mock_build, \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch = OrchestratorAgent()
            await orch.run_for_chat(
                task_type_str="unknown_invalid_type",
                tenant_id="test-tenant",
                db=db,
            )

        # 验证降级为 full_audit
        call_args = mock_build.call_args
        assert call_args[0][0] == "full_audit"      # 降级后使用 full_audit 图

    @pytest.mark.asyncio
    async def test_prefilled_element_passed_to_state(self):
        """预填充的票据要素（bill_element_dict）应写入 LangGraph 初始状态"""
        events = [node_event("parse", parsed_doc={})]
        mock_graph = make_mock_graph(events)
        db = make_mock_db()

        # 捕获 build_audit_graph 的调用参数（graph.astream 的第一个参数是 initial_state）
        captured_initial_state = {}

        async def _capture_astream(state, config=None):
            captured_initial_state.update(state)    # 捕获传入的初始状态
            yield node_event("parse", parsed_doc={})

        mock_graph.astream = _capture_astream

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch = OrchestratorAgent()
            await orch.run_for_chat(
                task_type_str="full_audit",
                tenant_id="test-tenant",
                db=db,
                bill_element_dict={"drawer": "公司A", "amount": "100万"},  # 预填充要素
            )

        # 验证预填充要素写入了初始状态（LangGraph 节点可跳过 OCR）
        assert captured_initial_state.get("bill_element") == {"drawer": "公司A", "amount": "100万"}

    @pytest.mark.asyncio
    async def test_audit_task_created_in_db(self):
        """run_for_chat() 应该在数据库中创建 AuditTask 记录"""
        events = [node_event("parse", parsed_doc={})]
        mock_graph = make_mock_graph(events)
        db = make_mock_db()

        with patch("app.agents.orchestrator_agent.build_audit_graph", return_value=mock_graph), \
             patch("app.agents.orchestrator_agent.get_checkpointer", return_value=None):

            orch = OrchestratorAgent()
            await orch.run_for_chat(
                task_type_str="full_audit",
                tenant_id="test-tenant",
                db=db,
            )

        # 验证 db.add() 被调用（AuditTask 添加到会话）
        db.add.assert_called_once()
        # 验证 db.flush() 被调用（确保 task_id 在 DB 中可用）
        db.flush.assert_called_once()
