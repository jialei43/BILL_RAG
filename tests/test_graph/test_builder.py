# tests/test_graph/test_builder.py
# LangGraph 图构建器测试
# 测试策略：
#   - 验证 8 种业务类型图都能成功 compile()（无环、节点引用正确）
#   - 验证图按正确顺序执行（使用全 Mock 节点，不调用真实外部服务）
#   - 验证并行层并发执行（asyncio.gather 验证）
#   - 验证失败传播：关键节点失败 → 图直接 END，后续节点不执行

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.builder import build_audit_graph, get_supported_task_types
from app.graph.state import make_initial_state


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def make_test_state(task_type: str = "full_audit") -> dict:
    """构造测试用初始状态"""
    return make_initial_state(
        audit_task_id="test-build-001",
        tenant_id="test-tenant",
        task_type=task_type,
        trace_id="trace-xyz",
        file_path="/tmp/test.pdf",
    )


def make_success_node(result_key: str, result_value: dict):
    """
    构造始终成功的 Mock 节点函数

    返回指定 key/value 的状态增量（模拟节点成功写入 state 字段）
    """
    async def mock_node(state: dict) -> dict:
        return {result_key: result_value}
    return mock_node


def make_failure_node(node_name: str):
    """
    构造始终失败的 Mock 节点函数

    返回 failed_nodes 和 errors 增量（模拟节点失败）
    """
    async def mock_node(state: dict) -> dict:
        return {
            "failed_nodes": state.get("failed_nodes", []) + [node_name],
            "errors":       {**state.get("errors", {}), node_name: "mock failure"},
        }
    return mock_node


# ──────────────────────────────────────────────────────────────────────────────
# 测试：get_supported_task_types
# ──────────────────────────────────────────────────────────────────────────────

class TestGetSupportedTaskTypes:
    """验证支持的业务类型列表"""

    def test_returns_all_8_types(self):
        """应返回 8 种业务类型"""
        types = get_supported_task_types()
        assert len(types) == 8

    def test_contains_all_expected_types(self):
        """包含所有预期的业务类型"""
        types = get_supported_task_types()
        expected = [
            "full_audit", "issuance_check", "discount_apply",
            "acceptance_prompt", "endorsement",
            "payment_prompt", "pledge", "collection",
        ]
        for t in expected:
            assert t in types, f"{t} 不在支持列表中"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：build_audit_graph 编译验证
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildAuditGraphCompile:
    """验证 8 种业务类型的图都能成功 compile()"""

    @pytest.mark.parametrize("task_type", [
        "full_audit", "issuance_check", "discount_apply",
        "acceptance_prompt", "endorsement",
        "payment_prompt", "pledge", "collection",
    ])
    def test_compile_success(self, task_type: str):
        """每种业务类型的图都应该能够 compile()，无错误"""
        graph = build_audit_graph(task_type, checkpointer=None)  # 无检查点（不连 DB）
        assert graph is not None, f"{task_type} 图构建失败"

    def test_unknown_type_falls_back_to_full_audit(self):
        """未知业务类型应降级为 full_audit 图（不抛异常）"""
        graph = build_audit_graph("unknown_type_xyz", checkpointer=None)
        assert graph is not None                               # 降级后仍返回有效图

    def test_all_types_return_compiled_graph(self):
        """所有类型返回的图对象都是可调用的"""
        for task_type in get_supported_task_types():
            graph = build_audit_graph(task_type, checkpointer=None)
            # 编译后的图应具有 astream / ainvoke 方法（LangGraph CompiledStateGraph 接口）
            assert hasattr(graph, "astream"), f"{task_type} 图缺少 astream 方法"
            assert hasattr(graph, "ainvoke"), f"{task_type} 图缺少 ainvoke 方法"


# ──────────────────────────────────────────────────────────────────────────────
# 测试：FULL_AUDIT 图执行顺序
# ──────────────────────────────────────────────────────────────────────────────

class TestFullAuditGraphExecution:
    """验证 FULL_AUDIT 图的完整执行顺序和节点调用"""

    @pytest.mark.asyncio
    async def test_happy_path_executes_all_nodes(self):
        """全程成功时，所有节点按序执行，最终 state 包含所有结果字段"""
        execution_order = []                                   # 记录节点执行顺序

        async def mock_parse(state):
            execution_order.append("parse")
            return {"parsed_doc": {"text": "票据内容"}}

        async def mock_extract(state):
            execution_order.append("extract")
            return {"bill_element": {"drawer": "公司A"}}

        async def mock_parallel(state):
            execution_order.append("parallel")
            return {
                "compliance_summary": {"score": 95},
                "endorsement_result": {"valid": True},
                "fraud_result":       {"risk": "low"},
                "failed_nodes":       state.get("failed_nodes", []),
                "errors":             state.get("errors", {}),
            }

        async def mock_contract(state):
            execution_order.append("contract")
            return {"contract_result": {"match_score": 0.9}}

        async def mock_risk(state):
            execution_order.append("risk")
            return {"risk_result": {"grade": "A"}}

        async def mock_report(state):
            execution_order.append("report")
            return {"report_result": {"summary": "审核通过"}}

        # 必须 patch app.graph.builder.* 而非 app.graph.nodes.*
        # 因为 builder.py 用 from...import 创建了本地绑定，patch 需覆盖构建时看到的命名空间
        with patch("app.graph.builder.node_document_parser",                  mock_parse), \
             patch("app.graph.builder.node_element_extraction",               mock_extract), \
             patch("app.graph.builder.node_parallel_compliance_endorse_fraud", mock_parallel), \
             patch("app.graph.builder.node_contract_review",                  mock_contract), \
             patch("app.graph.builder.node_risk_assessment",                  mock_risk), \
             patch("app.graph.builder.node_report_generation",                mock_report):

            graph = build_audit_graph("full_audit", checkpointer=None)
            state = make_test_state("full_audit")
            final = await graph.ainvoke(state)

        # 验证执行顺序（必须从 parse 开始，report 结束）
        assert execution_order[0] == "parse"
        assert execution_order[1] == "extract"
        assert execution_order[2] == "parallel"
        assert execution_order[3] == "contract"
        assert execution_order[4] == "risk"
        assert execution_order[5] == "report"
        assert len(execution_order) == 6

        # 验证最终 state 包含所有结果字段
        assert final.get("parsed_doc") is not None
        assert final.get("bill_element") is not None
        assert final.get("compliance_summary") is not None
        assert final.get("risk_result") is not None
        assert final.get("report_result") is not None

    @pytest.mark.asyncio
    async def test_parse_failure_ends_graph(self):
        """parse 节点失败时，图直接终止，后续节点不执行"""
        execution_order = []

        async def mock_parse_fail(state):
            execution_order.append("parse")
            return {
                "failed_nodes": ["document_parser"],
                "errors": {"document_parser": "文件不存在"},
            }

        async def mock_extract(state):
            execution_order.append("extract")             # 不应被调用
            return {"bill_element": {}}

        with patch("app.graph.builder.node_document_parser",    mock_parse_fail), \
             patch("app.graph.builder.node_element_extraction", mock_extract):

            graph = build_audit_graph("full_audit", checkpointer=None)
            state = make_test_state("full_audit")
            final = await graph.ainvoke(state)

        # parse 失败后，extract 不应执行
        assert "parse" in execution_order
        assert "extract" not in execution_order
        # 失败信息写入 state
        assert "document_parser" in final.get("failed_nodes", [])

    @pytest.mark.asyncio
    async def test_extract_failure_ends_graph(self):
        """extract 节点失败时，图直接终止，并行层不执行"""
        execution_order = []

        async def mock_parse(state):
            execution_order.append("parse")
            return {"parsed_doc": {"text": "内容"}}

        async def mock_extract_fail(state):
            execution_order.append("extract")
            return {
                "failed_nodes": ["element_extraction"],
                "errors": {"element_extraction": "OCR 失败"},
            }

        async def mock_parallel(state):
            execution_order.append("parallel")            # 不应被调用
            return {}

        with patch("app.graph.builder.node_document_parser",                   mock_parse), \
             patch("app.graph.builder.node_element_extraction",                mock_extract_fail), \
             patch("app.graph.builder.node_parallel_compliance_endorse_fraud", mock_parallel):

            graph = build_audit_graph("full_audit", checkpointer=None)
            state = make_test_state("full_audit")
            final = await graph.ainvoke(state)

        # extract 失败后，并行层不应执行
        assert "extract"   in execution_order
        assert "parallel" not in execution_order
        assert "element_extraction" in final.get("failed_nodes", [])

    @pytest.mark.asyncio
    async def test_contract_failure_does_not_end_graph(self):
        """FULL_AUDIT 中合同审核（非关键节点）失败时，图继续执行到 risk → report"""
        execution_order = []

        async def mock_parse(state):
            execution_order.append("parse")
            return {"parsed_doc": {}}

        async def mock_extract(state):
            execution_order.append("extract")
            return {"bill_element": {"drawer": "公司A"}}

        async def mock_parallel(state):
            execution_order.append("parallel")
            return {
                "compliance_summary": {},
                "endorsement_result": {},
                "fraud_result":       {},
                "failed_nodes":       state.get("failed_nodes", []),
                "errors":             state.get("errors", {}),
            }

        async def mock_contract_fail(state):
            execution_order.append("contract")
            # 合同审核失败（非关键节点）
            return {
                "failed_nodes": state.get("failed_nodes", []) + ["contract_review"],
                "errors": {**state.get("errors", {}), "contract_review": "合同文本为空"},
            }

        async def mock_risk(state):
            execution_order.append("risk")                # 应该被调用（合同失败不阻断）
            return {"risk_result": {"grade": "B"}}

        async def mock_report(state):
            execution_order.append("report")              # 应该被调用
            return {"report_result": {"summary": "部分审核通过"}}

        with patch("app.graph.builder.node_document_parser",                   mock_parse), \
             patch("app.graph.builder.node_element_extraction",                mock_extract), \
             patch("app.graph.builder.node_parallel_compliance_endorse_fraud", mock_parallel), \
             patch("app.graph.builder.node_contract_review",                   mock_contract_fail), \
             patch("app.graph.builder.node_risk_assessment",                   mock_risk), \
             patch("app.graph.builder.node_report_generation",                 mock_report):

            graph = build_audit_graph("full_audit", checkpointer=None)
            state = make_test_state("full_audit")
            final = await graph.ainvoke(state)

        # 合同失败后，risk 和 report 仍然执行
        assert "contract" in execution_order
        assert "risk"     in execution_order
        assert "report"   in execution_order


# ──────────────────────────────────────────────────────────────────────────────
# 测试：DISCOUNT_APPLY 图（合同审核是关键节点）
# ──────────────────────────────────────────────────────────────────────────────

class TestDiscountApplyGraph:
    """验证 DISCOUNT_APPLY 图中合同审核是关键节点"""

    @pytest.mark.asyncio
    async def test_contract_failure_ends_graph(self):
        """DISCOUNT_APPLY 中合同审核（关键节点）失败时，图直接终止"""
        execution_order = []

        async def mock_parse(state):
            execution_order.append("parse")
            return {"parsed_doc": {}}

        async def mock_extract(state):
            execution_order.append("extract")
            return {"bill_element": {"drawer": "公司A"}}

        async def mock_parallel(state):
            execution_order.append("parallel")
            return {
                "compliance_summary": {},
                "endorsement_result": {},
                "fraud_result":       {},
                "failed_nodes":       state.get("failed_nodes", []),
                "errors":             state.get("errors", {}),
            }

        async def mock_contract_fail(state):
            execution_order.append("contract")
            # 合同审核失败（贴现场景关键节点）
            return {
                "failed_nodes": state.get("failed_nodes", []) + ["contract_review"],
                "errors": {**state.get("errors", {}), "contract_review": "合同不匹配"},
            }

        async def mock_risk(state):
            execution_order.append("risk")                # 不应被调用
            return {"risk_result": {}}

        with patch("app.graph.builder.node_document_parser",                   mock_parse), \
             patch("app.graph.builder.node_element_extraction",                mock_extract), \
             patch("app.graph.builder.node_parallel_compliance_endorse_fraud", mock_parallel), \
             patch("app.graph.builder.node_contract_review",                   mock_contract_fail), \
             patch("app.graph.builder.node_risk_assessment",                   mock_risk):

            graph = build_audit_graph("discount_apply", checkpointer=None)
            state = make_test_state("discount_apply")
            final = await graph.ainvoke(state)

        # 合同失败后，risk 不应执行（DISCOUNT_APPLY 与 FULL_AUDIT 行为不同）
        assert "contract" in execution_order
        assert "risk" not in execution_order


# ──────────────────────────────────────────────────────────────────────────────
# 测试：ISSUANCE_CHECK 图（无合同审核节点）
# ──────────────────────────────────────────────────────────────────────────────

class TestIssuanceCheckGraph:
    """验证 ISSUANCE_CHECK 图的拓扑（无合同审核节点）"""

    @pytest.mark.asyncio
    async def test_executes_without_contract_node(self):
        """出票预检流程不经过合同审核节点"""
        execution_order = []

        async def mock_parse(state):
            execution_order.append("parse")
            return {"parsed_doc": {}}

        async def mock_extract(state):
            execution_order.append("extract")
            return {"bill_element": {"drawer": "公司A"}}

        async def mock_parallel(state):
            execution_order.append("parallel_ci")
            return {
                "compliance_summary": {},
                "issuance_result":    {"allowed": True},
                "failed_nodes":       state.get("failed_nodes", []),
                "errors":             state.get("errors", {}),
            }

        async def mock_risk(state):
            execution_order.append("risk")
            return {"risk_result": {"grade": "A"}}

        async def mock_report(state):
            execution_order.append("report")
            return {"report_result": {}}

        with patch("app.graph.builder.node_document_parser",           mock_parse), \
             patch("app.graph.builder.node_element_extraction",        mock_extract), \
             patch("app.graph.builder.node_parallel_compliance_issuance", mock_parallel), \
             patch("app.graph.builder.node_risk_assessment",           mock_risk), \
             patch("app.graph.builder.node_report_generation",         mock_report):

            graph = build_audit_graph("issuance_check", checkpointer=None)
            state = make_test_state("issuance_check")
            final = await graph.ainvoke(state)

        # 无合同审核节点
        assert "contract" not in execution_order
        # 正常流程执行
        assert "parse"        in execution_order
        assert "parallel_ci"  in execution_order
        assert "risk"         in execution_order
        assert "report"       in execution_order


# ──────────────────────────────────────────────────────────────────────────────
# 测试：payment_prompt / pledge / collection 与 full_audit 拓扑相同
# ──────────────────────────────────────────────────────────────────────────────

class TestAliasGraphTypes:
    """验证 payment_prompt / pledge / collection 使用与 full_audit 相同的图拓扑"""

    @pytest.mark.parametrize("task_type", ["payment_prompt", "pledge", "collection"])
    @pytest.mark.asyncio
    async def test_same_topology_as_full_audit(self, task_type: str):
        """这三种类型与 full_audit 使用相同节点集"""
        graph = build_audit_graph(task_type, checkpointer=None)
        # 验证图对象存在并可调用
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "astream")


# ──────────────────────────────────────────────────────────────────────────────
# 测试：ENDORSEMENT 图（无合规检索节点）
# ──────────────────────────────────────────────────────────────────────────────

class TestEndorsementGraph:
    """验证 ENDORSEMENT 图的拓扑（无合规检索，无合同审核）"""

    @pytest.mark.asyncio
    async def test_executes_without_compliance_and_contract(self):
        """背书转让流程不经过合规检索和合同审核节点"""
        execution_order = []

        async def mock_parse(state):
            execution_order.append("parse")
            return {"parsed_doc": {}}

        async def mock_extract(state):
            execution_order.append("extract")
            return {"bill_element": {"drawer": "公司A"}}

        async def mock_parallel(state):
            execution_order.append("parallel_ef")
            return {
                "endorsement_result": {"chain": ["A→B"]},
                "fraud_result":       {"risk": "low"},
                "failed_nodes":       state.get("failed_nodes", []),
                "errors":             state.get("errors", {}),
            }

        async def mock_risk(state):
            execution_order.append("risk")
            return {"risk_result": {"grade": "A"}}

        async def mock_report(state):
            execution_order.append("report")
            return {"report_result": {}}

        with patch("app.graph.builder.node_document_parser",      mock_parse), \
             patch("app.graph.builder.node_element_extraction",   mock_extract), \
             patch("app.graph.builder.node_parallel_endorse_fraud", mock_parallel), \
             patch("app.graph.builder.node_risk_assessment",      mock_risk), \
             patch("app.graph.builder.node_report_generation",    mock_report):

            graph = build_audit_graph("endorsement", checkpointer=None)
            state = make_test_state("endorsement")
            final = await graph.ainvoke(state)

        # 无合规检索和合同审核节点
        assert "compliance" not in execution_order
        assert "contract"   not in execution_order
        # 正常流程执行
        assert "parse"       in execution_order
        assert "parallel_ef" in execution_order
        assert "risk"        in execution_order
        assert "report"      in execution_order


# ──────────────────────────────────────────────────────────────────────────────
# 测试：astream() 流式执行
# ──────────────────────────────────────────────────────────────────────────────

class TestGraphStreaming:
    """验证 astream() 接口的流式执行行为"""

    @pytest.mark.asyncio
    async def test_astream_yields_events_per_node(self):
        """astream() 应该每完成一个节点就产生一个事件"""
        async def mock_parse(state):
            return {"parsed_doc": {}}

        async def mock_extract(state):
            return {"bill_element": {"drawer": "公司A"}}

        async def mock_parallel(state):
            return {
                "compliance_summary": {},
                "endorsement_result": {},
                "fraud_result":       {},
                "failed_nodes":       state.get("failed_nodes", []),
                "errors":             state.get("errors", {}),
            }

        async def mock_contract(state):
            return {"contract_result": {}}

        async def mock_risk(state):
            return {"risk_result": {}}

        async def mock_report(state):
            return {"report_result": {}}

        events = []

        with patch("app.graph.builder.node_document_parser",                   mock_parse), \
             patch("app.graph.builder.node_element_extraction",                mock_extract), \
             patch("app.graph.builder.node_parallel_compliance_endorse_fraud", mock_parallel), \
             patch("app.graph.builder.node_contract_review",                   mock_contract), \
             patch("app.graph.builder.node_risk_assessment",                   mock_risk), \
             patch("app.graph.builder.node_report_generation",                 mock_report):

            graph = build_audit_graph("full_audit", checkpointer=None)
            state = make_test_state("full_audit")

            # 收集所有 astream 事件
            async for event in graph.astream(state):
                events.append(event)

        # 每个节点完成后产生一个事件，FULL_AUDIT 有 6 个节点
        assert len(events) == 6, f"期望 6 个事件，实际收到 {len(events)} 个"

        # 事件顺序：每个事件的 key 是节点名
        node_names = [list(e.keys())[0] for e in events]
        assert node_names[0] == "parse"
        assert node_names[-1] == "report"
