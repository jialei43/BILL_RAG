# tests/test_graph/test_nodes.py
# LangGraph 节点函数测试
# 测试策略：Mock 所有 MCP 工具函数，只测试节点函数的状态转换逻辑
# 不依赖外部服务（数据库、Redis、Milvus），可在 CI 环境中运行

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

# 构造测试用初始状态
BASE_STATE = {
    "audit_task_id": "test-task-001",
    "tenant_id":     "test-tenant",
    "task_type":     "full_audit",
    "trace_id":      "trace-abc",
    "file_path":     "/tmp/test_bill.pdf",
    "failed_nodes":  [],
    "skipped_nodes": [],
    "errors":        {},
}


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def make_state(**overrides) -> dict:
    """基于 BASE_STATE 构造测试状态，支持字段覆盖"""
    state = dict(BASE_STATE)
    state.update(overrides)
    return state


def make_tool_success(data: dict) -> dict:
    """构造 MCP 工具成功返回值"""
    return {"success": True, "data": data, "error": None, "error_code": None}


def make_tool_failure(error: str, error_code: str = "TOOL_ERROR") -> dict:
    """构造 MCP 工具失败返回值"""
    return {"success": False, "data": None, "error": error, "error_code": error_code}


# ──────────────────────────────────────────────────────────────────────────────
# 测试：_node_failed 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeFailed:
    """测试私有辅助函数 _node_failed 的增量构造逻辑"""

    def test_empty_state_failed_nodes(self):
        """空状态时，正确创建 failed_nodes 和 errors"""
        from app.graph.nodes import _node_failed
        state = {"failed_nodes": [], "errors": {}}
        result = _node_failed(state, "test_node", "测试错误")
        assert result["failed_nodes"] == ["test_node"]          # 追加新节点名
        assert result["errors"] == {"test_node": "测试错误"}    # 追加错误信息

    def test_existing_failed_nodes_preserved(self):
        """已有失败节点时，追加新节点而非覆盖"""
        from app.graph.nodes import _node_failed
        state = {
            "failed_nodes": ["previous_node"],          # 已有失败节点
            "errors": {"previous_node": "之前的错误"},
        }
        result = _node_failed(state, "new_node", "新错误")
        assert "previous_node" in result["failed_nodes"]  # 保留原有节点
        assert "new_node" in result["failed_nodes"]       # 追加新节点
        assert result["errors"]["previous_node"] == "之前的错误"  # 保留原有错误
        assert result["errors"]["new_node"] == "新错误"           # 追加新错误

    def test_result_only_contains_failed_and_errors(self):
        """返回值只包含 failed_nodes 和 errors 两个字段（不覆盖其他状态）"""
        from app.graph.nodes import _node_failed
        state = make_state()
        result = _node_failed(state, "node_x", "错误 X")
        assert set(result.keys()) == {"failed_nodes", "errors"}  # 只有这两个键


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_document_parser
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeDocumentParser:
    """测试文档解析节点"""

    @pytest.mark.asyncio
    async def test_success_returns_parsed_doc(self):
        """工具成功时，节点返回 parsed_doc 字段"""
        mock_data = {"text": "票据内容", "tables": []}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))

        with patch("app.mcp.tools.document_tools.parse_bill_document", mock_tool):
            from app.graph.nodes import node_document_parser
            state = make_state()
            result = await node_document_parser(state)

        assert result == {"parsed_doc": mock_data}                # 只返回变更字段
        mock_tool.assert_called_once_with(                        # 验证调用参数
            file_path="/tmp/test_bill.pdf",
            audit_task_id="test-task-001",
            tenant_id="test-tenant",
        )

    @pytest.mark.asyncio
    async def test_failure_returns_failed_nodes(self):
        """工具失败时，节点返回 failed_nodes 和 errors 增量"""
        mock_tool = AsyncMock(return_value=make_tool_failure("文件不存在"))

        with patch("app.mcp.tools.document_tools.parse_bill_document", mock_tool):
            from app.graph.nodes import node_document_parser
            state = make_state()
            result = await node_document_parser(state)

        assert "document_parser" in result["failed_nodes"]        # 节点名写入失败列表
        assert "document_parser" in result["errors"]              # 错误信息记录
        assert "文件不存在" in result["errors"]["document_parser"]

    @pytest.mark.asyncio
    async def test_empty_file_path_fails(self):
        """file_path 为空时，不调用工具直接返回失败"""
        mock_tool = AsyncMock()

        with patch("app.mcp.tools.document_tools.parse_bill_document", mock_tool):
            from app.graph.nodes import node_document_parser
            state = make_state(file_path=None)                    # 覆盖为空路径
            result = await node_document_parser(state)

        mock_tool.assert_not_called()                             # 工具不应被调用
        assert "document_parser" in result["failed_nodes"]

    @pytest.mark.asyncio
    async def test_no_extra_fields_in_success(self):
        """成功时返回值不包含其他字段（不覆盖 bill_element 等）"""
        mock_data = {"text": "内容"}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))

        with patch("app.mcp.tools.document_tools.parse_bill_document", mock_tool):
            from app.graph.nodes import node_document_parser
            state = make_state()
            result = await node_document_parser(state)

        assert list(result.keys()) == ["parsed_doc"]              # 只有 parsed_doc


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_element_extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeElementExtraction:
    """测试要素抽取节点"""

    @pytest.mark.asyncio
    async def test_success_returns_bill_element(self):
        """工具成功时，节点返回 bill_element 字段"""
        mock_data = {"drawer": "公司A", "amount": "100万"}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))

        with patch("app.mcp.tools.extraction_tools.extract_bill_elements", mock_tool):
            from app.graph.nodes import node_element_extraction
            state = make_state()
            result = await node_element_extraction(state)

        assert result == {"bill_element": mock_data}

    @pytest.mark.asyncio
    async def test_prefilled_element_passed_to_tool(self):
        """预填充要素时，传递 prefilled_element 参数给工具（快速路径）"""
        prefilled = {"drawer": "公司B", "amount": "50万"}
        mock_data = {"drawer": "公司B", "amount": "50万", "id": "db-123"}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))

        with patch("app.mcp.tools.extraction_tools.extract_bill_elements", mock_tool):
            from app.graph.nodes import node_element_extraction
            state = make_state(bill_element=prefilled)            # 注入预填充要素
            result = await node_element_extraction(state)

        # 验证调用时传入了 prefilled_element（快速路径）
        call_kwargs = mock_tool.call_args.kwargs
        assert call_kwargs["prefilled_element"] == prefilled

    @pytest.mark.asyncio
    async def test_failure_returns_failed_nodes(self):
        """工具失败时，节点返回 failed_nodes 增量"""
        mock_tool = AsyncMock(return_value=make_tool_failure("OCR 识别失败"))

        with patch("app.mcp.tools.extraction_tools.extract_bill_elements", mock_tool):
            from app.graph.nodes import node_element_extraction
            state = make_state()
            result = await node_element_extraction(state)

        assert "element_extraction" in result["failed_nodes"]


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_compliance_retrieval
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeComplianceRetrieval:
    """测试合规检索节点"""

    @pytest.mark.asyncio
    async def test_success_returns_compliance_summary(self):
        """工具成功时，返回 compliance_summary 字段"""
        mock_data = {"issues": [], "score": 95}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state_with_element = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.compliance_tools.check_compliance", mock_tool):
            from app.graph.nodes import node_compliance_retrieval
            result = await node_compliance_retrieval(state_with_element)

        assert result == {"compliance_summary": mock_data}

    @pytest.mark.asyncio
    async def test_no_bill_element_fails(self):
        """bill_element 为空时，不调用工具直接返回失败"""
        mock_tool = AsyncMock()

        with patch("app.mcp.tools.compliance_tools.check_compliance", mock_tool):
            from app.graph.nodes import node_compliance_retrieval
            state = make_state()                                  # 无 bill_element
            result = await node_compliance_retrieval(state)

        mock_tool.assert_not_called()
        assert "compliance_retrieval" in result["failed_nodes"]


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_endorsement_chain
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeEndorsementChain:
    """测试背书链分析节点"""

    @pytest.mark.asyncio
    async def test_success_returns_endorsement_result(self):
        """工具成功时，返回 endorsement_result 字段"""
        mock_data = {"chain": ["A→B→C"], "valid": True}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state_with_element = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", mock_tool):
            from app.graph.nodes import node_endorsement_chain
            result = await node_endorsement_chain(state_with_element)

        assert result == {"endorsement_result": mock_data}

    @pytest.mark.asyncio
    async def test_no_bill_element_fails(self):
        """bill_element 为空时，不调用工具直接失败"""
        mock_tool = AsyncMock()
        with patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", mock_tool):
            from app.graph.nodes import node_endorsement_chain
            result = await node_endorsement_chain(make_state())

        mock_tool.assert_not_called()
        assert "endorsement_chain" in result["failed_nodes"]


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_fraud_detection
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeFraudDetection:
    """测试欺诈检测节点"""

    @pytest.mark.asyncio
    async def test_success_returns_fraud_result(self):
        """工具成功时，返回 fraud_result 字段"""
        mock_data = {"risk_level": "low", "indicators": []}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.fraud_tools.detect_fraud", mock_tool):
            from app.graph.nodes import node_fraud_detection
            result = await node_fraud_detection(state)

        assert result == {"fraud_result": mock_data}

    @pytest.mark.asyncio
    async def test_passes_endorsement_result_if_present(self):
        """state 中有 endorsement_result 时，工具调用时也传入此参数"""
        endorse_data = {"chain": ["A→B"]}
        mock_data = {"risk_level": "low"}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(
            bill_element={"drawer": "公司A"},
            endorsement_result=endorse_data,
        )

        with patch("app.mcp.tools.fraud_tools.detect_fraud", mock_tool):
            from app.graph.nodes import node_fraud_detection
            await node_fraud_detection(state)

        call_kwargs = mock_tool.call_args.kwargs
        assert call_kwargs["endorsement_result"] == endorse_data  # 传入了背书链结果

    @pytest.mark.asyncio
    async def test_no_bill_element_fails(self):
        """bill_element 为空时，直接失败"""
        mock_tool = AsyncMock()
        with patch("app.mcp.tools.fraud_tools.detect_fraud", mock_tool):
            from app.graph.nodes import node_fraud_detection
            result = await node_fraud_detection(make_state())

        mock_tool.assert_not_called()
        assert "fraud_detection" in result["failed_nodes"]


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_contract_review
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeContractReview:
    """测试合同审核节点"""

    @pytest.mark.asyncio
    async def test_success_with_contract_text(self):
        """有合同文本时，工具调用成功"""
        mock_data = {"match_score": 0.95, "issues": []}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(
            bill_element={"drawer": "公司A"},
            contract_text="合同文本内容...",
        )

        with patch("app.mcp.tools.contract_tools.review_contract", mock_tool):
            from app.graph.nodes import node_contract_review
            result = await node_contract_review(state)

        assert result == {"contract_result": mock_data}
        call_kwargs = mock_tool.call_args.kwargs
        assert call_kwargs["contract_text"] == "合同文本内容..."  # 传入合同文本

    @pytest.mark.asyncio
    async def test_no_bill_element_fails(self):
        """bill_element 为空时，直接失败"""
        mock_tool = AsyncMock()
        with patch("app.mcp.tools.contract_tools.review_contract", mock_tool):
            from app.graph.nodes import node_contract_review
            result = await node_contract_review(make_state())

        mock_tool.assert_not_called()
        assert "contract_review" in result["failed_nodes"]


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_bill_issuance
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeBillIssuance:
    """测试出票预检节点"""

    @pytest.mark.asyncio
    async def test_success_returns_issuance_result(self):
        """工具成功时，返回 issuance_result 字段"""
        mock_data = {"allowed": True, "checks": []}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.issuance_tools.check_bill_issuance", mock_tool):
            from app.graph.nodes import node_bill_issuance
            result = await node_bill_issuance(state)

        assert result == {"issuance_result": mock_data}

    @pytest.mark.asyncio
    async def test_failure_returns_failed_nodes(self):
        """工具失败时，节点返回 failed_nodes"""
        mock_tool = AsyncMock(return_value=make_tool_failure("额度不足"))
        state = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.issuance_tools.check_bill_issuance", mock_tool):
            from app.graph.nodes import node_bill_issuance
            result = await node_bill_issuance(state)

        assert "bill_issuance" in result["failed_nodes"]
        assert "额度不足" in result["errors"]["bill_issuance"]


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_risk_assessment
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeRiskAssessment:
    """测试风险评估节点"""

    @pytest.mark.asyncio
    async def test_success_returns_risk_result(self):
        """工具成功时，返回 risk_result 字段"""
        mock_data = {"risk_score": 0.2, "grade": "A"}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(
            compliance_summary={"score": 95},
            endorsement_result={"valid": True},
        )

        with patch("app.mcp.tools.risk_tools.assess_risk", mock_tool):
            from app.graph.nodes import node_risk_assessment
            result = await node_risk_assessment(state)

        assert result == {"risk_result": mock_data}

    @pytest.mark.asyncio
    async def test_passes_all_dimension_results(self):
        """节点将所有维度结果传给风险评估工具"""
        mock_data = {"risk_score": 0.3}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(
            compliance_summary={"score": 90},
            endorsement_result={"valid": True},
            contract_result={"match_score": 0.9},
            fraud_result={"risk_level": "low"},
        )

        with patch("app.mcp.tools.risk_tools.assess_risk", mock_tool):
            from app.graph.nodes import node_risk_assessment
            await node_risk_assessment(state)

        call_kwargs = mock_tool.call_args.kwargs
        assert call_kwargs["compliance_summary"] == {"score": 90}
        assert call_kwargs["endorsement_result"] == {"valid": True}
        assert call_kwargs["contract_result"] == {"match_score": 0.9}
        assert call_kwargs["fraud_result"] == {"risk_level": "low"}


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_report_generation
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeReportGeneration:
    """测试报告生成节点"""

    @pytest.mark.asyncio
    async def test_success_returns_report_result(self):
        """工具成功时，返回 report_result 字段"""
        mock_data = {"report_id": "rpt-001", "summary": "审核通过"}
        mock_tool = AsyncMock(return_value=make_tool_success(mock_data))
        state = make_state(
            bill_element={"drawer": "公司A"},
            risk_result={"grade": "A"},
        )

        with patch("app.mcp.tools.report_tools.generate_report", mock_tool):
            from app.graph.nodes import node_report_generation
            result = await node_report_generation(state)

        assert result == {"report_result": mock_data}

    @pytest.mark.asyncio
    async def test_is_final_always_true(self):
        """LangGraph 末尾节点生成终版报告，is_final 始终为 True"""
        mock_tool = AsyncMock(return_value=make_tool_success({"report_id": "rpt-002"}))
        state = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.report_tools.generate_report", mock_tool):
            from app.graph.nodes import node_report_generation
            await node_report_generation(state)

        call_kwargs = mock_tool.call_args.kwargs
        assert call_kwargs["is_final"] is True                    # 末尾节点始终生成终版报告


# ──────────────────────────────────────────────────────────────────────────────
# 测试：node_parallel_compliance_endorse_fraud（并行层聚合节点）
# ──────────────────────────────────────────────────────────────────────────────

class TestNodeParallelComplianceEndorseFraud:
    """测试并行层聚合节点（asyncio.gather）"""

    @pytest.mark.asyncio
    async def test_all_success_merges_all_results(self):
        """三个工具都成功时，合并三个节点的返回值到同一个 dict"""
        compliance_data = {"score": 95}
        endorse_data    = {"chain": ["A→B"]}
        fraud_data      = {"risk_level": "low"}

        state = make_state(bill_element={"drawer": "公司A"})

        # Mock 三个工具函数
        mock_compliance = AsyncMock(return_value=make_tool_success(compliance_data))
        mock_endorse    = AsyncMock(return_value=make_tool_success(endorse_data))
        mock_fraud      = AsyncMock(return_value=make_tool_success(fraud_data))

        with patch("app.mcp.tools.compliance_tools.check_compliance", mock_compliance), \
             patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", mock_endorse), \
             patch("app.mcp.tools.fraud_tools.detect_fraud", mock_fraud):

            from app.graph.nodes import node_parallel_compliance_endorse_fraud
            result = await node_parallel_compliance_endorse_fraud(state)

        # 三个结果字段都应该存在
        assert result["compliance_summary"] == compliance_data
        assert result["endorsement_result"] == endorse_data
        assert result["fraud_result"] == fraud_data

        # 三个工具都被调用了一次（并行）
        mock_compliance.assert_called_once()
        mock_endorse.assert_called_once()
        mock_fraud.assert_called_once()

    @pytest.mark.asyncio
    async def test_one_failure_does_not_block_others(self):
        """一个工具失败时，其他两个工具仍然执行并返回结果"""
        compliance_data = {"score": 95}
        fraud_data      = {"risk_level": "low"}

        state = make_state(bill_element={"drawer": "公司A"})

        mock_compliance = AsyncMock(return_value=make_tool_success(compliance_data))
        mock_endorse    = AsyncMock(return_value=make_tool_failure("背书链服务不可用"))  # 背书链失败
        mock_fraud      = AsyncMock(return_value=make_tool_success(fraud_data))

        with patch("app.mcp.tools.compliance_tools.check_compliance", mock_compliance), \
             patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", mock_endorse), \
             patch("app.mcp.tools.fraud_tools.detect_fraud", mock_fraud):

            from app.graph.nodes import node_parallel_compliance_endorse_fraud
            result = await node_parallel_compliance_endorse_fraud(state)

        # 合规和欺诈结果仍然存在
        assert result.get("compliance_summary") == compliance_data
        assert result.get("fraud_result") == fraud_data

        # 背书链节点失败，记录在 failed_nodes
        assert "endorsement_chain" in result["failed_nodes"]
        assert "endorsement_chain" in result["errors"]

        # 三个工具都被调用了（互不阻塞）
        mock_compliance.assert_called_once()
        mock_endorse.assert_called_once()
        mock_fraud.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_failed_nodes_merged(self):
        """多个工具同时失败时，所有失败节点名都记录到 failed_nodes"""
        state = make_state(bill_element={"drawer": "公司A"})

        # 三个工具全部失败
        mock_compliance = AsyncMock(return_value=make_tool_failure("合规服务超时"))
        mock_endorse    = AsyncMock(return_value=make_tool_failure("背书链服务超时"))
        mock_fraud      = AsyncMock(return_value=make_tool_failure("欺诈检测超时"))

        with patch("app.mcp.tools.compliance_tools.check_compliance", mock_compliance), \
             patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", mock_endorse), \
             patch("app.mcp.tools.fraud_tools.detect_fraud", mock_fraud):

            from app.graph.nodes import node_parallel_compliance_endorse_fraud
            result = await node_parallel_compliance_endorse_fraud(state)

        # 三个失败节点都应记录
        assert "compliance_retrieval" in result["failed_nodes"]
        assert "endorsement_chain"    in result["failed_nodes"]
        assert "fraud_detection"      in result["failed_nodes"]
        assert len(result["failed_nodes"]) == 3

    @pytest.mark.asyncio
    async def test_parallel_execution_order_independent(self):
        """验证三个工具以并发方式执行（总耗时约等于最慢工具耗时，而非累加）"""
        import time

        async def slow_compliance(**kwargs):
            """模拟 0.05s 耗时的合规检索"""
            await asyncio.sleep(0.05)
            return make_tool_success({"score": 95})

        async def slow_endorse(**kwargs):
            """模拟 0.05s 耗时的背书链分析"""
            await asyncio.sleep(0.05)
            return make_tool_success({"chain": []})

        async def slow_fraud(**kwargs):
            """模拟 0.05s 耗时的欺诈检测"""
            await asyncio.sleep(0.05)
            return make_tool_success({"risk_level": "low"})

        state = make_state(bill_element={"drawer": "公司A"})

        with patch("app.mcp.tools.compliance_tools.check_compliance", slow_compliance), \
             patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", slow_endorse), \
             patch("app.mcp.tools.fraud_tools.detect_fraud", slow_fraud):

            from app.graph.nodes import node_parallel_compliance_endorse_fraud
            start = time.time()
            await node_parallel_compliance_endorse_fraud(state)
            elapsed = time.time() - start

        # 串行执行需要 0.15s，并行执行应在 0.12s 内完成（留 70ms 余量）
        assert elapsed < 0.12, f"并行执行耗时 {elapsed:.3f}s，超过预期（疑似串行执行）"

    @pytest.mark.asyncio
    async def test_no_bill_element_all_fail(self):
        """bill_element 为空时，三个并行工具都应失败"""
        state = make_state()                                      # 无 bill_element

        mock_compliance = AsyncMock(return_value=make_tool_success({}))
        mock_endorse    = AsyncMock(return_value=make_tool_success({}))
        mock_fraud      = AsyncMock(return_value=make_tool_success({}))

        with patch("app.mcp.tools.compliance_tools.check_compliance", mock_compliance), \
             patch("app.mcp.tools.endorsement_tools.analyze_endorsement_chain", mock_endorse), \
             patch("app.mcp.tools.fraud_tools.detect_fraud", mock_fraud):

            from app.graph.nodes import node_parallel_compliance_endorse_fraud
            result = await node_parallel_compliance_endorse_fraud(state)

        # bill_element 为空，三个节点都应失败
        assert "compliance_retrieval" in result["failed_nodes"]
        assert "endorsement_chain"    in result["failed_nodes"]
        assert "fraud_detection"      in result["failed_nodes"]
        # 工具不应被调用（各节点内部校验 bill_element 后直接返回失败）
        mock_compliance.assert_not_called()
        mock_endorse.assert_not_called()
        mock_fraud.assert_not_called()
