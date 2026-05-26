# tests/test_agents/test_m10_api.py
# M10 模块测试：出票预检 Agent + 端到端集成测试
# 运行方式：python -m pytest tests/test_agents/test_m10_api.py -v

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.bill_issuance_agent import BillIssuanceAgent, ISSUANCE_REQUIRED_FIELDS


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mock_db():
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock())
    mock_db.add = MagicMock()
    return mock_db


def make_valid_bill_element(**overrides) -> dict:
    """创建满足所有出票预检条件的票据要素"""
    base = {
        "ticket_number":  "SHBH20240115001",
        "ticket_type":    "银行承兑汇票",
        "issue_date":     "2024-01-15",
        "due_date":       "2025-01-15",
        "amount_numeric": 1000000.0,   # 100 万，在合理范围内
        "amount_text":    "壹百万元整",
        "currency":       "人民币",
        "drawer":         "正常出票公司",
        "drawer_account": "6222021234567890",
        "drawer_bank":    "工商银行测试支行",
        "acceptor":       "工商银行测试支行",
        "payee":          "正常收款公司",
        "drawee_bank":    "工商银行测试支行",
        "endorsers":      [],
        "maturity_days":  365,         # 恰好在上限
        "trade_purpose":  "货款结算",
        "confidence_score": 0.95,
    }
    base.update(overrides)
    return base


def make_ctx_with_bill(element: dict, task_id: str = "m10-test-001") -> AgentContext:
    ctx = AgentContext(audit_task_id=task_id, tenant_id="t-001")
    ctx.shared_data["bill_element"] = element
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：BillIssuanceAgent 出票预检
# ──────────────────────────────────────────────────────────────────────────────

class TestBillIssuanceAgent(unittest.TestCase):
    """验证出票资格预检的各类场景"""

    def setUp(self):
        self.agent = BillIssuanceAgent()

    def test_valid_bill_passes_precheck(self):
        """验证合格票据通过所有预检"""
        ctx = make_ctx_with_bill(make_valid_bill_element())
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertTrue(result.data["is_eligible"], "合格票据应通过预检")
        self.assertEqual(result.data["failed_count"], 0)

    def test_missing_required_field_fails_precheck(self):
        """验证必填字段缺失时预检不通过"""
        element = make_valid_bill_element()
        element["ticket_number"] = None   # 票据号码为空

        ctx = make_ctx_with_bill(element, task_id="m10-missing-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertFalse(result.data["is_eligible"])
        self.assertGreater(result.data["failed_count"], 0)

    def test_all_required_fields_validated(self):
        """验证所有必填字段都在预检范围内"""
        # 确认 ISSUANCE_REQUIRED_FIELDS 包含出票关键字段
        self.assertIn("ticket_number", ISSUANCE_REQUIRED_FIELDS)
        self.assertIn("drawer",        ISSUANCE_REQUIRED_FIELDS)
        self.assertIn("acceptor",      ISSUANCE_REQUIRED_FIELDS)
        self.assertIn("payee",         ISSUANCE_REQUIRED_FIELDS)
        self.assertIn("amount_numeric", ISSUANCE_REQUIRED_FIELDS)

    def test_amount_below_minimum_fails(self):
        """验证金额低于最低限额时预检不通过"""
        element = make_valid_bill_element(amount_numeric=5000.0)   # 5000 < 10000 最低限额

        ctx = make_ctx_with_bill(element, task_id="m10-low-amount")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertFalse(result.data["is_eligible"])
        # 检查失败原因包含金额问题
        failure_checks = [c["check"] for c in result.data["failed_checks"]]
        self.assertIn("amount_range", failure_checks)

    def test_amount_exceeds_maximum_fails(self):
        """验证金额超过单张上限时预检不通过"""
        element = make_valid_bill_element(amount_numeric=600_000_000.0)   # 6 亿 > 5 亿上限

        ctx = make_ctx_with_bill(element, task_id="m10-high-amount")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertFalse(result.data["is_eligible"])
        failure_checks = [c["check"] for c in result.data["failed_checks"]]
        self.assertIn("amount_range", failure_checks)

    def test_maturity_days_exceeds_limit_fails(self):
        """验证期限超过 365 天时预检不通过"""
        element = make_valid_bill_element(maturity_days=400)   # 400 > 365 天

        ctx = make_ctx_with_bill(element, task_id="m10-maturity-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertFalse(result.data["is_eligible"])
        failure_checks = [c["check"] for c in result.data["failed_checks"]]
        self.assertIn("maturity_days", failure_checks)

    def test_blacklisted_drawer_fails(self):
        """验证出票人在黑名单中时预检不通过"""
        element = make_valid_bill_element(drawer="BLACKLIST-出票公司")

        ctx = make_ctx_with_bill(element, task_id="m10-blacklist-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertFalse(result.data["is_eligible"])
        failure_checks = [c["check"] for c in result.data["failed_checks"]]
        self.assertIn("blacklist", failure_checks)

    def test_blacklisted_payee_fails(self):
        """验证收款人在黑名单中时预检不通过"""
        element = make_valid_bill_element(payee="BLACKLIST-收款公司")

        ctx = make_ctx_with_bill(element, task_id="m10-bl-payee")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertFalse(result.data["is_eligible"])

    def test_credit_limit_exceeded_warning(self):
        """验证金额超过授信额度时产生警告（但不是严重失败）"""
        element = make_valid_bill_element(amount_numeric=150_000_000.0)   # 1.5亿 > 1亿授信

        ctx = make_ctx_with_bill(element, task_id="m10-credit-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        # 授信超额是 warning 级别，但仍导致 is_eligible=False
        self.assertFalse(result.data["is_eligible"])
        failure_checks = [c["check"] for c in result.data["failed_checks"]]
        self.assertIn("credit_limit", failure_checks)

    def test_result_written_to_shared_data(self):
        """验证预检结果写入 shared_data["issuance_result"]"""
        ctx = make_ctx_with_bill(make_valid_bill_element())
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        self.assertIn("issuance_result", ctx.shared_data)
        ir = ctx.shared_data["issuance_result"]
        self.assertIn("is_eligible",      ir)
        self.assertIn("failed_checks",    ir)
        self.assertIn("recommendation",   ir)
        self.assertIn("blacklist_passed", ir)
        self.assertIn("credit_passed",    ir)

    def test_recommendation_generated(self):
        """验证预检结论文本非空"""
        ctx = make_ctx_with_bill(make_valid_bill_element())
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertIsNotNone(result.data["recommendation"])
        self.assertGreater(len(result.data["recommendation"]), 0)

    def test_empty_bill_element_multiple_failures(self):
        """验证票据要素全空时产生多个失败项"""
        ctx = make_ctx_with_bill({}, task_id="m10-empty-001")   # 空要素
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertFalse(result.data["is_eligible"])
        # 空票据应产生多个失败项（至少包含所有必填字段）
        self.assertGreaterEqual(result.data["failed_count"], len(ISSUANCE_REQUIRED_FIELDS))


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：端到端集成验证（模拟完整 Agent 链路）
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEndIntegration(unittest.TestCase):
    """
    端到端集成测试：模拟从要素提取到风险评估的完整 Agent 链路
    使用 Mock 替代数据库和外部服务，验证数据在 shared_data 中的流转
    """

    def test_compliance_agent_reads_bill_element(self):
        """验证合规 Agent 能正确读取要素 Agent 写入的 bill_element"""
        from app.agents.compliance_retrieval_agent import ComplianceRetrievalAgent

        ctx = AgentContext(audit_task_id="e2e-test-001", tenant_id="t-001")
        ctx.shared_data["bill_element"] = {
            "ticket_number": "SHBH001", "drawer": "测试公司",
            "payee": "收款公司", "amount_numeric": 100000.0,
            "issue_date": "2024-01-15", "due_date": "2025-01-15",
            "acceptor": "银行", "drawee_bank": "银行",
        }

        agent = ComplianceRetrievalAgent()
        mock_db = make_mock_db()

        async def mock_rag(q):
            return "合规规定", 0.85

        with patch.object(agent, "_call_rag", side_effect=mock_rag):
            result = run_async(agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertIn("compliance_summary", ctx.shared_data)

    def test_risk_agent_reads_compliance_summary(self):
        """验证风险 Agent 能正确读取合规 Agent 写入的 compliance_summary"""
        from app.agents.risk_assessment_agent import RiskAssessmentAgent

        ctx = AgentContext(audit_task_id="e2e-test-002", tenant_id="t-001")
        # 模拟合规 Agent 的输出
        ctx.shared_data["compliance_summary"] = {
            "total_fields": 18, "violation_count": 0,
            "severe_count": 0, "compliance_rate": 1.0,
            "is_overall_compliant": True,
        }

        agent = RiskAssessmentAgent()
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertIn("risk_result", ctx.shared_data)
        self.assertGreater(ctx.shared_data["risk_result"]["compliance_score"], 0)

    def test_report_agent_reads_risk_result(self):
        """验证报告 Agent 能正确读取风险 Agent 写入的 risk_result"""
        import tempfile
        from app.agents.report_generation_agent import ReportGenerationAgent

        ctx = AgentContext(audit_task_id="e2e-test-003", tenant_id="t-001")
        # 模拟风险 Agent 的输出
        ctx.shared_data["risk_result"] = {
            "composite_score": 85.0, "risk_level": "LOW",
            "compliance_score": 100.0, "endorsement_score": 100.0,
            "contract_score": 85.0, "fraud_score": 95.0,
            "missing_dimensions": [],
        }

        temp_dir = tempfile.mkdtemp()
        agent = ReportGenerationAgent(output_dir=temp_dir)
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertIn("report_result", ctx.shared_data)
        self.assertEqual(ctx.shared_data["risk_result"]["risk_level"], "LOW")

    def test_full_chain_shared_data_flow(self):
        """
        验证四个核心 Agent 的数据流转：
        EndorsementChainAgent → ContractReviewAgent → RiskAssessmentAgent
        (每个 Agent 都能读到上一个写入的数据)
        """
        from app.agents.endorsement_chain_agent import EndorsementChainAgent
        from app.agents.contract_review_agent import ContractReviewAgent
        from app.agents.risk_assessment_agent import RiskAssessmentAgent

        ctx = AgentContext(audit_task_id="e2e-chain-001", tenant_id="t-001")
        ctx.shared_data["bill_element"] = {
            "ticket_number": "E2E001", "drawer": "出票公司",
            "payee": "收款公司", "amount_numeric": 500000.0,
            "issue_date": "2024-01-15", "due_date": "2025-01-15",
            "acceptor": "银行", "drawee_bank": "银行",
            "endorsers": [], "trade_purpose": "货款",
            "maturity_days": 365, "confidence_score": 0.92,
        }

        mock_db = make_mock_db()

        # 1. 背书链 Agent
        endorse_agent = EndorsementChainAgent()
        run_async(endorse_agent.run(ctx, mock_db))
        self.assertIn("endorsement_result", ctx.shared_data)

        # 2. 合同审核 Agent（无合同文本，降级处理）
        contract_agent = ContractReviewAgent()
        run_async(contract_agent.run(ctx, mock_db))
        self.assertIn("contract_result", ctx.shared_data)

        # 3. 风险评估 Agent（读取背书+合同结果）
        risk_agent = RiskAssessmentAgent()
        result = run_async(risk_agent.run(ctx, mock_db))
        self.assertIn("risk_result", ctx.shared_data)

        # 验证风险评估正确读取了前两个 Agent 的数据
        self.assertTrue(result.success)
        risk_result = ctx.shared_data["risk_result"]
        self.assertNotIn("endorsement", risk_result.get("missing_dimensions", []))
        self.assertNotIn("contract",    risk_result.get("missing_dimensions", []))

    def test_issuance_precheck_before_full_audit(self):
        """验证出票预检阻断不合格票据进入全流程"""
        issuance_agent = BillIssuanceAgent()
        ctx = AgentContext(audit_task_id="e2e-issuance-001", tenant_id="t-001")
        # 不合格票据：票据号码为空
        ctx.shared_data["bill_element"] = {
            "ticket_number": None,   # 空票据号码
            "drawer": "出票公司",
            "amount_numeric": 100000.0,
        }
        mock_db = make_mock_db()

        result = run_async(issuance_agent.run(ctx, mock_db))

        # 预检不通过
        self.assertFalse(result.data["is_eligible"])
        # 在实际业务中，这里会停止后续全流程审核
        issuance_result = ctx.shared_data["issuance_result"]
        self.assertFalse(issuance_result["is_eligible"])


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
