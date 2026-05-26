# tests/test_agents/test_m6_report.py
# M6 模块测试：验证报告 JSON 9 节结构完整性和 PDF 文件生成
# 运行方式：python -m pytest tests/test_agents/test_m6_report.py -v

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.report_generation_agent import ReportGenerationAgent
from app.agents.utils.report_renderer import ReportRenderer


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


def make_full_ctx(task_id="m6-test-001") -> AgentContext:
    """创建包含完整四维结果的上下文（模拟所有前置 Agent 已运行完毕）"""
    ctx = AgentContext(audit_task_id=task_id, tenant_id="t-001")

    # 模拟 ElementExtractionAgent 写入的 bill_element
    ctx.shared_data["bill_element"] = {
        "ticket_number":    "SHBH20240115",
        "ticket_type":      "银行承兑汇票",
        "issue_date":       "2024-01-15",
        "due_date":         "2025-01-15",
        "amount_numeric":   1000000.0,
        "amount_text":      "壹百万元整",
        "currency":         "人民币",
        "drawer":           "测试公司",
        "drawer_account":   "6222021234567890",
        "drawer_bank":      "工商银行测试支行",
        "acceptor":         "工商银行测试支行",
        "payee":            "收款方公司",
        "drawee_bank":      "工商银行测试支行",
        "endorsers":        [],
        "maturity_days":    365,
        "trade_purpose":    "货款结算",
        "acceptance_clause": None,
        "special_remarks":  None,
        "confidence_score": 0.95,
    }

    # 模拟 ComplianceRetrievalAgent 写入的合规汇总
    ctx.shared_data["compliance_summary"] = {
        "total_fields":    18,
        "violation_count": 1,
        "severe_count":    0,
        "compliance_rate": 0.944,
        "is_overall_compliant": True,
    }

    # 模拟 EndorsementChainAgent 写入的背书结果
    ctx.shared_data["endorsement_result"] = {
        "chain_id":       "test-chain-001",
        "endorser_count": 0,
        "is_continuous":  True,
        "has_cycle":      False,
        "violation_count": 0,
        "violation_codes": [],
    }

    # 模拟 ContractReviewAgent 写入的合同结果
    ctx.shared_data["contract_result"] = {
        "review_id":              "test-review-001",
        "match_score":            85.0,
        "trade_background_score": 70.0,
        "amount_match":           True,
        "party_match":            True,
        "mismatch_count":         0,
    }

    # 模拟 FraudDetectionAgent 写入的欺诈检测结果
    ctx.shared_data["fraud_result"] = {
        "detection_id":      "test-fraud-001",
        "overall_fraud_score": 0.05,
        "seal_score":         0.02,
        "duplicate_score":    0.0,
        "tamper_score":       0.05,
        "network_score":      0.01,
        "blacklist_score":    0.0,
    }

    # 模拟 RiskAssessmentAgent 写入的风险评估结果
    ctx.shared_data["risk_result"] = {
        "assessment_id":   "test-assess-001",
        "compliance_score":  94.4,
        "endorsement_score": 100.0,
        "contract_score":    85.0,
        "fraud_score":       95.0,
        "composite_score":   93.8,
        "risk_level":        "LOW",
        "missing_dimensions": [],
    }

    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：ReportRenderer 报告渲染器
# ──────────────────────────────────────────────────────────────────────────────

class TestReportRenderer(unittest.TestCase):
    """验证 PDF 文件生成"""

    def test_pdf_file_created(self):
        """验证 render() 生成了 PDF 文件"""
        renderer = ReportRenderer(output_dir=tempfile.gettempdir())

        report_json = {
            "summary":     {"task_id": "test-001", "composite_score": 85.0, "risk_level": "LOW", "generated_at": "2024-01-15T10:00:00"},
            "elements":    {"ticket_number": "SHBH001"},
            "compliance":  {"compliance_rate": 0.95, "violation_count": 1, "severe_count": 0},
            "endorsement": {"endorser_count": 0, "is_continuous": True, "has_cycle": False, "violation_count": 0, "violation_codes": []},
            "contract":    {"match_score": 85.0, "amount_match": True, "party_match": True},
            "risk":        {"composite_score": 85.0, "risk_level": "LOW", "compliance_score": 90.0, "endorsement_score": 100.0, "contract_score": 85.0, "fraud_score": 95.0},
            "fraud":       {"overall_fraud_score": 0.05},
            "flow":        {"task_status": None, "completed_steps": None},
            "conclusion":  {"decision": "APPROVE", "recommendations": ["No issues identified."]},
        }

        pdf_path = renderer.render(report_json, "test-task-001", training_mode=False)

        # 验证文件存在
        self.assertTrue(os.path.exists(pdf_path), f"PDF 文件未生成: {pdf_path}")
        # 验证文件大小 > 0
        file_size = os.path.getsize(pdf_path)
        self.assertGreater(file_size, 0, "PDF 文件大小为 0")

        # 清理测试文件
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    def test_pdf_file_size_nonzero(self):
        """验证生成的 PDF 文件大小合理（至少 1KB）"""
        renderer = ReportRenderer(output_dir=tempfile.gettempdir())

        report_json = {
            "summary":     {"task_id": "test-002", "composite_score": 50.0, "risk_level": "MEDIUM_HIGH", "generated_at": "2024-01-15T10:00:00"},
            "elements":    {},
            "compliance":  {"compliance_rate": 0.80, "violation_count": 3, "severe_count": 1},
            "endorsement": {"endorser_count": 2, "is_continuous": False, "has_cycle": False, "violation_count": 1, "violation_codes": ["EN01"]},
            "contract":    {"match_score": 50.0, "amount_match": False, "party_match": True},
            "risk":        {"composite_score": 50.0, "risk_level": "MEDIUM_HIGH"},
            "fraud":       {"overall_fraud_score": 0.2},
            "flow":        {},
            "conclusion":  {"decision": "MANUAL_REVIEW", "recommendations": ["Review required"]},
        }

        pdf_path = renderer.render(report_json, "test-task-002")
        file_size = os.path.getsize(pdf_path)

        # PDF 至少应包含页眉/页脚等基础内容，大小应 > 1000 字节
        self.assertGreater(file_size, 1000)

        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    def test_training_mode_pdf(self):
        """验证培训模式下 PDF 生成成功（附加注解层）"""
        renderer = ReportRenderer(output_dir=tempfile.gettempdir())

        report_json = {
            "summary":     {"task_id": "test-003", "composite_score": 80.0, "risk_level": "MEDIUM_LOW"},
            "elements":    {}, "compliance": {}, "endorsement": {}, "contract": {},
            "risk":        {}, "fraud": {}, "flow": {}, "conclusion": {"decision": "APPROVE"},
        }

        pdf_path = renderer.render(report_json, "test-task-003", training_mode=True)

        self.assertTrue(os.path.exists(pdf_path))
        # 培训模式比非培训模式多一页（注解），文件应更大一些
        self.assertGreater(os.path.getsize(pdf_path), 0)

        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：ReportGenerationAgent 报告生成 Agent
# ──────────────────────────────────────────────────────────────────────────────

class TestReportGenerationAgent(unittest.TestCase):
    """验证报告 JSON 9 节结构和 Agent 整体流程"""

    def setUp(self):
        # 使用临时目录存放测试 PDF
        self.temp_dir = tempfile.mkdtemp()
        self.agent = ReportGenerationAgent(output_dir=self.temp_dir)

    def tearDown(self):
        """清理测试生成的 PDF 文件"""
        for f in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, f))
            except Exception:
                pass

    def test_report_has_9_sections(self):
        """验证 report_json 包含 9 个必需节"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        # 从数据库 add 的对象中检查 report_json
        added = mock_db.add.call_args[0][0]
        report_json = added.report_json

        required_sections = [
            "summary", "elements", "compliance", "endorsement",
            "contract", "risk", "fraud", "flow", "conclusion",
        ]
        for section in required_sections:
            self.assertIn(section, report_json, f"报告缺少节：{section}")

    def test_report_summary_has_required_fields(self):
        """验证 summary 节包含关键字段"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        added = mock_db.add.call_args[0][0]
        summary = added.report_json["summary"]
        self.assertIn("task_id", summary)
        self.assertIn("composite_score", summary)
        self.assertIn("risk_level", summary)
        self.assertIn("generated_at", summary)

    def test_report_conclusion_has_decision(self):
        """验证 conclusion 节包含审核决策"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        added = mock_db.add.call_args[0][0]
        conclusion = added.report_json["conclusion"]
        self.assertIn("decision", conclusion)
        # LOW 风险对应 APPROVE 决策
        self.assertEqual(conclusion["decision"], "APPROVE")

    def test_pdf_generated_and_size_nonzero(self):
        """验证 PDF 文件生成且大小 > 0"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertTrue(result.data["has_pdf"], "PDF 文件未生成")
        self.assertGreater(result.data["pdf_size_bytes"], 0)

    def test_pdf_path_returned_in_result(self):
        """验证结果中包含 PDF 文件路径"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertIsNotNone(result.data["pdf_path"])
        self.assertTrue(os.path.exists(result.data["pdf_path"]))

    def test_audit_report_written_to_db(self):
        """验证 AuditReport 记录写入数据库"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        mock_db.add.assert_called_once()
        from app.models.agent_models import AuditReport
        added = mock_db.add.call_args[0][0]
        self.assertIsInstance(added, AuditReport)
        self.assertEqual(added.audit_task_id, "m6-test-001")

    def test_result_written_to_shared_data(self):
        """验证报告结果写入 shared_data["report_result"]"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        self.assertIn("report_result", ctx.shared_data)
        rr = ctx.shared_data["report_result"]
        self.assertIn("report_id", rr)
        self.assertIn("pdf_path", rr)
        self.assertIn("pdf_size_bytes", rr)
        self.assertIn("has_pdf", rr)

    def test_training_mode_flag_stored_in_db(self):
        """验证培训模式标志正确存储在数据库记录中"""
        ctx = make_full_ctx(task_id="m6-train-001")
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db, training_mode=True))

        added = mock_db.add.call_args[0][0]
        self.assertTrue(added.training_mode)

    def test_is_final_flag_stored_in_db(self):
        """验证终版报告标志正确存储"""
        ctx = make_full_ctx(task_id="m6-final-001")
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db, is_final=True))

        added = mock_db.add.call_args[0][0]
        self.assertTrue(added.is_final)

    def test_pdf_render_failure_does_not_crash(self):
        """验证 PDF 渲染失败时 Agent 仍成功返回（JSON 报告仍保存）"""
        from unittest.mock import patch

        ctx = make_full_ctx(task_id="m6-fail-001")
        mock_db = make_mock_db()

        # 模拟 PDF 渲染失败
        with patch.object(ReportRenderer, "render", side_effect=RuntimeError("PDF 渲染失败")):
            result = run_async(self.agent.run(ctx, mock_db))

        # Agent 应该成功（JSON 保存），但 has_pdf=False
        self.assertTrue(result.success)
        self.assertFalse(result.data["has_pdf"])
        self.assertEqual(result.data["pdf_size_bytes"], 0)

    def test_empty_shared_data_no_crash(self):
        """验证 shared_data 为空时 Agent 不崩溃（优雅降级）"""
        ctx = AgentContext(audit_task_id="m6-empty-001", tenant_id="t-001")
        # shared_data 为空（所有前置 Agent 均未运行）
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        # 即使没有任何中间结果，也应成功生成（空内容的）报告
        self.assertTrue(result.success)
        self.assertEqual(result.data["sections_count"], 9)

    def test_report_elements_maps_18_fields(self):
        """验证报告 elements 节包含 18 个票据字段"""
        ctx = make_full_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        added = mock_db.add.call_args[0][0]
        elements = added.report_json["elements"]

        # 18 个字段
        expected_fields = [
            "ticket_number", "ticket_type", "issue_date", "due_date",
            "amount_numeric", "amount_text", "currency", "drawer",
            "drawer_account", "drawer_bank", "acceptor", "payee",
            "drawee_bank", "endorsers", "maturity_days", "trade_purpose",
            "acceptance_clause", "special_remarks",
        ]
        for field in expected_fields:
            self.assertIn(field, elements, f"elements 节缺少字段: {field}")


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
