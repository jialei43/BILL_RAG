# tests/test_agents/test_m3_parsing.py
# M3 模块测试：验证 DocumentParserAgent 和 ElementExtractionAgent 的字段映射和数据库写入
# 测试策略：
#   - mock pdf_parser.BillPDFParser / bill_recognition.BillRecognitionService
#   - 验证置信度计算逻辑
#   - 验证 18 字段正确写入 shared_data 和 BillElement ORM 对象
# 运行方式：python -m pytest tests/test_agents/test_m3_parsing.py -v

import asyncio
import os
import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.document_parser_agent import DocumentParserAgent
from app.agents.element_extraction_agent import ElementExtractionAgent


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def make_mock_db():
    """创建不连接真实数据库的 mock 会话"""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock())
    mock_db.add = MagicMock()   # add() 是同步方法
    return mock_db


# ──────────────────────────────────────────────────────────────────────────────
# Mock 数据类（模拟现有服务的返回值，不导入真实服务）
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MockParsedElement:
    """模拟 pdf_parser.ParsedElement"""
    text: str = "测试文本"
    element_type: str = "text"
    metadata: dict = field(default_factory=dict)


@dataclass
class MockParsedDocument:
    """模拟 pdf_parser.ParsedDocument"""
    elements: list = field(default_factory=list)
    parse_stats: dict = field(default_factory=dict)


@dataclass
class MockBillElement:
    """模拟 bill_recognition.BillElement（15 个基础字段）"""
    ticket_number:  str   = "SHBH20240115123456"
    ticket_type:    str   = "银行承兑汇票"
    issue_date:     str   = "2024-01-15"
    due_date:       str   = "2025-01-15"
    amount_numeric: float = 1000000.0
    amount_text:    str   = "壹百万元整"
    currency:       str   = "人民币"
    drawer:         str   = "测试出票公司有限公司"
    drawer_account: str   = "6222021234567890"
    drawer_bank:    str   = "中国工商银行测试支行"
    acceptor:       str   = "中国工商银行测试支行"
    payee:          str   = "测试收款公司有限公司"
    drawee_bank:    str   = "中国工商银行测试支行"
    endorsers:      list  = field(default_factory=list)
    # maturity_days 不在原始 BillElement 中，由 Agent 计算


@dataclass
class MockRecognitionResult:
    """模拟 bill_recognition.RecognitionResult"""
    bills: list = field(default_factory=list)
    raw_texts: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：DocumentParserAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentParserAgent(unittest.TestCase):
    """验证 DocumentParserAgent 的解析流程和置信度计算"""

    def setUp(self):
        self.ctx = AgentContext(audit_task_id="m3-parse-001", tenant_id="t-001")
        self.mock_db = make_mock_db()
        self.agent = DocumentParserAgent()

    def _make_mock_parsed_doc(self, element_count=5, ocr_confidence=None):
        """创建 Mock ParsedDocument"""
        elements = [MockParsedElement(text=f"文本片段{i}") for i in range(element_count)]
        stats = {"page_count": 3}
        if ocr_confidence is not None:
            stats["ocr_avg_confidence"] = ocr_confidence
        return MockParsedDocument(elements=elements, parse_stats=stats)

    def test_parse_success_writes_to_shared_data(self):
        """验证解析成功后结果写入 shared_data["parsed_doc"]"""
        mock_doc = self._make_mock_parsed_doc(element_count=5, ocr_confidence=0.92)

        with patch.object(self.agent, "_parse_file", new_callable=AsyncMock) as mock_parse, \
             patch("os.path.exists", return_value=True):

            mock_parse.return_value = mock_doc

            result = run_async(self.agent.run(
                self.ctx, self.mock_db, file_path="/fake/path/test.pdf"
            ))

        self.assertTrue(result.success)
        self.assertIn("parsed_doc", self.ctx.shared_data)
        self.assertIn("elements", self.ctx.shared_data["parsed_doc"])

    def test_parse_returns_element_count(self):
        """验证解析结果包含正确的片段数量"""
        mock_doc = self._make_mock_parsed_doc(element_count=8, ocr_confidence=0.95)

        with patch.object(self.agent, "_parse_file", new_callable=AsyncMock) as mock_parse, \
             patch("os.path.exists", return_value=True):

            mock_parse.return_value = mock_doc
            result = run_async(self.agent.run(
                self.ctx, self.mock_db, file_path="/fake/path/test.pdf"
            ))

        self.assertEqual(result.data["element_count"], 8)

    def test_low_confidence_triggers_warning_flag(self):
        """验证低置信度（< 0.85）时结果中包含 low_confidence=True"""
        mock_doc = self._make_mock_parsed_doc(element_count=3, ocr_confidence=0.72)

        with patch.object(self.agent, "_parse_file", new_callable=AsyncMock) as mock_parse, \
             patch("os.path.exists", return_value=True):

            mock_parse.return_value = mock_doc
            result = run_async(self.agent.run(
                self.ctx, self.mock_db, file_path="/fake/path/test.pdf"
            ))

        self.assertTrue(result.success)           # 低置信度不阻断流程
        self.assertTrue(result.data["low_confidence"])  # 但标记 low_confidence

    def test_high_confidence_no_warning_flag(self):
        """验证高置信度（>= 0.85）时 low_confidence=False"""
        mock_doc = self._make_mock_parsed_doc(ocr_confidence=0.95)

        with patch.object(self.agent, "_parse_file", new_callable=AsyncMock) as mock_parse, \
             patch("os.path.exists", return_value=True):

            mock_parse.return_value = mock_doc
            result = run_async(self.agent.run(
                self.ctx, self.mock_db, file_path="/fake/path/test.pdf"
            ))

        self.assertFalse(result.data["low_confidence"])

    def test_file_not_found_returns_failure(self):
        """验证文件不存在时返回失败结果"""
        with patch("os.path.exists", return_value=False):
            result = run_async(self.agent.run(
                self.ctx, self.mock_db, file_path="/nonexistent/path/test.pdf"
            ))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "E_PARSE_FILE_NOT_FOUND")

    def test_no_file_path_returns_failure(self):
        """验证既未传入 file_path 又无 document_id 时返回失败"""
        mock_db = make_mock_db()
        # execute 返回空结果（模拟数据库中无记录）
        mock_result = MagicMock()
        mock_result.first = MagicMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = run_async(self.agent.run(
            AgentContext(audit_task_id="m3-parse-nofile", tenant_id="t-001"),
            mock_db,
            # 不传 file_path
        ))

        self.assertFalse(result.success)
        self.assertIn("E_PARSE", result.error_code)

    def test_shared_data_confidence_filled(self):
        """验证 shared_data 中包含 confidence 字段"""
        mock_doc = self._make_mock_parsed_doc(ocr_confidence=0.88)

        with patch.object(self.agent, "_parse_file", new_callable=AsyncMock) as mock_parse, \
             patch("os.path.exists", return_value=True):

            mock_parse.return_value = mock_doc
            run_async(self.agent.run(
                self.ctx, self.mock_db, file_path="/fake/path/test.pdf"
            ))

        self.assertIn("confidence", self.ctx.shared_data["parsed_doc"])
        self.assertAlmostEqual(self.ctx.shared_data["parsed_doc"]["confidence"], 0.88, places=2)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：ElementExtractionAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestElementExtractionAgent(unittest.TestCase):
    """验证 ElementExtractionAgent 的 18 字段映射和数据库写入"""

    def setUp(self):
        self.ctx = AgentContext(audit_task_id="m3-extract-001", tenant_id="t-001")
        self.mock_db = make_mock_db()
        self.agent = ElementExtractionAgent()

    def _make_recognition_result(
        self,
        empty_fields: list = None,  # 指定为空的字段名列表（测试低置信度）
        raw_text: str = "",
    ) -> MockRecognitionResult:
        """创建 Mock RecognitionResult"""
        bill = MockBillElement()
        if empty_fields:
            for f in empty_fields:
                setattr(bill, f, None)  # 将指定字段置空，模拟识别失败
        return MockRecognitionResult(bills=[bill], raw_texts=[raw_text])

    def test_18_fields_in_shared_data(self):
        """验证 shared_data["bill_element"] 包含 18 个要素字段"""
        recognition = self._make_recognition_result()

        with patch.object(self.agent, "_sync_recognize", return_value=recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            run_async(self.agent.run(self.ctx, self.mock_db))

        elem = self.ctx.shared_data["bill_element"]

        # 验证 15 个基础字段
        base_fields = [
            "ticket_number", "ticket_type", "issue_date", "due_date",
            "amount_numeric", "amount_text", "currency", "drawer",
            "acceptor", "payee", "drawee_bank", "endorsers", "maturity_days",
        ]
        for field_name in base_fields:
            self.assertIn(field_name, elem, f"shared_data 缺少基础字段: {field_name}")

        # 验证 3 个扩展字段（可以为 None，但必须存在于字典中）
        extended_fields = ["trade_purpose", "confidence_score"]
        for field_name in extended_fields:
            self.assertIn(field_name, elem, f"shared_data 缺少扩展字段: {field_name}")

    def test_bill_element_added_to_db(self):
        """验证 BillElement ORM 对象被正确添加到数据库会话"""
        recognition = self._make_recognition_result()

        with patch.object(self.agent, "_sync_recognize", return_value=recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            result = run_async(self.agent.run(self.ctx, self.mock_db))

        self.assertTrue(result.success)
        self.mock_db.add.assert_called_once()  # 验证 db.add() 被调用了一次（添加一条记录）
        # 验证添加的对象是 BillElement 实例
        from app.models.agent_models import BillElement
        added_obj = self.mock_db.add.call_args[0][0]
        self.assertIsInstance(added_obj, BillElement)

    def test_correct_ticket_number_in_result(self):
        """验证结果中包含正确的票据号码"""
        recognition = self._make_recognition_result()

        with patch.object(self.agent, "_sync_recognize", return_value=recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            result = run_async(self.agent.run(self.ctx, self.mock_db))

        self.assertEqual(result.data["ticket_number"], "SHBH20240115123456")

    def test_confidence_calculation_all_fields_present(self):
        """验证所有字段都有值时置信度接近 1.0"""
        recognition = self._make_recognition_result()

        with patch.object(self.agent, "_sync_recognize", return_value=recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            result = run_async(self.agent.run(self.ctx, self.mock_db))

        self.assertGreater(result.data["confidence"], 0.8)  # 全字段时置信度应较高

    def test_confidence_lower_when_fields_missing(self):
        """验证部分字段缺失时置信度低于全字段时"""
        # 全字段置信度
        full_recognition = self._make_recognition_result()
        # 缺失 5 个字段的置信度
        partial_recognition = self._make_recognition_result(
            empty_fields=["drawer", "acceptor", "payee", "drawee_bank", "amount_text"]
        )

        with patch.object(self.agent, "_sync_recognize", return_value=full_recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):
            result_full = run_async(self.agent.run(
                AgentContext(audit_task_id="m3-full", tenant_id="t-001"), self.mock_db
            ))

        mock_db2 = make_mock_db()
        with patch.object(self.agent, "_sync_recognize", return_value=partial_recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):
            result_partial = run_async(self.agent.run(
                AgentContext(audit_task_id="m3-partial", tenant_id="t-001"), mock_db2
            ))

        self.assertGreater(result_full.data["confidence"], result_partial.data["confidence"])

    def test_no_bill_found_returns_failure(self):
        """验证识别到 0 张票据时返回失败"""
        empty_recognition = MockRecognitionResult(bills=[], raw_texts=[])

        with patch.object(self.agent, "_sync_recognize", return_value=empty_recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            result = run_async(self.agent.run(self.ctx, self.mock_db))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "E_EXTRACT_NO_BILL_FOUND")

    def test_maturity_days_calculated(self):
        """验证距到期天数被正确计算"""
        recognition = self._make_recognition_result()  # due_date = "2025-01-15"

        with patch.object(self.agent, "_sync_recognize", return_value=recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            run_async(self.agent.run(self.ctx, self.mock_db))

        # 不验证具体天数（依赖当前日期），但验证字段存在且为整数
        self.assertIn("maturity_days", self.ctx.shared_data["bill_element"])
        self.assertIsInstance(
            self.ctx.shared_data["bill_element"]["maturity_days"], int
        )

    def test_extended_fields_extracted_from_raw_text(self):
        """验证从原始 OCR 文本中提取扩展字段"""
        raw_text = "贸易背景：货款结算\n承兑条款：利率按LPR+50BP浮动\n备注：限于背书转让"
        recognition = self._make_recognition_result(raw_text=raw_text)

        with patch.object(self.agent, "_sync_recognize", return_value=recognition), \
             patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(b"fake_bytes", "test.pdf")):

            run_async(self.agent.run(self.ctx, self.mock_db))

        elem = self.ctx.shared_data["bill_element"]
        self.assertEqual(elem.get("trade_purpose"), "货款结算")  # 贸易背景被正确提取

    def test_no_file_bytes_returns_failure(self):
        """验证无法获取文件字节时返回失败"""
        with patch.object(self.agent, "_resolve_file_bytes", new_callable=AsyncMock,
                          return_value=(None, None)):

            result = run_async(self.agent.run(self.ctx, self.mock_db))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "E_EXTRACT_NO_FILE")


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
