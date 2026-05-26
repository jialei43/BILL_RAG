# tests/test_agents/test_m4_compliance.py
# M4 模块测试：验证合规 RAG 检索并行查询和背书链 9 类违规检测
# 运行方式：python -m pytest tests/test_agents/test_m4_compliance.py -v

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.compliance_retrieval_agent import ComplianceRetrievalAgent, FIELD_QUERY_TEMPLATES
from app.agents.endorsement_chain_agent import EndorsementChainAgent


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


def make_bill_element_ctx(
    ticket_number="SHBH20240115",
    drawer="测试公司",
    payee="收款方公司",
    endorsers=None,
) -> AgentContext:
    """创建包含 bill_element 的上下文"""
    ctx = AgentContext(audit_task_id="m4-test-001", tenant_id="t-001")
    ctx.shared_data["bill_element"] = {
        "ticket_number":  ticket_number,
        "ticket_type":    "银行承兑汇票",
        "issue_date":     "2024-01-15",
        "due_date":       "2025-01-15",
        "amount_numeric": 1000000.0,
        "amount_text":    "壹百万元整",
        "currency":       "人民币",
        "drawer":         drawer,
        "drawer_account": "6222021234567890",
        "drawer_bank":    "工商银行测试支行",
        "acceptor":       "工商银行测试支行",
        "payee":          payee,
        "drawee_bank":    "工商银行测试支行",
        "endorsers":      endorsers or [],
        "maturity_days":  365,
        "trade_purpose":  "货款结算",
        "confidence_score": 0.95,
    }
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：ComplianceRetrievalAgent 并行查询
# ──────────────────────────────────────────────────────────────────────────────

class TestComplianceRetrievalAgent(unittest.TestCase):
    """验证合规检索的并行执行和结果写入"""

    def setUp(self):
        self.agent = ComplianceRetrievalAgent()
        self.mock_db = make_mock_db()

    def _mock_rag_call(self, answer="合规规定：符合要求", score=0.85):
        """创建固定返回值的 RAG mock"""
        return AsyncMock(return_value=(answer, score))

    def test_parallel_queries_for_all_fields(self):
        """验证 18 个字段都发起了 RAG 查询（并行执行）"""
        ctx = make_bill_element_ctx()
        call_count = []  # 记录 RAG 调用次数

        async def mock_rag(query):
            call_count.append(query)  # 记录每次调用的查询内容
            return "合规规定内容", 0.85

        with patch.object(self.agent, "_call_rag", side_effect=mock_rag):
            result = run_async(self.agent.run(ctx, self.mock_db))

        # 验证每个字段都发起了查询
        self.assertEqual(len(call_count), len(FIELD_QUERY_TEMPLATES),
                         f"应该有 {len(FIELD_QUERY_TEMPLATES)} 次 RAG 调用，实际 {len(call_count)} 次")

    def test_compliance_checks_added_to_db(self):
        """验证所有 18 个合规检查结果都写入了数据库"""
        ctx = make_bill_element_ctx()

        async def mock_rag(query):
            return "合规规定", 0.85

        with patch.object(self.agent, "_call_rag", side_effect=mock_rag):
            run_async(self.agent.run(ctx, self.mock_db))

        # db.add() 应该被调用 18 次（每字段一条记录）
        self.assertEqual(self.mock_db.add.call_count, len(FIELD_QUERY_TEMPLATES))

    def test_required_field_empty_triggers_severe_violation(self):
        """验证必填字段为空时产生 SEVERE 违规"""
        from app.models.agent_models import ViolationLevel
        ctx = make_bill_element_ctx(ticket_number=None)  # 票据号码为空

        async def mock_rag(query):
            return "合规规定", 0.85

        with patch.object(self.agent, "_call_rag", side_effect=mock_rag):
            result = run_async(self.agent.run(ctx, self.mock_db))

        self.assertTrue(result.success)
        # 验证违规计数大于 0
        self.assertGreater(result.data["violation_count"], 0)
        self.assertGreater(result.data["severe_count"], 0)

    def test_all_fields_present_high_compliance_rate(self):
        """验证所有字段都有值且 RAG 高分时合规率高"""
        ctx = make_bill_element_ctx()

        async def mock_rag(query):
            return "符合规定", 0.95  # 高分表示知识库有明确规定

        with patch.object(self.agent, "_call_rag", side_effect=mock_rag):
            result = run_async(self.agent.run(ctx, self.mock_db))

        self.assertTrue(result.success)
        self.assertGreater(result.data["compliance_rate"], 0.9)

    def test_result_written_to_shared_data(self):
        """验证合规汇总写入 shared_data["compliance_summary"]"""
        ctx = make_bill_element_ctx()

        async def mock_rag(query):
            return "合规规定", 0.80

        with patch.object(self.agent, "_call_rag", side_effect=mock_rag):
            run_async(self.agent.run(ctx, self.mock_db))

        self.assertIn("compliance_summary", ctx.shared_data)
        summary = ctx.shared_data["compliance_summary"]
        self.assertIn("compliance_rate", summary)
        self.assertIn("violation_count", summary)
        self.assertIn("is_overall_compliant", summary)

    def test_no_bill_element_returns_failure(self):
        """验证 shared_data 中缺少 bill_element 时返回失败"""
        ctx = AgentContext(audit_task_id="m4-noelem", tenant_id="t-001")
        # 不写入 bill_element

        result = run_async(self.agent.run(ctx, self.mock_db))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "C_NO_ELEMENT")

    def test_rag_failure_does_not_crash(self):
        """验证 RAG 服务异常时不崩溃，返回默认值"""
        ctx = make_bill_element_ctx()

        async def mock_rag_fail(query):
            raise ConnectionError("RAG 服务不可用")  # 模拟 RAG 服务崩溃

        with patch.object(self.agent, "_call_rag", side_effect=mock_rag_fail):
            result = run_async(self.agent.run(ctx, self.mock_db))

        # 即使 RAG 异常，Agent 也应该成功完成（不崩溃）
        self.assertTrue(result.success)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：EndorsementChainAgent 背书链违规检测
# ──────────────────────────────────────────────────────────────────────────────

class TestEndorsementChainAgent(unittest.TestCase):
    """验证背书链的 9 类违规检测逻辑"""

    def setUp(self):
        self.agent = EndorsementChainAgent()

    def _run_agent(self, drawer, payee, endorsers):
        """辅助方法：构建上下文并运行 Agent"""
        ctx = make_bill_element_ctx(drawer=drawer, payee=payee, endorsers=endorsers)
        ctx.audit_task_id = f"endorse-{drawer[:4]}"  # 防止 UUID 冲突
        mock_db = make_mock_db()
        return run_async(self.agent.run(ctx, mock_db)), ctx

    def test_clean_chain_no_violations(self):
        """验证正常背书链（无违规）"""
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=["第一背书人公司", "第二背书人公司"],
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["violation_count"], 0)
        self.assertTrue(result.data["is_continuous"])

    def test_duplicate_endorser_detected(self):
        """验证重复背书：同一主体出现两次（EN02）"""
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=["收款公司", "第三方公司"],  # 收款公司又出现在背书人列表（重复）
        )

        self.assertTrue(result.success)
        self.assertIn("EN02", result.data["violation_codes"])

    def test_cycle_in_endorsement_detected(self):
        """验证背书闭环（EN03）：有向图中存在环路"""
        # 构造闭环：出票人 → 收款人 → 背书人A → 收款人（形成环路）
        # 通过直接测试 _check_cycle 方法
        agent = EndorsementChainAgent()

        # 有环路的边列表
        edges_with_cycle = [
            {"from": "n0", "to": "n1"},
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n1"},  # 形成环路 n1 → n2 → n1
        ]
        self.assertTrue(agent._check_cycle(edges_with_cycle))

    def test_no_cycle_in_linear_chain(self):
        """验证线性背书链没有环路"""
        agent = EndorsementChainAgent()
        edges_linear = [
            {"from": "n0", "to": "n1"},
            {"from": "n1", "to": "n2"},
            {"from": "n2", "to": "n3"},
        ]
        self.assertFalse(agent._check_cycle(edges_linear))

    def test_blank_endorser_detected(self):
        """验证空白背书（EN04）：背书人列表中有空值"""
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=["第一背书人", "", "第三背书人"],  # 第二背书人为空
        )

        self.assertTrue(result.success)
        self.assertIn("EN04", result.data["violation_codes"])
        self.assertGreater(result.data.get("violation_count", 0), 0)

    def test_drawer_in_endorsers_detected(self):
        """验证出票人在背书人列表中（EN07）"""
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=["中间背书人", "出票公司", "最终持票人"],  # 出票公司又出现在背书人中
        )

        self.assertTrue(result.success)
        self.assertIn("EN07", result.data["violation_codes"])

    def test_self_endorsement_detected(self):
        """验证自我背书（EN09）：相邻背书人相同"""
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=["A公司", "A公司", "B公司"],  # A公司连续背书两次
        )

        self.assertTrue(result.success)
        self.assertIn("EN09", result.data["violation_codes"])

    def test_max_depth_violation(self):
        """验证超出最大背书层数（EN06）：超过 10 层"""
        # 构造 11 个背书人（超过 MAX_ENDORSEMENT_DEPTH=10）
        endorsers = [f"第{i+1}背书人公司" for i in range(11)]
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=endorsers,
        )

        self.assertTrue(result.success)
        self.assertIn("EN06", result.data["violation_codes"])

    def test_empty_endorsers_no_violation(self):
        """验证无背书人时不产生违规"""
        result, ctx = self._run_agent(
            drawer="出票公司",
            payee="收款公司",
            endorsers=[],  # 未背书转让
        )

        self.assertTrue(result.success)
        self.assertEqual(result.data["endorser_count"], 0)
        # 无背书人不应产生 EN04 等背书相关违规
        violation_codes = result.data["violation_codes"]
        self.assertNotIn("EN04", violation_codes)
        self.assertNotIn("EN06", violation_codes)

    def test_result_written_to_db(self):
        """验证背书链分析结果写入了数据库"""
        ctx = make_bill_element_ctx(endorsers=["背书人A", "背书人B"])
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        mock_db.add.assert_called_once()
        from app.models.agent_models import EndorsementChain
        added = mock_db.add.call_args[0][0]
        self.assertIsInstance(added, EndorsementChain)

    def test_result_written_to_shared_data(self):
        """验证分析结果写入 shared_data["endorsement_result"]"""
        ctx = make_bill_element_ctx(endorsers=["背书人A"])
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        self.assertIn("endorsement_result", ctx.shared_data)
        er = ctx.shared_data["endorsement_result"]
        self.assertIn("is_continuous", er)
        self.assertIn("has_cycle", er)
        self.assertIn("violation_codes", er)


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
