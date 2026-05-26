# tests/test_agents/test_m5_risk.py
# M5 模块测试：验证合同审核和四维风险评分
# 运行方式：python -m pytest tests/test_agents/test_m5_risk.py -v

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.contract_review_agent import ContractReviewAgent
from app.agents.risk_assessment_agent import RiskAssessmentAgent, DIMENSION_WEIGHTS
from app.models.agent_models import RiskLevel


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


def make_base_ctx(task_id="m5-test-001") -> AgentContext:
    """创建基础上下文"""
    return AgentContext(audit_task_id=task_id, tenant_id="t-001")


def make_ctx_with_bill_element(
    amount=1000000.0,
    drawer="测试公司",
    payee="收款方公司",
    trade_purpose="货款结算",
) -> AgentContext:
    """创建带有 bill_element 的上下文"""
    ctx = make_base_ctx()
    ctx.shared_data["bill_element"] = {
        "ticket_number":  "SHBH20240115",
        "ticket_type":    "银行承兑汇票",
        "issue_date":     "2024-01-15",
        "due_date":       "2025-01-15",
        "amount_numeric": amount,
        "amount_text":    "壹百万元整",
        "currency":       "人民币",
        "drawer":         drawer,
        "drawer_account": "6222021234567890",
        "drawer_bank":    "工商银行测试支行",
        "acceptor":       "工商银行测试支行",
        "payee":          payee,
        "drawee_bank":    "工商银行测试支行",
        "endorsers":      [],
        "maturity_days":  365,
        "trade_purpose":  trade_purpose,
        "confidence_score": 0.95,
    }
    return ctx


def make_full_shared_data(
    compliance_rate=1.0,
    severe_count=0,
    is_continuous=True,
    has_cycle=False,
    endorsement_violations=0,
    match_score=100.0,
    fraud_score=0.0,
) -> dict:
    """构建包含四维中间结果的 shared_data"""
    return {
        "compliance_summary": {
            "total_fields":    18,
            "violation_count": int((1 - compliance_rate) * 18),
            "severe_count":    severe_count,
            "compliance_rate": compliance_rate,
            "is_overall_compliant": severe_count == 0,
        },
        "endorsement_result": {
            "chain_id":        "test-chain-001",
            "endorser_count":  0,
            "is_continuous":   is_continuous,
            "has_cycle":       has_cycle,
            "violation_count": endorsement_violations,
            "violation_codes": [],
        },
        "contract_result": {
            "review_id":               "test-review-001",
            "match_score":             match_score,
            "trade_background_score":  80.0,
            "amount_match":            True,
            "party_match":             True,
            "mismatch_count":          0,
        },
        "fraud_result": {
            "detection_id":      "test-fraud-001",
            "overall_fraud_score": fraud_score,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：ContractReviewAgent 合同审核
# ──────────────────────────────────────────────────────────────────────────────

class TestContractReviewAgent(unittest.TestCase):
    """验证合同审核的 6 项要素比对逻辑"""

    def setUp(self):
        self.agent = ContractReviewAgent()
        self.mock_db = make_mock_db()

    def test_no_contract_returns_default_score(self):
        """验证无合同时返回默认中等分数（50分）"""
        ctx = make_ctx_with_bill_element()

        result = run_async(self.agent.run(ctx, self.mock_db))

        self.assertTrue(result.success)
        # 无合同时 match_score 应为默认值 50.0
        self.assertEqual(result.data["match_score"], 50.0)

    def test_contract_with_matching_amount(self):
        """验证合同金额与票面金额匹配时 amount_match=True"""
        ctx = make_ctx_with_bill_element(amount=1000000.0)
        # 合同金额与票面一致（100万元）
        contract_text = "甲方：测试公司\n乙方：收款方公司\n合同金额：100万元\n用途：货款"

        result = run_async(self.agent.run(ctx, self.mock_db, contract_text=contract_text))

        self.assertTrue(result.success)
        self.assertTrue(result.data["amount_match"])

    def test_contract_with_mismatched_amount(self):
        """验证合同金额与票面金额差超过 5% 时 amount_match=False"""
        ctx = make_ctx_with_bill_element(amount=1000000.0)
        # 合同金额 200 万，差距超过 5%
        contract_text = "甲方：测试公司\n乙方：收款方公司\n合同金额：200万元\n用途：货款"

        result = run_async(self.agent.run(ctx, self.mock_db, contract_text=contract_text))

        self.assertTrue(result.success)
        self.assertFalse(result.data["amount_match"])

    def test_contract_party_match(self):
        """验证票据交易方与合同甲乙方匹配时 party_match=True"""
        ctx = make_ctx_with_bill_element(drawer="测试公司", payee="收款方公司")
        contract_text = "甲方：测试公司\n乙方：收款方公司\n合同金额：100万元"

        result = run_async(self.agent.run(ctx, self.mock_db, contract_text=contract_text))

        self.assertTrue(result.success)
        self.assertTrue(result.data["party_match"])

    def test_contract_party_mismatch(self):
        """验证票据交易方与合同方不匹配时 party_match=False"""
        ctx = make_ctx_with_bill_element(drawer="测试公司", payee="收款方公司")
        # 合同甲乙方与票据出票人/收款人完全不同
        contract_text = "甲方：完全不相关公司A\n乙方：完全不相关公司B\n合同金额：100万元"

        result = run_async(self.agent.run(ctx, self.mock_db, contract_text=contract_text))

        self.assertTrue(result.success)
        self.assertFalse(result.data["party_match"])

    def test_full_match_score_100(self):
        """验证四项全部匹配时综合得分接近 100（满分：amount25+party35+date20+purpose20）"""
        ctx = make_ctx_with_bill_element(
            amount=1000000.0, drawer="测试公司", payee="收款方公司", trade_purpose="货款结算"
        )
        contract_text = "甲方：测试公司\n乙方：收款方公司\n合同金额：100万元\n用途：货款结算"

        result = run_async(self.agent.run(ctx, self.mock_db, contract_text=contract_text))

        self.assertTrue(result.success)
        # 全部匹配时综合分应为 100（25+35+20+20）
        self.assertEqual(result.data["match_score"], 100.0)

    def test_result_written_to_db(self):
        """验证合同审核结果写入了数据库"""
        ctx = make_ctx_with_bill_element()

        run_async(self.agent.run(ctx, self.mock_db))

        # ContractReview 记录应被添加到 db 会话
        self.mock_db.add.assert_called_once()
        from app.models.agent_models import ContractReview
        added = self.mock_db.add.call_args[0][0]
        self.assertIsInstance(added, ContractReview)

    def test_result_written_to_shared_data(self):
        """验证审核结果写入 shared_data["contract_result"]"""
        ctx = make_ctx_with_bill_element()

        run_async(self.agent.run(ctx, self.mock_db))

        self.assertIn("contract_result", ctx.shared_data)
        cr = ctx.shared_data["contract_result"]
        self.assertIn("match_score", cr)
        self.assertIn("amount_match", cr)
        self.assertIn("party_match", cr)
        self.assertIn("mismatch_count", cr)

    def test_trade_background_score_without_purpose(self):
        """验证无贸易背景说明时 trade_background_score 较低（40分）"""
        ctx = make_ctx_with_bill_element(trade_purpose=None)
        ctx.shared_data["bill_element"]["trade_purpose"] = None

        result = run_async(self.agent.run(ctx, self.mock_db))

        self.assertTrue(result.success)
        # 无贸易背景说明时评分应为 40
        self.assertEqual(result.data["trade_background_score"], 40.0)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：RiskAssessmentAgent 四维风险评分
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskAssessmentAgent(unittest.TestCase):
    """验证四维加权评分和五档风险阈值"""

    def setUp(self):
        self.agent = RiskAssessmentAgent()

    def _run_with_shared_data(self, shared_data: dict, task_id="risk-test") -> AgentResult:
        """辅助方法：注入 shared_data 并运行 Agent"""
        ctx = make_base_ctx(task_id=task_id)
        ctx.shared_data.update(shared_data)
        mock_db = make_mock_db()
        return run_async(self.agent.run(ctx, mock_db)), ctx

    # ── 五档风险等级临界值测试 ──────────────────────────────────────────────────

    def test_risk_level_low_above_85(self):
        """验证综合评分 ≥ 85 时风险等级为 LOW"""
        # 构造所有维度高分的场景：合规率100% + 背书无违规 + 合同全匹配 + 欺诈0分
        # 期望综合分：100×0.3 + 100×0.3 + 100×0.2 + 100×0.2 = 100
        data = make_full_shared_data(
            compliance_rate=1.0, severe_count=0,
            is_continuous=True, has_cycle=False, endorsement_violations=0,
            match_score=100.0, fraud_score=0.0,
        )
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.data["composite_score"], 85.0)
        self.assertEqual(result.data["risk_level"], RiskLevel.LOW.value)

    def test_risk_level_medium_low_70_to_84(self):
        """验证综合评分 70~84 时风险等级为 MEDIUM_LOW"""
        # 合规率 80%（无严重违规），背书无违规，合同匹配度 70，欺诈低
        # 合规得分：80（无扣分），背书：100，合同：70，欺诈：95
        # 综合：80×0.3 + 100×0.3 + 70×0.2 + 95×0.2 = 24+30+14+19 = 87 → 需要调低
        # 合规率 70%：70分，合同 60 分，欺诈 90 分
        # 综合：70×0.3 + 100×0.3 + 60×0.2 + 90×0.2 = 21+30+12+18 = 81
        data = make_full_shared_data(
            compliance_rate=0.70, severe_count=0,
            is_continuous=True, has_cycle=False, endorsement_violations=0,
            match_score=60.0, fraud_score=0.1,
        )
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        score = result.data["composite_score"]
        self.assertGreaterEqual(score, 70.0)
        self.assertLess(score, 85.0)
        self.assertEqual(result.data["risk_level"], RiskLevel.MEDIUM_LOW.value)

    def test_risk_level_medium_high_50_to_69(self):
        """验证综合评分 50~69 时风险等级为 MEDIUM_HIGH"""
        # 合规率 60%，背书断链（-40），合同低匹配度，欺诈中等
        # 合规得分：60，背书：60（断链扣40），合同：50，欺诈：70
        # 综合：60×0.3 + 60×0.3 + 50×0.2 + 70×0.2 = 18+18+10+14 = 60
        data = make_full_shared_data(
            compliance_rate=0.60, severe_count=0,
            is_continuous=False, has_cycle=False, endorsement_violations=1,
            match_score=50.0, fraud_score=0.30,
        )
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        score = result.data["composite_score"]
        self.assertGreaterEqual(score, 50.0)
        self.assertLess(score, 70.0)
        self.assertEqual(result.data["risk_level"], RiskLevel.MEDIUM_HIGH.value)

    def test_risk_level_high_30_to_49(self):
        """验证综合评分 30~49 时风险等级为 HIGH"""
        # 合规率 40%（有严重违规），背书断链+其他违规，合同低分，欺诈中高
        # 合规得分：40-15=25（1个严重），背书：100-40-10=50（断链+1违规），合同：30，欺诈：50
        # 综合：25×0.3 + 50×0.3 + 30×0.2 + 50×0.2 = 7.5+15+6+10 = 38.5
        data = make_full_shared_data(
            compliance_rate=0.40, severe_count=1,
            is_continuous=False, has_cycle=False, endorsement_violations=2,
            match_score=30.0, fraud_score=0.50,
        )
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        score = result.data["composite_score"]
        self.assertGreaterEqual(score, 30.0)
        self.assertLess(score, 50.0)
        self.assertEqual(result.data["risk_level"], RiskLevel.HIGH.value)

    def test_risk_level_critical_below_30(self):
        """验证综合评分 < 30 时风险等级为 CRITICAL"""
        # 合规率极低（有多个严重违规），背书存在闭环，合同不匹配，欺诈高
        # 合规得分：max(0, 20-45) = 0（3个严重），背书：max(0, 100-50-40) = 10（闭环+断链），
        # 合同：10，欺诈：10
        # 综合：0×0.3 + 10×0.3 + 10×0.2 + 10×0.2 = 0+3+2+2 = 7
        data = make_full_shared_data(
            compliance_rate=0.20, severe_count=3,
            is_continuous=False, has_cycle=True, endorsement_violations=2,
            match_score=10.0, fraud_score=0.90,
        )
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        score = result.data["composite_score"]
        self.assertLess(score, 30.0)
        self.assertEqual(result.data["risk_level"], RiskLevel.CRITICAL.value)

    # ── 维度缺失降级测试 ──────────────────────────────────────────────────────

    def test_missing_compliance_dimension_uses_default(self):
        """验证缺少 compliance_summary 时使用降级默认分"""
        # 只提供背书/合同/欺诈，缺少合规
        data = {
            "endorsement_result": {"is_continuous": True, "has_cycle": False, "violation_count": 0},
            "contract_result":    {"match_score": 80.0, "amount_match": True, "party_match": True, "mismatch_count": 0},
            "fraud_result":       {"overall_fraud_score": 0.1},
        }
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        self.assertIn("compliance", result.data["missing_dimensions"])
        # 合规维度应使用默认分 60.0
        self.assertEqual(result.data["compliance_score"], 60.0)

    def test_missing_contract_dimension_uses_default(self):
        """验证缺少 contract_result 时使用降级默认分（50分）"""
        data = {
            "compliance_summary": {"compliance_rate": 1.0, "severe_count": 0, "violation_count": 0},
            "endorsement_result": {"is_continuous": True, "has_cycle": False, "violation_count": 0},
            "fraud_result":       {"overall_fraud_score": 0.0},
        }
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        self.assertIn("contract", result.data["missing_dimensions"])
        # 合同维度降级分为 50.0
        self.assertEqual(result.data["contract_score"], 50.0)

    def test_all_dimensions_missing_uses_all_defaults(self):
        """验证四个维度全部缺失时全部使用降级分，流程不崩溃"""
        ctx = make_base_ctx(task_id="missing-all")
        # shared_data 中没有任何维度结果
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertEqual(len(result.data["missing_dimensions"]), 4)
        # 所有维度都应使用降级分
        self.assertIsNotNone(result.data["composite_score"])

    def test_missing_fraud_dimension_uses_default_80(self):
        """验证缺少欺诈检测结果时使用降级默认分（80分）"""
        data = {
            "compliance_summary": {"compliance_rate": 1.0, "severe_count": 0, "violation_count": 0},
            "endorsement_result": {"is_continuous": True, "has_cycle": False, "violation_count": 0},
            "contract_result":    {"match_score": 100.0, "amount_match": True, "party_match": True, "mismatch_count": 0},
        }
        result, ctx = self._run_with_shared_data(data)

        self.assertTrue(result.success)
        self.assertIn("fraud", result.data["missing_dimensions"])
        self.assertEqual(result.data["fraud_score"], 80.0)

    # ── 评分逻辑测试 ──────────────────────────────────────────────────────────

    def test_compliance_score_severe_violation_penalty(self):
        """验证合规维度：严重违规扣分逻辑（每个 SEVERE -15分）"""
        data = {
            "compliance_summary": {
                "compliance_rate": 1.0,   # 合规率 100%
                "severe_count":    2,      # 2 个严重违规
                "violation_count": 2,
            },
        }
        result, ctx = self._run_with_shared_data(data)
        # 合规得分：100 - 2×15 = 70
        self.assertEqual(result.data["compliance_score"], 70.0)

    def test_endorsement_score_cycle_deduction(self):
        """验证背书维度：闭环（-50）+ 断链（-40）的复合扣分"""
        data = {
            "endorsement_result": {
                "is_continuous":   False,   # 断链 -40
                "has_cycle":       True,    # 闭环 -50
                "violation_count": 2,       # 2个违规（断链+闭环均已计入扣分）
            },
        }
        result, ctx = self._run_with_shared_data(data)
        # 背书得分：100 - 40（断链） - 50（闭环） = 10，max(0, 10) = 10
        self.assertEqual(result.data["endorsement_score"], 10.0)

    def test_fraud_score_inversion(self):
        """验证欺诈维度分数翻转：欺诈风险 0.4 → 安全得分 60"""
        data = {
            "fraud_result": {"overall_fraud_score": 0.40},  # 欺诈风险 40%
        }
        result, ctx = self._run_with_shared_data(data)
        # 安全得分：(1 - 0.40) × 100 = 60.0
        self.assertEqual(result.data["fraud_score"], 60.0)

    def test_composite_score_formula(self):
        """验证综合评分公式：合规×0.3 + 背书×0.3 + 合同×0.2 + 欺诈×0.2"""
        # 构造已知各维度得分的数据
        # 合规：100分，背书：100分，合同：100分，欺诈：100分 → 综合应为 100
        data = make_full_shared_data(
            compliance_rate=1.0, severe_count=0,
            is_continuous=True, has_cycle=False, endorsement_violations=0,
            match_score=100.0, fraud_score=0.0,
        )
        result, ctx = self._run_with_shared_data(data)
        self.assertAlmostEqual(result.data["composite_score"], 100.0, places=1)

    def test_weight_sum_equals_1(self):
        """验证四维权重之和为 1.0"""
        total_weight = sum(DIMENSION_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)

    # ── 数据库和 shared_data 写入测试 ────────────────────────────────────────

    def test_result_written_to_db(self):
        """验证风险评估结果写入数据库"""
        ctx = make_base_ctx()
        ctx.shared_data.update(make_full_shared_data())
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        mock_db.add.assert_called_once()
        from app.models.agent_models import RiskAssessment
        added = mock_db.add.call_args[0][0]
        self.assertIsInstance(added, RiskAssessment)

    def test_result_written_to_shared_data(self):
        """验证风险评估结果写入 shared_data["risk_result"]"""
        ctx = make_base_ctx()
        ctx.shared_data.update(make_full_shared_data())
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        self.assertIn("risk_result", ctx.shared_data)
        rr = ctx.shared_data["risk_result"]
        self.assertIn("composite_score", rr)
        self.assertIn("risk_level", rr)
        self.assertIn("compliance_score", rr)
        self.assertIn("endorsement_score", rr)
        self.assertIn("contract_score", rr)
        self.assertIn("fraud_score", rr)
        self.assertIn("missing_dimensions", rr)

    def test_assessor_notes_generated(self):
        """验证评估备注文本非空"""
        ctx = make_base_ctx()
        ctx.shared_data.update(make_full_shared_data())
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        added = mock_db.add.call_args[0][0]
        self.assertIsNotNone(added.assessor_notes)
        self.assertGreater(len(added.assessor_notes), 0)

    def test_dimension_details_has_all_keys(self):
        """验证 dimension_details 包含四个维度的详细说明"""
        ctx = make_base_ctx()
        ctx.shared_data.update(make_full_shared_data())
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        added = mock_db.add.call_args[0][0]
        self.assertIn("compliance",  added.dimension_details)
        self.assertIn("endorsement", added.dimension_details)
        self.assertIn("contract",    added.dimension_details)
        self.assertIn("fraud",       added.dimension_details)

    # ── 边界值测试 ────────────────────────────────────────────────────────────

    def test_score_exactly_85_is_low_risk(self):
        """验证边界值：综合评分恰好 85 时为 LOW 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(85.0)
        self.assertEqual(level, RiskLevel.LOW)

    def test_score_84_99_is_medium_low(self):
        """验证边界值：综合评分 84.99 时为 MEDIUM_LOW 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(84.99)
        self.assertEqual(level, RiskLevel.MEDIUM_LOW)

    def test_score_exactly_70_is_medium_low(self):
        """验证边界值：综合评分恰好 70 时为 MEDIUM_LOW 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(70.0)
        self.assertEqual(level, RiskLevel.MEDIUM_LOW)

    def test_score_exactly_50_is_medium_high(self):
        """验证边界值：综合评分恰好 50 时为 MEDIUM_HIGH 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(50.0)
        self.assertEqual(level, RiskLevel.MEDIUM_HIGH)

    def test_score_exactly_30_is_high(self):
        """验证边界值：综合评分恰好 30 时为 HIGH 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(30.0)
        self.assertEqual(level, RiskLevel.HIGH)

    def test_score_29_99_is_critical(self):
        """验证边界值：综合评分 29.99 时为 CRITICAL 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(29.99)
        self.assertEqual(level, RiskLevel.CRITICAL)

    def test_score_zero_is_critical(self):
        """验证边界值：综合评分 0 时为 CRITICAL 风险"""
        agent = RiskAssessmentAgent()
        level = agent._determine_risk_level(0.0)
        self.assertEqual(level, RiskLevel.CRITICAL)


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
