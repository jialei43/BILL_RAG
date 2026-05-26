# tests/test_agents/test_m8_fraud.py
# M8 模块测试：验证五维并行欺诈检测和重复票据强制置 1.0 逻辑
# 运行方式：python -m pytest tests/test_agents/test_m8_fraud.py -v

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.fraud_detection_agent import (
    FraudDetectionAgent,
    FRAUD_DIMENSION_WEIGHTS,
    DIMENSION_TIMEOUT_SECONDS,
)
from app.models.agent_models import FraudDetection


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


def make_ctx(
    ticket_number="SHBH20240115",
    drawer="测试公司",
    payee="收款方公司",
    confidence_score=0.95,
    task_id="m8-test-001",
) -> AgentContext:
    """创建包含 bill_element 和 endorsement_result 的上下文"""
    ctx = AgentContext(audit_task_id=task_id, tenant_id="t-001")
    ctx.shared_data["bill_element"] = {
        "ticket_number":  ticket_number,
        "ticket_type":    "银行承兑汇票",
        "amount_numeric": 1000000.0,
        "drawer":         drawer,
        "payee":          payee,
        "confidence_score": confidence_score,
    }
    ctx.shared_data["endorsement_result"] = {
        "is_continuous":   True,
        "has_cycle":       False,
        "violation_count": 0,
        "violation_codes": [],
    }
    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：五维并行执行验证
# ──────────────────────────────────────────────────────────────────────────────

class TestFraudDetectionParallel(unittest.TestCase):
    """验证五维检测并行执行和结果汇总"""

    def setUp(self):
        self.agent = FraudDetectionAgent()

    def test_all_5_dimensions_executed(self):
        """验证五维检测全部执行（通过验证返回结果中所有维度得分）"""
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        # 检查五维得分都在结果中
        self.assertIn("seal_score",      result.data)
        self.assertIn("duplicate_score", result.data)
        self.assertIn("tamper_score",    result.data)
        self.assertIn("network_score",   result.data)
        self.assertIn("blacklist_score", result.data)

    def test_all_dimensions_scores_between_0_and_1(self):
        """验证所有维度得分在 0~1 范围内"""
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        for dim in ["seal_score", "duplicate_score", "tamper_score", "network_score", "blacklist_score"]:
            score = result.data[dim]
            self.assertGreaterEqual(score, 0.0, f"{dim} 得分低于 0")
            self.assertLessEqual(score, 1.0, f"{dim} 得分超过 1.0")

    def test_overall_score_in_valid_range(self):
        """验证综合欺诈评分在 0~1 范围内"""
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertGreaterEqual(result.data["overall_fraud_score"], 0.0)
        self.assertLessEqual(result.data["overall_fraud_score"], 1.0)

    def test_weight_sum_equals_1(self):
        """验证五维权重之和为 1.0"""
        total = sum(FRAUD_DIMENSION_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_clean_ticket_low_fraud_score(self):
        """验证正常票据（高置信度，无异常）的欺诈分接近 0"""
        ctx = make_ctx(confidence_score=0.98)   # 高置信度=印章可信
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        # 正常票据综合欺诈分应该很低
        self.assertLess(result.data["overall_fraud_score"], 0.3)
        self.assertEqual(result.data["fraud_level"], "clean")

    def test_fraud_detection_written_to_db(self):
        """验证 FraudDetection 记录写入数据库"""
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        mock_db.add.assert_called_once()
        added = mock_db.add.call_args[0][0]
        self.assertIsInstance(added, FraudDetection)
        self.assertEqual(added.audit_task_id, "m8-test-001")

    def test_result_written_to_shared_data(self):
        """验证检测结果写入 shared_data["fraud_result"]"""
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        self.assertIn("fraud_result", ctx.shared_data)
        fr = ctx.shared_data["fraud_result"]
        self.assertIn("overall_fraud_score", fr)
        self.assertIn("fraud_level", fr)
        self.assertIn("seal_score", fr)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：重复票据强制置 1.0
# ──────────────────────────────────────────────────────────────────────────────

class TestDuplicateTicketRule(unittest.TestCase):
    """验证重复票据的硬性规则：命中时综合分强制为 1.0"""

    def setUp(self):
        self.agent = FraudDetectionAgent()

    def test_duplicate_ticket_forces_score_1(self):
        """验证重复票据（DUP- 前缀）命中时综合欺诈分强制为 1.0"""
        ctx = make_ctx(ticket_number="DUP-SHBH20240115", task_id="m8-dup-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertEqual(result.data["overall_fraud_score"], 1.0)   # 强制 1.0
        self.assertEqual(result.data["duplicate_score"], 1.0)
        self.assertEqual(result.data["fraud_level"], "fraud")

    def test_duplicate_check_result_contains_detail(self):
        """验证重复票据命中时检测详情包含历史任务 ID"""
        ctx = make_ctx(ticket_number="DUP-SHBH20240115", task_id="m8-dup-detail")
        mock_db = make_mock_db()

        run_async(self.agent.run(ctx, mock_db))

        added = mock_db.add.call_args[0][0]
        dup_result = added.duplicate_check_result
        self.assertTrue(dup_result["is_duplicate"])
        self.assertIsNotNone(dup_result["duplicate_task_id"])

    def test_non_duplicate_has_zero_duplicate_score(self):
        """验证正常票据的重复得分为 0"""
        ctx = make_ctx(ticket_number="NORMAL-SHBH20240115", task_id="m8-normal-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertEqual(result.data["duplicate_score"], 0.0)

    def test_duplicate_overrides_other_clean_dimensions(self):
        """验证重复票据命中后，即使其他维度全部 0 分，综合分仍为 1.0"""
        # 高置信度（印章无问题）+ 无背书违规 + 但票据是重复的
        ctx = make_ctx(
            ticket_number="DUP-OVERRIDE-001",
            confidence_score=0.99,   # 印章高可信
            task_id="m8-dup-override",
        )
        ctx.shared_data["endorsement_result"] = {
            "is_continuous": True, "has_cycle": False, "violation_count": 0
        }
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        # 无论其他维度多么干净，重复票据必须是 1.0
        self.assertEqual(result.data["overall_fraud_score"], 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3：各维度检测逻辑
# ──────────────────────────────────────────────────────────────────────────────

class TestFraudDimensions(unittest.TestCase):
    """验证各维度的具体检测逻辑"""

    def setUp(self):
        self.agent = FraudDetectionAgent()

    def test_seal_check_low_confidence_raises_score(self):
        """验证低置信度票据（疑似印章伪造）得分较高"""
        ctx = make_ctx(confidence_score=0.60, task_id="m8-seal-001")   # 低置信度
        mock_db = make_mock_db()

        # 绕过缓存，强制执行真实检测（相同票据指纹可能命中前一测试缓存）
        with patch("app.agents.fraud_detection_agent.bill_cache.get_fraud", AsyncMock(return_value=None)):
            result = run_async(self.agent.run(ctx, mock_db))

        # 低置信度时印章得分应 > 高置信度
        self.assertGreater(result.data["seal_score"], 0.1)

    def test_network_check_cycle_high_score(self):
        """验证背书链存在闭环时网络维度得分高"""
        ctx = make_ctx(task_id="m8-network-001")
        ctx.shared_data["endorsement_result"] = {
            "is_continuous": True,
            "has_cycle":     True,   # 存在闭环
            "violation_count": 2,
            "violation_codes": ["EN03"],
        }
        mock_db = make_mock_db()

        # 绕过缓存，确保使用当前 endorsement_result（不同 endorsement_result 但相同 bill_fp）
        with patch("app.agents.fraud_detection_agent.bill_cache.get_fraud", AsyncMock(return_value=None)):
            result = run_async(self.agent.run(ctx, mock_db))

        # 闭环时网络维度得分应 = 0.80
        self.assertAlmostEqual(result.data["network_score"], 0.80, places=2)

    def test_blacklist_check_hit_entities(self):
        """验证出票人在黑名单中时黑名单维度得分 > 0"""
        ctx = make_ctx(drawer="BLACKLIST-COMPANY", task_id="m8-blacklist-001")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertGreater(result.data["blacklist_score"], 0.0)

    def test_blacklist_no_hit_score_zero(self):
        """验证正常主体黑名单得分为 0"""
        ctx = make_ctx(drawer="正常测试公司", payee="正常收款公司", task_id="m8-bl-nohit")
        mock_db = make_mock_db()

        result = run_async(self.agent.run(ctx, mock_db))

        self.assertEqual(result.data["blacklist_score"], 0.0)

    def test_fraud_level_clean(self):
        """验证综合欺诈分 < 0.3 时等级为 clean"""
        level = self.agent._determine_fraud_level(0.29)
        self.assertEqual(level, "clean")

    def test_fraud_level_suspicious(self):
        """验证综合欺诈分 0.3~0.6 时等级为 suspicious"""
        level_low  = self.agent._determine_fraud_level(0.30)
        level_high = self.agent._determine_fraud_level(0.59)
        self.assertEqual(level_low,  "suspicious")
        self.assertEqual(level_high, "suspicious")

    def test_fraud_level_high_risk(self):
        """验证综合欺诈分 0.6~0.8 时等级为 high_risk"""
        level = self.agent._determine_fraud_level(0.60)
        self.assertEqual(level, "high_risk")
        level2 = self.agent._determine_fraud_level(0.79)
        self.assertEqual(level2, "high_risk")

    def test_fraud_level_fraud(self):
        """验证综合欺诈分 >= 0.8 时等级为 fraud"""
        level = self.agent._determine_fraud_level(0.80)
        self.assertEqual(level, "fraud")
        level2 = self.agent._determine_fraud_level(1.0)
        self.assertEqual(level2, "fraud")


# ──────────────────────────────────────────────────────────────────────────────
# 测试 4：超时和异常容错
# ──────────────────────────────────────────────────────────────────────────────

class TestFraudTimeoutAndResilience(unittest.TestCase):
    """验证单维度超时时其他维度不受影响"""

    def setUp(self):
        self.agent = FraudDetectionAgent()

    def test_single_dimension_timeout_does_not_crash(self):
        """验证单个维度超时（得分归零）时整体 Agent 不崩溃"""
        async def slow_seal(bill_element, task_id):
            await asyncio.sleep(10)   # 10秒 > 2秒超时阈值
            return 0.5, {}

        ctx = make_ctx(task_id="m8-timeout-001")
        mock_db = make_mock_db()

        # 绕过缓存，确保 _check_seal 的 patch 能实际生效
        with patch("app.agents.fraud_detection_agent.bill_cache.get_fraud", AsyncMock(return_value=None)), \
             patch.object(self.agent, "_check_seal", side_effect=slow_seal):
            result = run_async(self.agent.run(ctx, mock_db))

        # 整体不崩溃，印章维度因超时归零
        self.assertTrue(result.success)
        self.assertEqual(result.data["seal_score"], 0.0)   # 超时→归零

    def test_exception_in_dimension_does_not_crash(self):
        """验证单个维度抛出异常时整体 Agent 不崩溃"""
        async def failing_blacklist(bill_element, task_id):
            raise RuntimeError("黑名单数据库连接失败")

        ctx = make_ctx(task_id="m8-except-001")
        mock_db = make_mock_db()

        with patch.object(self.agent, "_check_blacklist", side_effect=failing_blacklist):
            result = run_async(self.agent.run(ctx, mock_db))

        # 整体不崩溃，黑名单维度归零
        self.assertTrue(result.success)
        self.assertEqual(result.data["blacklist_score"], 0.0)

    def test_all_dimensions_timeout_returns_zero_score(self):
        """验证所有维度超时时综合欺诈分为 0（无欺诈嫌疑，保守处理）"""
        async def always_timeout(*args, **kwargs):
            await asyncio.sleep(10)   # 超过所有超时阈值
            return 0.5, {}

        ctx = make_ctx(task_id="m8-all-timeout")
        mock_db = make_mock_db()

        # 绕过缓存，确保所有维度的 patch 能实际生效
        with patch("app.agents.fraud_detection_agent.bill_cache.get_fraud", AsyncMock(return_value=None)), \
             patch.object(self.agent, "_check_seal",      side_effect=always_timeout), \
             patch.object(self.agent, "_check_duplicate", side_effect=always_timeout), \
             patch.object(self.agent, "_check_tamper",    side_effect=always_timeout), \
             patch.object(self.agent, "_check_network",   side_effect=always_timeout), \
             patch.object(self.agent, "_check_blacklist", side_effect=always_timeout):
            result = run_async(self.agent.run(ctx, mock_db))

        # 所有维度超时 → 综合分为 0 → 等级为 clean（保守处理）
        self.assertTrue(result.success)
        self.assertEqual(result.data["overall_fraud_score"], 0.0)
        self.assertEqual(result.data["fraud_level"], "clean")


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
