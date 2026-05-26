# tests/test_agents/test_m7_flow.py
# M7 模块测试：验证七步状态机转换和超时预警级别计算
# 运行方式：python -m pytest tests/test_agents/test_m7_flow.py -v

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.flow_tracking_agent import FlowTrackingAgent, TIMEOUT_THRESHOLDS_MS, TOTAL_STEPS
from app.agents.utils.flow_channel_mock import FlowChannelMock, FLOW_STEP_NODES
from app.models.agent_models import FlowTrackingStatus, FlowMessageStatus


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


def make_ctx(task_id="m7-test-001") -> AgentContext:
    ctx = AgentContext(audit_task_id=task_id, tenant_id="t-001")
    ctx.shared_data["bill_element"] = {"ticket_number": "SHBH20240115"}
    return ctx


def make_success_step_result(step_index: int, processing_ms: int = 100) -> dict:
    """创建模拟成功步骤应答"""
    return {
        "step_index":    step_index,
        "send_node":     FLOW_STEP_NODES[step_index]["send"],
        "ack_node":      FLOW_STEP_NODES[step_index]["ack"],
        "status":        "success",
        "processing_ms": processing_ms,
        "arrived_at":    "2024-01-15T10:00:00+00:00",
        "msg_content":   {"msg_id": f"TEST-S{step_index:02d}"},
        "is_anomaly":    False,
        "anomaly_reason": None,
    }


def make_error_step_result(step_index: int, status: str = "timeout") -> dict:
    """创建模拟异常步骤应答"""
    return {
        "step_index":    step_index,
        "send_node":     FLOW_STEP_NODES[step_index]["send"],
        "ack_node":      FLOW_STEP_NODES[step_index]["ack"],
        "status":        status,
        "processing_ms": 300_000,   # 超时耗时
        "arrived_at":    "2024-01-15T10:00:00+00:00",
        "msg_content":   {"error": "timeout"},
        "is_anomaly":    True,
        "anomaly_reason": f"Step {step_index} {status}",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：FlowChannelMock 适配层
# ──────────────────────────────────────────────────────────────────────────────

class TestFlowChannelMock(unittest.TestCase):
    """验证 Mock 适配层的接口格式和随机行为"""

    def test_send_step_returns_required_fields(self):
        """验证 send_step 返回包含所有必需字段"""
        channel = FlowChannelMock(failure_rate_multiplier=0)   # 0=强制成功
        result = run_async(channel.send_step(1, "SHBH001", "ACCEPTANCE_PROMPT"))

        required_fields = [
            "step_index", "send_node", "ack_node", "status",
            "processing_ms", "arrived_at", "msg_content", "is_anomaly",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"返回缺少字段: {field}")

    def test_send_step_success_when_failure_rate_zero(self):
        """验证失败率为 0 时步骤永远成功"""
        channel = FlowChannelMock(failure_rate_multiplier=0)

        for step in range(1, 8):   # 测试所有 7 个步骤
            result = run_async(channel.send_step(step, "SHBH001", "ACCEPTANCE_PROMPT"))
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["is_anomaly"])

    def test_send_step_invalid_step_raises_error(self):
        """验证无效步骤序号抛出异常"""
        channel = FlowChannelMock()
        with self.assertRaises(ValueError):
            run_async(channel.send_step(8, "SHBH001", "ACCEPTANCE_PROMPT"))   # 步骤 8 不存在

    def test_send_step_node_names_correct(self):
        """验证步骤节点名称与协议定义一致"""
        channel = FlowChannelMock(failure_rate_multiplier=0)

        for step_index in range(1, 8):
            result = run_async(channel.send_step(step_index, "SHBH001", "ACCEPTANCE_PROMPT"))
            expected_send = FLOW_STEP_NODES[step_index]["send"]
            self.assertEqual(result["send_node"], expected_send)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：FlowTrackingAgent 七步状态机
# ──────────────────────────────────────────────────────────────────────────────

class TestFlowTrackingAgent(unittest.TestCase):
    """验证七步状态机转换和整体流程"""

    def _make_always_success_channel(self, ms_per_step: int = 50) -> FlowChannelMock:
        """创建始终返回成功的通道（控制耗时）"""
        mock_channel = AsyncMock(spec=FlowChannelMock)
        # 每次调用返回对应步骤的成功结果
        async def mock_send(step_index, ticket_number, business_type, metadata=None):
            return make_success_step_result(step_index, processing_ms=ms_per_step)
        mock_channel.send_step = mock_send
        return mock_channel

    def _make_fail_at_step_channel(self, fail_step: int, fail_status: str = "timeout"):
        """创建在指定步骤失败的通道"""
        async def mock_send(step_index, ticket_number, business_type, metadata=None):
            if step_index == fail_step:
                return make_error_step_result(step_index, fail_status)
            return make_success_step_result(step_index)
        mock_channel = AsyncMock(spec=FlowChannelMock)
        mock_channel.send_step = mock_send
        return mock_channel

    def test_all_7_steps_complete_successfully(self):
        """验证七步全部成功时状态为 COMPLETED"""
        channel = self._make_always_success_channel()
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertEqual(result.data["completed_steps"], 7)
        self.assertEqual(result.data["total_steps"], 7)
        self.assertTrue(result.data["is_completed"])
        self.assertEqual(result.data["status"], FlowTrackingStatus.COMPLETED.value)

    def test_timeout_at_step_3_stops_flow(self):
        """验证第 3 步超时后流转终止，已完成 2 步"""
        channel = self._make_fail_at_step_channel(fail_step=3, fail_status="timeout")
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx(task_id="m7-timeout-003")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertEqual(result.data["completed_steps"], 2)   # 1、2步成功
        self.assertFalse(result.data["is_completed"])
        self.assertEqual(result.data["status"], FlowTrackingStatus.TIMEOUT.value)

    def test_error_at_step_5_stops_flow(self):
        """验证第 5 步报错后流转终止，已完成 4 步"""
        channel = self._make_fail_at_step_channel(fail_step=5, fail_status="error")
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx(task_id="m7-error-005")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        self.assertTrue(result.success)
        self.assertEqual(result.data["completed_steps"], 4)
        self.assertEqual(result.data["status"], FlowTrackingStatus.ERROR.value)

    def test_flow_tracking_task_written_to_db(self):
        """验证 FlowTrackingTask 主记录写入数据库"""
        channel = self._make_always_success_channel()
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db))

        # db.add 应该被调用 1次（主记录）+ 7次（步骤记录）= 8 次
        # 主记录（FlowTrackingTask）+ 7 个 FlowMessage
        self.assertGreaterEqual(mock_db.add.call_count, 8)

    def test_flow_messages_written_for_each_step(self):
        """验证每个步骤都写入了 FlowMessage 记录"""
        channel = self._make_always_success_channel()
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db))

        from app.models.agent_models import FlowMessage
        # 检查有 7 条 FlowMessage 记录
        message_count = sum(
            1 for call in mock_db.add.call_args_list
            if isinstance(call[0][0], FlowMessage)
        )
        self.assertEqual(message_count, 7)

    def test_result_written_to_shared_data(self):
        """验证流转结果写入 shared_data["flow_result"]"""
        channel = self._make_always_success_channel()
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db))

        self.assertIn("flow_result", ctx.shared_data)
        fr = ctx.shared_data["flow_result"]
        self.assertIn("flow_task_id", fr)
        self.assertIn("completed_steps", fr)
        self.assertIn("status", fr)
        self.assertIn("timeout_level", fr)

    def test_channel_exception_does_not_crash(self):
        """验证通道接口抛出异常时 Agent 不崩溃"""
        mock_channel = AsyncMock(spec=FlowChannelMock)
        mock_channel.send_step = AsyncMock(side_effect=ConnectionError("通道不可用"))
        agent = FlowTrackingAgent(channel=mock_channel)
        ctx = make_ctx(task_id="m7-channel-error")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        # Agent 应该成功返回（虽然流转异常终止）
        self.assertTrue(result.success)
        self.assertEqual(result.data["completed_steps"], 0)
        self.assertEqual(result.data["status"], FlowTrackingStatus.ERROR.value)

    def test_ticket_number_from_shared_data(self):
        """验证票据号码从 shared_data["bill_element"] 读取"""
        channel = self._make_always_success_channel()
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx()
        ctx.shared_data["bill_element"]["ticket_number"] = "CUSTOM-TICKET-001"
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db))

        # 验证 FlowTrackingTask 使用了正确的票据号码
        from app.models.agent_models import FlowTrackingTask
        flow_task = None
        for call in mock_db.add.call_args_list:
            if isinstance(call[0][0], FlowTrackingTask):
                flow_task = call[0][0]
                break
        self.assertIsNotNone(flow_task)
        self.assertEqual(flow_task.ticket_number, "CUSTOM-TICKET-001")

    def test_error_step_message_has_anomaly_flag(self):
        """验证异常步骤的 FlowMessage 记录 is_anomaly=True"""
        channel = self._make_fail_at_step_channel(fail_step=2, fail_status="timeout")
        agent = FlowTrackingAgent(channel=channel)
        ctx = make_ctx(task_id="m7-anomaly-002")
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db))

        from app.models.agent_models import FlowMessage
        anomaly_messages = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], FlowMessage) and call[0][0].is_anomaly
        ]
        self.assertGreater(len(anomaly_messages), 0)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3：超时预警级别计算
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeoutLevelCalculation(unittest.TestCase):
    """验证超时预警级别的五档阈值计算"""

    def setUp(self):
        # 使用简单 Mock 通道，只测试超时计算逻辑
        self.agent = FlowTrackingAgent(channel=None)

    def test_normal_level_below_30s(self):
        """验证 0~29999ms 对应 NORMAL 级别"""
        level = self.agent._calc_timeout_level(0)
        self.assertEqual(level, "NORMAL")

        level = self.agent._calc_timeout_level(29_999)
        self.assertEqual(level, "NORMAL")

    def test_watch_level_30s_to_60s(self):
        """验证 30000~59999ms 对应 WATCH 级别"""
        level = self.agent._calc_timeout_level(30_000)
        self.assertEqual(level, "WATCH")

        level = self.agent._calc_timeout_level(59_999)
        self.assertEqual(level, "WATCH")

    def test_warning_level_60s_to_120s(self):
        """验证 60000~119999ms 对应 WARNING 级别"""
        level = self.agent._calc_timeout_level(60_000)
        self.assertEqual(level, "WARNING")

        level = self.agent._calc_timeout_level(119_999)
        self.assertEqual(level, "WARNING")

    def test_urgent_level_120s_to_300s(self):
        """验证 120000~299999ms 对应 URGENT 级别"""
        level = self.agent._calc_timeout_level(120_000)
        self.assertEqual(level, "URGENT")

        level = self.agent._calc_timeout_level(299_999)
        self.assertEqual(level, "URGENT")

    def test_overdue_level_above_300s(self):
        """验证 >= 300000ms 对应 OVERDUE 级别"""
        level = self.agent._calc_timeout_level(300_000)
        self.assertEqual(level, "OVERDUE")

        level = self.agent._calc_timeout_level(1_000_000)
        self.assertEqual(level, "OVERDUE")

    def test_timeout_level_integration_slow_steps(self):
        """验证耗时较长的完整流转会触发更高预警级别"""
        # 构造高耗时步骤：每步 60秒，7步共 420秒 > OVERDUE 阈值
        async def slow_send(step_index, ticket_number, business_type, metadata=None):
            return make_success_step_result(step_index, processing_ms=60_000)

        mock_channel = AsyncMock(spec=FlowChannelMock)
        mock_channel.send_step = slow_send
        agent = FlowTrackingAgent(channel=mock_channel)
        ctx = make_ctx(task_id="m7-slow-001")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db))

        # 7步 × 60000ms = 420000ms > OVERDUE 阈值
        self.assertEqual(result.data["timeout_level"], "OVERDUE")


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
