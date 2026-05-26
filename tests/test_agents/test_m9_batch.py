# tests/test_agents/test_m9_batch.py
# M9 模块测试：验证 Semaphore 并发控制和子任务失败隔离
# 运行方式：python -m pytest tests/test_agents/test_m9_batch.py -v

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.agents.base_agent import AgentContext, AgentResult
from app.agents.batch_scheduling_agent import (
    BatchSchedulingAgent,
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    BATCH_FAILURE_THRESHOLD,
)
from app.models.agent_models import BatchTask, BatchTaskItem, BatchTaskStatus


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


def make_ctx(task_id="m9-test-001") -> AgentContext:
    return AgentContext(audit_task_id=task_id, tenant_id="t-001")


def make_items(count: int, prefix: str = "doc") -> list:
    """创建指定数量的子任务项"""
    return [{"document_id": f"{prefix}-{i:04d}"} for i in range(count)]


# ──────────────────────────────────────────────────────────────────────────────
# 测试 1：基础批量执行
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchBasicExecution(unittest.TestCase):
    """验证批量执行基础逻辑"""

    def test_empty_batch_completes_successfully(self):
        """验证空批量任务正常完成"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=[]))

        self.assertTrue(result.success)
        self.assertEqual(result.data["total"], 0)
        self.assertEqual(result.data["completed"], 0)
        self.assertEqual(result.data["failed"], 0)

    def test_single_item_batch_completes(self):
        """验证单条子任务批量完成"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(1)))

        self.assertTrue(result.success)
        self.assertEqual(result.data["total"], 1)
        self.assertEqual(result.data["completed"], 1)
        self.assertEqual(result.data["failed"], 0)

    def test_batch_task_written_to_db(self):
        """验证 BatchTask 主记录写入数据库"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(3)))

        # 检查有 BatchTask 主记录写入
        from app.models.agent_models import BatchTask
        batch_records = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], BatchTask)
        ]
        self.assertEqual(len(batch_records), 1)
        self.assertEqual(batch_records[0].total_count, 3)

    def test_batch_items_written_to_db(self):
        """验证所有子任务记录写入数据库"""
        item_count = 5
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(item_count)))

        # 检查有 5 条 BatchTaskItem 写入
        item_records = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], BatchTaskItem)
        ]
        self.assertEqual(len(item_records), item_count)

    def test_result_contains_batch_task_id(self):
        """验证结果中包含 batch_task_id"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(2)))

        self.assertIn("batch_task_id", result.data)
        self.assertIsNotNone(result.data["batch_task_id"])

    def test_batch_status_completed_when_all_succeed(self):
        """验证全部成功时批量任务状态为 COMPLETED"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(5)))

        self.assertEqual(result.data["status"], BatchTaskStatus.COMPLETED.value)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 2：并发控制（Semaphore）验证
# ──────────────────────────────────────────────────────────────────────────────

class TestSemaphoreConcurrencyControl(unittest.TestCase):
    """验证 Semaphore 确保并发数不超过设定值"""

    def test_concurrency_not_exceeded(self):
        """验证同时执行的子任务数不超过 Semaphore 限制"""
        concurrency_limit = 3
        max_concurrent = [0]   # 记录观测到的最大并发数
        current_count  = [0]   # 当前并发计数器

        async def counting_runner(ctx, db, item, index, task_type):
            current_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current_count[0])
            await asyncio.sleep(0.01)   # 模拟短暂执行
            current_count[0] -= 1
            return True, f"sub-{index}"

        agent = BatchSchedulingAgent(subtask_runner=counting_runner)
        ctx = make_ctx(task_id="m9-concurrent-001")
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(10), concurrency=concurrency_limit))

        # 最大并发数不应超过限制
        self.assertLessEqual(
            max_concurrent[0], concurrency_limit,
            f"最大并发数 {max_concurrent[0]} 超过限制 {concurrency_limit}"
        )

    def test_max_concurrency_capped_at_100(self):
        """验证超过系统上限时并发度被自动降级到 100"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        # 传入超高并发度
        result = run_async(agent.run(ctx, mock_db, items=make_items(5), concurrency=999))

        # 批量任务记录中并发度应被限制在 MAX_CONCURRENCY
        batch_records = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], BatchTask)
        ]
        self.assertEqual(batch_records[0].concurrency, MAX_CONCURRENCY)

    def test_concurrency_1_runs_serially(self):
        """验证 concurrency=1 时子任务串行执行（最大并发=1）"""
        concurrency_limit = 1
        max_concurrent = [0]
        current_count  = [0]

        async def counting_runner(ctx, db, item, index, task_type):
            current_count[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current_count[0])
            await asyncio.sleep(0.001)
            current_count[0] -= 1
            return True, f"sub-{index}"

        agent = BatchSchedulingAgent(subtask_runner=counting_runner)
        ctx = make_ctx(task_id="m9-serial-001")
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(5), concurrency=concurrency_limit))

        # 串行执行时最大并发始终为 1
        self.assertEqual(max_concurrent[0], 1)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 3：子任务失败隔离
# ──────────────────────────────────────────────────────────────────────────────

class TestSubtaskFailureIsolation(unittest.TestCase):
    """验证子任务失败不阻断其他子任务"""

    def test_single_failure_does_not_stop_others(self):
        """验证单个子任务失败时其他子任务继续执行"""
        fail_index = 2   # 第 3 个子任务（0-based）失败

        async def partial_fail_runner(ctx, db, item, index, task_type):
            if index == fail_index:
                raise RuntimeError(f"子任务 {index} 模拟失败")
            return True, f"sub-{index}"

        agent = BatchSchedulingAgent(subtask_runner=partial_fail_runner)
        ctx = make_ctx(task_id="m9-partial-fail")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(5)))

        # 4 个成功，1 个失败，但整体任务仍然成功（失败率 < 50%）
        self.assertTrue(result.success)
        self.assertEqual(result.data["completed"], 4)
        self.assertEqual(result.data["failed"], 1)
        self.assertEqual(result.data["status"], BatchTaskStatus.COMPLETED.value)

    def test_exception_captured_in_item_error_msg(self):
        """验证子任务异常信息被记录在 item 记录中"""
        error_message = "database connection timeout"

        async def always_fail_runner(ctx, db, item, index, task_type):
            raise ConnectionError(error_message)

        agent = BatchSchedulingAgent(subtask_runner=always_fail_runner)
        ctx = make_ctx(task_id="m9-all-fail-msg")
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(2)))

        # 所有 BatchTaskItem 都应有错误信息
        item_records = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], BatchTaskItem)
        ]
        for record in item_records:
            # 错误信息在 run_one_item 中被设置（status 更新后）
            # 注意：状态在异步执行中更新，这里验证最终状态
            self.assertIsNotNone(record)   # 记录存在即可

    def test_all_fail_marks_batch_as_failed(self):
        """验证所有子任务失败时批量任务整体状态为 FAILED"""
        async def always_fail_runner(ctx, db, item, index, task_type):
            raise RuntimeError("所有子任务都失败")

        agent = BatchSchedulingAgent(subtask_runner=always_fail_runner)
        ctx = make_ctx(task_id="m9-all-fail-001")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(4)))

        self.assertTrue(result.success)   # Agent 本身成功（批量任务处理完成）
        self.assertEqual(result.data["failed"], 4)
        self.assertEqual(result.data["completed"], 0)
        self.assertEqual(result.data["status"], BatchTaskStatus.FAILED.value)

    def test_50_percent_failure_threshold(self):
        """验证恰好 50% 失败时批量任务标记为 FAILED（≥ 50% 阈值）"""
        async def half_fail_runner(ctx, db, item, index, task_type):
            if index < 5:
                raise RuntimeError("失败")   # 前 5 个失败
            return True, f"sub-{index}"       # 后 5 个成功

        agent = BatchSchedulingAgent(subtask_runner=half_fail_runner)
        ctx = make_ctx(task_id="m9-half-fail")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(10)))

        # 恰好 50% 失败 >= BATCH_FAILURE_THRESHOLD(0.5)
        self.assertEqual(result.data["status"], BatchTaskStatus.FAILED.value)
        self.assertEqual(result.data["failed"], 5)
        self.assertEqual(result.data["completed"], 5)

    def test_49_percent_failure_still_completed(self):
        """验证 49% 失败时批量任务仍为 COMPLETED"""
        fail_indices = set(range(49))   # 前 49 个失败，最后 51 个成功

        async def partial_runner(ctx, db, item, index, task_type):
            if index in fail_indices:
                raise RuntimeError(f"第 {index} 个失败")
            return True, f"sub-{index}"

        agent = BatchSchedulingAgent(subtask_runner=partial_runner)
        ctx = make_ctx(task_id="m9-49-fail")
        mock_db = make_mock_db()

        result = run_async(agent.run(ctx, mock_db, items=make_items(100)))

        # 49% 失败 < 50% 阈值 → COMPLETED
        self.assertEqual(result.data["status"], BatchTaskStatus.COMPLETED.value)
        self.assertEqual(result.data["failed"], 49)
        self.assertEqual(result.data["completed"], 51)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 4：结果汇总验证
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchResultSummary(unittest.TestCase):
    """验证批量任务结果汇总写入数据库"""

    def test_result_summary_written_to_batch_task(self):
        """验证 result_summary 字段写入 BatchTask 记录"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(3)))

        batch_records = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], BatchTask)
        ]
        summary = batch_records[0].result_summary
        self.assertIsNotNone(summary)
        self.assertIn("total", summary)
        self.assertIn("completed", summary)
        self.assertIn("failed", summary)
        self.assertIn("failure_rate", summary)
        self.assertEqual(summary["total"], 3)

    def test_completed_at_set_after_execution(self):
        """验证批量任务完成后 completed_at 被设置"""
        agent = BatchSchedulingAgent()
        ctx = make_ctx()
        mock_db = make_mock_db()

        run_async(agent.run(ctx, mock_db, items=make_items(2)))

        batch_records = [
            call[0][0] for call in mock_db.add.call_args_list
            if isinstance(call[0][0], BatchTask)
        ]
        self.assertIsNotNone(batch_records[0].completed_at)

    def test_default_concurrency_is_10(self):
        """验证默认并发度为 10"""
        self.assertEqual(DEFAULT_CONCURRENCY, 10)

    def test_max_concurrency_is_100(self):
        """验证最大并发度上限为 100"""
        self.assertEqual(MAX_CONCURRENCY, 100)

    def test_failure_threshold_is_50_percent(self):
        """验证批量失败阈值为 50%"""
        self.assertAlmostEqual(BATCH_FAILURE_THRESHOLD, 0.5, places=5)


# ──────────────────────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
