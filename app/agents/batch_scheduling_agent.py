# app/agents/batch_scheduling_agent.py
# BatchSchedulingAgent：批量审核调度专项 Agent
# 职责：
#   1. 接收批量审核任务（多个文档/票据），创建 batch_tasks 主记录
#   2. 使用 asyncio.Semaphore(concurrency) 控制并发上限（默认10，最大100）
#   3. 为每个子任务创建独立的 AgentContext 并调用 OrchestratorAgent
#   4. 子任务失败不阻断其他子任务（错误隔离）
#   5. 将结果汇总写入 batch_tasks.result_summary

from __future__ import annotations

import asyncio          # asyncio.Semaphore：控制并发槽位
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.models.agent_models import (
    AuditTaskStatus,
    BatchTask,
    BatchTaskItem,
    BatchTaskStatus,
)


# ── 并发度限制（Semaphore 上限）──────────────────────────────────────────────
MAX_CONCURRENCY = 100    # 系统允许的最大并发槽位（防止过载）
DEFAULT_CONCURRENCY = 10  # 默认并发度（平衡吞吐量和稳定性）

# ── 整体失败率阈值（子任务失败率超过此值，批量任务整体标记为 FAILED）──────────
BATCH_FAILURE_THRESHOLD = 0.5  # 50% 子任务失败时整批标记失败


class BatchSchedulingAgent(BaseAgent):
    """
    批量调度 Agent：通过 asyncio.Semaphore 控制并发，安全分发大批量审核任务
    设计原则：子任务完全隔离（单个失败不影响其他），立即返回 batch_id（异步执行）
    """

    agent_name = "batch_scheduling_agent"

    def __init__(
        self,
        subtask_runner: Optional[Callable] = None,   # 子任务执行函数（可注入，便于测试）
    ):
        # subtask_runner：执行单个审核子任务的函数；None 时使用内置 OrchestratorAgent
        self._subtask_runner = subtask_runner or self._default_subtask_runner

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        items: Optional[List[Dict[str, Any]]] = None,   # 子任务列表，每项包含 document_id 等
        concurrency: int = DEFAULT_CONCURRENCY,           # 并发槽位数
        task_type: str = "FULL_AUDIT",                    # 批量任务的业务类型
        **kwargs
    ) -> AgentResult:
        """
        批量调度主逻辑

        Args:
            ctx:         上下文（audit_task_id 作为父任务 ID）
            db:          数据库会话
            items:       子任务列表（[{document_id: str, ...}, ...]）
            concurrency: Semaphore 并发槽位数（限制同时执行的子任务数）
            task_type:   批量任务的业务类型

        Returns:
            AgentResult: data 包含 batch_task_id / total / completed / failed
        """
        items = items or []
        # 并发度不能超过系统上限
        concurrency = min(concurrency, MAX_CONCURRENCY)

        # 步骤 1：创建 BatchTask 主记录
        batch_task_id = str(uuid.uuid4())
        batch_task = BatchTask(
            id=batch_task_id,
            tenant_id=ctx.tenant_id,
            task_type=task_type,
            total_count=len(items),
            completed_count=0,
            failed_count=0,
            concurrency=concurrency,
            status=BatchTaskStatus.RUNNING,
            submitted_by=ctx.audit_task_id,   # 用父任务 ID 标记提交来源
            started_at=datetime.now(timezone.utc),
        )
        db.add(batch_task)

        # 步骤 2：创建所有子任务记录（批量写入）
        item_records: List[BatchTaskItem] = []
        for idx, item in enumerate(items):
            item_record = BatchTaskItem(
                id=str(uuid.uuid4()),
                batch_task_id=batch_task_id,
                document_id=item.get("document_id"),
                item_index=idx,
                status=AuditTaskStatus.PENDING,
            )
            db.add(item_record)
            item_records.append(item_record)

        # 步骤 3：创建 Semaphore，并发执行所有子任务
        semaphore = asyncio.Semaphore(concurrency)
        logger.info(
            f"[{self.agent_name}] 开始批量执行 "
            f"total={len(items)} concurrency={concurrency} "
            f"batch_id={batch_task_id[:8]} task={ctx.audit_task_id}"
        )

        async def run_one_item(item: dict, item_record: BatchTaskItem, index: int):
            """
            单个子任务执行函数（在 Semaphore 保护下并发运行）
            Semaphore 确保同时执行的子任务数不超过 concurrency
            """
            async with semaphore:   # 获取 Semaphore 槽位（超过上限时自动等待）
                try:
                    item_record.status = AuditTaskStatus.RUNNING
                    # 调用子任务执行函数（默认为 OrchestratorAgent）
                    success, audit_task_id = await self._subtask_runner(
                        ctx=ctx,
                        db=db,
                        item=item,
                        index=index,
                        task_type=task_type,
                    )
                    item_record.audit_task_id = audit_task_id
                    item_record.status = AuditTaskStatus.COMPLETED if success else AuditTaskStatus.FAILED
                    return success
                except Exception as e:
                    # 子任务异常：记录错误，标记失败，不影响其他子任务
                    logger.warning(
                        f"[{self.agent_name}] 子任务 {index} 异常: {e} "
                        f"batch={batch_task_id[:8]}"
                    )
                    item_record.status = AuditTaskStatus.FAILED
                    item_record.error_msg = str(e)[:500]   # 截断过长的错误信息
                    return False

        # asyncio.gather：并发执行所有子任务（Semaphore 内部控制并发数）
        task_results = await asyncio.gather(
            *[run_one_item(item, record, idx) for idx, (item, record) in enumerate(zip(items, item_records))],
            return_exceptions=False,  # 子任务已捕获异常，此处不需要 return_exceptions
        )

        # 步骤 4：统计结果
        completed_count = sum(1 for r in task_results if r)
        failed_count    = sum(1 for r in task_results if not r)
        failure_rate    = failed_count / len(items) if items else 0.0

        # 步骤 5：更新批量任务状态
        batch_task.completed_count = completed_count
        batch_task.failed_count    = failed_count
        batch_task.completed_at    = datetime.now(timezone.utc)
        batch_task.status = (
            BatchTaskStatus.FAILED    if failure_rate >= BATCH_FAILURE_THRESHOLD else
            BatchTaskStatus.COMPLETED
        )
        batch_task.result_summary = {
            "total":      len(items),
            "completed":  completed_count,
            "failed":     failed_count,
            "failure_rate": round(failure_rate, 3),
            "concurrency_used": concurrency,
        }

        logger.info(
            f"[{self.agent_name}] 批量执行完成 "
            f"completed={completed_count}/{len(items)} "
            f"failed={failed_count} "
            f"status={batch_task.status.value} "
            f"batch_id={batch_task_id[:8]}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "batch_task_id": batch_task_id,
                "total":         len(items),
                "completed":     completed_count,
                "failed":        failed_count,
                "failure_rate":  failure_rate,
                "status":        batch_task.status.value,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 默认子任务执行器
    # ──────────────────────────────────────────────────────────────────────────

    async def _default_subtask_runner(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        item: dict,
        index: int,
        task_type: str,
    ) -> tuple:
        """
        默认子任务执行器：创建独立 AgentContext 并运行（简化实现）
        生产环境替换为完整的 OrchestratorAgent 调用链

        Returns:
            (success: bool, audit_task_id: str)
        """
        # 每个子任务使用独立的 audit_task_id（避免共享状态）
        sub_task_id = str(uuid.uuid4())

        # 创建子任务上下文（继承父任务的 tenant_id）
        sub_ctx = AgentContext(
            audit_task_id=sub_task_id,
            tenant_id=ctx.tenant_id,
            document_id=item.get("document_id"),
        )

        # 简化实现：模拟处理延迟
        await asyncio.sleep(0)   # 让出事件循环（让其他子任务有机会运行）

        # 实际场景：调用 OrchestratorAgent（此处简化为直接成功）
        return True, sub_task_id
