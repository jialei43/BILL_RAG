# app/agents/flow_tracking_agent.py
# FlowTrackingAgent：票据业务七步报文流转追踪专项 Agent
# 职责：
#   1. 实现七步报文流转状态机（发起行→前置机→票交所→对手行→...→业务完成）
#   2. 每步调用 Mock 适配层（生产替换为真实接口）获取报文状态
#   3. 计算超时预警级别（NORMAL/WATCH/WARNING/URGENT/OVERDUE）
#   4. 将追踪主记录写入 flow_tracking_tasks，14 个节点状态写入 flow_messages
#   5. 生成自然语言摘要（描述当前流转状态）

from __future__ import annotations

import asyncio          # asyncio.gather：并行执行各步骤（但流转步骤有顺序依赖，此处串行）
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.agents.utils.flow_channel_mock import FlowChannelMock, FLOW_STEP_NODES
from app.models.agent_models import (
    FlowMessage,
    FlowMessageStatus,
    FlowTrackingStatus,
    FlowTrackingTask,
)


# ── 超时预警阈值（毫秒）─────────────────────────────────────────────────────
TIMEOUT_THRESHOLDS_MS = {
    "NORMAL":  0,       # 0~30秒：正常范围
    "WATCH":   30_000,  # 30~60秒：开始关注
    "WARNING": 60_000,  # 60~120秒：发出警告
    "URGENT":  120_000, # 120~300秒：紧急处理
    "OVERDUE": 300_000, # >300秒：已超期
}

# ── 七步流转协议的总步骤数（固定值）─────────────────────────────────────────
TOTAL_STEPS = 7


class FlowTrackingAgent(BaseAgent):
    """
    流转追踪 Agent：通过驱动七步状态机，记录票据业务报文的完整流转过程
    采用适配器模式隔离外部接口依赖，可无缝切换 Mock/真实实现
    """

    agent_name = "flow_tracking_agent"

    def __init__(self, channel: Optional[FlowChannelMock] = None):
        # channel：报文通道适配器；None 时自动创建 Mock 实例（开发/测试环境）
        self._channel = channel or FlowChannelMock()

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        business_type: str = "ACCEPTANCE_PROMPT",   # 默认业务类型：提示承兑
        ticket_number: Optional[str] = None,         # 被追踪的票据号码
        metadata: Optional[dict] = None,             # 附加元数据（金额/期限等）
        **kwargs
    ) -> AgentResult:
        """
        流转追踪主逻辑：驱动七步状态机并记录所有节点状态

        Args:
            ctx:           上下文
            db:            数据库会话
            business_type: 业务类型（决定报文流转路径）
            ticket_number: 票据号码（从 shared_data 读取或直接传入）
            metadata:      附加报文元数据

        Returns:
            AgentResult: 包含 completed_steps / status / timeout_level
        """
        # 步骤 1：获取票据号码（优先从参数取，否则从 shared_data 读取）
        if not ticket_number:
            bill_element = ctx.shared_data.get("bill_element", {})
            ticket_number = bill_element.get("ticket_number") or f"MOCK-{ctx.audit_task_id[:8]}"

        # 步骤 2：创建流转追踪主记录
        flow_task_id = str(uuid.uuid4())
        flow_task = FlowTrackingTask(
            id=flow_task_id,
            audit_task_id=ctx.audit_task_id,
            business_type=business_type,
            ticket_number=ticket_number,
            total_steps=TOTAL_STEPS,
            completed_steps=0,
            current_node=FLOW_STEP_NODES[1]["send"],   # 初始节点
            status=FlowTrackingStatus.INITIATED,
            timeout_level="NORMAL",
        )
        db.add(flow_task)

        # 步骤 3：驱动七步状态机，逐步执行并记录
        completed_steps = 0
        total_elapsed_ms = 0
        final_status = FlowTrackingStatus.COMPLETED
        messages: List[FlowMessage] = []

        for step_index in range(1, TOTAL_STEPS + 1):
            # 向适配层发送当前步骤（Mock 或真实接口）
            try:
                step_result = await self._channel.send_step(
                    step_index=step_index,
                    ticket_number=ticket_number,
                    business_type=business_type,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"[{self.agent_name}] 步骤 {step_index} 通道异常: {e}")
                # 接口异常：记录错误状态并终止流转
                step_result = {
                    "step_index":    step_index,
                    "send_node":     FLOW_STEP_NODES[step_index]["send"],
                    "ack_node":      FLOW_STEP_NODES[step_index]["ack"],
                    "status":        "error",
                    "processing_ms": 0,
                    "arrived_at":    datetime.now(timezone.utc).isoformat(),
                    "msg_content":   {"error": str(e)},
                    "is_anomaly":    True,
                    "anomaly_reason": f"通道异常: {e}",
                }

            # 确定本步骤报文状态
            msg_status, is_anomaly = self._map_step_status(step_result["status"])

            # 计算累计耗时，用于超时预警
            total_elapsed_ms += step_result.get("processing_ms", 0)

            # 创建报文节点记录（每步骤一条）
            msg = FlowMessage(
                id=str(uuid.uuid4()),
                flow_task_id=flow_task_id,
                step_index=step_index,
                node_name=step_result["send_node"],   # 记录发送节点名称
                msg_type="send",                       # 该步骤的发送方向
                msg_content=step_result.get("msg_content"),
                status=msg_status,
                processing_ms=step_result.get("processing_ms", 0),
                is_anomaly=is_anomaly,
                anomaly_reason=step_result.get("anomaly_reason"),
            )
            db.add(msg)
            messages.append(msg)

            if is_anomaly:
                # 发生异常：终止后续步骤，标记整体状态
                if step_result["status"] == "timeout":
                    final_status = FlowTrackingStatus.TIMEOUT
                else:
                    final_status = FlowTrackingStatus.ERROR
                logger.warning(
                    f"[{self.agent_name}] 步骤 {step_index} 异常终止 "
                    f"status={step_result['status']} task={ctx.audit_task_id}"
                )
                break   # 异常步骤后不再继续
            else:
                completed_steps += 1

        # 步骤 4：计算超时预警级别
        timeout_level = self._calc_timeout_level(total_elapsed_ms)

        # 步骤 5：更新流转追踪主记录
        flow_task.completed_steps = completed_steps
        flow_task.status = final_status if completed_steps < TOTAL_STEPS else FlowTrackingStatus.COMPLETED
        flow_task.timeout_level = timeout_level
        flow_task.current_node = messages[-1].node_name if messages else None
        flow_task.summary_text = self._gen_summary(
            completed_steps, TOTAL_STEPS, timeout_level, final_status
        )
        if completed_steps == TOTAL_STEPS:
            flow_task.completed_at = datetime.now(timezone.utc)

        # 步骤 6：写入 shared_data 供后续 Agent 使用
        ctx.shared_data["flow_result"] = {
            "flow_task_id":    flow_task_id,
            "completed_steps": completed_steps,
            "total_steps":     TOTAL_STEPS,
            "status":          flow_task.status.value,
            "timeout_level":   timeout_level,
            "current_node":    flow_task.current_node,
            "total_elapsed_ms": total_elapsed_ms,
        }

        logger.info(
            f"[{self.agent_name}] 流转追踪完成 "
            f"steps={completed_steps}/{TOTAL_STEPS} "
            f"status={flow_task.status.value} "
            f"timeout_level={timeout_level} "
            f"task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "flow_task_id":    flow_task_id,
                "completed_steps": completed_steps,
                "total_steps":     TOTAL_STEPS,
                "status":          flow_task.status.value,
                "timeout_level":   timeout_level,
                "is_completed":    completed_steps == TOTAL_STEPS,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 私有辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _map_step_status(self, raw_status: str) -> Tuple[FlowMessageStatus, bool]:
        """
        将通道返回的原始状态映射为 FlowMessageStatus 枚举 + 是否异常标志

        Returns:
            (FlowMessageStatus, is_anomaly)
        """
        mapping = {
            "success": (FlowMessageStatus.RESPONDED, False),
            "timeout": (FlowMessageStatus.TIMEOUT,   True),
            "error":   (FlowMessageStatus.ERROR,      True),
        }
        return mapping.get(raw_status, (FlowMessageStatus.ERROR, True))

    def _calc_timeout_level(self, total_elapsed_ms: int) -> str:
        """
        根据累计耗时计算超时预警级别
        阈值从高到低遍历：第一个满足的阈值即为预警级别
        """
        # 反向遍历（从最高级别开始）
        for level in ["OVERDUE", "URGENT", "WARNING", "WATCH", "NORMAL"]:
            if total_elapsed_ms >= TIMEOUT_THRESHOLDS_MS[level]:
                return level
        return "NORMAL"

    def _gen_summary(
        self,
        completed_steps: int,
        total_steps: int,
        timeout_level: str,
        status: FlowTrackingStatus,
    ) -> str:
        """生成自然语言流转状态摘要（模拟 LLM 输出）"""
        if status == FlowTrackingStatus.COMPLETED:
            return (
                f"票据流转已全部完成（{completed_steps}/{total_steps}步），"
                f"超时预警级别：{timeout_level}，业务处理正常。"
            )
        elif status == FlowTrackingStatus.TIMEOUT:
            step_num = completed_steps + 1
            return (
                f"票据流转在第 {step_num} 步发生超时，"
                f"已完成 {completed_steps}/{total_steps} 步，"
                f"建议联系对应节点排查网络或系统问题。"
            )
        elif status == FlowTrackingStatus.ERROR:
            step_num = completed_steps + 1
            return (
                f"票据流转在第 {step_num} 步发生异常，"
                f"已完成 {completed_steps}/{total_steps} 步，"
                f"错误可能由报文格式或接口异常引起，请人工介入处理。"
            )
        else:
            return (
                f"票据流转进行中（{completed_steps}/{total_steps}步），"
                f"当前状态：{status.value}。"
            )
