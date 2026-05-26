# app/api/tracking_router.py
# 票据流转追踪 API 路由
# 路由组前缀：/api/v1/tracking
# 覆盖接口：
#   POST /api/v1/tracking/tasks                         - 创建流转追踪任务（同步执行7步）
#   GET  /api/v1/tracking/tasks/{task_id}               - 查询流转追踪状态
#   GET  /api/v1/tracking/tasks/{task_id}/messages      - 获取七步报文节点详情

import uuid  # 生成任务 ID
from typing import Optional  # 可选类型

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel  # 请求/响应 Schema
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import AgentContext  # Agent 上下文
from app.core.auth import get_current_user, TokenData  # 认证依赖
from app.core.database import get_db  # 数据库会话依赖
from app.models.agent_models import (
    AuditTaskType,          # 业务类型枚举（流转追踪需要知道是哪种业务）
    FlowTrackingTask,       # 流转追踪主表 ORM
    FlowMessage,            # 报文节点状态 ORM
    FlowTrackingStatus,     # 追踪任务状态枚举
)


# ── 路由注册 ───────────────────────────────────────────────────────────────────
tracking_router = APIRouter(prefix="/tracking", tags=["票据流转追踪"])


# ── Pydantic 请求/响应 Schema ──────────────────────────────────────────────────

class TrackingTaskCreateRequest(BaseModel):
    """创建流转追踪任务的请求体"""
    ticket_number: str                                  # 被追踪的票据号码（必填）
    business_type: str = AuditTaskType.FULL_AUDIT.value  # 业务类型（决定报文路径）
    audit_task_id: Optional[str] = None                  # 可选：关联的审核任务 ID


class TrackingTaskResponse(BaseModel):
    """流转追踪任务响应体"""
    flow_task_id: str         # 流转追踪任务 ID
    ticket_number: str        # 票据号码
    status: str               # 追踪状态
    total_steps: int          # 总步骤数（固定为 7）
    completed_steps: int      # 已完成步骤数
    current_node: Optional[str]   # 当前报文所在节点
    timeout_level: str            # 超时预警级别
    summary: Optional[str]        # 自然语言摘要


class FlowMessageResponse(BaseModel):
    """单个报文节点响应体"""
    id: str
    step_index: int         # 步骤序号（1~7）
    node_name: str          # 节点名称
    msg_type: str           # 报文类型：send/ack
    status: str             # 节点状态
    processing_ms: int      # 处理耗时（毫秒）
    is_anomaly: bool        # 是否异常
    anomaly_reason: Optional[str]   # 异常原因


# ── 接口实现 ───────────────────────────────────────────────────────────────────

@tracking_router.post("/tasks", response_model=TrackingTaskResponse, status_code=201)
async def create_tracking_task(
    request: TrackingTaskCreateRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    创建流转追踪任务（同步执行，返回完整7步结果）
    POST /api/v1/tracking/tasks

    FlowTrackingAgent 在此接口中**同步**执行七步报文流转模拟，
    因此接口可能耗时数秒（取决于 FlowChannelMock 的模拟延迟）。
    生产环境可改为异步，通过 GET 接口轮询结果。

    请求示例：
    {
        "ticket_number": "SHBH20240115001",
        "business_type": "acceptance_prompt"
    }
    """
    # 校验 business_type 合法性
    try:
        AuditTaskType(request.business_type)
    except ValueError:
        valid_types = [t.value for t in AuditTaskType]
        raise HTTPException(
            status_code=400,
            detail=f"无效的 business_type: {request.business_type}，有效值: {valid_types}"
        )

    from app.agents.flow_tracking_agent import FlowTrackingAgent  # 延迟导入

    # 创建追踪任务上下文（使用提供的 audit_task_id 或生成新 ID）
    task_id = request.audit_task_id or str(uuid.uuid4())
    ctx = AgentContext(
        audit_task_id=task_id,
        tenant_id=current_user.tenant_id,
    )
    # 将票据号码和业务类型注入 shared_data，FlowTrackingAgent 从此读取
    ctx.shared_data["ticket_number"] = request.ticket_number
    ctx.shared_data["business_type"] = request.business_type

    agent = FlowTrackingAgent()
    result = await agent.run(ctx, db)

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"流转追踪执行失败: {result.error_msg}"
        )

    flow_result = result.data or {}

    logger.info(
        f"[tracking_router] 流转追踪完成 "
        f"ticket={request.ticket_number} "
        f"status={flow_result.get('status')} "
        f"steps={flow_result.get('completed_steps')}/7"
    )

    return TrackingTaskResponse(
        flow_task_id=flow_result.get("flow_task_id", task_id),
        ticket_number=request.ticket_number,
        status=flow_result.get("status", "unknown"),
        total_steps=7,
        completed_steps=flow_result.get("completed_steps", 0),
        current_node=flow_result.get("current_node"),
        timeout_level=flow_result.get("timeout_level", "NORMAL"),
        summary=flow_result.get("summary"),
    )


@tracking_router.get("/tasks/{task_id}", response_model=TrackingTaskResponse)
async def get_tracking_task(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    查询流转追踪任务状态
    GET /api/v1/tracking/tasks/{task_id}

    返回七步流转的整体进度和当前节点位置。
    """
    task = await db.get(FlowTrackingTask, task_id)

    if not task:
        raise HTTPException(status_code=404, detail="流转追踪任务不存在")

    # 注意：FlowTrackingTask 不直接关联 tenant_id（通过 AuditTask 间接关联）
    # 这里简化处理：super_admin 不做限制，其他用户通过 audit_task_id 验证归属
    if not current_user.is_super_admin and task.audit_task_id:
        audit_task = await db.get(
            __import__("app.models.agent_models", fromlist=["AuditTask"]).AuditTask,
            task.audit_task_id
        )
        if audit_task and audit_task.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=403, detail="无权访问此流转追踪任务")

    return TrackingTaskResponse(
        flow_task_id=task.id,
        ticket_number=task.ticket_number or "",
        status=task.status.value if task.status else "unknown",
        total_steps=task.total_steps or 7,
        completed_steps=task.completed_steps or 0,
        current_node=task.current_node,
        timeout_level=task.timeout_level or "NORMAL",
        summary=task.summary_text,
    )


@tracking_router.get("/tasks/{task_id}/messages")
async def get_tracking_messages(
    task_id: str,
    page: int = Query(default=1, ge=1),                 # 页码（从 1 开始）
    page_size: int = Query(default=14, ge=1, le=100),    # 默认 14（7步×2节点=14条）
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取七步报文节点详情
    GET /api/v1/tracking/tasks/{task_id}/messages

    返回每个报文节点的到达时间、处理耗时和异常标记，
    供前端渲染流转时间轴图。
    """
    # 验证任务存在
    task = await db.get(FlowTrackingTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="流转追踪任务不存在")

    # 查询该追踪任务的所有报文节点，按步骤序号升序排列
    stmt = (
        select(FlowMessage)
        .where(FlowMessage.flow_task_id == task_id)
        .order_by(FlowMessage.step_index, FlowMessage.created_at)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    messages = (await db.execute(stmt)).scalars().all()

    # 查询节点总数
    from sqlalchemy import func
    count_stmt = (
        select(func.count(FlowMessage.id))
        .where(FlowMessage.flow_task_id == task_id)
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    return {
        "flow_task_id": task_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "messages": [
            FlowMessageResponse(
                id=msg.id,
                step_index=msg.step_index,
                node_name=msg.node_name,
                msg_type=msg.msg_type or "send",
                status=msg.status.value if msg.status else "unknown",
                processing_ms=msg.processing_ms or 0,
                is_anomaly=msg.is_anomaly or False,
                anomaly_reason=msg.anomaly_reason,
            ).model_dump()
            for msg in messages
        ],
    }
