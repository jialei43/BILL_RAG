# app/api/batch_router.py
# 批量审核任务 API 路由
# 路由组前缀：/api/v1/batch
# 覆盖接口：
#   POST /api/v1/batch/tasks                         - 创建批量审核任务
#   GET  /api/v1/batch/tasks/{batch_id}              - 查询批次进度
#   GET  /api/v1/batch/tasks/{batch_id}/items        - 获取批次子项列表（分页）

import uuid  # 生成批次 ID
from typing import List, Optional  # 类型注解

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field  # 请求/响应 Schema
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import AgentContext  # Agent 上下文
from app.core.auth import get_current_user, TokenData  # 认证依赖
from app.core.database import get_db  # 数据库会话依赖
from app.models.agent_models import (
    AuditTaskType,     # 任务类型枚举
    BatchTask,         # 批量任务主表 ORM
    BatchTaskItem,     # 批量子任务表 ORM
    BatchTaskStatus,   # 批量任务状态枚举
    AuditTaskStatus,   # 子任务状态枚举（批量子项复用此枚举）
)


# ── 路由注册 ───────────────────────────────────────────────────────────────────
batch_router = APIRouter(prefix="/batch", tags=["批量审核任务"])


# ── Pydantic 请求/响应 Schema ──────────────────────────────────────────────────

class BatchItemInput(BaseModel):
    """批量任务中单个子任务的输入"""
    document_id: Optional[str] = None      # 文档 ID（与全流程审核对应）
    bill_element: Optional[dict] = None    # 或直接传入票据要素（快速模式）


class BatchTaskCreateRequest(BaseModel):
    """创建批量审核任务的请求体"""
    items: List[BatchItemInput] = Field(..., min_length=1, max_length=1000)  # 子任务列表（1~1000条）
    task_type: str = AuditTaskType.FULL_AUDIT.value  # 业务类型（所有子任务同类型）
    concurrency: int = Field(default=10, ge=1, le=100)  # 并发度（1~100）


class BatchTaskResponse(BaseModel):
    """批量任务响应体（创建和查询通用）"""
    batch_task_id: str    # 批次 ID
    status: str           # 批次状态
    total: int            # 子任务总数
    completed: int        # 已完成数
    failed: int           # 失败数
    failure_rate: float   # 失败率（0.0~1.0）
    message: str = ""     # 补充说明


class BatchItemResponse(BaseModel):
    """批量子任务响应体"""
    id: str               # 子任务记录 ID
    item_index: int       # 批次内序号
    document_id: Optional[str]     # 文档 ID
    audit_task_id: Optional[str]   # 关联的审核任务 ID
    status: str           # 子任务状态
    error_msg: Optional[str]       # 失败时的错误信息


# ── 后台任务：异步执行批量审核 ────────────────────────────────────────────────

async def _run_batch_audit(
    batch_task_id: str,
    tenant_id: str,
    items: List[dict],
    task_type: str,
    concurrency: int,
):
    """
    后台任务：调用 BatchSchedulingAgent 执行批量审核
    HTTP 响应发送后在后台异步执行，不阻塞请求
    """
    from app.agents.batch_scheduling_agent import BatchSchedulingAgent  # 延迟导入
    from app.core.database import AsyncSessionLocal  # 后台任务使用独立会话

    logger.info(
        f"[batch_router] 开始批量审核 batch={batch_task_id[:8]} "
        f"count={len(items)} concurrency={concurrency}"
    )

    async with AsyncSessionLocal() as db:
        ctx = AgentContext(
            audit_task_id=batch_task_id,  # 用批次 ID 作为父任务 ID
            tenant_id=tenant_id,
        )

        try:
            agent = BatchSchedulingAgent()
            await agent.run(
                ctx=ctx,
                db=db,
                items=items,
                concurrency=concurrency,
                task_type=task_type,
            )
            logger.info(f"[batch_router] 批量审核完成 batch={batch_task_id[:8]}")

        except Exception as e:
            # 批量调度本身异常：更新批次状态为失败
            logger.error(f"[batch_router] 批量审核异常 batch={batch_task_id[:8]}: {e}")
            try:
                task_record = await db.get(BatchTask, batch_task_id)
                if task_record:
                    task_record.status = BatchTaskStatus.FAILED
                    await db.commit()
            except Exception:
                pass  # 状态更新失败不影响日志记录


# ── 接口实现 ───────────────────────────────────────────────────────────────────

@batch_router.post("/tasks", response_model=BatchTaskResponse, status_code=202)
async def create_batch_task(
    request: BatchTaskCreateRequest,
    background_tasks: BackgroundTasks,       # 后台任务队列
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    创建批量审核任务
    POST /api/v1/batch/tasks

    接受最多 1000 个子任务，立即返回 batch_task_id。
    实际审核在后台异步执行，通过 GET /batch/tasks/{id} 轮询进度。

    请求示例：
    {
        "items": [{"document_id": "xxx"}, {"document_id": "yyy"}],
        "task_type": "full_audit",
        "concurrency": 10
    }
    """
    # 校验 task_type
    try:
        AuditTaskType(request.task_type)
    except ValueError:
        valid_types = [t.value for t in AuditTaskType]
        raise HTTPException(
            status_code=400,
            detail=f"无效的 task_type: {request.task_type}，有效值: {valid_types}"
        )

    # 每个子项至少有一种输入方式
    items_as_dicts = []
    for idx, item in enumerate(request.items):
        if not item.document_id and not item.bill_element:
            raise HTTPException(
                status_code=400,
                detail=f"第 {idx} 个子任务缺少 document_id 或 bill_element"
            )
        items_as_dicts.append({
            "document_id":  item.document_id,
            "bill_element": item.bill_element,
        })

    # 创建批量任务主记录（PENDING 状态，后台任务创建子记录）
    batch_task_id = str(uuid.uuid4())
    batch_task = BatchTask(
        id=batch_task_id,
        tenant_id=current_user.tenant_id,
        task_type=AuditTaskType(request.task_type),
        total_count=len(request.items),
        completed_count=0,
        failed_count=0,
        concurrency=min(request.concurrency, 100),  # 强制不超过系统上限
        status=BatchTaskStatus.PENDING,
        submitted_by=current_user.user_id,
    )
    db.add(batch_task)
    await db.flush()  # 让 batch_task_id 可用

    # 注册后台批量审核任务
    background_tasks.add_task(
        _run_batch_audit,
        batch_task_id=batch_task_id,
        tenant_id=current_user.tenant_id,
        items=items_as_dicts,
        task_type=request.task_type,
        concurrency=request.concurrency,
    )

    logger.info(
        f"[batch_router] 批量任务已提交 batch={batch_task_id[:8]} "
        f"count={len(request.items)} type={request.task_type} "
        f"tenant={current_user.tenant_id}"
    )

    return BatchTaskResponse(
        batch_task_id=batch_task_id,
        status=BatchTaskStatus.PENDING.value,
        total=len(request.items),
        completed=0,
        failed=0,
        failure_rate=0.0,
        message=f"批量任务已提交（{len(request.items)} 个子任务），正在后台处理",
    )


@batch_router.get("/tasks/{batch_id}", response_model=BatchTaskResponse)
async def get_batch_task(
    batch_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    查询批量任务进度
    GET /api/v1/batch/tasks/{batch_id}

    前端轮询此接口，直到 status 变为 completed 或 failed。
    """
    task = await db.get(BatchTask, batch_id)

    # 权限检查：只能查看本租户批量任务
    if not task or (
        not current_user.is_super_admin and task.tenant_id != current_user.tenant_id
    ):
        raise HTTPException(status_code=404, detail="批量任务不存在")

    # 计算实时失败率（兜底 0.0，避免除以零）
    failure_rate = (
        task.failed_count / task.total_count
        if task.total_count > 0 else 0.0
    )

    # 根据状态生成提示信息
    if task.status == BatchTaskStatus.COMPLETED:
        msg = f"批量审核已完成：{task.completed_count}/{task.total_count} 成功"
    elif task.status == BatchTaskStatus.FAILED:
        msg = f"批量审核失败：失败率 {failure_rate:.1%}，超过 50% 阈值"
    elif task.status == BatchTaskStatus.RUNNING:
        done = (task.completed_count or 0) + (task.failed_count or 0)
        msg = f"批量审核进行中：{done}/{task.total_count} 已处理"
    else:
        msg = "批量任务已提交，等待调度"

    return BatchTaskResponse(
        batch_task_id=task.id,
        status=task.status.value,
        total=task.total_count or 0,
        completed=task.completed_count or 0,
        failed=task.failed_count or 0,
        failure_rate=failure_rate,
        message=msg,
    )


@batch_router.get("/tasks/{batch_id}/items")
async def get_batch_task_items(
    batch_id: str,
    page: int = Query(default=1, ge=1),                   # 页码（从 1 开始）
    page_size: int = Query(default=20, ge=1, le=100),      # 每页条数（最多 100）
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取批次子项列表（分页）
    GET /api/v1/batch/tasks/{batch_id}/items?page=1&page_size=20

    返回每个子任务的处理状态，便于排查哪些文件失败。
    """
    # 先验证批次归属（防止跨租户枚举子任务）
    task = await db.get(BatchTask, batch_id)
    if not task or (
        not current_user.is_super_admin and task.tenant_id != current_user.tenant_id
    ):
        raise HTTPException(status_code=404, detail="批量任务不存在")

    # 分页查询子任务，按 item_index 升序（保持提交顺序）
    stmt = (
        select(BatchTaskItem)
        .where(BatchTaskItem.batch_task_id == batch_id)
        .order_by(BatchTaskItem.item_index)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = (await db.execute(stmt)).scalars().all()

    # 查询子任务总数（用于前端分页组件）
    from sqlalchemy import func
    count_stmt = (
        select(func.count(BatchTaskItem.id))
        .where(BatchTaskItem.batch_task_id == batch_id)
    )
    total = (await db.execute(count_stmt)).scalar() or 0

    return {
        "batch_task_id": batch_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "items": [
            BatchItemResponse(
                id=item.id,
                item_index=item.item_index,
                document_id=item.document_id,
                audit_task_id=item.audit_task_id,
                status=item.status.value if item.status else "unknown",
                error_msg=item.error_msg,
            ).model_dump()
            for item in items
        ],
    }
