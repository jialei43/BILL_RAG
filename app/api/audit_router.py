# app/api/audit_router.py
# 审核任务 API 路由：提交、查询、报告下载
# 路由组前缀：/api/v1/audit
# 覆盖接口：
#   POST /api/v1/audit/tasks              - 提交审核任务（全流程 or 指定类型）
#   GET  /api/v1/audit/tasks/{task_id}    - 查询任务状态和进度
#   GET  /api/v1/audit/tasks/{task_id}/report     - 下载审核报告 JSON
#   GET  /api/v1/audit/tasks/{task_id}/report/pdf - 下载 PDF 报告
#   POST /api/v1/audit/issuance/check     - 出票预检（独立调用，无需全流程）

import uuid  # 生成任务 ID
from pathlib import Path  # 路径处理（PDF 文件读取）
from typing import Optional  # 可选类型

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse  # 文件下载响应
from loguru import logger
from pydantic import BaseModel  # 请求/响应 Schema
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import AgentContext  # Agent 上下文
from app.core.auth import get_current_user, TokenData  # 认证依赖
from app.core.database import get_db, AsyncSessionLocal  # 数据库会话依赖
from app.core.cache import bill_cache, compute_bill_fingerprint  # 缓存层
from app.models.agent_models import (
    AuditTask, AuditTaskStatus, AuditTaskType,  # 审核任务 ORM
    AuditReport,                                # 报告 ORM
)


# ── 路由注册 ───────────────────────────────────────────────────────────────────
audit_router = APIRouter(prefix="/audit", tags=["审核任务"])


# ── Pydantic 请求/响应 Schema ──────────────────────────────────────────────────

class AuditTaskCreateRequest(BaseModel):
    """创建审核任务的请求体"""
    document_id: Optional[str] = None          # 关联的文档 ID（上传后获得）
    bill_record_id: Optional[str] = None        # 或直接传入已有票据主档 ID
    task_type: str = AuditTaskType.FULL_AUDIT.value  # 业务类型（默认全流程审核）
    bill_element: Optional[dict] = None         # 可选：直接传入票据要素（跳过文档解析）
    priority: int = 5                           # 调度优先级：1(最高)~10(最低)


class IssuanceCheckRequest(BaseModel):
    """出票预检请求体"""
    bill_element: dict  # 票据要素字典（必填）


class AuditTaskResponse(BaseModel):
    """审核任务响应体"""
    task_id: str        # 任务 ID
    status: str         # 当前状态
    progress_pct: float  # 完成百分比
    current_agent: Optional[str] = None  # 当前执行的 Agent
    task_type: str      # 业务类型
    message: str = ""   # 补充说明


# ── 后台任务：异步执行完整审核流程 ───────────────────────────────────────────────

async def _run_full_audit(
    task_id: str,
    tenant_id: str,
    document_id: Optional[str],
    bill_record_id: Optional[str],
    bill_element: Optional[dict],
    task_type: str,
):
    """
    后台任务：创建 AgentContext 并调用 OrchestratorAgent 执行完整审核
    由 FastAPI BackgroundTasks 在响应发送后异步执行，不阻塞 HTTP 请求
    """
    from app.agents.orchestrator_agent import OrchestratorAgent  # 延迟导入，避免循环依赖
    from app.core.database import AsyncSessionLocal  # 后台任务需独立会话

    logger.info(f"[audit_router] 开始后台审核 task={task_id} type={task_type}")

    # 后台任务使用独立的数据库会话（与 HTTP 请求的会话生命周期不同）
    async with AsyncSessionLocal() as db:
        ctx = AgentContext(
            audit_task_id=task_id,
            tenant_id=tenant_id,
            document_id=document_id,
            bill_record_id=bill_record_id,
        )

        # 若调用方直接传入了票据要素，注入到 shared_data 供 Agent 读取
        if bill_element:
            ctx.shared_data["bill_element"] = bill_element

        try:
            orchestrator = OrchestratorAgent()
            try:
                audit_task_type = AuditTaskType(task_type)
            except ValueError:
                audit_task_type = AuditTaskType.FULL_AUDIT

            await orchestrator.run(ctx, db, task_type=audit_task_type)
            logger.info(f"[audit_router] 后台审核完成 task={task_id}")

            # 审核成功完成后，写入最高层审核任务缓存
            if bill_element:
                bill_fp = compute_bill_fingerprint(bill_element)
                await bill_cache.set_audit(tenant_id, bill_fp, task_type, task_id)

        except Exception as e:
            # 兜底：使用独立 session 更新状态为 FAILED（原 db 可能在异常后状态不可靠）
            logger.error(f"[audit_router] 后台审核异常 task={task_id}: {e}", exc_info=True)
            try:
                async with AsyncSessionLocal() as fail_db:
                    from sqlalchemy import update as sa_update
                    await fail_db.execute(
                        sa_update(AuditTask)
                        .where(AuditTask.id == task_id)
                        .values(
                            status=AuditTaskStatus.FAILED,
                            error_msg=str(e)[:500],
                        )
                    )
                    await fail_db.commit()
            except Exception:
                pass


# ── 接口实现 ───────────────────────────────────────────────────────────────────

@audit_router.post("/tasks", response_model=AuditTaskResponse, status_code=202)
async def create_audit_task(
    request: AuditTaskCreateRequest,
    background_tasks: BackgroundTasks,       # FastAPI 后台任务队列
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    提交审核任务
    POST /api/v1/audit/tasks

    支持三种调用方式：
    1. 传入 document_id（最常用）：从已上传文档开始全流程审核
    2. 传入 bill_element（测试/开发）：直接传入票据要素，跳过文档解析步骤
    3. 传入 bill_record_id：基于已入库的票据主档重新审核

    任务立即返回 task_id，实际审核在后台异步执行。
    前端通过 GET /api/v1/audit/tasks/{task_id} 轮询进度。
    """
    # 参数校验：三种方式必须提供至少一种
    if not request.document_id and not request.bill_element and not request.bill_record_id:
        raise HTTPException(
            status_code=400,
            detail="必须提供 document_id、bill_element 或 bill_record_id 之一"
        )

    # 校验 task_type 合法性
    try:
        AuditTaskType(request.task_type)
    except ValueError:
        valid_types = [t.value for t in AuditTaskType]
        raise HTTPException(
            status_code=400,
            detail=f"无效的 task_type: {request.task_type}，有效值: {valid_types}"
        )

    # ── 最高层缓存查询（仅当直接传入 bill_element 时才可计算指纹）──────────────────
    # 若 document_id 或 bill_record_id 方式提交，指纹需等解析完成后才可知，此处不拦截
    if request.bill_element:
        bill_fp = compute_bill_fingerprint(request.bill_element)
        cached_audit = await bill_cache.get_audit(
            current_user.tenant_id, bill_fp, request.task_type
        )
        if cached_audit is not None:
            existing_task_id = cached_audit["task_id"]
            # 验证历史任务仍然存在且已完成（防止缓存指向已删除的任务）
            existing_task = await db.get(AuditTask, existing_task_id)
            if existing_task and existing_task.status == AuditTaskStatus.COMPLETED:
                logger.info(
                    f"[audit_router] 审核任务缓存命中 "
                    f"bill_fp={bill_fp[:8]} existing_task={existing_task_id[:8]} "
                    f"tenant={current_user.tenant_id}"
                )
                return AuditTaskResponse(
                    task_id=existing_task_id,
                    status=AuditTaskStatus.COMPLETED.value,
                    progress_pct=100.0,
                    task_type=request.task_type,
                    message="该票据已有完成的审核结果（缓存命中），可直接下载报告",
                )

    # 创建 AuditTask 主记录（PENDING 状态）
    task_id = str(uuid.uuid4())
    audit_task = AuditTask(
        id=task_id,
        tenant_id=current_user.tenant_id,
        document_id=request.document_id,
        bill_record_id=request.bill_record_id,
        task_type=AuditTaskType(request.task_type),
        status=AuditTaskStatus.PENDING,
        priority=request.priority,
        submitted_by=current_user.user_id,
    )
    db.add(audit_task)
    # flush 让 task_id 可用，但事务尚未提交（BackgroundTask 提交后才执行）
    await db.flush()

    # 注册后台审核任务（HTTP 响应发送后才真正执行）
    background_tasks.add_task(
        _run_full_audit,
        task_id=task_id,
        tenant_id=current_user.tenant_id,
        document_id=request.document_id,
        bill_record_id=request.bill_record_id,
        bill_element=request.bill_element,
        task_type=request.task_type,
    )

    logger.info(
        f"[audit_router] 审核任务已提交 task={task_id} "
        f"type={request.task_type} tenant={current_user.tenant_id}"
    )

    return AuditTaskResponse(
        task_id=task_id,
        status=AuditTaskStatus.PENDING.value,
        progress_pct=0.0,
        task_type=request.task_type,
        message="审核任务已提交，正在后台处理。通过 GET /api/v1/audit/tasks/{task_id} 查询进度",
    )


@audit_router.get("/tasks/{task_id}", response_model=AuditTaskResponse)
async def get_audit_task(
    task_id: str,                            # 任务 ID（URL 路径参数）
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    查询审核任务状态和进度
    GET /api/v1/audit/tasks/{task_id}

    前端轮询此接口，直到 status 变为 completed 或 failed。
    progress_pct 从 0 增长到 100，每个 Agent 完成后由 OrchestratorAgent 更新。
    """
    task = await db.get(AuditTask, task_id)

    # 权限检查：只能查看本租户任务（super_admin 除外）
    if not task or (
        not current_user.is_super_admin and task.tenant_id != current_user.tenant_id
    ):
        raise HTTPException(status_code=404, detail="审核任务不存在")

    # 组装响应（区分不同终态给出不同提示）
    if task.status == AuditTaskStatus.COMPLETED:
        msg = "审核已完成，可通过 GET /report 下载报告"
    elif task.status == AuditTaskStatus.FAILED:
        msg = f"审核失败: {task.error_msg or '未知错误'}"
    elif task.status == AuditTaskStatus.RUNNING:
        msg = f"正在执行: {task.current_agent or '调度中'}"
    else:
        msg = "任务已提交，等待调度"

    return AuditTaskResponse(
        task_id=task.id,
        status=task.status.value,
        progress_pct=task.progress_pct or 0.0,
        current_agent=task.current_agent,
        task_type=task.task_type.value if task.task_type else "unknown",
        message=msg,
    )


@audit_router.get("/tasks/{task_id}/report")
async def get_audit_report(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取审核报告（JSON 格式）
    GET /api/v1/audit/tasks/{task_id}/report

    返回 ReportGenerationAgent 生成的 9 节结构化 JSON 报告。
    任务未完成时返回 404。
    """
    # 验证任务归属
    task = await db.get(AuditTask, task_id)
    if not task or (
        not current_user.is_super_admin and task.tenant_id != current_user.tenant_id
    ):
        raise HTTPException(status_code=404, detail="审核任务不存在")

    if task.status != AuditTaskStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"任务状态为 {task.status.value}，报告尚未生成（等待完成后再下载）"
        )

    # 查询最新的终版报告（is_final=True 优先，否则取最新）
    stmt = (
        select(AuditReport)
        .where(AuditReport.audit_task_id == task_id)
        .order_by(AuditReport.is_final.desc(), AuditReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(stmt)).scalar_one_or_none()

    if not report or not report.report_json:
        raise HTTPException(status_code=404, detail="报告尚未生成，请稍后重试")

    # 直接返回 JSON 内容（FastAPI 自动序列化）
    return {
        "task_id": task_id,
        "report_version": report.version,
        "is_final": report.is_final,
        "training_mode": report.training_mode,
        "generated_at": report.created_at.isoformat() if report.created_at else None,
        "report": report.report_json,  # 9 节完整报告内容
    }


@audit_router.get("/tasks/{task_id}/report/pdf")
async def download_audit_report_pdf(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    下载审核报告 PDF
    GET /api/v1/audit/tasks/{task_id}/report/pdf

    返回 reportlab 生成的 PDF 文件，Content-Disposition 为附件下载。
    PDF 生成失败时返回 404，客户端可降级到 JSON 报告。
    """
    # 验证任务归属
    task = await db.get(AuditTask, task_id)
    if not task or (
        not current_user.is_super_admin and task.tenant_id != current_user.tenant_id
    ):
        raise HTTPException(status_code=404, detail="审核任务不存在")

    # 查询有 PDF 路径的报告记录
    stmt = (
        select(AuditReport)
        .where(
            AuditReport.audit_task_id == task_id,
            AuditReport.pdf_path.isnot(None),  # 必须有 PDF 路径
        )
        .order_by(AuditReport.is_final.desc(), AuditReport.created_at.desc())
        .limit(1)
    )
    report = (await db.execute(stmt)).scalar_one_or_none()

    if not report or not report.pdf_path:
        raise HTTPException(
            status_code=404,
            detail="PDF 报告不存在，请使用 JSON 报告接口，或联系管理员重新生成"
        )

    pdf_path = Path(report.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF 文件已不在磁盘上，可能已被清理，请联系管理员重新生成"
        )

    # 以附件形式返回 PDF 文件（浏览器会自动下载）
    filename = f"audit_report_{task_id[:8]}.pdf"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@audit_router.post("/issuance/check")
async def issuance_precheck(
    request: IssuanceCheckRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    出票资格预检（独立调用，无需提交全流程任务）
    POST /api/v1/audit/issuance/check

    快速验证票据要素是否满足出票条件，适用于：
    - 出票前实时验证（用户填写完表单即刻校验）
    - 批量校验（无需等待完整审核报告）

    返回 is_eligible=true/false 和 failed_checks 列表。
    """
    from app.agents.bill_issuance_agent import BillIssuanceAgent  # 延迟导入

    # 创建临时任务上下文（预检不创建 AuditTask 记录，避免产生脏数据）
    tmp_task_id = str(uuid.uuid4())
    ctx = AgentContext(
        audit_task_id=tmp_task_id,
        tenant_id=current_user.tenant_id,
    )
    ctx.shared_data["bill_element"] = request.bill_element

    agent = BillIssuanceAgent()
    result = await agent.run(ctx, db)

    logger.info(
        f"[audit_router] 出票预检完成 eligible={result.data.get('is_eligible')} "
        f"failed={result.data.get('failed_count')} tenant={current_user.tenant_id}"
    )

    return {
        "is_eligible":    result.data.get("is_eligible"),
        "failed_count":   result.data.get("failed_count"),
        "failed_checks":  result.data.get("failed_checks"),
        "recommendation": result.data.get("recommendation"),
    }
