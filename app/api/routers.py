# app/api/routers.py
# FastAPI 路由汇总文件
# 这是系统所有 HTTP 接口的定义处，相当于"API 接待台"
# 每个路由函数对应一个 HTTP 接口：前端调用→本函数执行→返回结果
#
# 接口分组：
# /api/v1/auth      - 认证（登录、注册）
# /api/v1/documents - 文档管理（上传/列表/查询/删除）
# /api/v1/query     - 智能问答
# /api/v1/tenants   - 租户管理（仅管理员）
# /metrics          - Prometheus 监控指标
# /health           - 健康检查

import os                # 文件操作（获取文件大小、删除文件）
import shutil            # 高级文件操作（复制上传的文件）
import uuid              # 生成唯一 ID
from datetime import timedelta    # 时间差（用于 Token 有效期）
from pathlib import Path          # 跨平台路径处理
from typing import Optional       # 可选类型

from fastapi import (
    APIRouter,           # 路由分组器（把相关接口放在一起）
    Depends,             # 依赖注入（自动执行认证、数据库会话等）
    HTTPException,       # HTTP 错误（如 404、403）
    UploadFile,          # 上传文件对象
    File,                # 文件参数标记
    Form,                # multipart/form-data 文本字段标记
    BackgroundTasks,     # 后台任务（不阻塞当前请求的异步处理）
    Query,               # URL 查询参数（如 ?page=1&size=20）
    status               # HTTP 状态码常量
)
from fastapi.security import OAuth2PasswordRequestForm  # 兼容 Swagger UI 的 OAuth2 表单登录
from fastapi import Request as FastAPIRequest           # 用于读取原始请求体
from fastapi.responses import StreamingResponse, Response, JSONResponse
# StreamingResponse：流式响应（用于 SSE 实时推送）
# Response：通用响应（可自定义 Content-Type）

import threading

from sqlalchemy.ext.asyncio import AsyncSession   # 异步数据库会话
from sqlalchemy import select, func               # select：构建查询；func：SQL 函数（COUNT、SUM）
from sqlalchemy import Numeric, update as sa_update
from loguru import logger                         # 日志

# ── 批次追踪（内存级，服务重启后清零，批次生命周期短暂，无需持久化）────────────────
# batch_id → [doc_id, ...]：记录每个批次包含的文档 ID
_batch_registry: dict[str, list[str]] = {}
# 已取消的批次集合（某文件失败后整批终止）
_batch_cancelled: set[str] = set()
_batch_lock = threading.Lock()

from config.settings import settings              # 配置
from app.core.auth import (
    get_current_user,               # 依赖：验证 Token，返回当前用户
    get_admin_user,                 # 依赖：向后兼容别名 → get_super_admin
    get_super_admin,                # 依赖：仅超级管理员
    get_tenant_admin_or_above,      # 依赖：租户管理员或超级管理员
    TokenData,                      # Token 数据结构
    verify_password,                # 密码验证函数
    get_password_hash,              # 密码哈希函数
    create_access_token,            # 生成 JWT Token
)
from app.core.database import get_db              # 依赖：获取数据库会话
from app.models.db_models import (
    Tenant, User, Document, DocumentChunk, QueryLog, DocumentStatus,
    BillRecord, BillVersion,  # 票据生命周期模型
    UserRole,
)
# 导入所有数据库模型类
from app.models.schemas import *                  # 导入所有 Pydantic Schema（请求/响应结构）
from app.services.ingestion import ingestion_pipeline  # 文档入库流水线
from app.services.rag_service import rag_service       # RAG 问答服务
from app.services.rate_limiter import rate_limiter     # 限流器
from app.services.metrics import metrics               # 监控

# 允许上传的文件类型白名单
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".tiff"}


# ════════════════════════════════════════════════════════════════════════════════
# 认证路由（/api/v1/auth/...）
# ════════════════════════════════════════════════════════════════════════════════
auth_router = APIRouter(prefix="/auth", tags=["认证"])
# prefix：该路由组的 URL 前缀（所有接口 URL 都以 /auth 开头）
# tags：API 文档中的分组标签


async def _parse_login_credentials(request: FastAPIRequest) -> tuple[str, str]:
    """自动识别 Content-Type，同时支持 JSON 和 form 表单两种登录格式。"""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        username = body.get("username", "")
        password = body.get("password", "")
    else:
        # application/x-www-form-urlencoded（Swagger UI / curl -d）
        form = await request.form()
        username = form.get("username", "")
        password = form.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=422, detail="username 和 password 不能为空")
    return username, password


@auth_router.post("/login", response_model=TokenResponse)
async def login(
    request: FastAPIRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    用户登录接口
    POST /api/v1/auth/login
    同时支持：
      - JSON body:  {"username": "...", "password": "..."}
      - Form 表单:  username=...&password=...  （Swagger UI Authorize 按钮）
    """
    username, password = await _parse_login_credentials(request)

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="账户已被禁用")

    token = create_access_token({
        "user_id": user.id,
        "tenant_id": user.tenant_id,
        "role": user.role.value if hasattr(user.role, "value") else str(user.role),
    })

    return TokenResponse(
        access_token=token,
        tenant_id=user.tenant_id,
        user_id=user.id,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@auth_router.post("/register", response_model=UserResponse, status_code=201)
async def register_user(
    request: UserCreate,                              # 请求体：新用户信息
    current_user: TokenData = Depends(get_current_user),  # 必须已登录
    db: AsyncSession = Depends(get_db),
):
    """
    注册新用户
    POST /api/v1/auth/register

    权限规则：
    - super_admin：可在任意租户创建 tenant_admin 或 user（通过 tenant_id 指定目标租户）
    - tenant_admin：只能在自己的租户创建 user（role 只能是 user）
    - user：无权创建账号（403）
    """
    # 普通用户无权创建账号
    if not current_user.is_tenant_admin_or_above:
        raise HTTPException(status_code=403, detail="无权创建用户，至少需要租户管理员权限")

    # 确定目标租户：super_admin 可指定，其他角色只能在自己的租户内创建
    if request.tenant_id:
        if not current_user.is_super_admin:
            raise HTTPException(status_code=403, detail="只有超级管理员可以指定目标租户")
        target_tenant_id = request.tenant_id
        # 验证目标租户存在
        if not await db.get(Tenant, target_tenant_id):
            raise HTTPException(status_code=404, detail="目标租户不存在")
    else:
        target_tenant_id = current_user.tenant_id

    # 租户管理员只能创建 user 角色，不能创建 tenant_admin
    target_role = request.role  # 已由 Schema 限制为 tenant_admin 或 user
    if current_user.role == "tenant_admin" and target_role != "user":
        raise HTTPException(status_code=403, detail="租户管理员只能创建普通用户（user）账号")

    # 检查用户名是否已存在
    existing = await db.execute(select(User).where(User.username == request.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="用户名已存在")

    user = User(
        id=str(uuid.uuid4()),
        tenant_id=target_tenant_id,
        username=request.username,
        email=request.email,
        hashed_password=get_password_hash(request.password),
        role=UserRole(target_role),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


# ════════════════════════════════════════════════════════════════════════════════
# 文档管理路由（/api/v1/documents/...）
# ════════════════════════════════════════════════════════════════════════════════
docs_router = APIRouter(prefix="/documents", tags=["文档管理"])


def _save_upload(file: UploadFile, tenant_id: str) -> str:
    """
    把上传的文件保存到本地磁盘
    存储路径：data/uploads/{租户ID}/{UUID}{扩展名}
    用 UUID 命名避免文件名冲突（两个用户上传同名文件不会互相覆盖）
    """
    upload_dir = Path(settings.UPLOAD_DIR) / tenant_id  # 每个租户有独立子目录
    upload_dir.mkdir(parents=True, exist_ok=True)        # 创建目录（如不存在）
    # parents=True：同时创建所有父目录；exist_ok=True：目录已存在时不报错

    ext = Path(file.filename).suffix.lower()  # 提取文件扩展名并转小写（.PDF → .pdf）
    filename = f"{uuid.uuid4()}{ext}"         # UUID + 扩展名（如：550e8400....pdf）
    file_path = upload_dir / filename

    # 把上传的文件内容写到磁盘
    with open(file_path, "wb") as f:          # wb=以二进制写入模式打开
        shutil.copyfileobj(file.file, f)      # 流式复制（不占用大量内存）

    return str(file_path)   # 返回文件的完整路径


async def _cancel_batch(batch_id: str, failed_doc_id: str, reason: str):
    """某文件失败时，将同批所有仍在 PROCESSING 的文件改为 FAILED，并标记批次取消。"""
    with _batch_lock:
        if batch_id in _batch_cancelled:
            return  # 已经处理过，避免重复
        _batch_cancelled.add(batch_id)
        sibling_ids = [d for d in _batch_registry.get(batch_id, []) if d != failed_doc_id]

    if not sibling_ids:
        return

    from app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        await session.execute(
            sa_update(Document)
            .where(Document.id.in_(sibling_ids), Document.status == DocumentStatus.PROCESSING)
            .values(
                status=DocumentStatus.FAILED,
                error_msg=f"批次中其他文件解析失败，已终止。失败文件: {failed_doc_id[:8]}… 原因: {reason[:200]}",
            )
        )
        await session.commit()

    logger.warning(
        f"[batch] 批次 {batch_id} 已取消，{len(sibling_ids)} 个文件标记为失败 "
        f"触发文件={failed_doc_id[:8]}"
    )


async def _update_doc_status(
    doc_id: str,
    status,
    result: dict = None,
    error: str = None,
):
    """更新数据库中文档的处理状态，在主事件循环内执行，避免跨循环操作 asyncpg 连接。"""
    from app.core.database import AsyncSessionLocal
    from app.services.metrics import metrics as _metrics
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, doc_id)
        if doc:
            doc.status = status
            if result:
                doc.chunk_count = result.get("chunk_count", 0)
                doc.page_count = result.get("page_count", 0)
                doc.parse_meta = result.get("parse_stats", {})
            if error:
                doc.error_msg = error
            await session.commit()

            # 文档处理完成后，从数据库汇总该租户的总 chunk 数，更新 Prometheus Gauge
            if status == DocumentStatus.COMPLETED:
                try:
                    total_q = select(func.coalesce(func.sum(Document.chunk_count), 0)).where(
                        Document.tenant_id == doc.tenant_id,
                        Document.status == DocumentStatus.COMPLETED,
                    )
                    total_chunks = (await session.execute(total_q)).scalar() or 0
                    _metrics.set_chunk_count(doc.tenant_id, int(total_chunks))
                except Exception:
                    pass

                # 写入 document_chunks 表（PostgreSQL 镜像，供 SQL 分析）
                # uuid5 保证相同 document_id+chunk_index 始终生成同一个 PK，天然幂等
                if result and result.get("_chunks"):
                    try:
                        from sqlalchemy.dialects.postgresql import insert as pg_insert
                        doc_uuid = uuid.UUID(doc_id)
                        def _strip_null(s):
                            """PostgreSQL UTF-8 不允许 \x00 空字节，PDF 解析偶尔会引入，统一过滤。"""
                            return s.replace("\x00", "") if isinstance(s, str) else s

                        chunk_rows = [
                            {
                                "id": str(uuid.uuid5(doc_uuid, str(c["chunk_index"]))),
                                "document_id": doc_id,
                                "tenant_id": doc.tenant_id,
                                "chunk_index": c["chunk_index"],
                                "content": _strip_null(c["content"]),
                                "section_path": _strip_null(c.get("section_path")),
                                "page_num": c.get("page_num"),
                                "chunk_type": c.get("chunk_type"),
                                "token_count": c.get("token_count"),
                                "milvus_id": f"{doc_id}_{c['chunk_index']}",
                                "chunk_metadata": None,
                            }
                            for c in result["_chunks"]
                        ]
                        stmt = pg_insert(DocumentChunk).values(chunk_rows).on_conflict_do_nothing(
                            index_elements=["id"]
                        )
                        await session.execute(stmt)
                        await session.commit()
                        logger.info(
                            f"[chunks] 写入 document_chunks 成功: doc={doc_id} count={len(chunk_rows)}"
                        )
                    except Exception as e:
                        logger.warning(f"[chunks] document_chunks 写入失败 doc={doc_id}: {e}")


async def _run_ingestion(
    file_path: str,
    tenant_id: str,
    document_id: str,
    batch_id: str = None,
):
    """
    后台任务：执行文档入库（解析→分块→向量化→存储）

    改为 async def，由 FastAPI 在主事件循环中调度：
    - 同步的 ingestion_pipeline.ingest() 通过 run_in_executor 放到线程池执行，不阻塞事件循环
    - 数据库状态更新留在主事件循环，asyncpg 连接池不会出现跨循环错误
    """
    import asyncio
    loop = asyncio.get_event_loop()

    # 批次已被取消（同批其他文件先失败），直接标记失败跳过处理
    if batch_id and batch_id in _batch_cancelled:
        await _update_doc_status(document_id, DocumentStatus.FAILED,
                                  error="同批次其他文件解析失败，已终止")
        logger.info(f"[batch] 跳过已取消批次内的文件 doc={document_id} batch={batch_id}")
        return

    try:
        result = await loop.run_in_executor(
            None,                          # 使用默认线程池
            ingestion_pipeline.ingest,     # 同步函数
            file_path, tenant_id, document_id,
        )
        await _update_doc_status(document_id, DocumentStatus.COMPLETED, result)
        metrics.record_ingest(tenant_id, "success")
        logger.info(f"Background ingest completed: {document_id}")

    except Exception as e:
        await _update_doc_status(document_id, DocumentStatus.FAILED, error=str(e))
        metrics.record_ingest(tenant_id, "failed")
        logger.error(f"Background ingest failed: {document_id} - {e}")
        # 触发批次级联取消：将同批其他仍在 PROCESSING 的文件全部标为 FAILED
        if batch_id:
            await _cancel_batch(batch_id, document_id, str(e))


async def _ingest_one(
    file: UploadFile,
    tenant_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
    batch_id: str = None,
) -> Document:
    """单文件保存 + 建库记录 + 注册后台入库任务，供单传和批量接口共用。"""
    ext = Path(file.filename).suffix.lower()  # 提取小写扩展名
    if ext not in ALLOWED_EXTENSIONS:  # 文件类型白名单校验
        raise HTTPException(
            status_code=400,
            detail=f"文件 {file.filename!r} 类型不支持: {ext}。支持: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    file_path = _save_upload(file, tenant_id)          # 保存文件到磁盘
    file_size = os.path.getsize(file_path)             # 获取文件大小（字节）

    doc = Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        filename=file.filename,                        # 原始文件名
        file_path=file_path,                           # 服务器存储路径
        file_type=ext.lstrip("."),                     # 去掉点号（.pdf → pdf）
        file_size=file_size,
        md5_hash="pending",                            # 临时占位，入库时计算真实 MD5
        status=DocumentStatus.PROCESSING,              # 状态设为"处理中"
    )
    db.add(doc)
    await db.flush()  # flush 让 doc.id 可用，但事务尚未提交

    background_tasks.add_task(_run_ingestion, file_path, tenant_id, doc.id, batch_id)
    return doc


@docs_router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,              # FastAPI 后台任务队列
    file: UploadFile = File(...),                   # 单文件上传，Swagger 可直接测试
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    单文档上传接口（异步入库）
    POST /api/v1/documents/upload
    文件立即保存，解析在后台异步进行，不阻塞响应
    批量上传请使用 POST /api/v1/documents/upload/batch
    """
    current_count = rate_limiter.get_doc_count(current_user.tenant_id)  # 查 Redis 配额计数
    if not rate_limiter.check_doc_quota(current_user.tenant_id, current_count):
        raise HTTPException(status_code=429, detail="文档配额已满")

    batch_id = str(uuid.uuid4())
    logger.info(f"[upload] 单文件上传 tenant={current_user.tenant_id} file={file.filename} batch={batch_id}")
    doc = await _ingest_one(file, current_user.tenant_id, background_tasks, db, batch_id)
    with _batch_lock:
        _batch_registry[batch_id] = [doc.id]
    # DocumentResponse 的 batch_id 字段来自 Schema 而非 ORM，需手动注入
    return DocumentResponse.model_validate(doc).model_copy(update={"batch_id": batch_id})


@docs_router.post("/upload/batch", response_model=BatchUploadResponse, status_code=202)
async def upload_documents_batch(
    background_tasks: BackgroundTasks,              # FastAPI 后台任务队列
    files: list[UploadFile] = File(...),            # 多文件，前端 form-data 用同名字段 files
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    批量文档上传接口（异步入库）
    POST /api/v1/documents/upload/batch
    所有文件整体校验通过后才入库，有一个不合法则全部拒绝
    每个文件独立注册后台任务；任意一个文件解析失败会终止整批，其余文件标记为 FAILED

    curl 示例：
        curl -X POST .../upload/batch \\
          -H "Authorization: Bearer $TOKEN" \\
          -F "files=@a.pdf" -F "files=@b.pdf"
    """
    if not files:  # 防止传空列表
        raise HTTPException(status_code=400, detail="至少上传一个文件")

    # 先整体校验配额，不够则直接拒绝，避免部分保存后再报错
    current_count = rate_limiter.get_doc_count(current_user.tenant_id)
    if not rate_limiter.check_doc_quota(current_user.tenant_id, current_count + len(files) - 1):
        raise HTTPException(
            status_code=429,
            detail=f"文档配额不足，当前已用 {current_count}，本次上传 {len(files)} 个，超出限制"
        )

    # 先整体校验文件类型，有一个不合法则全部拒绝，避免部分入库产生脏数据
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {f.filename!r} 类型不支持: {ext}。支持: {', '.join(ALLOWED_EXTENSIONS)}"
            )

    batch_id = str(uuid.uuid4())
    logger.info(
        f"[upload_batch] tenant={current_user.tenant_id} "
        f"count={len(files)} files={[f.filename for f in files]} batch={batch_id}"
    )

    docs = []
    for file in files:
        doc = await _ingest_one(file, current_user.tenant_id, background_tasks, db, batch_id)
        docs.append(doc)

    # 批次注册：需在后台任务运行（响应发送后）之前完成
    with _batch_lock:
        _batch_registry[batch_id] = [d.id for d in docs]

    logger.info(f"[upload_batch] 登记完成 batch={batch_id} ids={[d.id for d in docs]}")
    return BatchUploadResponse(
        batch_id=batch_id,
        total=len(docs),
        documents=[
            DocumentResponse.model_validate(d).model_copy(update={"batch_id": batch_id})
            for d in docs
        ],
    )


async def _ingest_from_path(
    file_path: Path,
    tenant_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
    batch_id: str = None,
) -> Document:
    """从服务器本地路径创建入库记录（文件夹批量入库专用，文件不复制，原路径直接入库）。"""
    file_size = os.path.getsize(file_path)  # 获取文件大小
    ext = file_path.suffix.lower()          # 提取扩展名

    doc = Document(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        filename=file_path.name,           # 只取文件名，不含目录部分
        file_path=str(file_path),          # 保留完整路径，入库任务通过此路径读取文件
        file_type=ext.lstrip("."),
        file_size=file_size,
        md5_hash="pending",                # 真实 MD5 在后台入库时计算
        status=DocumentStatus.PROCESSING,
    )
    db.add(doc)
    await db.flush()  # 让 doc.id 可用，事务尚未提交

    background_tasks.add_task(_run_ingestion, str(file_path), tenant_id, doc.id, batch_id)
    return doc


@docs_router.post("/upload/folder", response_model=BatchUploadResponse, status_code=202)
async def upload_folder(
    background_tasks: BackgroundTasks,
    request: FolderUploadRequest,          # JSON body：{"folder_path": "...", "recursive": false}
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    文件夹批量入库接口
    POST /api/v1/documents/upload/folder
    传入服务器上已存在的文件夹路径，自动扫描所有支持格式的文件并异步入库。
    文件不会被复制，Document.file_path 直接指向原始路径。
    recursive=true 时递归扫描所有子目录。
    任意一个文件解析失败会终止整批，其余文件标记为 FAILED。

    请求示例：
        {"folder_path": "/data/bills/2024", "recursive": true}
    """
    folder = Path(request.folder_path)

    # 路径安全校验：必须是绝对路径且实际存在的目录
    if not folder.is_absolute():
        raise HTTPException(status_code=400, detail="folder_path 必须是绝对路径")
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"路径不存在: {request.folder_path}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"路径不是文件夹: {request.folder_path}")

    # 扫描文件：recursive=True 则用 rglob 递归，否则只扫当前层
    glob_fn = folder.rglob if request.recursive else folder.glob
    matched = sorted(  # 按路径排序，保证每次入库顺序一致
        p for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )

    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"文件夹中未找到支持的文件（支持: {', '.join(ALLOWED_EXTENSIONS)}）"
        )

    # 整体检查配额，不足则全部拒绝
    current_count = rate_limiter.get_doc_count(current_user.tenant_id)
    if not rate_limiter.check_doc_quota(current_user.tenant_id, current_count + len(matched) - 1):
        raise HTTPException(
            status_code=429,
            detail=f"文档配额不足，当前已用 {current_count}，文件夹含 {len(matched)} 个文件，超出限制"
        )

    batch_id = str(uuid.uuid4())
    logger.info(
        f"[upload_folder] tenant={current_user.tenant_id} "
        f"folder={request.folder_path} matched={len(matched)} recursive={request.recursive} batch={batch_id}"
    )

    docs = []
    for file_path in matched:
        doc = await _ingest_from_path(file_path, current_user.tenant_id, background_tasks, db, batch_id)
        docs.append(doc)

    with _batch_lock:
        _batch_registry[batch_id] = [d.id for d in docs]

    logger.info(
        f"[upload_folder] 登记完成 batch={batch_id} count={len(docs)} "
        f"ids={[d.id for d in docs]}"
    )
    return BatchUploadResponse(
        batch_id=batch_id,
        total=len(docs),
        documents=[
            DocumentResponse.model_validate(d).model_copy(update={"batch_id": batch_id})
            for d in docs
        ],
    )


@docs_router.get("/", response_model=PaginatedResponse)
async def list_documents(
    page: int = Query(default=1, ge=1),                           # 页码（从1开始）
    page_size: int = Query(default=20, ge=1, le=100),             # 每页条数（最多100）
    status_filter: Optional[str] = Query(default=None),           # 可选：按状态过滤
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取文档列表（支持分页和状态过滤）
    - 管理员：可查看所有租户的文档
    - 普通用户：只能查看本租户的文档
    GET /api/v1/documents/?page=1&page_size=20&status_filter=completed
    """
    # super_admin 查全部；tenant_admin 和 user 只查自己租户
    base_filter = [] if current_user.is_super_admin else [Document.tenant_id == current_user.tenant_id]

    # 构建主查询
    query = select(Document)
    if base_filter:
        query = query.where(*base_filter)
    if status_filter:
        query = query.where(Document.status == status_filter)

    # 用相同条件查总数（与主查询保持一致，包含 status_filter）
    count_base = select(Document)
    if base_filter:
        count_base = count_base.where(*base_filter)
    if status_filter:
        count_base = count_base.where(Document.status == status_filter)
    count_q = select(func.count()).select_from(count_base.subquery())
    total = (await db.execute(count_q)).scalar()

    # 分页：按上传时间倒序
    query = query.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    docs = (await db.execute(query)).scalars().all()

    return PaginatedResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, (total + page_size - 1) // page_size),
    )


@docs_router.get("/stats")
async def document_stats(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取文档统计数据（分块数、今日查询数）
    GET /api/v1/documents/stats
    """
    from datetime import date as _date

    tenant_filter = [] if current_user.is_super_admin else [Document.tenant_id == current_user.tenant_id]

    chunk_q = select(func.coalesce(func.sum(Document.chunk_count), 0))
    if tenant_filter:
        chunk_q = chunk_q.where(*tenant_filter)
    total_chunks = (await db.execute(chunk_q)).scalar()

    today_q = select(func.count(QueryLog.id)).where(
        func.date(QueryLog.created_at) == _date.today()
    )
    if not current_user.is_super_admin:
        today_q = today_q.where(QueryLog.tenant_id == current_user.tenant_id)
    today_queries = (await db.execute(today_q)).scalar()

    return {"total_chunks": int(total_chunks), "today_queries": int(today_queries)}


@docs_router.get("/batch/{batch_id}", summary="查询批次内所有文件的处理状态")
async def get_batch_status(
    batch_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    轮询批次处理进度
    GET /api/v1/documents/batch/{batch_id}
    返回 overall_status: processing / completed / failed
    前端可每隔 2-3 秒轮询，直到 overall_status 不再是 processing
    """
    with _batch_lock:
        doc_ids = _batch_registry.get(batch_id)
        is_cancelled = batch_id in _batch_cancelled

    if doc_ids is None:
        raise HTTPException(status_code=404, detail="批次不存在或已过期（服务重启后内存清零）")

    # 查询数据库中该批次所有文档的当前状态
    stmt = select(Document).where(Document.id.in_(doc_ids))
    if not current_user.is_super_admin:
        stmt = stmt.where(Document.tenant_id == current_user.tenant_id)
    docs = (await db.execute(stmt)).scalars().all()

    doc_map = {d.id: d for d in docs}
    items = []
    for doc_id in doc_ids:
        doc = doc_map.get(doc_id)
        if doc:
            items.append(BatchDocItem(
                id=doc.id,
                filename=doc.filename,
                status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
                error_msg=doc.error_msg,
                chunk_count=doc.chunk_count or 0,
            ))

    statuses = {item.status for item in items}
    if "failed" in statuses:
        overall = "failed"
    elif "processing" in statuses or "pending" in statuses:
        overall = "processing"
    else:
        overall = "completed"

    return {
        "batch_id":       batch_id,
        "total":          len(items),
        "overall_status": overall,
        "is_cancelled":   is_cancelled,
        "documents":      [item.model_dump() for item in items],
    }


@docs_router.post("/batch/{batch_id}/retry", response_model=BatchUploadResponse, status_code=202,
                  summary="重试批次内所有失败文件")
async def retry_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    批次整体重试：将批次内所有 FAILED 文件重置为 PROCESSING 并重新触发入库任务
    POST /api/v1/documents/batch/{batch_id}/retry

    - 生成全新的 batch_id，避免触发旧批次的取消标记
    - 已成功（COMPLETED）的文件跳过，不重复处理
    - 原始文件必须仍在磁盘上，否则该文件跳过并保持 FAILED
    """
    with _batch_lock:
        doc_ids = _batch_registry.get(batch_id)

    if doc_ids is None:
        raise HTTPException(status_code=404, detail="批次不存在或已过期（服务重启后内存清零）")

    stmt = select(Document).where(Document.id.in_(doc_ids))
    if not current_user.is_super_admin:
        stmt = stmt.where(Document.tenant_id == current_user.tenant_id)
    docs = (await db.execute(stmt)).scalars().all()

    failed_docs = [d for d in docs if d.status == DocumentStatus.FAILED]
    if not failed_docs:
        raise HTTPException(status_code=409, detail="批次内没有失败的文件，无需重试")

    # 校验磁盘文件：原始文件不在就无法重试
    retryable = []
    skipped = []
    for doc in failed_docs:
        if doc.file_path and os.path.exists(doc.file_path):
            retryable.append(doc)
        else:
            skipped.append(doc.filename)

    if skipped:
        logger.warning(f"[batch_retry] 以下文件原始文件已不存在，跳过: {skipped}")

    if not retryable:
        raise HTTPException(status_code=422, detail=f"所有失败文件的原始文件均已不存在，请重新上传。文件: {skipped}")

    # 生成新批次 ID，旧批次的取消标记不影响新任务
    new_batch_id = str(uuid.uuid4())

    for doc in retryable:
        doc.status = DocumentStatus.PROCESSING
        doc.error_msg = None
        background_tasks.add_task(_run_ingestion, doc.file_path, doc.tenant_id, doc.id, new_batch_id)

    with _batch_lock:
        _batch_registry[new_batch_id] = [d.id for d in retryable]

    logger.info(
        f"[batch_retry] 旧批次={batch_id} 新批次={new_batch_id} "
        f"重试={len(retryable)} 跳过={len(skipped)}"
    )

    return BatchUploadResponse(
        batch_id=new_batch_id,
        total=len(retryable),
        documents=[
            DocumentResponse.model_validate(d).model_copy(update={"batch_id": new_batch_id})
            for d in retryable
        ],
    )


@docs_router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,    # URL 路径参数（/documents/{document_id}）
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取单个文档详情
    GET /api/v1/documents/{document_id}
    """
    doc = await db.get(Document, document_id)

    # super_admin 可查任意租户文档；其他用户只能查自己租户的文档
    if not doc or (not current_user.is_super_admin and doc.tenant_id != current_user.tenant_id):
        raise HTTPException(status_code=404, detail="文档不存在")

    return doc


@docs_router.post("/{document_id}/retry", response_model=DocumentResponse, status_code=202)
async def retry_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    重新触发失败文档的入库流程（断点续传）
    POST /api/v1/documents/{document_id}/retry
    - 仅允许对 status=failed 的文档重试
    - 有解析 checkpoint 时跳过 OCR/Vision，直接从向量化续传
    - 无需重新上传文件，复用磁盘上已保存的原始文件
    """
    doc = await db.get(Document, document_id)

    # super_admin 可操作任意租户文档；其他用户只能操作自己租户的文档
    if not doc or (not current_user.is_super_admin and doc.tenant_id != current_user.tenant_id):
        raise HTTPException(status_code=404, detail="文档不存在")

    # 只允许重试失败的文档，避免对正在处理或已完成的文档重复触发
    if doc.status != DocumentStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail=f"文档当前状态为 {doc.status}，仅 failed 状态可重试",
        )

    # 原始文件必须还在磁盘上，否则无法重新入库
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=422, detail="原始文件已不存在，请重新上传")

    # 重置状态为 processing，清空上次的错误信息
    doc.status = DocumentStatus.PROCESSING
    doc.error_msg = None

    # 触发后台入库任务（复用相同的 document_id，checkpoint 和 chunk 级幂等均生效）
    background_tasks.add_task(_run_ingestion, doc.file_path, current_user.tenant_id, document_id)

    return doc


@docs_router.delete("/{document_id}", response_model=SuccessResponse)
async def delete_document(
    document_id: str,
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    删除文档（同时删除：Milvus向量 + 磁盘文件 + 数据库记录）
    DELETE /api/v1/documents/{document_id}
    """
    doc = await db.get(Document, document_id)

    # super_admin 可删除任意租户文档；其他用户只能删除自己租户的文档
    if not doc or (not current_user.is_super_admin and doc.tenant_id != current_user.tenant_id):
        raise HTTPException(status_code=404, detail="文档不存在")

    # 第一步：从 Milvus 删除该文档的所有向量
    ingestion_pipeline.delete_document(current_user.tenant_id, document_id)

    # 第二步：删除磁盘上的文件
    if doc.file_path and os.path.exists(doc.file_path):
        os.remove(doc.file_path)   # 删除原始文件

    # 第三步：先删除 document_chunks（FK NOT NULL，无级联删除配置，需显式清理）
    from sqlalchemy import delete as sa_delete
    await db.execute(sa_delete(DocumentChunk).where(DocumentChunk.document_id == document_id))

    # 第四步：从数据库删除文档记录
    await db.delete(doc)

    return SuccessResponse(message=f"文档 {document_id} 已删除")


# ════════════════════════════════════════════════════════════════════════════════
# 问答路由（/api/v1/query/...）
# ════════════════════════════════════════════════════════════════════════════════
query_router = APIRouter(prefix="/query", tags=["智能问答"])


@query_router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    票据业务智能问答（非流式）
    POST /api/v1/query/
    完整 RAG 流程：检索→精排→LLM生成→返回
    """
    # QPS 限流检查：防止单个租户频繁请求
    allowed, rl_info = rate_limiter.check_rate_limit(current_user.tenant_id)
    if not allowed:
        metrics.record_rate_limited(current_user.tenant_id)  # 上报限流事件
        raise HTTPException(
            status_code=429,
            detail=f"请求频率超限，请 {rl_info['reset_in']}s 后重试",
            headers={"X-RateLimit-Reset": str(rl_info["reset_in"])},
            # 在响应头中告知客户端多久后可以重试
        )

    logger.info(  # 非流式请求入口日志
        f"[query] 收到问答请求 tenant={current_user.tenant_id} "
        f"query={request.query[:60]!r}"
    )
    try:
        # 执行 RAG 问答
        result = await rag_service.query(
            tenant_id=current_user.tenant_id,
            query=request.query,
            top_k=request.top_k,   # 可选：覆盖默认检索数量
        )

        # 问候语响应不写日志（retrieved_count=0 代表早返回，未经过检索）
        if result.get("retrieved_count", 0) > 0:
            log = QueryLog(
                id=result["query_id"],
                tenant_id=current_user.tenant_id,
                user_id=current_user.user_id,
                query=request.query,
                answer=result["answer"],
                retrieved_chunks=result["sources"],
                retrieval_ms=result["retrieval_ms"],
                llm_ms=result["llm_ms"],
                total_ms=result["total_ms"],
                route_type=result.get("route_type"),
            )
            db.add(log)

        logger.info(  # 问答成功日志，记录耗时和检索数量
            f"[query] 问答完成 tenant={current_user.tenant_id} "
            f"retrieved={result['retrieved_count']} "
            f"total_ms={result['total_ms']}"
        )
        # 上报监控指标
        metrics.record_query(current_user.tenant_id, "success")
        metrics.record_total_time(result["total_ms"], current_user.tenant_id)

        # 构建响应（把 sources 列表转换为 SourceChunk 对象列表）
        return QueryResponse(
            **{k: v for k, v in result.items() if k != "sources"},
            sources=[SourceChunk(**s) for s in result["sources"]],
        )
        # **result：把字典展开为关键字参数

    except Exception as e:
        metrics.record_query(current_user.tenant_id, "error")
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@query_router.post("/stream")
async def query_stream(
    request: QueryRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    流式智能问答（SSE：Server-Sent Events）
    POST /api/v1/query/stream
    像 ChatGPT 一样逐步输出答案，用户不用等全部生成完
    """
    allowed, _ = rate_limiter.check_rate_limit(current_user.tenant_id)  # 限流检查
    if not allowed:
        raise HTTPException(status_code=429, detail="请求频率超限")

    logger.info(  # 流式请求入口日志，方便定位"有没有进到这个接口"
        f"[stream] 收到流式查询 tenant={current_user.tenant_id} "
        f"query={request.query[:60]!r}"
    )

    async def event_generator():
        """异步生成器：逐步产生 SSE 格式的数据"""
        token_count = 0  # 统计实际输出 token 数，用于调试
        try:
            logger.info(f"[stream] event_generator 启动，开始调用 rag_service")  # 确认生成器真正开始执行
            async for token in rag_service.query_stream(
                tenant_id=current_user.tenant_id,
                query=request.query,
                user_id=current_user.user_id,
            ):
                token_count += 1  # 每个 token 计数
                yield f"data: {token}\n\n"  # SSE 格式：每条数据以 "data: " 开头，\n\n 结尾
            yield "data: [DONE]\n\n"  # 生成完毕，发送结束信号（前端检测到 [DONE] 停止接收）
            logger.info(  # 流式完成日志，记录输出量
                f"[stream] 流式生成完毕 tenant={current_user.tenant_id} "
                f"token_count={token_count}"
            )
        except Exception as e:  # 捕获生成器内部异常，避免静默关闭连接
            logger.error(  # 异常日志，包含 exc_info 以便追栈
                f"[stream] 生成器异常 tenant={current_user.tenant_id} error={e}",
                exc_info=True,
            )
            yield f"data: [ERROR] {str(e)}\n\n"  # 把错误信息推送给客户端

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",  # SSE 的 Content-Type
        headers={
            # Content-Encoding: identity 告知 GZipMiddleware 不要压缩此响应
            # GZipMiddleware 会缓冲整个流再压缩，导致 SSE 客户端迟迟收不到数据
            "Content-Encoding": "identity",
            "Cache-Control": "no-cache",       # 禁止中间代理缓存 SSE 事件
            "X-Accel-Buffering": "no",         # 禁止 nginx 对 SSE 做响应缓冲
        },
    )


@query_router.get("/stats")
async def query_stats(
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    检索统计：专项/模糊 × 成功/失败 四维度 + 耗时 + 近期日志
    GET /api/v1/query/stats
    """
    from datetime import date as _date

    tenant_filter = [] if current_user.is_super_admin else [QueryLog.tenant_id == current_user.tenant_id]

    def _apply(q):
        return q.where(*tenant_filter) if tenant_filter else q

    # ── 核心四维度：route_type × top_k_hit ────────────────────────────────
    matrix_q = _apply(
        select(
            QueryLog.route_type,
            QueryLog.top_k_hit,
            func.count(QueryLog.id).label("cnt"),
        ).group_by(QueryLog.route_type, QueryLog.top_k_hit)
    )
    matrix_rows = (await db.execute(matrix_q)).fetchall()

    # 初始化各维度计数
    counts = {
        "specialized_success": 0,
        "specialized_fail":    0,
        "fuzzy_success":       0,
        "fuzzy_fail":          0,
        "off_topic":           0,  # 无关咨询（未经检索，直接拒绝）
        "unknown":             0,  # top_k_hit=NULL（历史数据兜底）
    }
    for row in matrix_rows:
        rt  = row.route_type or "fuzzy"
        hit = row.top_k_hit
        if rt == "off_topic":
            counts["off_topic"] += row.cnt
        elif rt == "specialized":
            if hit is True:  counts["specialized_success"] += row.cnt
            elif hit is False: counts["specialized_fail"]  += row.cnt
            else:              counts["unknown"]            += row.cnt
        else:  # fuzzy
            if hit is True:  counts["fuzzy_success"] += row.cnt
            elif hit is False: counts["fuzzy_fail"]  += row.cnt
            else:              counts["unknown"]      += row.cnt

    total       = sum(counts.values())
    specialized = counts["specialized_success"] + counts["specialized_fail"]
    fuzzy       = counts["fuzzy_success"]       + counts["fuzzy_fail"]
    success     = counts["specialized_success"] + counts["fuzzy_success"]
    fail        = counts["specialized_fail"]    + counts["fuzzy_fail"]
    off_topic   = counts["off_topic"]

    # ── 今日查询数 ─────────────────────────────────────────────────────────
    today_q = _apply(
        select(func.count(QueryLog.id)).where(
            func.date(QueryLog.created_at) == _date.today()
        )
    )
    today = (await db.execute(today_q)).scalar() or 0

    # ── 平均耗时（仅检索成功的条目）──────────────────────────────────────
    avg_q = _apply(
        select(
            func.round(func.avg(QueryLog.retrieval_ms).cast(Numeric), 1).label("avg_retrieval"),
            func.round(func.avg(QueryLog.total_ms).cast(Numeric), 1).label("avg_total"),
        ).where(QueryLog.top_k_hit.is_(True))
    )
    avg_row = (await db.execute(avg_q)).fetchone()

    # ── 最近 50 条查询日志 ─────────────────────────────────────────────────
    recent_q = _apply(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(50)
    )
    recent_logs = (await db.execute(recent_q)).scalars().all()

    def _log_dict(log):
        return {
            "id":             log.id,
            "query":          log.query,
            "route_type":     log.route_type,
            "intent_id":      log.intent_id,
            "top_k_hit":      log.top_k_hit,
            "retrieval_ms":   log.retrieval_ms,
            "total_ms":       log.total_ms,
            "chunk_count":    len(log.retrieved_chunks) if log.retrieved_chunks else 0,
            "feedback_score": log.feedback_score,
            "created_at":     log.created_at.isoformat() if log.created_at else None,
        }

    return {
        "total":                total,
        "specialized":          specialized,
        "fuzzy":                fuzzy,
        "success":              success,
        "fail":                 fail,
        "off_topic":            off_topic,
        "specialized_success":  counts["specialized_success"],
        "specialized_fail":     counts["specialized_fail"],
        "fuzzy_success":        counts["fuzzy_success"],
        "fuzzy_fail":           counts["fuzzy_fail"],
        "today":                today,
        "avg_retrieval_ms":     float(avg_row.avg_retrieval) if avg_row and avg_row.avg_retrieval else 0,
        "avg_total_ms":         float(avg_row.avg_total)     if avg_row and avg_row.avg_total     else 0,
        "recent":               [_log_dict(l) for l in recent_logs],
    }


@query_router.post("/feedback", response_model=SuccessResponse)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    提交对答案的评分反馈（1-5星）
    POST /api/v1/query/feedback
    """
    log = await db.get(QueryLog, request.query_id)  # 查找对应的查询记录

    # 查询记录不存在，或不属于当前租户（防止对其他租户的记录打分）
    if not log or log.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="查询记录不存在")

    log.feedback_score = request.score   # 更新评分（1-5）
    return SuccessResponse(message="反馈已提交")


# ════════════════════════════════════════════════════════════════════════════════
# 租户管理路由（/api/v1/tenants/...）仅管理员可访问
# ════════════════════════════════════════════════════════════════════════════════
tenant_router = APIRouter(prefix="/tenants", tags=["租户管理"])


@tenant_router.post("/", response_model=TenantResponse, status_code=201)
async def create_tenant(
    request: TenantCreate,
    _: TokenData = Depends(get_admin_user),  # _ 表示这个参数只用于权限检查，不使用其值
    db: AsyncSession = Depends(get_db),
):
    """
    创建新租户（只有管理员才能调用）
    POST /api/v1/tenants/
    每个票据机构入驻系统时需要管理员先创建租户
    """
    # 若填写了统一社会信用代码，先查库是否已有同一家公司
    if request.license_no:
        dup = await db.execute(select(Tenant).where(Tenant.license_no == request.license_no))
        existing_tenant = dup.scalar_one_or_none()
        if existing_tenant:
            # 租户已存在，直接返回，前端会用返回的 id 去注册用户
            return existing_tenant

    # 生成唯一机构代码（未填则取 UUID 前 8 位大写）
    code = (request.code or uuid.uuid4().hex[:8].upper()).upper()
    dup_code = await db.execute(select(Tenant).where(Tenant.code == code))
    if dup_code.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"机构代码 {code!r} 已存在")

    tenant = Tenant(
        id=str(uuid.uuid4()),
        name=request.name,
        code=code,
        license_no=request.license_no,
        doc_quota=request.doc_quota,
        qps_limit=request.qps_limit,
        milvus_partition=f"tenant_{code}",
    )
    db.add(tenant)
    await db.flush()           # 触发 INSERT，让数据库填充 server_default 字段
    await db.refresh(tenant)   # 重新加载，使 status/created_at 等字段有值
    return tenant


@tenant_router.get("/", response_model=list[TenantResponse])
async def list_tenants(
    current_user: TokenData = Depends(get_tenant_admin_or_above),
    db: AsyncSession = Depends(get_db),
):
    """
    获取租户列表
    GET /api/v1/tenants/
    - super_admin：返回全部租户
    - tenant_admin：仅返回自己所属租户
    """
    if current_user.is_super_admin:
        result = await db.execute(select(Tenant).order_by(Tenant.created_at.desc()))
        tenants = result.scalars().all()
    else:
        tenant = await db.get(Tenant, current_user.tenant_id)
        tenants = [tenant] if tenant else []

    if not tenants:
        return []

    # 批量查询各租户文档数和查询数，附加到响应
    tenant_ids = [t.id for t in tenants]
    doc_counts = dict(
        (await db.execute(
            select(Document.tenant_id, func.count(Document.id).label("cnt"))
            .where(Document.tenant_id.in_(tenant_ids))
            .group_by(Document.tenant_id)
        )).fetchall()
    )
    query_counts = dict(
        (await db.execute(
            select(QueryLog.tenant_id, func.count(QueryLog.id).label("cnt"))
            .where(QueryLog.tenant_id.in_(tenant_ids))
            .group_by(QueryLog.tenant_id)
        )).fetchall()
    )

    result_list = []
    for t in tenants:
        data = TenantResponse.model_validate(t)
        data.doc_count = doc_counts.get(t.id, 0)
        data.query_count = query_counts.get(t.id, 0)
        result_list.append(data)
    return result_list


@tenant_router.get("/{tenant_id}/stats", response_model=TenantStats)
async def get_tenant_stats(
    tenant_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取租户使用统计（文档数、查询数、配额使用率等）
    GET /api/v1/tenants/{tenant_id}/stats
    权限：管理员可查看任意租户，普通用户只能查看自己的租户
    """
    # 权限检查：super_admin 可查任意租户；tenant_admin 只能查自己的租户；普通用户拒绝
    if not current_user.is_tenant_admin_or_above:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    if not current_user.is_super_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="无权访问其他租户数据")

    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="租户不存在")

    # 文档统计（直接查库，避免 Redis 计数不一致）
    doc_rows = (await db.execute(
        select(Document.status, func.count(Document.id).label("cnt"))
        .where(Document.tenant_id == tenant_id)
        .group_by(Document.status)
    )).fetchall()
    doc_count = sum(r.cnt for r in doc_rows)
    completed_count = next((r.cnt for r in doc_rows if str(r.status) in ("completed", "DocumentStatus.COMPLETED")), 0)
    failed_count    = next((r.cnt for r in doc_rows if str(r.status) in ("failed",    "DocumentStatus.FAILED")),    0)

    # 分块总数
    chunk_count = (await db.execute(
        select(func.coalesce(func.sum(Document.chunk_count), 0))
        .where(Document.tenant_id == tenant_id)
    )).scalar() or 0

    # 查询统计
    query_count = (await db.execute(
        select(func.count(QueryLog.id)).where(QueryLog.tenant_id == tenant_id)
    )).scalar() or 0

    today_query_count = (await db.execute(
        select(func.count(QueryLog.id)).where(
            QueryLog.tenant_id == tenant_id,
            func.date(QueryLog.created_at) == func.current_date()
        )
    )).scalar() or 0

    # 平均响应耗时（仅成功检索的请求）
    avg_raw = (await db.execute(
        select(func.avg(QueryLog.total_ms))
        .where(QueryLog.tenant_id == tenant_id)
        .where(QueryLog.total_ms.isnot(None))
    )).scalar()
    avg_latency_ms = round(float(avg_raw), 1) if avg_raw else None

    quota_pct = round(doc_count / tenant.doc_quota * 100, 1) if tenant.doc_quota else 0.0

    return TenantStats(
        tenant_id=tenant_id,
        doc_count=doc_count,
        completed_count=completed_count,
        failed_count=failed_count,
        chunk_count=chunk_count,
        query_count=query_count,
        today_query_count=today_query_count,
        doc_quota=tenant.doc_quota,
        qps_limit=tenant.qps_limit,
        quota_used_pct=quota_pct,
        avg_latency_ms=avg_latency_ms,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 票据识别路由（/api/v1/bill-recognition/...）
# ════════════════════════════════════════════════════════════════════════════════
bill_router = APIRouter(prefix="/bill-recognition", tags=["票据识别"])
# 独立于文档管理路由，专注于票据要素的结构化提取
# 不写库、不入 Milvus，只做识别并返回结果

# 允许上传到此接口的文件类型（仅图片和 PDF，不含 Excel/Word）
BILL_ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


@bill_router.post(
    "/recognize",
    response_model=BillRecognitionResponse,
    summary="票据要素识别",
    description=(
        "上传票据图片或 PDF，自动识别并返回结构化要素：\n"
        "出票日期、到期日、票面金额、出票人、承兑人、背书人等。\n"
        "**不会入库**，仅做一次性识别，结果直接返回。\n"
        "PDF 支持多页（正面 + 背书页），每页单独识别。"
    ),
)
async def recognize_bill(
    file: UploadFile = File(..., description="票据图片（PNG/JPG/TIFF）或 PDF 文件"),
    current_user: TokenData = Depends(get_current_user),   # 需要登录才能调用
):
    """
    票据要素识别接口
    POST /api/v1/bill-recognition/recognize
    不写数据库，同步返回识别结果（调用视觉大模型，单张约 2-5 秒）
    """
    from app.services.bill_recognition import bill_recognition_service
    from dataclasses import asdict   # 把 dataclass 转为字典，方便构建 Pydantic Schema

    # 校验文件类型（只允许图片和 PDF）
    suffix = Path(file.filename or "").suffix.lower()   # 提取并小写化文件后缀
    if suffix not in BILL_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,   # 415 Unsupported Media Type
            detail=f"不支持的文件类型 '{suffix}'，请上传 PDF 或图片文件",
        )

    # 读取文件内容（FastAPI UploadFile 需要 await 读取）
    file_bytes = await file.read()   # 文件字节内容
    if not file_bytes:
        raise HTTPException(status_code=400, detail="文件内容为空")   # 空文件拒绝处理

    logger.info(
        f"[bill_recognition] user={current_user.user_id} "
        f"file={file.filename} size={len(file_bytes)}"
    )

    try:
        # 调用票据识别服务（同步操作，内部调用视觉 API）
        result = bill_recognition_service.recognize_file(file_bytes, file.filename or "")

        # 把 dataclass BillElement 列表转换为 Pydantic Schema 列表
        bill_schemas = []
        for b in result.bills:
            d = asdict(b)                          # BillElement dataclass → Python 字典
            bill_schemas.append(BillElementSchema(**d))   # 字典 → Pydantic Schema

        return BillRecognitionResponse(
            filename=file.filename or "",          # 原始文件名
            bill_count=len(bill_schemas),          # 识别到的票据数量
            bills=bill_schemas,                    # 所有票据要素
            page_count=result.page_count,          # 处理的页数
            model_used=result.model_used,          # 使用的视觉模型
            elapsed_ms=result.elapsed_ms,          # 总耗时
        )

    except Exception as e:
        logger.error(f"[bill_recognition] failed: {e}")
        raise HTTPException(status_code=500, detail=f"票据识别失败: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════════
# 票据生命周期路由（/api/v1/bills/...）
# 职责：上传票据图片/PDF → 识别要素 → 写入 BillRecord/BillVersion + Milvus 向量
#       支持流转追踪（同一张票多次上传只追加版本和增量背书人向量）
# ════════════════════════════════════════════════════════════════════════════════
bills_router = APIRouter(prefix="/bills", tags=["票据生命周期管理"])

# 票据识别支持的文件类型（图片+PDF，不含 Excel/Word）
BILL_LIFECYCLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


@bills_router.post(
    "/upload",
    response_model=list[BillUpsertResponse],
    status_code=202,
    summary="上传票据并入生命周期管理",
    description=(
        "上传票据图片（PNG/JPG/TIFF）或 PDF，自动识别要素并写入 BillRecord + BillVersion 表和 Milvus 向量库。\n"
        "**同一票据号码多次上传**会创建新版本并仅追加新增背书人的向量块，不会重复入库。\n"
        "返回每张识别到票据的入库结果，含风险标记和版本号。"
    ),
)
async def upload_bill_lifecycle(
    file: UploadFile = File(..., description="票据图片或 PDF"),
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    票据上传生命周期管理入口
    POST /api/v1/bills/upload
    同步识别后立即入库，不走后台任务队列
    """
    from app.services.bill_recognition import bill_recognition_service  # 票据视觉识别服务
    from app.services.bill_lifecycle import bill_lifecycle_service       # 生命周期管理服务

    suffix = Path(file.filename or "").suffix.lower()  # 提取文件后缀（统一小写）
    if suffix not in BILL_LIFECYCLE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型 '{suffix}'，请上传 PDF 或图片文件",
        )

    file_bytes = await file.read()  # 读取文件字节内容（用于识别和存盘）
    if not file_bytes:
        raise HTTPException(status_code=400, detail="文件内容为空")

    # ── 保存原始文件到磁盘（存盘路径与普通文档上传一致）────────────────────────
    upload_dir = Path(settings.UPLOAD_DIR) / current_user.tenant_id
    upload_dir.mkdir(parents=True, exist_ok=True)       # 创建租户子目录（若不存在）
    saved_name = f"{uuid.uuid4()}{suffix}"              # UUID 命名，避免冲突
    saved_path = upload_dir / saved_name
    saved_path.write_bytes(file_bytes)                  # 同步写字节到磁盘

    # ── 创建 Document 记录（用于追溯原始文件来源）──────────────────────────────
    doc = Document(
        id=str(uuid.uuid4()),
        tenant_id=current_user.tenant_id,
        filename=file.filename or saved_name,
        file_path=str(saved_path),
        file_type=suffix.lstrip("."),
        file_size=len(file_bytes),
        md5_hash="pending",            # 票据场景不需要精确 MD5，先占位
        status=DocumentStatus.COMPLETED,  # 票据识别是同步的，写入即完成
    )
    db.add(doc)
    await db.flush()  # 让 doc.id 可用，事务尚未提交

    logger.info(
        f"[bills_upload] tenant={current_user.tenant_id} "
        f"file={file.filename} size={len(file_bytes)} doc_id={doc.id}"
    )

    # ── 同步识别票据要素 ─────────────────────────────────────────────────────
    try:
        recog_result = bill_recognition_service.recognize_file(file_bytes, file.filename or "")
    except Exception as e:
        logger.error(f"[bills_upload] 识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"票据识别失败: {str(e)}")

    if not recog_result.bills:
        raise HTTPException(status_code=422, detail="未能从文件中识别出有效票据要素")

    # ── 逐张票据执行生命周期管理（支持多张/PDF多页） ──────────────────────────
    upsert_results = []
    for element in recog_result.bills:
        try:
            ret = await bill_lifecycle_service.upsert_bill(
                db=db,
                element=element,
                document_id=doc.id,
                tenant_id=current_user.tenant_id,
                user_id=current_user.user_id,
            )
            upsert_results.append(BillUpsertResponse(**ret))
        except Exception as e:
            logger.error(f"[bills_upload] 生命周期入库失败 ticket={element.ticket_number}: {e}")
            raise HTTPException(status_code=500, detail=f"票据入库失败: {str(e)}")

    logger.info(
        f"[bills_upload] 入库完成 count={len(upsert_results)} "
        f"tickets={[r.ticket_number for r in upsert_results]}"
    )
    return upsert_results


@bills_router.get(
    "/",
    response_model=PaginatedResponse,
    summary="分页查询票据列表",
)
async def list_bills(
    page: int = Query(default=1, ge=1),                 # 页码（从1开始）
    page_size: int = Query(default=20, ge=1, le=100),   # 每页条数
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取当前租户的票据主档列表（分页）
    GET /api/v1/bills/?page=1&page_size=20
    列表视图不展开版本历史，仅返回主档核心字段
    """
    from app.services.bill_lifecycle import bill_lifecycle_service

    records, total = await bill_lifecycle_service.get_all_records(
        db, current_user.tenant_id, page, page_size
    )

    # 列表视图：版本列表留空，减少响应体大小
    items = [
        BillRecordResponse(
            id=r.id,
            ticket_number=r.ticket_number,
            ticket_type=r.ticket_type,
            issue_date=r.issue_date,
            due_date=r.due_date,
            amount_numeric=r.amount_numeric,
            amount_text=r.amount_text,
            drawer=r.drawer,
            acceptor=r.acceptor,
            payee=r.payee,
            risk_flags=r.risk_flags or [],
            latest_version=r.latest_version,
            versions=[],               # 列表页不加载版本历史
            created_at=r.created_at,
        )
        for r in records
    ]

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@bills_router.get(
    "/{ticket_number}",
    response_model=BillRecordResponse,
    summary="查询票据详情（含流转历史）",
)
async def get_bill(
    ticket_number: str,                                # URL 路径参数：票据号码
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    按票据号码查询主档详情，含完整的流转版本历史
    GET /api/v1/bills/{ticket_number}
    """
    from app.services.bill_lifecycle import bill_lifecycle_service
    from sqlalchemy import select as sa_select

    record = await bill_lifecycle_service.get_bill_record(
        db, current_user.tenant_id, ticket_number
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"票据 {ticket_number} 不存在")

    # 单独查询版本列表（避免 async ORM 懒加载问题）
    ver_stmt = (
        sa_select(BillVersion)
        .where(BillVersion.bill_record_id == record.id)
        .order_by(BillVersion.version)
    )
    versions = (await db.execute(ver_stmt)).scalars().all()

    version_responses = [
        BillVersionResponse(
            version=v.version,
            endorsers=v.endorsers or [],
            new_endorsers=v.new_endorsers or [],
            upload_time=v.upload_time,
            uploaded_by=v.uploaded_by,
        )
        for v in versions
    ]

    return BillRecordResponse(
        id=record.id,
        ticket_number=record.ticket_number,
        ticket_type=record.ticket_type,
        issue_date=record.issue_date,
        due_date=record.due_date,
        amount_numeric=record.amount_numeric,
        amount_text=record.amount_text,
        drawer=record.drawer,
        acceptor=record.acceptor,
        payee=record.payee,
        risk_flags=record.risk_flags or [],
        latest_version=record.latest_version,
        versions=version_responses,
        created_at=record.created_at,
    )


# ── 票据咨询接口（聊天窗口发起业务前的可行性审核）───────────────────────────────────
@query_router.post(
    "/consult",
    response_model=None,      # 动态响应类型（rag_answer / agent_report），此处不限定 Schema
    summary="票据业务咨询（上传票据 + 意图路由）",
    description=(
        "聊天窗口一站式咨询接口，业务员在发起业务流程前调用。\n\n"
        "**使用方式（multipart/form-data）：**\n"
        "- `query`：自然语言问题（必填），如：这张票据能否贴现\n"
        "- `bill_file`：票据图片/PDF（可选），上传后自动识别要素\n"
        "- `bill_record_id`：已入库的票据主档 ID（可替代文件上传）\n\n"
        "**响应类型（response_type）：**\n"
        "- `rag_answer`：知识性问题，返回 RAG 检索答案\n"
        "- `agent_report`：票据审核类问题，返回结构化审核报告\n"
        "- `off_topic`：与票据业务无关，返回引导语"
    ),
)
async def consult(
    query: str = Form(..., description="用户问题，如'这张票据能否贴现？'"),
    bill_file: Optional[UploadFile] = File(default=None, description="票据图片或PDF（可选）"),
    bill_record_id: Optional[str] = Form(default=None, description="已入库票据主档ID（与文件上传二选一）"),
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    票据业务咨询接口（发起业务流程前的可行性报告）
    POST /api/v1/query/consult

    核心流程：
    1. 若上传了 bill_file → 调用视觉模型识别票据要素（命中缓存则跳过）
    2. 意图识别（关键词 → LLM）判断是 RAG知识问答 还是 Agent专项审核
    3. Agent意图 → 调用 OrchestratorAgent.run_for_chat()，同步等待审核结论
       RAG意图 → 调用 rag_service.query_v2()，返回知识检索答案
    4. 统一封装为 ConsultResponse 返回
    """
    import time
    import uuid as _uuid
    from app.models.schemas import ConsultIssue, ConsultReport, ConsultResponse
    from app.services.intent_router import intent_router

    # ── 限流检查 ────────────────────────────────────────────────────────────────
    allowed, rl_info = rate_limiter.check_rate_limit(current_user.tenant_id)
    if not allowed:
        metrics.record_rate_limited(current_user.tenant_id)
        raise HTTPException(
            status_code=429,
            detail=f"请求频率超限，请 {rl_info['reset_in']}s 后重试",
            headers={"X-RateLimit-Reset": str(rl_info["reset_in"])},
        )

    query_id = _uuid.uuid4().hex[:16]  # 本次请求唯一 ID（用于日志追踪）
    t_start = time.perf_counter()

    logger.info(
        f"[consult] 收到咨询请求 tenant={current_user.tenant_id} "
        f"query_id={query_id} query={query[:60]!r} "
        f"has_file={bill_file is not None} bill_record_id={bill_record_id}"
    )

    # ── 步骤 1：票据文件上传 → 要素识别 ─────────────────────────────────────────
    bill_elements_dict: Optional[dict] = None  # 识别出的票据要素（供后续使用）
    bill_element_for_rag: Optional[dict] = None  # RAG 路径使用的上下文
    bill_file_bytes: Optional[bytes] = None
    bill_filename: str = ""

    if bill_file is not None:
        # 校验文件类型（仅允许图片和 PDF）
        suffix = Path(bill_file.filename or "").suffix.lower()
        if suffix not in BILL_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"不支持的文件类型 '{suffix}'，请上传 PDF 或图片文件",
            )
        bill_file_bytes = await bill_file.read()
        if not bill_file_bytes:
            raise HTTPException(status_code=400, detail="上传的票据文件内容为空")

        bill_filename = bill_file.filename or ""

        # 调用视觉识别服务（内部已集成 Redis 缓存，同一文件不重复识别）
        try:
            from app.services.bill_recognition import bill_recognition_service
            from dataclasses import asdict
            recog = bill_recognition_service.recognize_file(bill_file_bytes, bill_filename)
            if recog.bills:
                b = recog.bills[0]  # 取第一张票据（主票据）
                bill_elements_dict = asdict(b)  # 转为 dict，注入 shared_data
                bill_elements_dict["confidence_score"] = 1.0  # 默认置信度
                # RAG 路径用的轻量上下文（仅核心字段）
                bill_element_for_rag = {
                    "ticket_number":  b.ticket_number,
                    "ticket_type":    b.ticket_type,
                    "issue_date":     b.issue_date,
                    "due_date":       b.due_date,
                    "amount_numeric": b.amount_numeric,
                    "amount_text":    b.amount_text,
                    "drawer":         b.drawer,
                    "acceptor":       b.acceptor,
                    "payee":          b.payee,
                }
                logger.info(
                    f"[consult] 票据识别成功 ticket={b.ticket_number} "
                    f"elapsed={recog.elapsed_ms:.0f}ms"
                )
        except Exception as e:
            logger.warning(f"[consult] 票据识别失败，将仅走RAG路径: {e}")
            # 识别失败不中断流程，降级为纯 RAG 问答

    elif bill_record_id:
        # 从数据库加载已入库票据的要素
        bill_record = await db.get(BillRecord, bill_record_id)
        if bill_record and bill_record.tenant_id == current_user.tenant_id:
            bill_element_for_rag = {
                "ticket_number":  bill_record.ticket_number,
                "ticket_type":    bill_record.ticket_type,
                "issue_date":     bill_record.issue_date,
                "due_date":       bill_record.due_date,
                "amount_numeric": bill_record.amount_numeric,
                "amount_text":    bill_record.amount_text,
                "drawer":         bill_record.drawer,
                "acceptor":       bill_record.acceptor,
                "payee":          bill_record.payee,
            }
            # 已入库票据，也用于 Agent 路径（但没有完整18字段，用精简版）
            bill_elements_dict = bill_element_for_rag

    # ── 步骤 2：意图识别 ──────────────────────────────────────────────────────
    intent_id, confidence, intent_method = await intent_router.classify(query)

    logger.info(
        f"[consult] 意图识别完成 intent={intent_id} confidence={confidence:.2f} "
        f"method={intent_method} is_agent={intent_router.is_agent_intent(intent_id)}"
    )

    # ── 步骤 3a：Agent 审核路径 ───────────────────────────────────────────────
    # 核心原则：Agent 路由必须同时满足两个条件：
    #   1. 意图识别为 Agent 类意图
    #   2. 有票据文件/bill_record_id（有明确的审核对象）
    # 缺少票据时，即使意图是 Agent 类，也降级为 RAG 知识查询：
    #   用户可能只是在咨询某类业务的知识，还没准备好上传票据做审核
    #   此时强制要求上传文件会打断用户的知识咨询流程
    if intent_router.is_agent_intent(intent_id) and bill_elements_dict:
        # 有票据 + Agent意图 → 执行专项 Agent 审核

        # 获取对应的 task_type 和报告标签
        task_type_str = intent_router.get_agent_task_type(intent_id)
        audit_label   = intent_router.get_agent_label(intent_id)

        # 懒加载 OrchestratorAgent（避免启动时加载所有 Agent，首次调用时从注册表实例化）
        from app.agents.orchestrator_agent import OrchestratorAgent
        orchestrator = OrchestratorAgent()  # 无参构造：内部自动从 AgentRegistry 懒加载

        try:
            # 调用聊天审核模式（同步等待，预填充要素跳过 OCR）
            report_data = await orchestrator.run_for_chat(
                task_type_str=task_type_str,
                tenant_id=current_user.tenant_id,
                db=db,
                bill_element_dict=bill_elements_dict,
                audit_label=audit_label,
                timeout_seconds=180.0,
            )
        except Exception as e:
            logger.error(f"[consult] Agent 审核失败: {e}")
            raise HTTPException(status_code=500, detail=f"审核服务暂时不可用: {str(e)}")

        t_elapsed = (time.perf_counter() - t_start) * 1000

        # 构建结构化报告
        issues = [
            ConsultIssue(
                field=i.get("field", "unknown"),
                level=i.get("level", "warning"),
                description=i.get("description", ""),
                recommendation=i.get("recommendation"),
            )
            for i in report_data.get("issues", [])
        ]

        report = ConsultReport(
            audit_task_type=task_type_str,
            audit_label=audit_label,
            conclusion=report_data.get("conclusion", "审核中"),
            risk_level=report_data.get("risk_level", "UNKNOWN"),
            overall_score=report_data.get("overall_score"),
            issues=issues,
            summary=report_data.get("summary", ""),
            task_id=report_data.get("task_id", ""),
            elapsed_ms=report_data.get("elapsed_ms", 0.0),
        )

        logger.info(
            f"[consult] Agent审核完成 query_id={query_id} "
            f"task_id={report_data.get('task_id', '')[:8]} "
            f"conclusion={report.conclusion} elapsed={t_elapsed:.0f}ms"
        )
        metrics.record_query(current_user.tenant_id, "success")

        return ConsultResponse(
            query_id=query_id,
            intent_id=intent_id,
            response_type="agent_report",
            answer=report.summary,          # 摘要作为主答案，方便前端直接展示
            report=report,
            bill_elements=bill_elements_dict,
            sources=[],
            retrieval_ms=0.0,
            llm_ms=report_data.get("elapsed_ms", 0.0),
            total_ms=t_elapsed,
        )

    # ── 步骤 3b：离题处理 ────────────────────────────────────────────────────
    if intent_router.is_off_topic(intent_id, query):
        t_elapsed = (time.perf_counter() - t_start) * 1000
        return {
            "query_id":      query_id,
            "intent_id":     intent_id,
            "response_type": "off_topic",
            "answer":        "抱歉，您的问题超出了票据业务范围。本系统专注于票据合规审核、背书链分析、贴现申请审核等票据相关服务，请提问票据业务相关问题。",
            "report":        None,
            "bill_elements": bill_elements_dict,
            "sources":       [],
            "retrieval_ms":  0.0,
            "llm_ms":        0.0,
            "total_ms":      t_elapsed,
        }

    # ── 步骤 3c：RAG 知识问答路径 ─────────────────────────────────────────────
    try:
        rag_result = await rag_service.query_v2(
            tenant_id=current_user.tenant_id,
            query=query,
            top_k=None,
            user_id=current_user.user_id,
            bill_context=bill_element_for_rag,  # 将票据上下文注入检索（若有文件上传）
            db=db,
        )
        metrics.record_query(current_user.tenant_id, "success")
    except Exception as e:
        metrics.record_query(current_user.tenant_id, "error")
        logger.error(f"[consult] RAG 问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答服务失败: {str(e)}")

    t_elapsed = (time.perf_counter() - t_start) * 1000
    sources = [SourceChunk(**s) for s in rag_result.get("sources", [])]

    return ConsultResponse(
        query_id=query_id,
        intent_id=intent_id,
        response_type="rag_answer",
        answer=rag_result.get("answer", ""),
        report=None,
        bill_elements=bill_elements_dict,
        sources=sources,
        retrieval_ms=rag_result.get("retrieval_ms", 0.0),
        llm_ms=rag_result.get("llm_ms", 0.0),
        total_ms=t_elapsed,
    )


# ── 增强版问答接口（挂载在 query_router 下）──────────────────────────────────────
@query_router.post(
    "/v2",
    response_model=QueryResponseV2,
    summary="增强版智能问答（意图路由 + 质量评估）",
    description=(
        "增强版问答接口，支持：\n"
        "- **F1 引导**：空 query 返回友情引导菜单，不计入日志\n"
        "- **F2 路由**：意图识别（关键词 + qwen-max）→ 专项检索或模糊检索\n"
        "- **质量评估**：检索结果不足时触发转人工，并记录未命中日志\n"
        "- **票据上下文**：可选传入 bill_record_id，将已入库票据要素注入检索 query"
    ),
)
async def query_v2_endpoint(
    request: QueryRequestV2,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    增强版智能问答
    POST /api/v1/query/v2
    """
    allowed, rl_info = rate_limiter.check_rate_limit(current_user.tenant_id)  # 限流检查
    if not allowed:
        metrics.record_rate_limited(current_user.tenant_id)
        raise HTTPException(
            status_code=429,
            detail=f"请求频率超限，请 {rl_info['reset_in']}s 后重试",
            headers={"X-RateLimit-Reset": str(rl_info["reset_in"])},
        )

    # ── 可选：从数据库加载票据上下文（注入 bill_context）────────────────────────
    bill_context = None
    if request.bill_record_id:
        bill_record = await db.get(BillRecord, request.bill_record_id)
        # 仅允许访问本租户的票据主档，防止跨租户数据泄露
        if bill_record and bill_record.tenant_id == current_user.tenant_id:
            bill_context = {
                "ticket_number": bill_record.ticket_number,
                "ticket_type":   bill_record.ticket_type,
                "issue_date":    bill_record.issue_date,
                "due_date":      bill_record.due_date,
                "amount_numeric": bill_record.amount_numeric,
                "amount_text":   bill_record.amount_text,
                "drawer":        bill_record.drawer,
                "drawer_bank":   bill_record.drawer_bank,
                "acceptor":      bill_record.acceptor,
                "payee":         bill_record.payee,
            }

    logger.info(
        f"[query_v2] tenant={current_user.tenant_id} "
        f"query={request.query[:60]!r} ctx={'yes' if bill_context else 'no'}"
    )

    try:
        result = await rag_service.query_v2(
            tenant_id=current_user.tenant_id,
            query=request.query,
            top_k=request.top_k,
            user_id=current_user.user_id,
            bill_context=bill_context,
            db=db,
        )
        metrics.record_query(current_user.tenant_id, "success")

        # sources 是 dict 列表，转为 SourceChunk Pydantic 对象
        sources = [SourceChunk(**s) for s in result.get("sources", [])]
        return QueryResponseV2(
            query_id=result["query_id"],
            answer_type=result.get("answer_type", "answer"),
            answer=result["answer"],
            sources=sources,
            intent_id=result.get("intent_id"),
            route_type=result.get("route_type"),
            retrieval_quality=result.get("retrieval_quality"),
            transfer_to_human=result.get("transfer_to_human", False),
            contact_info=result.get("contact_info"),
            query_saved=result.get("query_saved", False),
            retrieval_ms=result.get("retrieval_ms", 0.0),
            llm_ms=result.get("llm_ms", 0.0),
            total_ms=result.get("total_ms", 0.0),
        )

    except Exception as e:
        metrics.record_query(current_user.tenant_id, "error")
        logger.error(f"[query_v2] 失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════════
# 监控路由（无前缀）
# ════════════════════════════════════════════════════════════════════════════════
metrics_router = APIRouter(tags=["监控"])


@metrics_router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus 指标端点
    GET /metrics
    Prometheus 服务器定期来抓取这个接口（默认每15秒一次）
    返回的是 Prometheus 文本格式，不是 JSON
    """
    data, content_type = metrics.generate_metrics()  # 生成 Prometheus 格式文本
    return Response(content=data, media_type=content_type)
    # 用 Response 而不是 JSONResponse，因为内容是特殊格式


@metrics_router.get("/health")
async def health_check():
    """
    系统健康检查接口（增强版）
    GET /health
    K8s / 负载均衡器定期调用，判断是否继续转发流量。
    检查所有关键依赖：Milvus、Redis、PostgreSQL、MCP Server、LangGraph 检查点。
    全部正常返回 HTTP 200，任意一个异常返回 HTTP 503。
    """
    checks = {
        "milvus":         _check_milvus(),            # 向量数据库连通性
        "redis":          _check_redis(),             # 缓存服务连通性
        "postgres":       await _check_postgres(),    # 关系型数据库连通性
        "mcp_server":     _check_mcp_tools(),         # MCP 工具是否已注册
        "langgraph":      _check_langgraph(),         # LangGraph 检查点是否已初始化
    }

    # 任意一个依赖不可用则整体标记为 degraded（K8s 可据此决定是否重启）
    all_ok = all(v == "ok" for v in checks.values())
    status = "healthy" if all_ok else "degraded"

    return JSONResponse(
        status_code=200 if all_ok else 503,           # 503 告知负载均衡停止转发
        content={
            "status":  status,
            "version": settings.APP_VERSION,
            "checks":  checks,
        }
    )


def _check_milvus() -> str:
    """检查 Milvus 是否可以连接，返回 "ok" 或 "unavailable" """
    try:
        from pymilvus import connections
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        return "ok"
    except Exception:
        return "unavailable"


def _check_redis() -> str:
    """检查 Redis 是否可以连接，返回 "ok" 或 "unavailable" """
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()         # ping：Redis 心跳命令，返回 PONG 表示正常
        return "ok"
    except Exception:
        return "unavailable"


async def _check_postgres() -> str:
    """检查 PostgreSQL 是否可以连接（异步），返回 "ok" 或 "unavailable" """
    try:
        from sqlalchemy import text
        from app.core.database import engine           # 模块级异步引擎（在 database.py 中创建）
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))       # 最轻量的心跳查询
        return "ok"
    except Exception:
        return "unavailable"


def _check_mcp_tools() -> str:
    """检查 MCP Server 是否已注册工具，返回注册数量或 "unavailable" """
    try:
        from app.mcp.server import mcp              # 获取全局 FastMCP 实例
        tool_count = len(list(mcp._tool_manager._tools))  # 读取已注册工具数量
        return "ok" if tool_count > 0 else "unavailable"
    except Exception:
        return "unavailable"


def _check_langgraph() -> str:
    """检查 LangGraph 检查点（PostgresSaver）是否已初始化，返回 "ok" 或 "unavailable" """
    try:
        from app.graph import get_checkpointer
        checkpointer = get_checkpointer()           # 获取全局 PostgresSaver 实例
        return "ok" if checkpointer is not None else "unavailable"
    except Exception:
        return "unavailable"
