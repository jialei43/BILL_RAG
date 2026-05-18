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
    BackgroundTasks,     # 后台任务（不阻塞当前请求的异步处理）
    Query,               # URL 查询参数（如 ?page=1&size=20）
    status               # HTTP 状态码常量
)
from fastapi.security import OAuth2PasswordRequestForm  # 兼容 Swagger UI 的 OAuth2 表单登录
from fastapi import Request as FastAPIRequest           # 用于读取原始请求体
from fastapi.responses import StreamingResponse, Response
# StreamingResponse：流式响应（用于 SSE 实时推送）
# Response：通用响应（可自定义 Content-Type）

from sqlalchemy.ext.asyncio import AsyncSession   # 异步数据库会话
from sqlalchemy import select, func               # select：构建查询；func：SQL 函数（COUNT、SUM）
from loguru import logger                         # 日志

from config.settings import settings              # 配置
from app.core.auth import (
    get_current_user,        # 依赖：验证 Token，返回当前用户
    get_admin_user,          # 依赖：验证 Token 且必须是管理员
    TokenData,               # Token 数据结构
    verify_password,         # 密码验证函数
    get_password_hash,       # 密码哈希函数
    create_access_token,     # 生成 JWT Token
)
from app.core.database import get_db              # 依赖：获取数据库会话
from app.models.db_models import (
    Tenant, User, Document, DocumentChunk, QueryLog, DocumentStatus,
    BillRecord, BillVersion,  # 票据生命周期模型
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
        "is_admin": user.is_admin,
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
    注册新用户（需要已登录，只能在自己的租户下注册）
    POST /api/v1/auth/register
    status_code=201：HTTP 201 Created（资源创建成功）
    """
    # 检查用户名是否已存在
    existing = await db.execute(select(User).where(User.username == request.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="用户名已存在")
        # 409 Conflict：资源冲突（已存在同名资源）

    # 创建新用户（归属于当前登录用户的租户）
    user = User(
        id=str(uuid.uuid4()),
        tenant_id=current_user.tenant_id,              # 新用户自动归属于当前租户
        username=request.username,
        email=request.email,
        hashed_password=get_password_hash(request.password),  # 哈希后再存储
        is_admin=request.is_admin and current_user.is_admin,
        # 只有当前用户是管理员时，才能创建管理员（防止权限提升）
    )
    db.add(user)        # 加入数据库会话（未提交）
    return user         # FastAPI 自动调用 db.commit()（通过 get_db 依赖）


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


async def _update_doc_status(
    doc_id: str,
    status,
    result: dict = None,
    error: str = None,
):
    """更新数据库中文档的处理状态，在主事件循环内执行，避免跨循环操作 asyncpg 连接。"""
    from app.core.database import AsyncSessionLocal
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


async def _run_ingestion(
    file_path: str,
    tenant_id: str,
    document_id: str,
):
    """
    后台任务：执行文档入库（解析→分块→向量化→存储）

    改为 async def，由 FastAPI 在主事件循环中调度：
    - 同步的 ingestion_pipeline.ingest() 通过 run_in_executor 放到线程池执行，不阻塞事件循环
    - 数据库状态更新留在主事件循环，asyncpg 连接池不会出现跨循环错误
    """
    import asyncio
    loop = asyncio.get_event_loop()

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


async def _ingest_one(
    file: UploadFile,
    tenant_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
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

    background_tasks.add_task(_run_ingestion, file_path, tenant_id, doc.id)  # 注册后台入库任务
    return doc


@docs_router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,              # FastAPI 后台任务队列
    file: UploadFile = File(...),                   # 单文件上传，Swagger 可直接测试
    current_user: TokenData = Depends(get_current_user),
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

    logger.info(f"[upload] 单文件上传 tenant={current_user.tenant_id} file={file.filename}")  # 入口日志
    doc = await _ingest_one(file, current_user.tenant_id, background_tasks, db)
    return doc


@docs_router.post("/upload/batch", response_model=list[DocumentResponse], status_code=202)
async def upload_documents_batch(
    background_tasks: BackgroundTasks,              # FastAPI 后台任务队列
    files: list[UploadFile] = File(...),            # 多文件，前端 form-data 用同名字段 files
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    批量文档上传接口（异步入库）
    POST /api/v1/documents/upload/batch
    所有文件整体校验通过后才入库，有一个不合法则全部拒绝
    每个文件独立注册后台任务，并发入库互不阻塞

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

    logger.info(  # 批量上传入口日志
        f"[upload_batch] tenant={current_user.tenant_id} "
        f"count={len(files)} files={[f.filename for f in files]}"
    )

    docs = []
    for file in files:
        doc = await _ingest_one(file, current_user.tenant_id, background_tasks, db)  # 复用单文件逻辑
        docs.append(doc)

    logger.info(f"[upload_batch] 登记完成 ids={[d.id for d in docs]}")  # 全部写入事务
    return docs


async def _ingest_from_path(
    file_path: Path,
    tenant_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
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

    background_tasks.add_task(_run_ingestion, str(file_path), tenant_id, doc.id)  # 注册后台任务
    return doc


@docs_router.post("/upload/folder", response_model=list[DocumentResponse], status_code=202)
async def upload_folder(
    background_tasks: BackgroundTasks,
    request: FolderUploadRequest,          # JSON body：{"folder_path": "...", "recursive": false}
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    文件夹批量入库接口
    POST /api/v1/documents/upload/folder
    传入服务器上已存在的文件夹路径，自动扫描所有支持格式的文件并异步入库。
    文件不会被复制，Document.file_path 直接指向原始路径。
    recursive=true 时递归扫描所有子目录。

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

    logger.info(  # 文件夹入库入口日志
        f"[upload_folder] tenant={current_user.tenant_id} "
        f"folder={request.folder_path} matched={len(matched)} recursive={request.recursive}"
    )

    docs = []
    for file_path in matched:
        doc = await _ingest_from_path(file_path, current_user.tenant_id, background_tasks, db)
        docs.append(doc)

    logger.info(  # 登记完成日志
        f"[upload_folder] 登记完成 count={len(docs)} "
        f"ids={[d.id for d in docs]}"
    )
    return docs


@docs_router.get("/", response_model=PaginatedResponse)
async def list_documents(
    page: int = Query(default=1, ge=1),                           # 页码（从1开始）
    page_size: int = Query(default=20, ge=1, le=100),             # 每页条数（最多100）
    status_filter: Optional[str] = Query(default=None),           # 可选：按状态过滤
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    获取当前租户的文档列表（支持分页和状态过滤）
    GET /api/v1/documents/?page=1&page_size=20&status_filter=completed
    """
    # 构建基础查询（只查当前租户的文档）
    query = select(Document).where(Document.tenant_id == current_user.tenant_id)

    # 如果有状态过滤，添加过滤条件
    if status_filter:
        query = query.where(Document.status == status_filter)

    # 查询总记录数（用于计算总页数）
    count_q = select(func.count()).select_from(
        select(Document).where(Document.tenant_id == current_user.tenant_id).subquery()
        # subquery()：把查询作为子查询，再在外层做 COUNT
    )
    total = (await db.execute(count_q)).scalar()  # scalar()：取查询结果的第一个值

    # 分页查询（按上传时间倒序，最新的在前面）
    query = query.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    # offset：跳过前 N 条（如第2页跳过前20条）；limit：最多返回 N 条
    docs = (await db.execute(query)).scalars().all()   # all()：取所有结果

    return PaginatedResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],  # 转换为 Schema 对象
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,  # 向上取整：ceil(total/page_size)
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
    doc = await db.get(Document, document_id)   # 按主键查询（最高效）

    # 文档不存在 或 不属于当前租户（防止A租户查B租户的文档）
    if not doc or doc.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    return doc


@docs_router.post("/{document_id}/retry", response_model=DocumentResponse, status_code=202)
async def retry_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user),
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

    # 文档不存在或不属于当前租户
    if not doc or doc.tenant_id != current_user.tenant_id:
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
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    删除文档（同时删除：Milvus向量 + 磁盘文件 + 数据库记录）
    DELETE /api/v1/documents/{document_id}
    """
    doc = await db.get(Document, document_id)
    if not doc or doc.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 第一步：从 Milvus 删除该文档的所有向量
    ingestion_pipeline.delete_document(current_user.tenant_id, document_id)

    # 第二步：删除磁盘上的文件
    if doc.file_path and os.path.exists(doc.file_path):
        os.remove(doc.file_path)   # 删除原始文件

    # 第三步：从数据库删除文档记录（关联的 chunks 记录由数据库外键级联删除）
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

        # 记录本次查询到数据库（用于分析和反馈）
        log = QueryLog(
            id=result["query_id"],
            tenant_id=current_user.tenant_id,
            user_id=current_user.user_id,
            query=request.query,
            answer=result["answer"],
            retrieved_chunks=result["sources"],   # 引用的文档片段（JSON 格式存储）
            retrieval_ms=result["retrieval_ms"],
            llm_ms=result["llm_ms"],
            total_ms=result["total_ms"],
        )
        db.add(log)   # 添加到数据库（自动提交）

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
    # 检查机构代码是否已存在（机构代码全系统唯一）
    existing = await db.execute(select(Tenant).where(Tenant.code == request.code))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="机构代码已存在")

    tenant = Tenant(
        id=str(uuid.uuid4()),
        name=request.name,
        code=request.code,
        license_no=request.license_no,
        doc_quota=request.doc_quota,
        qps_limit=request.qps_limit,
        milvus_partition=f"tenant_{request.code}",  # 自动生成 Milvus 分区名（如 tenant_ABC_BANK）
    )
    db.add(tenant)
    return tenant


@tenant_router.get("/", response_model=list[TenantResponse])
async def list_tenants(
    _: TokenData = Depends(get_admin_user),   # 仅管理员
    db: AsyncSession = Depends(get_db),
):
    """
    获取所有租户列表（仅管理员）
    GET /api/v1/tenants/
    """
    result = await db.execute(select(Tenant).order_by(Tenant.created_at.desc()))
    # 按创建时间倒序，最新注册的租户排在前面
    return result.scalars().all()   # 返回所有租户对象


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
    # 权限检查：非管理员只能查看自己的租户
    if not current_user.is_admin and current_user.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="无权访问")

    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="租户不存在")

    # 从 Redis 读取文档计数（比查数据库快）
    doc_count = rate_limiter.get_doc_count(tenant_id)

    # 查询该租户所有文档的 chunk 总数
    chunk_count_q = await db.execute(
        select(func.sum(Document.chunk_count)).where(Document.tenant_id == tenant_id)
        # func.sum：SQL 的 SUM 函数，统计所有文档的 chunk_count 之和
    )
    chunk_count = chunk_count_q.scalar() or 0   # 如果没有文档，SUM 返回 None，用 or 0 转为 0

    # 查询今天的查询次数
    today_query_q = await db.execute(
        select(func.count(QueryLog.id)).where(
            QueryLog.tenant_id == tenant_id,
            func.date(QueryLog.created_at) == func.current_date()
            # func.date()：提取日期部分；func.current_date()：今天的日期
        )
    )
    today_queries = today_query_q.scalar() or 0

    return TenantStats(
        tenant_id=tenant_id,
        doc_count=doc_count,
        chunk_count=chunk_count,
        query_count_today=today_queries,
        doc_quota=tenant.doc_quota,
        qps_limit=tenant.qps_limit,
        quota_used_pct=round(doc_count / tenant.doc_quota * 100, 1),
        # 配额使用百分比，保留1位小数（如 45.3）
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
    系统健康检查接口
    GET /health
    监控系统（如 K8s、负载均衡器）定期调用，返回"healthy"就继续转发流量
    同时检查依赖服务（Milvus、Redis）的状态
    """
    return {
        "status": "healthy",              # 应用本身正常运行
        "version": settings.APP_VERSION,  # 当前版本
        "services": {
            "milvus": _check_milvus(),    # Milvus 连通性
            "redis": _check_redis(),      # Redis 连通性
        }
    }


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
        r.ping()         # ping：Redis 的心跳命令，返回 PONG 表示正常
        return "ok"
    except Exception:
        return "unavailable"
