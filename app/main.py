# app/main.py
# 这是整个 Web 服务的"大门"——FastAPI 应用的主入口文件
# 负责：启动/关闭初始化、注册路由、添加中间件、处理全局异常

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os                                # Python 标准库，用于设置进程级环境变量
import time                              # Python 标准库，用于计时（记录每个请求的处理时长）
import uuid                              # 生成唯一请求 ID，用于全链路日志追踪
from contextlib import asynccontextmanager  # 用于定义"异步上下文管理器"，实现启动/关闭钩子

# 必须在任何 tokenizer 导入之前设置，防止 uvicorn 多进程 fork 后 HuggingFace
# tokenizers 检测到"fork 后使用了并行 tokenizer"而打印 TOKENIZERS_PARALLELISM 警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, Request        # FastAPI：Web 框架主类；Request：代表一个 HTTP 请求
from fastapi.middleware.cors import CORSMiddleware   # CORS 中间件：允许跨域请求（前端调用后端时需要）
from fastapi.middleware.gzip import GZipMiddleware   # GZip 中间件：自动压缩响应体，节省带宽
from fastapi.responses import JSONResponse           # 用于返回 JSON 格式的响应
from fastapi.staticfiles import StaticFiles          # 静态文件服务：将前端 HTML/CSS/JS 托管到 API 服务器
from loguru import logger                            # loguru：更好用的日志库，比 print 更专业

from config.settings import settings                 # 导入全局配置对象
from app.core.logging import setup_logging, contextualize  # 生产级日志配置
from app.core.tracing import trace_id_middleware     # Trace ID 中间件（多容器可观测性）

# 日志系统必须在所有模块导入之前初始化，确保第三方库的 stdlib logging 也被拦截
setup_logging()
from app.core.database import init_db                # 导入数据库初始化函数
from app.services.embedding import init_jieba        # 导入 jieba 分词初始化函数
from app.services.vector_store import connect_milvus, get_or_create_collection  # Milvus 连接和集合管理
from app.api.routers import (
    auth_router, docs_router, query_router,   # 各功能模块的路由（认证、文档、问答）
    tenant_router, metrics_router,            # 租户管理、监控指标路由
    bill_router,                              # 票据要素识别路由（独立接口，不写库）
    bills_router,                             # 票据生命周期路由（入库+流转+查询）
)
from app.core.cache import bill_cache  # 业务缓存实例（lifespan 关闭时断开连接）
# 多智能体系统路由（M9~M10 阶段新增）
from app.api.audit_router import audit_router       # 审核任务路由：提交/查询/报告下载
from app.api.batch_router import batch_router       # 批量审核路由：批次提交/进度查询/子项列表
from app.api.tracking_router import tracking_router  # 流转追踪路由：创建追踪/查询状态/报文详情


# ── 启动 / 关闭生命周期管理 ────────────────────────────────────────────────────
@asynccontextmanager                       # 装饰器：把下面的函数变成"异步上下文管理器"
async def lifespan(app: FastAPI):
    # yield 之前的代码：应用启动时执行（相当于"开门前的准备工作"）
    logger.info(f"🚀 启动 {settings.APP_NAME} v{settings.APP_VERSION}")

    # 第一步：初始化 jieba 分词的票据行业词典
    # jieba 默认不认识"承兑汇票"、"贴现"等专业词，加载词典后才能正确分词
    logger.info("初始化 jieba 票据领域词典...")
    init_jieba()

    # 第二步：连接 PostgreSQL 数据库并创建表结构
    # 如果数据库中还没有 users、documents 等表，这里会自动建表
    logger.info("初始化数据库...")
    await init_db()

    # 第三步：连接 Milvus 向量数据库
    # Milvus 用于存储文档的向量表示，支持语义相似度搜索
    logger.info("连接 Milvus 向量库...")
    if connect_milvus():                          # 如果连接成功
        get_or_create_collection()                # 确保向量集合（相当于"表"）存在
    else:
        logger.warning("⚠️ Milvus 未连接，向量检索不可用")  # 连接失败时发出警告，系统仍可运行（只是无法检索）

    # 第四步：初始化 LangGraph PostgresSaver（创建检查点表结构）
    # 多容器部署时，每个容器共享同一 PostgreSQL 的 checkpoints 表
    logger.info("初始化 LangGraph PostgresSaver（检查点持久化）...")
    try:
        from app.graph import setup_checkpointer
        await setup_checkpointer()                     # 幂等操作：表已存在时跳过建表
    except Exception as e:
        logger.warning(f"⚠️ LangGraph 检查点初始化失败（不影响服务启动）: {e}")

    # 第五步：初始化 MCP Server（注册所有工具）
    # 导入 server.py 触发所有 @mcp.tool() 装饰器的工具注册
    logger.info("初始化 MCP Server（注册工具）...")
    try:
        from app.mcp.server import mcp as _mcp        # 触发所有工具的注册（副作用导入）
        logger.info(f"MCP Server 已就绪，共注册工具：{len(list(_mcp._tool_manager._tools))} 个")
    except Exception as e:
        logger.warning(f"⚠️ MCP Server 初始化失败（不影响其他服务）: {e}")

    # 第六步：预热业务缓存 Redis 连接（懒加载，首次 get/set 时才真正建连接）
    # 这里只记录日志，实际连接在第一次缓存操作时建立
    from config.settings import settings as _s
    logger.info(f"业务缓存已配置 redis={_s.REDIS_URL} enabled={_s.CACHE_ENABLED}")

    logger.info("✅ 系统启动完成")
    yield   # 程序运行阶段：yield 之后暂停，等待应用正常运行……

    # yield 之后的代码：应用关闭时执行（相当于"关门前的收尾工作"）
    logger.info("🛑 系统关闭")
    # 关闭 LangGraph PostgresSaver 连接池（释放 psycopg3 连接，避免连接泄漏）
    try:
        from app.graph import teardown_checkpointer
        await teardown_checkpointer()
    except Exception as e:
        logger.warning(f"⚠️ LangGraph 连接池关闭失败: {e}")
    # 关闭业务缓存 Redis 连接（释放连接池资源，避免 TCP 连接泄漏）
    await bill_cache.close()


# ── 创建 FastAPI 应用实例 ─────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,      # API 文档的标题
    version=settings.APP_VERSION, # API 版本号
    description="""
## 票据业务智能顾问 RAG 系统

面向持牌票据经纪机构及银行票据部门的企业级智能问答与分析平台。

### 核心能力
- 🔍 **混合检索**：BGE-M3 稠密向量 + 稀疏语义 + BM25 三路融合 + Reranker 精排
- 📄 **多模态解析**：PDF/Word/Excel/图像，含扫描件 OCR（倾斜矫正）
- 🏢 **多租户隔离**：Milvus partition_key 物理隔离，零跨租户数据泄露
- ⚡ **高性能**：单文档入库 P95 < 45s，支持 20+ 机构并发
- 📊 **可观测性**：Prometheus + Grafana 全链路监控
    """,             # 显示在 /api/docs 页面的详细说明
    lifespan=lifespan,            # 指定上面定义的启动/关闭函数
    docs_url="/api/docs",         # Swagger UI 文档地址（可以在浏览器中直接测试 API）
    redoc_url="/api/redoc",       # ReDoc 格式文档地址（另一种 API 文档样式）
    openapi_url="/api/openapi.json",  # OpenAPI 规范 JSON 文件地址（供工具导入使用）
)

# ── 中间件配置 ────────────────────────────────────────────────────────────────
# 中间件：每个请求进来和出去时都会经过的"拦截器"

# 添加 CORS 跨域中间件
# 作用：允许其他域名（如前端网页）调用本 API，否则浏览器会拒绝跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有域名访问（生产环境建议限制为特定域名）
    allow_credentials=True,       # 允许携带 Cookie 和认证头
    allow_methods=["*"],          # 允许所有 HTTP 方法（GET、POST、DELETE 等）
    allow_headers=["*"],          # 允许所有请求头
)

# 添加 GZip 压缩中间件
# 作用：当响应体超过 1000 字节时，自动用 GZip 压缩，减少网络传输量
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Trace ID 中间件：为每个请求绑定追踪 ID，写入响应头（多容器日志聚合关键）
# 支持外部传入 X-Trace-ID（API 网关/上游服务透传）或自动生成
app.middleware("http")(trace_id_middleware)


# 自定义中间件：给每个响应添加"处理时间"响应头
@app.middleware("http")                    # 装饰器：注册为 HTTP 中间件
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()            # 记录请求开始时间（高精度计时器）
    response = await call_next(request)   # 调用下一个处理函数（实际的路由处理）
    elapsed = (time.perf_counter() - start) * 1000  # 计算耗时，转换为毫秒
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"  # 把耗时写入响应头
    return response                        # 返回带有耗时头的响应


# 自定义中间件：为每个请求生成唯一 ID 并注入日志上下文
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]   # 12位十六进制，兼顾唯一性与可读性
    async with contextualize(request_id=request_id):
        logger.debug(f"{request.method} {request.url.path}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id  # 透传给客户端，便于问题定位
        return response


# ── 全局异常处理 ──────────────────────────────────────────────────────────────
# 当任何地方抛出未捕获的异常时，统一在这里处理，避免把错误堆栈直接暴露给用户
@app.exception_handler(Exception)         # 捕获所有 Exception 类型的异常
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {type(exc).__name__}")  # 在日志中记录完整错误信息
    return JSONResponse(
        status_code=500,                  # HTTP 500 = 服务器内部错误
        content={"success": False, "error": "内部服务器错误", "detail": str(exc)},
        # 返回统一格式的错误响应，不暴露内部堆栈信息
    )


# ── 注册路由 ──────────────────────────────────────────────────────────────────
# 把各个模块的路由挂载到统一的 URL 前缀下
API_PREFIX = "/api/v1"                    # 所有 API 的公共前缀（版本号方便未来升级为 v2）

app.include_router(auth_router, prefix=API_PREFIX)    # 认证相关接口：/api/v1/auth/...
app.include_router(docs_router, prefix=API_PREFIX)    # 文档管理接口：/api/v1/documents/...
app.include_router(query_router, prefix=API_PREFIX)   # 智能问答接口：/api/v1/query/...
app.include_router(tenant_router, prefix=API_PREFIX)  # 租户管理接口：/api/v1/tenants/...
app.include_router(bill_router, prefix=API_PREFIX)    # 票据识别接口：/api/v1/bill-recognition/...
app.include_router(bills_router, prefix=API_PREFIX)   # 票据生命周期接口：/api/v1/bills/...
app.include_router(metrics_router)                    # 监控指标接口：/metrics、/health（无前缀）

# 多智能体系统路由（M9~M10 阶段新增）
app.include_router(audit_router,    prefix=API_PREFIX)    # 审核任务接口：/api/v1/audit/...
app.include_router(batch_router,    prefix=API_PREFIX)    # 批量任务接口：/api/v1/batch/...
app.include_router(tracking_router, prefix=API_PREFIX)    # 流转追踪接口：/api/v1/tracking/...

# MCP Server 挂载：将 MCP 作为 ASGI 子应用挂载到 /mcp 路径
# POST /mcp   → MCP 协议请求（initialize / list_tools / call_tool）
# GET  /mcp/sse → SSE 事件流（Streamable HTTP 模式）
# 外部 LLM（Claude API / Claude Desktop）通过此端点调用票据审核工具
try:
    from app.mcp.server import get_mcp_asgi_app
    app.mount("/mcp", get_mcp_asgi_app())              # 挂载 MCP Streamable HTTP ASGI 应用
    logger.info("MCP Server 已挂载到 /mcp")
except Exception as e:
    logger.warning(f"⚠️ MCP Server 挂载失败: {e}")

# 前端静态文件托管：将 frontend/ 目录挂载到 /frontend 路径
# 访问 http://localhost:8000/frontend/index.html 即可打开登录页
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# 根路径接口：访问 http://localhost:8000/ 时返回的欢迎信息
@app.get("/", tags=["Root"])
async def root():
    return {
        "app": settings.APP_NAME,         # 应用名称
        "version": settings.APP_VERSION,  # 版本号
        "docs": "/api/docs",              # 告知用户 API 文档在哪里
        "health": "/health",              # 告知用户健康检查接口在哪里
    }


# ── 直接运行入口 ──────────────────────────────────────────────────────────────
# 当执行 `python app/main.py` 时才运行下面的代码（通过 uvicorn 命令启动时不会执行）
if __name__ == "__main__":
    import uvicorn                        # uvicorn 是运行 FastAPI 的 ASGI 服务器
    uvicorn.run(
        "app.main:app",                   # 指定要运行的 FastAPI 应用（模块路径:变量名）
        host="0.0.0.0",                   # 监听所有网卡（0.0.0.0 表示外网也能访问）
        port=8002,                        # 监听 8000 端口
        reload=settings.DEBUG,            # 调试模式下开启热重载（改代码自动重启，不用手动）
        workers=1 if settings.DEBUG else 4,  # 调试时 1 个进程，生产时 4 个进程并发处理请求
        log_level="debug" if settings.DEBUG else "info",  # 日志详细程度
    )
