# mcp_server_main.py
# MCP Server 独立部署入口
#
# 启动方式：
#   uvicorn mcp_server_main:app --host 0.0.0.0 --port 8001 --workers 2
#
# 职责：
#   - 仅挂载 /mcp 端点，不包含主应用的业务 API 路由
#   - 初始化 Agent 所需的资源：PostgreSQL、Redis、Milvus、Jieba
#   - 注册所有 @mcp.tool() 工具（通过导入 app.mcp.server 触发）
#
# 与主应用的关系：
#   - 主应用（main.py）通过 MCPClient 发 HTTP 请求到本服务
#   - 本服务执行工具函数（实例化 Agent、写库、读 Redis shared_data）
#   - 两个服务共享同一 PostgreSQL 和 Redis，但进程隔离

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from config.settings import settings
from app.core.logging import setup_logging

setup_logging()

from app.core.database import init_db
from app.services.embedding import init_jieba
from app.services.vector_store import connect_milvus, get_or_create_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 MCP Server 启动中 url={settings.MCP_SERVER_URL}")

    # 初始化 jieba（合规检索等 Agent 可能用到中文分词）
    logger.info("初始化 jieba 票据领域词典...")
    init_jieba()

    # 初始化 PostgreSQL（Agent 写库依赖）
    logger.info("初始化数据库连接...")
    await init_db()

    # 初始化 Milvus（ComplianceRetrievalAgent 使用向量检索）
    logger.info("连接 Milvus 向量库...")
    if connect_milvus():
        get_or_create_collection()
    else:
        logger.warning("⚠️ Milvus 未连接，合规向量检索不可用")

    # 预连接 Redis shared_data 缓存层
    logger.info("预连接 shared_data Redis 缓存...")
    from app.core.shared_data_cache import _get_redis
    await _get_redis()
    logger.info(f"shared_data Redis 已就绪 url={settings.REDIS_URL}")

    # 注册所有 MCP 工具（导入 server.py 触发所有 @mcp.tool() 装饰器）
    logger.info("注册 MCP 工具...")
    from app.mcp.server import mcp as _mcp
    tool_count = len(list(_mcp._tool_manager._tools))
    logger.info(f"✅ MCP Server 启动完成，已注册工具：{tool_count} 个")

    # 启动 StreamableHTTPSessionManager（FastAPI mount 不会触发子应用 lifespan，须手动调用）
    async with _mcp.session_manager.run():
        yield

    logger.info("🛑 MCP Server 关闭")
    from app.core.cache import bill_cache
    await bill_cache.close()


app = FastAPI(
    title="票据审核 MCP Server",
    version=settings.APP_VERSION,
    description="票据多智能体工具服务，供主应用通过 MCP 协议调用",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# 挂载 MCP ASGI 子应用（Streamable HTTP 传输模式）
# streamable_http_app() 内部已注册 /mcp 路由，挂载到根路径避免路径双重前缀（/mcp/mcp）
from app.mcp.server import get_mcp_asgi_app
app.mount("/", get_mcp_asgi_app())