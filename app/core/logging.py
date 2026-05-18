# app/core/logging.py
# 生产级日志配置，基于 loguru
#
# 三路输出：
#   stderr   — 彩色人类可读格式（开发/容器日志收集）
#   app.log  — INFO+ JSON 结构化，按大小轮转 + gzip 压缩
#   error.log — ERROR+ JSON，带完整回溯，专供告警系统消费
#
# 额外特性：
#   - 通过 ContextVar 为每个请求注入 request_id，JSON 日志自动携带
#   - InterceptHandler 将 uvicorn / sqlalchemy 的 stdlib logging 重定向到 loguru
#   - 生产环境关闭 diagnose，防止敏感变量值泄露到日志

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from loguru import logger

from config.settings import settings


# ── stdlib logging → loguru 拦截器 ────────────────────────────────────────────
class _InterceptHandler(logging.Handler):
    """将标准库 logging 的所有输出转发给 loguru，保留原始调用位置。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到真实调用帧，使 loguru 显示正确的 file:line
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# ── 控制台格式（彩色） ─────────────────────────────────────────────────────────
_CONSOLE_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{message}"
)


# ── 公共接口：在请求中间件中调用 ──────────────────────────────────────────────
@asynccontextmanager
async def contextualize(**kwargs):
    """异步上下文管理器，将字段注入同一协程的 loguru 日志上下文。

    loguru.contextualize() 返回的是同步 _GeneratorContextManager，
    直接用 async with 会报 TypeError。这里用 @asynccontextmanager
    包一层，使其可跨 anyio 任务组边界正常工作。

    用法：
        async with contextualize(request_id="abc123"):
            logger.info("处理请求")   # JSON 中自动出现 request_id
    """
    with logger.contextualize(**kwargs):
        yield


# ── 核心初始化函数 ─────────────────────────────────────────────────────────────
def setup_logging() -> None:
    """配置 loguru，应在应用启动最早期调用一次（main.py 模块级别）。"""

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 移除 loguru 默认的 stderr handler，避免重复输出
    logger.remove()

    level = settings.LOG_LEVEL

    # ── Sink 1: stderr 彩色控制台 ──────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        format=_CONSOLE_FMT,
        colorize=True,
        enqueue=True,           # 多进程/线程安全
    )

    # ── Sink 2: app.log — INFO+ JSON，带自动轮转、保留、压缩 ──────────────
    logger.add(
        str(log_dir / "app.log"),
        level=level,
        rotation=settings.LOG_ROTATION,    # 超过 50 MB 自动切分
        retention=settings.LOG_RETENTION,  # 30 天后自动删除旧文件
        compression=settings.LOG_COMPRESSION,  # 切分后 gzip 压缩
        enqueue=True,
        serialize=True,         # 输出 JSON，便于 ELK / Loki 等聚合平台消费
        backtrace=True,         # 异常时输出完整调用链
        diagnose=settings.DEBUG,  # 仅调试模式显示局部变量，防止生产泄露敏感值
    )

    # ── Sink 3: error.log — ERROR+ 专用，告警系统消费 ────────────────────
    logger.add(
        str(log_dir / "error.log"),
        level="ERROR",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression=settings.LOG_COMPRESSION,
        enqueue=True,
        serialize=True,
        backtrace=True,
        diagnose=settings.DEBUG,
    )

    # ── 拦截 stdlib logging（uvicorn / sqlalchemy / fastapi 内部日志）────
    _intercept_stdlib_loggers()

    logger.info(
        f"日志系统初始化完成: level={level} "
        f"dir={log_dir} rotation={settings.LOG_ROTATION} "
        f"retention={settings.LOG_RETENTION}"
    )


def _intercept_stdlib_loggers() -> None:
    """将常见第三方库的 stdlib logging 重定向到 loguru。"""
    handler = _InterceptHandler()
    # 根级别拦截，确保没有漏网之鱼
    logging.basicConfig(handlers=[handler], level=0, force=True)

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
    ):
        log = logging.getLogger(name)
        log.handlers = [handler]
        log.propagate = False
