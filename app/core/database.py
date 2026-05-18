# app/core/database.py
# 数据库连接与 Session（会话）管理
# 这个文件解决一个核心问题：如何让整个程序安全、高效地访问 PostgreSQL 数据库

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# create_async_engine：创建"异步数据库引擎"（异步=等数据库响应时不阻塞其他请求）
# AsyncSession：异步数据库会话，每次和数据库交互都需要一个会话
# async_sessionmaker：会话工厂，用于批量创建会话对象

from sqlalchemy import event   # event 用于监听数据库事件（这里未直接使用，但保留以备将来扩展）
from loguru import logger      # 日志库

from config.settings import settings         # 导入配置（获取 DATABASE_URL）
from app.models.db_models import Base        # 导入所有数据库模型的基类（用于自动建表）


# 创建异步数据库引擎（整个应用只创建一次，全局复用）
engine = create_async_engine(
    settings.DATABASE_URL,    # 数据库连接地址（从配置读取）
    pool_pre_ping=True,       # 每次使用连接前先 ping 一下，自动丢弃已断开的连接
    pool_size=20,             # 连接池大小：最多保持 20 个数据库连接（避免频繁建立/断开）
    max_overflow=10,          # 连接池满了还可以临时额外创建 10 个连接
    pool_recycle=1800,        # 连接超过 30 分钟强制回收，防止协议状态机错乱 (asyncpg InternalClientError)
    echo=settings.DEBUG,      # 调试模式下打印所有 SQL 语句，方便排查问题
)

# 创建"会话工厂"（用 AsyncSessionLocal() 就能得到一个数据库会话）
AsyncSessionLocal = async_sessionmaker(
    engine,                   # 绑定到上面创建的引擎
    class_=AsyncSession,      # 使用异步会话类
    expire_on_commit=False,   # 提交后不让对象"过期"，可以继续访问对象属性
    autoflush=False,          # 不自动刷新（手动控制何时写入数据库，更安全）
    autocommit=False,         # 不自动提交（需要显式调用 commit()，出错可以回滚）
)


async def get_db() -> AsyncSession:
    """
    FastAPI 依赖注入函数：为每个 HTTP 请求提供一个独立的数据库会话
    使用方式：在路由函数参数中写 db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:  # 创建一个新的数据库会话
        try:
            yield session                       # 把会话"借给"路由函数使用
            await session.commit()             # 路由函数执行完毕后，提交所有数据库操作
        except Exception as e:
            await session.rollback()           # 如果出错，回滚所有未提交的操作（保证数据一致性）
            logger.warning(  # 回滚日志：记录异常类型，方便定位哪个接口触发了事务回滚
                f"[database] 事务回滚 error={type(e).__name__}: {e}"
            )
            raise                              # 把异常继续向上抛出，让全局异常处理器处理
        finally:
            await session.close()             # 无论成功还是失败，最后都关闭会话、释放连接回连接池


async def init_db():
    """
    初始化数据库表结构
    应用启动时调用，会自动检查并创建所有在 db_models.py 中定义的表
    如果表已存在则跳过（不会清空数据）
    """
    logger.info(f"[database] 初始化数据库表 url={settings.DATABASE_URL.split('@')[-1]}")  # 只打印主机+库名，不暴露密码
    try:
        async with engine.begin() as conn:                  # 开启一个数据库事务
            await conn.run_sync(Base.metadata.create_all)  # 同步执行"创建所有表"操作
            # run_sync：SQLAlchemy 的某些操作不支持异步，用 run_sync 包装后可以在异步环境中执行
        logger.info("[database] 数据库表初始化成功")         # 确认所有表已就绪
    except Exception as e:
        logger.error(f"[database] 数据库表初始化失败: {e}", exc_info=True)  # 含堆栈，方便定位连接/权限问题
        raise  # 初始化失败属于启动致命错误，继续向上抛出让程序停止
