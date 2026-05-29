# config/settings.py
# 这个文件负责管理整个项目的所有配置项（相当于项目的"总控制面板"）
# 所有配置都可以通过 .env 文件或环境变量来覆盖，不需要改源代码

from pydantic_settings import BaseSettings  # 从 pydantic 库导入"基础设置类"，它能自动从环境变量读取配置
from pydantic import Field                   # Field 用于给配置项设置默认值、验证规则等
from typing import Optional                  # Optional 表示这个配置项"可以为空"（即不是必填的）
import os                                    # Python 标准库，用于操作系统相关功能（这里备用，未直接使用）


class Settings(BaseSettings):
    # 定义 Settings 类，继承自 BaseSettings
    # 好处：程序启动时会自动读取 .env 文件和环境变量，非常方便

    # ── App 基础信息 ──────────────────────────────────────────────────────────
    APP_NAME: str = "票据业务智能顾问 RAG 系统"   # 应用名称，显示在 API 文档首页
    APP_VERSION: str = "1.0.0"                    # 版本号
    DEBUG: bool = False                            # 调试模式开关：True=开发模式（详细日志），False=生产模式
    SECRET_KEY: str = Field(default="change-me-in-production-super-secret-key")
    # SECRET_KEY：JWT Token 的加密密钥，生产环境必须改成随机长字符串，否则有安全风险
    ALGORITHM: str = "HS256"                       # JWT 加密算法：HS256 是最常用的对称加密算法
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24    # Token 有效期：60分钟×24=1天，过期后需重新登录

    # ── 数据库配置 ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/bill_rag"
        # 格式：协议+驱动://用户名:密码@主机:端口/数据库名
        # postgresql：数据库类型（PostgreSQL）
        # asyncpg：异步驱动（让程序在等数据库时不阻塞其他请求）
    )

    # ── Milvus 向量数据库配置 ──────────────────────────────────────────────────
    # Milvus 是专门存储"向量"的数据库，用于相似度搜索（找到语义相近的文档片段）
    MILVUS_HOST: str = "localhost"                 # Milvus 服务器地址
    MILVUS_PORT: int = 19530                       # Milvus 默认端口
    MILVUS_COLLECTION: str = "bill_documents"      # 集合名（类似关系数据库中的"表名"）
    MILVUS_DENSE_DIM: int = 1024                   # 稠密向量的维度：BGE-M3 模型输出 1024 个数字表示一段文本
    MILVUS_INDEX_TYPE: str = "IVF_FLAT"            # 索引类型：IVF_FLAT 是倒排文件索引，检索速度快
    MILVUS_METRIC_TYPE: str = "IP"                 # 相似度计算方式：IP=内积（两向量越相似，内积越大）
    MILVUS_NLIST: int = 128                        # IVF 索引的聚类数量，影响检索速度和精度的平衡

    # ── Redis 缓存配置 ─────────────────────────────────────────────────────────
    # Redis 是内存数据库，用于：限流计数、BM25索引缓存、文档计数、业务结果缓存
    REDIS_URL: str = "redis://localhost:6379/0"    # Redis 连接地址，/0 表示使用第 0 号数据库
    REDIS_CACHE_TTL: int = 3600                    # 通用缓存过期时间（旧字段，保留兼容）
    SHARED_DATA_TTL: int = 3600                    # MCP 跨服务 shared_data 缓存过期时间（秒）
    TENANT_QPS_LIMIT: int = 20                     # 每个租户每秒最多发 20 个请求（防止滥用）
    TENANT_QPS_WINDOW: int = 60                    # 限流滑动窗口大小：60秒内统计请求数
    TENANT_DOC_QUOTA: int = 10000                  # 每个租户最多上传 10000 个文档（防止存储爆炸）

    # ── 业务缓存 TTL 配置（各层缓存过期时间）────────────────────────────────────
    # 设计原则：结果越稳定 TTL 越长；依赖实时数据（黑名单/法规）的 TTL 较短
    CACHE_ENABLED: bool = True                     # 全局缓存开关（False=完全禁用，用于调试）
    CACHE_AUDIT_TTL: int = 14400                   # 审核任务级缓存：4 小时（工作日内结果可复用）
    CACHE_COMPLIANCE_TTL: int = 21600              # 合规 RAG 检索缓存：6 小时（法规白天不变）
    CACHE_FRAUD_TTL: int = 7200                    # 欺诈检测缓存：2 小时（黑名单可能更新）
    CACHE_RAG_TTL: int = 7200                      # RAG 问答缓存：2 小时（知识库短期稳定）
    CACHE_INTENT_TTL: int = 86400                  # 意图识别缓存：24 小时（同问题意图不变）
    CACHE_RECOGNITION_TTL: int = 86400             # 票据图片识别缓存：24 小时（文件内容不变）

    # ── 向量嵌入模型配置 ──────────────────────────────────────────────────────
    # 这些模型把文字转换成数字向量，让计算机能"理解"文本语义
    BGE_M3_MODEL_PATH: str = "/Users/jialei/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"  # 本地快照绝对路径，避免 snapshot_download 发起网络请求触发代理报错
    BGE_RERANKER_MODEL_PATH: str = "/Users/jialei/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # Reranker 本地快照路径，同上
    EMBEDDING_BATCH_SIZE: int = 32                         # 每批处理 32 条文本（避免显存溢出）
    RERANKER_TOP_N: int = 5                                # 精排后只保留最相关的 5 条结果

    # ── 检索参数 ──────────────────────────────────────────────────────────────
    # RAG 的核心参数：先粗检索多个候选，再精排取最好的几个
    RETRIEVAL_TOP_K: int = 20             # 第一步粗检索：从向量库取 20 个候选文档片段
    RERANK_TOP_N: int = 5                 # 第二步精排：从 20 个候选中选出最好的 5 个
    HYBRID_DENSE_WEIGHT: float = 0.6      # 稠密向量检索的权重（占 60%）
    HYBRID_SPARSE_WEIGHT: float = 0.4     # 稀疏向量检索的权重（占 40%）
    RRF_K: int = 60                       # RRF融合公式中的常数 k，控制排名靠前的文档的加分幅度

    # ── 视觉模型兜底配置（Qwen-VL API）────────────────────────────────────────
    # 当 PDF 中的表格提取质量不够好时，用视觉大模型来"看图识字"
    VISION_ENABLED: bool = True                            # 是否启用视觉兜底功能
    VISION_CONFIDENCE_THRESHOLD: float = 0.8              # 置信度阈值：低于 0.8 才触发视觉兜底
    VISION_MODEL: str = "qwen-vl-max"                     # 使用的视觉模型（qwen-vl-plus 便宜，qwen-vl-max 更准）

    # ── OCR 识别配置（PaddleOCR）──────────────────────────────────────────────
    # OCR = 光学字符识别，把扫描版 PDF 图片中的文字识别出来
    OCR_LANG: str = "ch"                  # 识别语言：ch=中文（同时支持英文）
    OCR_USE_ANGLE_CLS: bool = True        # 自动纠正倾斜：扫描件歪了也能正确识别
    OCR_DPI_SCALE: int = 2               # 图像放大倍数：放大 2 倍后再识别，提高准确率
    OCR_USE_GPU: bool = False             # 是否使用 GPU 加速 OCR（没有 GPU 就用 CPU）
    OCR_POOL_SIZE: int = 5               # OCR 实例池大小：允许 N 路并发，每实例约 600MB 内存

    # ── PDF 解析配置 ──────────────────────────────────────────────────────────
    PDF_RESOLUTION_SCALE: int = 2                  # PDF 渲染分辨率倍数
    CAMELOT_LATTICE_THRESHOLD: float = 0.8         # 有框线表格的置信度合格线（80%以上才认为提取成功）
    CAMELOT_STREAM_THRESHOLD: float = 0.6          # 无框线表格的置信度合格线（要求略低）

    # ── 文本分块配置 ──────────────────────────────────────────────────────────
    # 长文档需要切成小块才能放进向量模型，这里控制切块的大小
    CHUNK_MAX_TOKENS: int = 512           # 每个文本块最多 512 个 token（约 256-512 个中文字）
    CHUNK_OVERLAP_TOKENS: int = 64        # 相邻块之间重叠 64 个 token，防止语义被截断
    CHUNK_MIN_TOKENS: int = 50            # 太短的块（不足 50 token）直接丢弃，避免噪声

    # ── 大语言模型（LLM）配置 ─────────────────────────────────────────────────
    # LLM 负责根据检索到的文档片段生成最终答案
    LLM_PROVIDER: str = "openai"          # LLM 提供商：openai（兼容通义千问）/ anthropic / local
    OPENAI_API_KEY: Optional[str] = None  # OpenAI 或通义千问的 API Key（可以为空，但没有就无法生成答案）
    OPENAI_BASE_URL: Optional[str] = None # 自定义 API 地址：通义千问用此接口实现 OpenAI 兼容
    OPENAI_MODEL: str = "gpt-4o"          # 使用的模型名称（通义千问时填 qwen-max 等）
    ANTHROPIC_API_KEY: Optional[str] = None        # Anthropic Claude 的 API Key
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"  # Claude 模型版本
    LLM_MAX_TOKENS: int = 2048            # LLM 生成答案的最大 token 数（约 1000 个中文字）
    LLM_TEMPERATURE: float = 0.1          # 生成随机性：0=完全确定，1=很随机，0.1 接近确定性答案

    # ── 文件存储路径 ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"    # 用户上传的原始文件存放目录
    PROCESSED_DIR: str = "./data/processed"        # 解析处理后的文件存放目录
    JIEBA_USER_DICT: str = "./config/bill_dict.txt"  # 票据行业专用词典路径（用于中文分词）

    # ── 日志配置 ──────────────────────────────────────────────────────────────
    LOG_DIR: str = "./logs"              # 日志文件存放目录
    LOG_LEVEL: str = "INFO"              # 日志级别：DEBUG / INFO / WARNING / ERROR
    LOG_ROTATION: str = "50 MB"          # 单文件超过 50MB 时自动轮转
    LOG_RETENTION: str = "30 days"       # 保留最近 30 天的日志，更早的自动删除
    LOG_COMPRESSION: str = "gz"          # 轮转后用 gzip 压缩，节省磁盘空间

    # ── MCP 独立部署配置 ──────────────────────────────────────────────────────
    MCP_SERVER_URL: str = "http://localhost:8001/mcp"  # 独立 MCP Server 地址（主应用通过此地址调用工具）

    # ── 监控配置 ──────────────────────────────────────────────────────────────
    PROMETHEUS_ENABLED: bool = True       # 是否开启 Prometheus 监控数据采集
    GRAFANA_URL: str = "http://localhost:3000"  # Grafana 监控面板地址

    class Config:
        # 内部配置类，告诉 pydantic 如何读取配置
        env_file = ".env"         # 自动读取项目根目录下的 .env 文件
        case_sensitive = True     # 环境变量名区分大小写（DATABASE_URL ≠ database_url）


# 创建全局配置对象，整个项目通过 `from config.settings import settings` 导入使用
settings = Settings()
