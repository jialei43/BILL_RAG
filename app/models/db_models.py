# app/models/db_models.py
# 数据库表结构定义文件（ORM 模型）
# ORM = Object-Relational Mapping（对象关系映射）：用 Python 类来描述数据库表
# 好处：不用手写 SQL，直接操作 Python 对象，SQLAlchemy 自动转换为 SQL

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    # Column：列（字段）；String：字符串类型；Integer：整数；Float：浮点数
    # Boolean：布尔值（True/False）；DateTime：日期时间；Text：长文本（无长度限制）
    ForeignKey, JSON, Enum as SAEnum
    # ForeignKey：外键（关联另一张表的主键）；JSON：JSON 数据类型；Enum：枚举类型
)
from sqlalchemy.orm import relationship, DeclarativeBase
# relationship：定义两张表之间的关联关系（如"用户属于某个租户"）
# DeclarativeBase：所有模型类的基类

from sqlalchemy.sql import func  # func 提供 SQL 函数（如 NOW()、COUNT()）
import enum                      # Python 标准枚举库


class Base(DeclarativeBase):
    """所有数据库模型类的基类，继承此类的类会自动映射到数据库表"""
    pass


class TenantStatus(str, enum.Enum):
    """租户状态枚举：描述一个机构账户当前的状态"""
    ACTIVE = "active"        # 正常使用中
    SUSPENDED = "suspended"  # 已被暂停（违规或欠费）
    TRIAL = "trial"          # 试用期（功能或配额受限）


class DocumentStatus(str, enum.Enum):
    """文档处理状态枚举：追踪文档从上传到入库的全过程"""
    PENDING = "pending"          # 等待处理（刚上传，还没开始）
    PROCESSING = "processing"    # 正在处理（解析、分块、向量化中）
    COMPLETED = "completed"      # 处理完成（可以被检索了）
    FAILED = "failed"            # 处理失败（查看 error_msg 了解原因）


class Tenant(Base):
    """
    租户表：每个票据机构（银行/经纪公司）是一个租户
    多租户隔离：不同机构的数据完全独立，互不可见
    """
    __tablename__ = "tenants"   # 映射到数据库中名为 "tenants" 的表

    id = Column(String(36), primary_key=True)
    # 主键：UUID 格式（36字符），如 "550e8400-e29b-41d4-a716-446655440000"

    name = Column(String(200), nullable=False, comment="机构名称")
    # nullable=False 表示必填，不能为空；comment 是数据库字段注释

    code = Column(String(50), unique=True, nullable=False, comment="机构代码")
    # unique=True：机构代码唯一，不能重复（如 "ABC_BANK"）

    license_no = Column(String(100), comment="牌照编号")
    # 允许为空（没有 nullable=False），持牌金融机构才有牌照编号

    status = Column(SAEnum(TenantStatus), default=TenantStatus.TRIAL)
    # 状态字段，使用上面定义的枚举，新建租户默认是"试用"状态

    doc_quota = Column(Integer, default=10000, comment="文档配额")
    # 该租户最多可上传多少个文档，防止一个机构占用所有存储空间

    qps_limit = Column(Integer, default=20, comment="QPS限制")
    # 每秒最多发多少个请求，防止接口被刷爆

    milvus_partition = Column(String(100), comment="Milvus分区名")
    # 在向量数据库中，每个租户有独立的数据分区，彻底隔离数据

    created_at = Column(DateTime, server_default=func.now())
    # 创建时间：由数据库服务器自动填入当前时间（server_default 表示在数据库层面设置默认值）

    updated_at = Column(DateTime, onupdate=func.now())
    # 更新时间：每次记录被修改时，数据库自动更新此字段

    # 关联关系：一个租户有多个用户、多个文档
    users = relationship("User", back_populates="tenant")       # 通过 user.tenant 可以访问租户
    documents = relationship("Document", back_populates="tenant")


class User(Base):
    """用户表：系统的使用者，每个用户属于某一个租户"""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)                   # 用户唯一 ID（UUID）
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    # 外键：关联 tenants 表的 id，表示"该用户属于哪个机构"

    username = Column(String(100), unique=True, nullable=False)  # 用户名，全系统唯一
    email = Column(String(200), unique=True)                     # 邮箱，可以为空，但不能重复
    hashed_password = Column(String(200), nullable=False)        # 哈希后的密码（绝不存明文）
    is_admin = Column(Boolean, default=False)                    # 是否是管理员，默认不是
    is_active = Column(Boolean, default=True)                    # 账号是否激活，可以禁用某个用户

    created_at = Column(DateTime, server_default=func.now())    # 注册时间

    tenant = relationship("Tenant", back_populates="users")     # 通过 user.tenant 访问租户信息


class Document(Base):
    """
    文档记录表：追踪每一个上传文档的元数据和处理状态
    注意：实际内容（向量）存在 Milvus，这里只存元信息
    """
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)                   # 文档唯一 ID（UUID）
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    # 所属租户，用于数据隔离

    filename = Column(String(500), nullable=False)              # 原始文件名（用户上传时的名字）
    file_path = Column(String(1000))                            # 文件在服务器上的存储路径
    file_type = Column(String(20), comment="pdf/docx/xlsx/image")  # 文件类型
    file_size = Column(Integer, comment="bytes")                # 文件大小（字节）

    md5_hash = Column(String(32), nullable=False, comment="幂等去重")
    # MD5 哈希：32位字符串，用于检测重复文件
    # 相同内容的文件 MD5 相同，避免重复入库（幂等性）

    status = Column(SAEnum(DocumentStatus), default=DocumentStatus.PENDING)
    # 文档处理状态（见 DocumentStatus 枚举）

    chunk_count = Column(Integer, default=0)                    # 文档被切成了多少个片段
    page_count = Column(Integer, default=0)                     # 文档的总页数
    error_msg = Column(Text)                                    # 如果处理失败，这里记录错误原因
    parse_meta = Column(JSON, comment="解析元数据")              # 解析过程的统计信息（JSON格式）

    created_at = Column(DateTime, server_default=func.now())    # 上传时间
    updated_at = Column(DateTime, onupdate=func.now())          # 最后更新时间

    tenant = relationship("Tenant", back_populates="documents") # 关联到所属租户
    chunks = relationship("DocumentChunk", back_populates="document")  # 关联到所有片段


class DocumentChunk(Base):
    """
    文档片段表：记录文档切片的元数据
    文档被切成小块后，每块的文本和位置信息存在这里（向量存在 Milvus）
    """
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True)                   # 片段唯一 ID（UUID）
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    # 所属文档，外键关联

    tenant_id = Column(String(36), nullable=False)              # 所属租户（冗余字段，方便查询过滤）
    chunk_index = Column(Integer)                               # 片段序号（第几块，从0开始）
    content = Column(Text, nullable=False)                      # 片段的实际文本内容
    section_path = Column(String(500), comment="章节路径前缀")  # 所属章节（如"第一条 总则"）
    page_num = Column(Integer)                                  # 在原文档中的页码
    chunk_type = Column(String(50), comment="text/table/image_ocr")  # 片段类型
    token_count = Column(Integer)                               # 这个片段大约有多少个 token
    milvus_id = Column(String(100), comment="Milvus向量ID")     # 在 Milvus 中对应的向量 ID
    chunk_metadata = Column(JSON)                               # 其他元数据（JSON格式，灵活扩展）

    created_at = Column(DateTime, server_default=func.now())    # 创建时间

    document = relationship("Document", back_populates="chunks") # 关联到所属文档


class QueryLog(Base):
    """
    查询日志表：记录用户每次问答的详情
    用途：分析用户行为、计算检索命中率、收集用户反馈
    """
    __tablename__ = "query_logs"

    id = Column(String(36), primary_key=True)                   # 查询唯一 ID
    tenant_id = Column(String(36), nullable=False)              # 哪个机构的用户查询的
    user_id = Column(String(36))                                # 哪个用户查询的
    query = Column(Text, nullable=False)                        # 用户的原始问题
    answer = Column(Text)                                       # LLM 生成的答案
    retrieved_chunks = Column(JSON)                             # 检索到的文档片段列表（JSON格式）
    retrieval_ms = Column(Float)                                # 检索耗时（毫秒）
    rerank_ms = Column(Float)                                   # 精排耗时（毫秒）
    llm_ms = Column(Float)                                      # LLM 生成耗时（毫秒）
    total_ms = Column(Float)                                    # 全链路总耗时（毫秒）
    top_k_hit = Column(Boolean, comment="Top-5是否命中")         # 前5个检索结果是否包含答案
    feedback_score = Column(Integer, comment="用户反馈 1-5")     # 用户对答案的评分（1-5星）
    route_type = Column(String(20), comment="specialized/fuzzy")  # 检索路由类型
    intent_id = Column(String(50), comment="识别到的意图ID")      # 意图路由结果
    created_at = Column(DateTime, server_default=func.now())    # 查询时间


# ──────────────────────────────────────────────────────────────────────────────
# 票据生命周期管理（P1）
# ──────────────────────────────────────────────────────────────────────────────
class BillRecord(Base):
    """
    票据主档表：以票据号码为唯一业务键，记录票据全生命周期的核心字段。
    同一张票据无论上传多少次（流转），都对应同一条 BillRecord。
    """
    __tablename__ = "bill_records"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    ticket_number = Column(String(100), nullable=False, comment="票据号码，业务唯一键")
    ticket_type = Column(String(50), comment="银行承兑汇票/商业承兑汇票/本票")
    issue_date = Column(String(20), comment="出票日期 YYYY-MM-DD")
    due_date = Column(String(20), comment="到期日 YYYY-MM-DD")
    amount_numeric = Column(Float, comment="票面金额（数字）")
    amount_text = Column(String(100), comment="大写金额")
    currency = Column(String(20), default="人民币", comment="币种")
    drawer = Column(String(200), comment="出票人名称")
    drawer_account = Column(String(100), comment="出票人账号")
    drawer_bank = Column(String(200), comment="出票人开户行")
    acceptor = Column(String(200), comment="承兑人")
    payee = Column(String(200), comment="收款人（初始）")
    drawee_bank = Column(String(200), comment="付款行")
    risk_flags = Column(JSON, default=list, comment="矛盾/风险标记列表")
    latest_version = Column(Integer, default=1, comment="当前最新版本号")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    tenant = relationship("Tenant")
    versions = relationship("BillVersion", back_populates="bill_record", order_by="BillVersion.version")


class BillVersion(Base):
    """
    票据流转版本表：每次上传同一张票据时新增一条版本记录，保留完整流转历史。
    不删除旧版本，仅追加，支持任意时间点的版本对比。
    """
    __tablename__ = "bill_versions"

    id = Column(String(36), primary_key=True)
    bill_record_id = Column(String(36), ForeignKey("bill_records.id"), nullable=False)
    version = Column(Integer, nullable=False, comment="版本号，从1递增")
    document_id = Column(String(36), ForeignKey("documents.id"), comment="对应的原始文件记录")
    endorsers = Column(JSON, default=list, comment="本次识别到的完整背书人列表")
    new_endorsers = Column(JSON, default=list, comment="相比上一版本新增的背书人")
    upload_time = Column(DateTime, server_default=func.now())
    uploaded_by = Column(String(36), comment="操作人 user_id")
    recognition_raw = Column(JSON, comment="BillElement 原始识别结果，调试备用")

    bill_record = relationship("BillRecord", back_populates="versions")
    document = relationship("Document")


# ──────────────────────────────────────────────────────────────────────────────
# 意图识别与查询行为日志（P2 / P4）
# ──────────────────────────────────────────────────────────────────────────────
class IntentLog(Base):
    """
    意图识别日志：记录每次 IntentRouter 的分类结果。
    用于分析关键词/LLM 两条通道的覆盖率，指导词表扩充和阈值调优。
    """
    __tablename__ = "intent_logs"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False)
    user_id = Column(String(36))
    query = Column(Text, nullable=False, comment="用户原始输入")
    intent_id = Column(String(50), comment="识别到的意图，如 INTENT_PRICING")
    confidence = Column(Float, comment="置信度 0.0~1.0")
    classify_method = Column(String(20), comment="keyword 或 llm，哪条通道命中")
    route_type = Column(String(20), comment="specialized/fuzzy/no_result")
    created_at = Column(DateTime, server_default=func.now())


class SearchMissLog(Base):
    """
    检索未命中日志：专项或模糊检索质量不足时写入此表。
    按 intent_id 聚合高频未命中问题，指导知识库扩充优先级。
    """
    __tablename__ = "search_miss_logs"

    id = Column(String(36), primary_key=True)
    tenant_id = Column(String(36), nullable=False)
    query = Column(Text, nullable=False, comment="用户原始输入")
    intent_id = Column(String(50), comment="意图 ID")
    route_type = Column(String(20), comment="specialized 或 fuzzy，哪种检索未命中")
    top1_score = Column(Float, comment="最高检索相似度分数")
    retrieval_ms = Column(Float, comment="检索耗时毫秒")
    transferred = Column(Boolean, default=False, comment="是否触发了转人工")
    created_at = Column(DateTime, server_default=func.now())
