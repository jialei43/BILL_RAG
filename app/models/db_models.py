# app/models/db_models.py
# 数据库表结构定义（SQLAlchemy ORM 模型）
# comment= 参数会同步写入 PostgreSQL 的字段注释（\d+ 表名 可查看）

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    pass


class UserRole(str, enum.Enum):
    """用户角色枚举：三级权限体系"""
    SUPER_ADMIN  = "super_admin"   # 超级管理员：系统唯一，跨租户管理
    TENANT_ADMIN = "tenant_admin"  # 租户管理员：管理本租户文档与用户
    USER         = "user"          # 普通用户：问答聊天、上传票据


class TenantStatus(str, enum.Enum):
    """租户状态枚举"""
    ACTIVE    = "active"     # 正常使用
    SUSPENDED = "suspended"  # 已暂停（违规或欠费）
    TRIAL     = "trial"      # 试用期（功能或配额受限）


class DocumentStatus(str, enum.Enum):
    """文档处理状态枚举"""
    PENDING    = "pending"     # 等待处理（刚上传）
    PROCESSING = "processing"  # 解析/分块/向量化进行中
    COMPLETED  = "completed"   # 处理完成，可被检索
    FAILED     = "failed"      # 处理失败，error_msg 记录原因


# ──────────────────────────────────────────────────────────────────────────────
# tenants — 租户表
# 每个票据机构（银行/经纪公司）是一个租户，数据完全隔离
# ──────────────────────────────────────────────────────────────────────────────
class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(
        String(36), primary_key=True,
        comment="租户唯一标识，UUID 格式"
    )
    name = Column(
        String(200), nullable=False,
        comment="机构名称，如「中国建设银行上海分行」"
    )
    code = Column(
        String(50), unique=True, nullable=False,
        comment="机构代码，全局唯一，大写字母/数字，如「CCB_SH」"
    )
    license_no = Column(
        String(100),
        comment="统一社会信用代码（18位），用于幂等防重复创建"
    )
    status = Column(
        SAEnum(TenantStatus), default=TenantStatus.TRIAL,
        comment="租户状态：trial=试用期 / active=正常 / suspended=已暂停"
    )
    doc_quota = Column(
        Integer, default=10000,
        comment="文档配额上限，该租户最多可入库的文档数量"
    )
    qps_limit = Column(
        Integer, default=20,
        comment="每秒最大请求数（QPS），防止接口被滥用"
    )
    milvus_partition = Column(
        String(100),
        comment="Milvus 分区名，格式为 tenant_<CODE>，向量数据物理隔离"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="租户创建时间，由数据库自动填入"
    )
    updated_at = Column(
        DateTime, onupdate=func.now(),
        comment="最后修改时间，任意字段更新时自动刷新"
    )

    users     = relationship("User",     back_populates="tenant")
    documents = relationship("Document", back_populates="tenant")


# ──────────────────────────────────────────────────────────────────────────────
# users — 用户表
# 系统的实际使用者，每个用户归属于某一租户
# ──────────────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(
        String(36), primary_key=True,
        comment="用户唯一标识，UUID 格式"
    )
    tenant_id = Column(
        String(36), ForeignKey("tenants.id"), nullable=False,
        comment="所属租户 ID，外键关联 tenants.id"
    )
    username = Column(
        String(100), unique=True, nullable=False,
        comment="登录用户名，全系统唯一"
    )
    email = Column(
        String(200), unique=True,
        comment="电子邮箱，可为空，不可重复"
    )
    hashed_password = Column(
        String(200), nullable=False,
        comment="bcrypt 哈希后的密码，绝不存储明文"
    )
    role = Column(
        String(20), default=UserRole.USER.value, nullable=False,
        comment="角色：super_admin=超级管理员 / tenant_admin=租户管理员 / user=普通用户"
    )
    is_active = Column(
        Boolean, default=True,
        comment="账号是否启用，False 时拒绝登录"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="注册时间"
    )

    tenant = relationship("Tenant", back_populates="users")


# ──────────────────────────────────────────────────────────────────────────────
# documents — 文档记录表
# 追踪每个上传文档的元数据与处理状态
# 实际文本向量存储在 Milvus，此表只存元信息
# ──────────────────────────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    id = Column(
        String(36), primary_key=True,
        comment="文档唯一标识，UUID 格式"
    )
    tenant_id = Column(
        String(36), ForeignKey("tenants.id"), nullable=False,
        comment="所属租户 ID，数据隔离依据"
    )
    filename = Column(
        String(500), nullable=False,
        comment="用户上传时的原始文件名"
    )
    file_path = Column(
        String(1000),
        comment="文件在服务器磁盘上的存储路径（相对于项目根目录）"
    )
    file_type = Column(
        String(20),
        comment="文件类型：pdf / docx / xlsx / doc / image"
    )
    file_size = Column(
        Integer,
        comment="文件大小，单位字节（Bytes）"
    )
    md5_hash = Column(
        String(32), nullable=False,
        comment="文件内容的 MD5 哈希值，用于幂等去重，相同内容不重复入库"
    )
    status = Column(
        SAEnum(DocumentStatus), default=DocumentStatus.PENDING,
        comment="处理状态：pending / processing / completed / failed"
    )
    chunk_count = Column(
        Integer, default=0,
        comment="文档被切分成的片段数量"
    )
    page_count = Column(
        Integer, default=0,
        comment="文档总页数（PDF/Word 按页，Excel 按 Sheet）"
    )
    error_msg = Column(
        Text,
        comment="处理失败时的错误信息，供排查使用"
    )
    parse_meta = Column(
        JSON,
        comment="解析统计元数据，JSON 格式，如各类型表格数量、OCR 页数等"
    )
    batch_id = Column(
        String(36),
        comment="批量上传时的批次 ID，同批文件共享，用于进度轮询"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="文档上传时间"
    )
    updated_at = Column(
        DateTime, onupdate=func.now(),
        comment="最后状态更新时间"
    )

    tenant = relationship("Tenant", back_populates="documents")
    chunks  = relationship("DocumentChunk", back_populates="document", passive_deletes=True)


# ──────────────────────────────────────────────────────────────────────────────
# document_chunks — 文档片段表
# 文档切分后每个片段的文本与位置信息
# 对应的向量表示存储在 Milvus bill_documents 集合中
# ──────────────────────────────────────────────────────────────────────────────
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(
        String(36), primary_key=True,
        comment="片段唯一标识，UUID 格式"
    )
    document_id = Column(
        String(36), ForeignKey("documents.id"), nullable=False,
        comment="所属文档 ID，外键关联 documents.id"
    )
    tenant_id = Column(
        String(36), nullable=False,
        comment="所属租户 ID，冗余字段，避免跨表关联，加速过滤查询"
    )
    chunk_index = Column(
        Integer,
        comment="片段在文档中的序号，从 0 开始"
    )
    content = Column(
        Text, nullable=False,
        comment="片段的实际文本内容"
    )
    section_path = Column(
        String(500),
        comment="所属章节路径，如「第三条 贴现业务 > 3.1 申请条件」"
    )
    page_num = Column(
        Integer,
        comment="片段在原文档中的起始页码"
    )
    chunk_type = Column(
        String(50),
        comment="片段类型：text=正文 / table=表格 / image_ocr=图片OCR文字"
    )
    token_count = Column(
        Integer,
        comment="片段的 token 数量，用于控制 LLM 上下文长度"
    )
    milvus_id = Column(
        String(100),
        comment="对应 Milvus 中的向量 ID，格式为「document_id_chunk_index」"
    )
    chunk_metadata = Column(
        JSON,
        comment="扩展元数据，JSON 格式，存储表格行列数、图片尺寸等额外信息"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="片段入库时间"
    )

    document = relationship("Document", back_populates="chunks")


# ──────────────────────────────────────────────────────────────────────────────
# query_logs — 查询日志表
# 记录用户每次问答的完整链路数据，用于统计分析和效果优化
# ──────────────────────────────────────────────────────────────────────────────
class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(
        String(36), primary_key=True,
        comment="查询唯一标识，UUID 格式，也作为用户提交反馈的凭据"
    )
    tenant_id = Column(
        String(36), nullable=False,
        comment="发起查询的租户 ID"
    )
    user_id = Column(
        String(36),
        comment="发起查询的用户 ID，匿名查询时为 NULL"
    )
    query = Column(
        Text, nullable=False,
        comment="用户输入的原始问题文本"
    )
    answer = Column(
        Text,
        comment="LLM 生成的最终答案文本"
    )
    retrieved_chunks = Column(
        JSON,
        comment="本次检索到的文档片段列表，JSON 数组，含 chunk_id / score / content_preview"
    )
    retrieval_ms = Column(
        Float,
        comment="向量检索阶段耗时，单位毫秒"
    )
    rerank_ms = Column(
        Float,
        comment="BGE-Reranker 精排阶段耗时，单位毫秒"
    )
    llm_ms = Column(
        Float,
        comment="LLM 生成答案阶段耗时，单位毫秒"
    )
    total_ms = Column(
        Float,
        comment="全链路总耗时（检索 + 精排 + LLM），单位毫秒"
    )
    top_k_hit = Column(
        Boolean,
        comment="Top-K 检索是否命中：True=检索到相关片段，False=未命中（触发兜底或转人工）"
    )
    feedback_score = Column(
        Integer,
        comment="用户对答案的评分，1~5 星，NULL 表示用户未评分"
    )
    route_type = Column(
        String(20),
        comment="检索路由类型：specialized=专项知识库检索 / fuzzy=全库模糊检索 / off_topic=无关咨询"
    )
    intent_id = Column(
        String(50),
        comment="意图路由识别到的意图 ID，如 INTENT_PRICING / INTENT_ENDORSE"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="查询发生时间"
    )


# ──────────────────────────────────────────────────────────────────────────────
# bill_records — 票据主档表
# 以票据号码为业务唯一键，记录票据全生命周期的核心要素
# 同一张票据多次流转只对应一条记录，版本历史见 bill_versions
# ──────────────────────────────────────────────────────────────────────────────
class BillRecord(Base):
    __tablename__ = "bill_records"

    id = Column(
        String(36), primary_key=True,
        comment="主档唯一标识，UUID 格式"
    )
    tenant_id = Column(
        String(36), ForeignKey("tenants.id"), nullable=False,
        comment="所属租户 ID"
    )
    ticket_number = Column(
        String(100), nullable=False,
        comment="票据号码，同一租户内业务唯一键，用于跨版本关联"
    )
    ticket_type = Column(
        String(50),
        comment="票据类型：银行承兑汇票 / 商业承兑汇票 / 本票 / 支票"
    )
    issue_date = Column(
        String(20),
        comment="出票日期，格式 YYYY-MM-DD"
    )
    due_date = Column(
        String(20),
        comment="到期日（付款期限），格式 YYYY-MM-DD"
    )
    amount_numeric = Column(
        Float,
        comment="票面金额，数字形式，便于比较和排序，单位元"
    )
    amount_text = Column(
        String(100),
        comment="票面金额大写，如「壹佰万元整」"
    )
    currency = Column(
        String(20), default="人民币",
        comment="币种，默认人民币"
    )
    drawer = Column(
        String(200),
        comment="出票人名称（开票方公司名）"
    )
    drawer_account = Column(
        String(100),
        comment="出票人银行账号"
    )
    drawer_bank = Column(
        String(200),
        comment="出票人开户行名称"
    )
    acceptor = Column(
        String(200),
        comment="承兑人：银行承兑汇票填承兑银行，商业承兑汇票填付款企业"
    )
    payee = Column(
        String(200),
        comment="初始收款人（第一手持票人）"
    )
    drawee_bank = Column(
        String(200),
        comment="付款行（承兑行），负责最终兑付"
    )
    risk_flags = Column(
        JSON, default=list,
        comment="风险标记列表，JSON 数组，记录要素矛盾/异常，如金额大小写不一致"
    )
    latest_version = Column(
        Integer, default=1,
        comment="当前最新版本号，每次流转入库加 1"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="票据首次入库时间"
    )
    updated_at = Column(
        DateTime, onupdate=func.now(),
        comment="最后一次流转更新时间"
    )

    tenant   = relationship("Tenant")
    versions = relationship("BillVersion", back_populates="bill_record", order_by="BillVersion.version")


# ──────────────────────────────────────────────────────────────────────────────
# bill_versions — 票据流转版本表
# 每次上传同一张票据新增一条版本记录，保留完整流转历史
# 只追加不删除，支持任意时间点的版本对比
# ──────────────────────────────────────────────────────────────────────────────
class BillVersion(Base):
    __tablename__ = "bill_versions"

    id = Column(
        String(36), primary_key=True,
        comment="版本记录唯一标识，UUID 格式"
    )
    bill_record_id = Column(
        String(36), ForeignKey("bill_records.id"), nullable=False,
        comment="所属票据主档 ID，外键关联 bill_records.id"
    )
    version = Column(
        Integer, nullable=False,
        comment="版本号，从 1 开始单调递增，同一票据每次流转加 1"
    )
    document_id = Column(
        String(36), ForeignKey("documents.id"),
        comment="本次上传对应的原始文件记录 ID，关联 documents.id"
    )
    endorsers = Column(
        JSON, default=list,
        comment="本次识别到的完整背书人列表，JSON 字符串数组，按背书顺序排列"
    )
    new_endorsers = Column(
        JSON, default=list,
        comment="相比上一版本新增的背书人，用于快速判断流转新增了哪些持票方"
    )
    upload_time = Column(
        DateTime, server_default=func.now(),
        comment="本次流转上传时间"
    )
    uploaded_by = Column(
        String(36),
        comment="操作人的 user_id，记录是谁上传了这次流转"
    )
    recognition_raw = Column(
        JSON,
        comment="视觉模型原始识别结果（BillElementSchema），供调试和人工核查使用"
    )

    bill_record = relationship("BillRecord", back_populates="versions")
    document    = relationship("Document")


# ──────────────────────────────────────────────────────────────────────────────
# intent_logs — 意图识别日志表
# 记录每次意图路由的分类结果，用于分析关键词/LLM 两条通道的覆盖率
# ──────────────────────────────────────────────────────────────────────────────
class IntentLog(Base):
    __tablename__ = "intent_logs"

    id = Column(
        String(36), primary_key=True,
        comment="日志唯一标识，UUID 格式"
    )
    tenant_id = Column(
        String(36), nullable=False,
        comment="所属租户 ID"
    )
    user_id = Column(
        String(36),
        comment="发起查询的用户 ID"
    )
    query = Column(
        Text, nullable=False,
        comment="用户原始输入文本"
    )
    intent_id = Column(
        String(50),
        comment="识别到的意图 ID，如 INTENT_PRICING / INTENT_ENDORSE / INTENT_RISK"
    )
    confidence = Column(
        Float,
        comment="意图置信度，范围 0.0~1.0，低于阈值时降级为模糊检索"
    )
    classify_method = Column(
        String(20),
        comment="命中通道：keyword=关键词词表匹配 / llm=大模型分类"
    )
    route_type = Column(
        String(20),
        comment="最终路由结果：specialized=专项检索 / fuzzy=模糊检索 / no_result=无结果"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="意图识别时间"
    )


# ──────────────────────────────────────────────────────────────────────────────
# search_miss_logs — 检索未命中日志表
# 专项或模糊检索质量不足时写入此表，用于定向扩充知识库
# ──────────────────────────────────────────────────────────────────────────────
class SearchMissLog(Base):
    __tablename__ = "search_miss_logs"

    id = Column(
        String(36), primary_key=True,
        comment="日志唯一标识，UUID 格式"
    )
    tenant_id = Column(
        String(36), nullable=False,
        comment="所属租户 ID"
    )
    query = Column(
        Text, nullable=False,
        comment="用户原始输入文本"
    )
    intent_id = Column(
        String(50),
        comment="本次查询识别到的意图 ID，NULL 表示未命中任何意图"
    )
    route_type = Column(
        String(20),
        comment="未命中发生在哪种检索：specialized=专项检索未命中 / fuzzy=模糊检索未命中"
    )
    top1_score = Column(
        Float,
        comment="检索结果中最高的相似度分数，低于质量阈值时才写入此表"
    )
    retrieval_ms = Column(
        Float,
        comment="本次检索耗时，单位毫秒"
    )
    transferred = Column(
        Boolean, default=False,
        comment="是否触发了转人工流程：True=已转人工 / False=系统兜底回答"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="未命中发生时间"
    )
