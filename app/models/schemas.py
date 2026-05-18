# app/models/schemas.py
# API 请求/响应的数据结构定义（Pydantic Schema）
# 作用：定义"前端发来的数据长什么样"以及"后端返回的数据长什么样"
# Pydantic 的好处：自动验证数据格式，如果数据不合法，自动返回错误信息

from pydantic import BaseModel, Field, EmailStr
# BaseModel：所有 Schema 的基类
# Field：给字段添加验证规则（最小长度、最大值等）
# EmailStr：专门验证邮箱格式的类型（自动检测是否是合法邮箱）

from typing import Optional, Any   # Optional：可以为空；Any：任意类型
from datetime import datetime      # 日期时间类型
from enum import Enum              # Python 枚举（这里仅导入备用）


# ── 认证相关 Schema ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    """用户登录请求体：前端 POST /api/v1/auth/login 时发送的数据"""
    username: str   # 用户名
    password: str   # 明文密码（传输过程中走 HTTPS 加密，服务器收到后再哈希对比）

class TokenResponse(BaseModel):
    """登录成功后返回给前端的数据"""
    access_token: str        # JWT Token 字符串（前端需要保存，之后每次请求都带上）
    token_type: str = "bearer"  # Token 类型，固定为 "bearer"
    tenant_id: str           # 用户所属租户 ID（前端可能需要展示机构信息）
    user_id: str             # 当前登录用户的 ID
    expires_in: int          # Token 有效时长（秒），告诉前端多久后需要重新登录


# ── 租户相关 Schema ──────────────────────────────────────────────────────────
class TenantCreate(BaseModel):
    """创建新租户的请求体（只有管理员才能调用）"""
    name: str = Field(..., min_length=2, max_length=200)
    # ...：表示必填字段；min_length=2：最少 2 个字符；max_length=200：最多 200 个字符
    code: str = Field(..., min_length=2, max_length=50)  # 机构代码，如 "ABC_BANK"
    license_no: Optional[str] = None    # 牌照编号，可以不填
    doc_quota: int = Field(default=10000, ge=100)
    # ge=100：大于等于 100（不能给太小的配额）；默认 10000
    qps_limit: int = Field(default=20, ge=1, le=200)
    # ge=1：至少 1 个请求/秒；le=200：最多 200 个请求/秒（防止无限速）

class TenantResponse(BaseModel):
    """租户信息响应体（查询租户时返回这些字段）"""
    id: str             # 租户唯一 ID
    name: str           # 机构名称
    code: str           # 机构代码
    license_no: Optional[str]  # 牌照编号（可能为空）
    status: str         # 状态：active/suspended/trial
    doc_quota: int      # 文档配额
    qps_limit: int      # QPS 限制
    created_at: datetime  # 注册时间

    class Config:
        from_attributes = True  # 允许从 SQLAlchemy 模型对象直接创建此 Schema（ORM 模式）


# ── 用户相关 Schema ──────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    """注册新用户的请求体"""
    username: str = Field(..., min_length=3, max_length=100)  # 用户名至少 3 个字符
    email: Optional[str] = None          # 邮箱（可选）
    password: str = Field(..., min_length=8)  # 密码至少 8 个字符（基础安全要求）
    is_admin: bool = False               # 是否创建为管理员（默认不是）

class UserResponse(BaseModel):
    """用户信息响应体"""
    id: str             # 用户 ID
    username: str       # 用户名
    email: Optional[str]  # 邮箱（可能为空）
    is_admin: bool      # 是否是管理员
    is_active: bool     # 账号是否激活
    tenant_id: str      # 所属租户 ID
    created_at: datetime  # 注册时间

    class Config:
        from_attributes = True  # 支持从 ORM 对象直接转换


# ── 文档相关 Schema ──────────────────────────────────────────────────────────
class DocumentResponse(BaseModel):
    """文档信息响应体（上传文档或查询文档列表时返回）"""
    id: str              # 文档唯一 ID
    filename: str        # 原始文件名
    file_type: str       # 文件类型（pdf/docx/xlsx/image）
    file_size: int       # 文件大小（字节）
    status: str          # 处理状态（pending/processing/completed/failed）
    chunk_count: int     # 被切成了多少块
    page_count: int      # 总页数
    md5_hash: str        # 文件 MD5（用于去重）
    parse_meta: Optional[dict]   # 解析统计信息（如各类型表格数量）
    error_msg: Optional[str]     # 如果处理失败，这里是错误原因
    created_at: datetime  # 上传时间

    class Config:
        from_attributes = True

class FolderUploadRequest(BaseModel):
    """文件夹批量入库请求体：传入服务器上的文件夹路径，自动扫描所有支持的文件"""
    folder_path: str = Field(..., description="服务器上的文件夹绝对路径")  # 必填，绝对路径
    recursive: bool = Field(default=False, description="是否递归扫描子目录，默认只扫当前层")  # 可选递归


class IngestResponse(BaseModel):
    """文档入库完成后的详细响应（供内部调用使用）"""
    document_id: str     # 文档 ID
    md5_hash: str        # MD5 哈希值
    chunk_count: int     # 切块数量
    page_count: int      # 页数
    parse_stats: dict    # 解析统计（如：有几个框线表格、几个OCR页面）
    elapsed_ms: float    # 整个入库过程耗时（毫秒）
    status: str          # 入库状态：completed=成功，duplicate=已存在


# ── 问答相关 Schema ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    """智能问答请求体：用户发送的问题"""
    query: str = Field(..., min_length=2, max_length=2000)
    # 问题内容：至少 2 个字符（避免空问题），最多 2000 字符（防止超长输入）
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    # 要检索多少个候选片段（不填则用默认值）
    stream: bool = False  # 是否使用流式响应（True=像 ChatGPT 一样逐字返回）

class SourceChunk(BaseModel):
    """引用的文档片段信息（答案是基于哪些片段生成的）"""
    chunk_id: str         # 片段唯一 ID
    document_id: str      # 所属文档 ID
    section_path: str     # 所在章节（如"第一条 总则"）
    page_num: int         # 在原文档中的页码
    chunk_type: str       # 片段类型（text/table/image_ocr）
    rerank_score: float   # 精排分数（越高表示与问题越相关）
    content_preview: str  # 片段内容的前 200 个字（供用户查看参考依据）

class QueryResponse(BaseModel):
    """智能问答响应体：完整的问答结果"""
    query_id: str              # 本次查询的唯一 ID（用于提交反馈）
    answer: str                # LLM 生成的答案
    sources: list[SourceChunk] # 参考的文档片段列表（来源依据）
    retrieval_ms: float        # 检索耗时（毫秒）
    llm_ms: float              # LLM 生成耗时（毫秒）
    total_ms: float            # 全链路总耗时（毫秒）
    retrieved_count: int       # 检索到的片段数量

class FeedbackRequest(BaseModel):
    """用户对答案的反馈请求体"""
    query_id: str              # 要评价的查询 ID
    score: int = Field(..., ge=1, le=5)  # 评分：1-5 星（ge=1 至少1分，le=5 最多5分）
    comment: Optional[str] = None       # 文字评论（可选）


# ── 统计相关 Schema ──────────────────────────────────────────────────────────
class TenantStats(BaseModel):
    """租户使用统计信息"""
    tenant_id: str         # 租户 ID
    doc_count: int         # 已上传的文档数量
    chunk_count: int       # 总片段数量（文档被切成了多少块）
    query_count_today: int # 今天的查询次数
    doc_quota: int         # 文档总配额上限
    qps_limit: int         # QPS 限制
    quota_used_pct: float  # 配额使用百分比（如 45.3 表示用了 45.3%）


# ── 通用响应 Schema ──────────────────────────────────────────────────────────
class SuccessResponse(BaseModel):
    """操作成功的通用响应（如删除文档成功）"""
    success: bool = True       # 固定为 True（表示操作成功）
    message: str = "操作成功"  # 成功提示信息

class ErrorResponse(BaseModel):
    """操作失败的通用响应"""
    success: bool = False  # 固定为 False（表示操作失败）
    error: str             # 错误类型描述
    detail: Optional[Any] = None  # 详细错误信息（可以是任何类型）

class PaginatedResponse(BaseModel):
    """分页列表响应（查询多条记录时使用）"""
    items: list        # 当前页的数据列表
    total: int         # 总记录数
    page: int          # 当前页码（从 1 开始）
    page_size: int     # 每页显示多少条
    total_pages: int   # 总页数（= ceil(total / page_size)）


# ── 票据识别相关 Schema ───────────────────────────────────────────────────────
class BillElementSchema(BaseModel):
    """单张票据的结构化要素（视觉模型识别结果）"""
    is_bill_found: bool                    # 是否成功识别到票据
    ticket_type: Optional[str] = None      # 票据类型：银行承兑汇票 / 商业承兑汇票 / 本票等
    ticket_number: Optional[str] = None    # 票据号码
    issue_date: Optional[str] = None       # 出票日期（YYYY-MM-DD 或原始格式）
    due_date: Optional[str] = None         # 到期日（付款期限）
    amount_numeric: Optional[float] = None # 票面金额（数字，便于计算和排序）
    amount_text: Optional[str] = None      # 大写金额（如：壹佰万元整）
    currency: Optional[str] = "人民币"     # 币种
    drawer: Optional[str] = None           # 出票人名称
    drawer_account: Optional[str] = None   # 出票人银行账号
    drawer_bank: Optional[str] = None      # 出票人开户行
    acceptor: Optional[str] = None         # 承兑人（银行或企业）
    payee: Optional[str] = None            # 收款人（第一手持票人）
    endorsers: list[str] = []              # 背书人列表（按背书顺序排列，可为空）
    drawee_bank: Optional[str] = None      # 付款行（承兑行）

class BillRecognitionResponse(BaseModel):
    """票据识别接口的完整响应体"""
    filename: str                          # 上传的原始文件名
    bill_count: int                        # 识别到的票据数量（一个 PDF 可能有多张票据）
    bills: list[BillElementSchema]         # 所有票据要素列表
    page_count: int                        # 处理的页数（PDF 为总页数，图片固定为 1）
    model_used: str                        # 实际使用的视觉模型名称
    elapsed_ms: float                      # 总识别耗时（毫秒）


# ── 票据生命周期 Schema（P1）────────────────────────────────────────────────
class BillUpsertResponse(BaseModel):
    """上传票据后的入库响应：包含生命周期管理结果"""
    bill_record_id: str       # 票据主档 ID
    ticket_number: str        # 票据号码
    version: int              # 本次入库后的最新版本号
    is_new_bill: bool         # True=首次入库，False=流转更新
    new_endorsers: list[str]  # 本次新增的背书人（流转场景下有值）
    risk_flags: list          # 检测到的要素矛盾/风险列表
    document_id: str          # 原始文件记录 ID
    elapsed_ms: float         # 总处理耗时（毫秒）


class BillVersionResponse(BaseModel):
    """单个流转版本信息"""
    version: int
    endorsers: list[str]
    new_endorsers: list[str]
    upload_time: datetime
    uploaded_by: Optional[str]

    class Config:
        from_attributes = True


class BillRecordResponse(BaseModel):
    """票据主档详情响应"""
    id: str
    ticket_number: str
    ticket_type: Optional[str]
    issue_date: Optional[str]
    due_date: Optional[str]
    amount_numeric: Optional[float]
    amount_text: Optional[str]
    drawer: Optional[str]
    acceptor: Optional[str]
    payee: Optional[str]
    risk_flags: list
    latest_version: int
    versions: list[BillVersionResponse]
    created_at: datetime

    class Config:
        from_attributes = True


# ── 问答链路扩展 Schema（P2/P3）──────────────────────────────────────────────
class QueryRequestV2(BaseModel):
    """增强版问答请求体：支持空 query 触发引导、携带票据上下文"""
    query: str = Field(default="", max_length=2000)  # 允许空字符串（触发友情引导）
    bill_record_id: Optional[str] = None  # 可选：关联已入库的票据，用于专项检索上下文
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    stream: bool = False


class QueryResponseV2(BaseModel):
    """增强版问答响应体：包含意图路由和检索质量信息"""
    query_id: str
    answer_type: str                   # answer/guide_prompt/transfer_human
    answer: str
    sources: list[SourceChunk] = []
    intent_id: Optional[str] = None    # 识别到的意图
    route_type: Optional[str] = None   # specialized/fuzzy/no_result
    retrieval_quality: Optional[str] = None  # SUFFICIENT/PARTIAL/INSUFFICIENT
    transfer_to_human: bool = False
    contact_info: Optional[dict] = None  # 转人工时附带的联系信息
    query_saved: bool = False          # 未命中时是否已保存到优化日志
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
