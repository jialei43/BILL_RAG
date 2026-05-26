# app/models/agent_models.py
# 多智能体审核系统的数据库表结构定义（与 db_models.py 分离，避免改动原有表）
# 共 14 张表，覆盖审核任务全链路：任务 → 要素 → 合规 → 背书 → 合同 → 风险 → 报告 → 流转 → 欺诈 → 批量 → 字典
# 每个字段均附 comment= 参数，PostgreSQL \d+ 表名可查看中文说明

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, Enum as SAEnum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship  # 定义表间关系（一对多/多对一）
from sqlalchemy.sql import func          # func.now() 生成数据库端的当前时间戳
import enum                              # Python 内置枚举，配合 SAEnum 写入数据库

from app.models.db_models import Base   # 复用原有 Base，保证所有表在同一个 metadata 中


# ══════════════════════════════════════════════════════════════════════════════
# 枚举类型定义
# ══════════════════════════════════════════════════════════════════════════════

class AuditTaskStatus(str, enum.Enum):
    """审核任务整体状态：贯穿任务生命周期"""
    PENDING    = "pending"    # 待处理：已提交但尚未分配 Agent 执行
    RUNNING    = "running"    # 执行中：DAG 正在按序/并行调度各 Agent
    COMPLETED  = "completed"  # 已完成：所有 Agent 均返回结果
    FAILED     = "failed"     # 失败：有关键 Agent 抛出未恢复异常
    CANCELLED  = "cancelled"  # 已取消：用户主动中止或超时熔断


class AuditTaskType(str, enum.Enum):
    """审核任务业务类型：决定 OrchestratorAgent 生成哪条 DAG 路径"""
    FULL_AUDIT        = "full_audit"         # 全流程审核：依次执行所有 12 个 Agent
    ISSUANCE_CHECK    = "issuance_check"     # 出票合规预检：仅执行出票相关子集
    DISCOUNT_APPLY    = "discount_apply"     # 贴现申请审核：侧重合同审核 + 贸易背景
    ACCEPTANCE_PROMPT = "acceptance_prompt"  # 提示承兑：侧重背书链 + 承兑行资质
    ENDORSEMENT       = "endorsement"        # 背书转让：侧重背书连续性核查
    PAYMENT_PROMPT    = "payment_prompt"     # 提示付款：到期要素 + 账户校验
    PLEDGE            = "pledge"             # 质押背书：法律要素 + 质押登记
    COLLECTION        = "collection"         # 托收委托：委托链 + 资金流向


class RiskLevel(str, enum.Enum):
    """综合风险等级：五档分级，供报告和预警使用"""
    LOW         = "LOW"          # 低风险：综合评分 ≥ 85
    MEDIUM_LOW  = "MEDIUM_LOW"   # 中低风险：综合评分 70~84
    MEDIUM_HIGH = "MEDIUM_HIGH"  # 中高风险：综合评分 50~69，需人工复核
    HIGH        = "HIGH"         # 高风险：综合评分 30~49，建议拒绝
    CRITICAL    = "CRITICAL"     # 极高风险：综合评分 < 30，强制拒绝


class ViolationLevel(str, enum.Enum):
    """合规违规严重程度：对应处理动作不同"""
    INFO    = "info"    # 提示：不影响通过，记录备案
    WARNING = "warning" # 警告：需人工确认后可放行
    SEVERE  = "severe"  # 严重：必须修正，阻断审核通过


class FlowTrackingStatus(str, enum.Enum):
    """流转追踪任务状态：七步报文流转的整体进度"""
    INITIATED  = "initiated"   # 已发起：任务创建，第 1 步尚未发送
    IN_FLIGHT  = "in_flight"   # 流转中：部分步骤已完成
    COMPLETED  = "completed"   # 全部完成：7/7 步骤均收到应答
    TIMEOUT    = "timeout"     # 超时：某步骤等待应答超过阈值
    ERROR      = "error"       # 异常：报文解析失败或系统异常


class FlowMessageStatus(str, enum.Enum):
    """单个报文节点的状态：描述某一步骤的当前处理状态"""
    SENT          = "sent"           # 已发送：报文已离开发起方
    ACK_RECEIVED  = "ack_received"   # 已收到回执：目标方已签收（但可能未处理）
    PROCESSING    = "processing"     # 处理中：目标方正在执行业务逻辑
    RESPONDED     = "responded"      # 已应答：目标方返回业务结果
    TIMEOUT       = "timeout"        # 超时：等待应答超过规定时长
    ERROR         = "error"          # 报文异常：格式错误或路由失败
    CANCELLED     = "cancelled"      # 已撤销：发起方主动取消该报文


class BatchTaskStatus(str, enum.Enum):
    """批量任务整体状态"""
    PENDING    = "pending"    # 等待调度：已提交但未开始执行子任务
    RUNNING    = "running"    # 执行中：Semaphore 正在并发分发子任务
    COMPLETED  = "completed"  # 全部完成（含部分失败）
    FAILED     = "failed"     # 全部失败或致命异常


class EntityType(str, enum.Enum):
    """黑名单实体类型"""
    COMPANY = "company"  # 企业法人（出票人/背书人/承兑人）
    PERSON  = "person"   # 自然人（法定代表人/实际控制人）
    ACCOUNT = "account"  # 银行账号（涉案账户）


# ══════════════════════════════════════════════════════════════════════════════
# 1. audit_tasks — 审核主任务表
#    一笔票据审核对应一条记录，其他所有 Agent 结果表都外键关联此表
# ══════════════════════════════════════════════════════════════════════════════
class AuditTask(Base):
    __tablename__ = "audit_tasks"

    id = Column(
        String(36), primary_key=True,
        comment="任务唯一标识，UUID 格式，贯穿整个多 Agent 调用链路"
    )
    tenant_id = Column(
        String(36), ForeignKey("tenants.id"), nullable=False,
        comment="所属租户 ID，审核数据按租户隔离，不可跨租户查询"
    )
    document_id = Column(
        String(36), ForeignKey("documents.id"),
        comment="关联的原始文档 ID（上传的票据图片/PDF），可为空（测试任务无文档）"
    )
    bill_record_id = Column(
        String(36), ForeignKey("bill_records.id"),
        comment="关联的票据主档 ID，要素抽取成功后回填，用于幂等防重审"
    )
    task_type = Column(
        SAEnum(AuditTaskType), nullable=False, default=AuditTaskType.FULL_AUDIT,
        comment="业务类型：决定 OrchestratorAgent 生成哪条 DAG，不同类型调用不同子集 Agent"
    )
    status = Column(
        SAEnum(AuditTaskStatus), nullable=False, default=AuditTaskStatus.PENDING,
        comment="任务状态：pending→running→completed/failed，前端轮询此字段判断进度"
    )
    priority = Column(
        Integer, default=5,
        comment="调度优先级：1(最高)~10(最低)，高优先级任务优先分配 Semaphore 槽位"
    )
    dag_plan = Column(
        JSON,
        comment="OrchestratorAgent 生成的 DAG 执行图，JSON 结构：{nodes:[...], edges:[...]}"
    )
    progress_pct = Column(
        Float, default=0.0,
        comment="任务完成百分比：0.0~100.0，每个 Agent 完成后更新，前端进度条读取"
    )
    current_agent = Column(
        String(100),
        comment="当前正在执行的 Agent 名称，如 'element_extraction_agent'，调试时有用"
    )
    submitted_by = Column(
        String(36),
        comment="提交任务的用户 ID，NULL 表示系统自动触发（如批量任务子任务）"
    )
    error_code = Column(
        String(50),
        comment="失败时的错误码，对应 error_code_mappings.error_code，可快速定位问题类别"
    )
    error_msg = Column(
        Text,
        comment="失败时的详细错误信息，包含 Agent 名称、异常类型、堆栈摘要"
    )
    started_at = Column(
        DateTime,
        comment="任务开始执行时间（第一个 Agent 启动时写入），NULL 表示尚未启动"
    )
    completed_at = Column(
        DateTime,
        comment="任务结束时间（最后一个 Agent 完成/失败时写入）"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="任务提交时间，由数据库自动填入"
    )
    updated_at = Column(
        DateTime, onupdate=func.now(),
        comment="最后一次状态变更时间"
    )

    # 关系：一个审核任务对应多个子结果表记录
    bill_elements      = relationship("BillElement",      back_populates="audit_task", uselist=False)
    compliance_checks  = relationship("ComplianceCheck",  back_populates="audit_task")
    endorsement_chains = relationship("EndorsementChain", back_populates="audit_task", uselist=False)
    contract_reviews   = relationship("ContractReview",   back_populates="audit_task", uselist=False)
    risk_assessments   = relationship("RiskAssessment",   back_populates="audit_task", uselist=False)
    audit_reports      = relationship("AuditReport",      back_populates="audit_task")
    flow_tracking      = relationship("FlowTrackingTask", back_populates="audit_task", uselist=False)
    fraud_detection    = relationship("FraudDetection",   back_populates="audit_task", uselist=False)


# ══════════════════════════════════════════════════════════════════════════════
# 2. bill_elements — 票据要素抽取结果表
#    ElementExtractionAgent 的输出，扩展到 18 个要素字段
# ══════════════════════════════════════════════════════════════════════════════
class BillElement(Base):
    __tablename__ = "bill_elements"

    id = Column(
        String(36), primary_key=True,
        comment="要素记录唯一标识，UUID 格式"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务对应一条要素记录（一对一）"
    )
    # ── 基础 15 字段（与 bill_recognition.BillElement 对齐）─────────────────
    ticket_number = Column(String(100), comment="票据号码，业务唯一键")
    ticket_type   = Column(String(50),  comment="票据类型：银行承兑汇票/商业承兑汇票等")
    issue_date    = Column(String(20),  comment="出票日期，格式 YYYY-MM-DD")
    due_date      = Column(String(20),  comment="到期日，格式 YYYY-MM-DD")
    amount_numeric = Column(Float,      comment="票面金额（数字），单位元")
    amount_text   = Column(String(100), comment="票面金额大写，如「壹佰万元整」")
    currency      = Column(String(20),  comment="币种，默认人民币")
    drawer        = Column(String(200), comment="出票人名称")
    drawer_account = Column(String(100), comment="出票人银行账号")
    drawer_bank   = Column(String(200), comment="出票人开户行")
    acceptor      = Column(String(200), comment="承兑人（银行或企业）")
    payee         = Column(String(200), comment="初始收款人")
    drawee_bank   = Column(String(200), comment="付款行（最终兑付行）")
    endorsers     = Column(JSON,        comment="背书人列表，JSON 字符串数组，按顺序排列")
    maturity_days = Column(Integer,     comment="距到期天数，由 due_date 计算得出")
    # ── 新增 3 字段（超出原 BillElement 的扩展）──────────────────────────────
    trade_purpose = Column(
        Text,
        comment="贸易背景说明：出票时填写的交易目的，合同审核时与合同条款比对"
    )
    acceptance_clause = Column(
        Text,
        comment="承兑条款：承兑行/承兑企业的特殊说明，如利率浮动、抵押条件"
    )
    special_remarks = Column(
        Text,
        comment="其他特殊记载事项：印章注释、限制流通说明等"
    )
    # ── 质量控制字段 ──────────────────────────────────────────────────────────
    confidence_score = Column(
        Float, default=0.0,
        comment="整体识别置信度：18 个字段置信度的加权平均，< 0.7 触发人工核查"
    )
    field_confidences = Column(
        JSON,
        comment="各字段置信度详情：{字段名: 置信度}，供人工核查时快速定位低置信字段"
    )
    raw_ocr_result = Column(
        JSON,
        comment="OCR/视觉模型的原始输出，保留用于调试和重新解析"
    )
    extraction_method = Column(
        String(50),
        comment="抽取方式：ocr_only=纯OCR / vision_llm=视觉大模型 / hybrid=混合"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="要素写入时间"
    )

    audit_task = relationship("AuditTask", back_populates="bill_elements")


# ══════════════════════════════════════════════════════════════════════════════
# 3. compliance_checks — 合规检查项表
#    ComplianceRetrievalAgent 的输出，每个要素字段对应一条合规检查记录
# ══════════════════════════════════════════════════════════════════════════════
class ComplianceCheck(Base):
    __tablename__ = "compliance_checks"

    id = Column(
        String(36), primary_key=True,
        comment="合规检查项唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务对应多条合规检查记录（一对多）"
    )
    element_field = Column(
        String(100), nullable=False,
        comment="被检查的要素字段名，如 'ticket_number'/'amount_numeric'，与 bill_elements 对齐"
    )
    element_value = Column(
        Text,
        comment="被检查的要素字段值（字符串化），与下方 RAG 结果对照"
    )
    regulation_ref = Column(
        String(500),
        comment="相关法规引用：如「票据法第 22 条」「中国人民银行令[2016]第 3 号第 15 条」"
    )
    rag_query = Column(
        Text,
        comment="向知识库发起的检索问句，由 Agent 自动构造，如「出票日期格式要求是什么？」"
    )
    rag_answer = Column(
        Text,
        comment="RAG 返回的合规依据文本，直接引用知识库原文"
    )
    rag_score = Column(
        Float,
        comment="RAG 检索相关性分数：0.0~1.0，分数越高表示知识库中有明确对应规定"
    )
    is_compliant = Column(
        Boolean, nullable=False, default=True,
        comment="是否合规：True=通过 / False=违规，主判断字段"
    )
    violation_level = Column(
        SAEnum(ViolationLevel), default=ViolationLevel.INFO,
        comment="违规严重程度：info=提示 / warning=警告 / severe=严重阻断"
    )
    violation_desc = Column(
        Text,
        comment="违规描述：具体说明哪里不符合规定，供审核人员阅读理解"
    )
    suggestion = Column(
        Text,
        comment="修正建议：告知申请方如何修改才能通过，提升审核效率"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="合规检查完成时间"
    )

    audit_task = relationship("AuditTask", back_populates="compliance_checks")

    __table_args__ = (
        # 联合索引：按任务查询该任务的所有合规检查项（最常见查询模式）
        Index("ix_compliance_audit_task", "audit_task_id"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. endorsement_chains — 背书链路分析表
#    EndorsementChainAgent 的输出，重建背书有向图并检测 9 类违规
# ══════════════════════════════════════════════════════════════════════════════
class EndorsementChain(Base):
    __tablename__ = "endorsement_chains"

    id = Column(
        String(36), primary_key=True,
        comment="背书链路记录唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务对应一条背书链路分析记录（一对一）"
    )
    chain_graph = Column(
        JSON,
        comment="背书有向图的 JSON 表示：{nodes:[{id,name,type}], edges:[{from,to,date}]}"
    )
    endorser_count = Column(
        Integer, default=0,
        comment="背书人总数，即从出票人到当前持票人的中间流转次数"
    )
    is_continuous = Column(
        Boolean, default=True,
        comment="背书链是否连续：False=存在断链，即某个背书人未出现在前一手的收款人栏"
    )
    violation_codes = Column(
        JSON, default=list,
        comment="检测到的违规代码列表：如['E001','E003']，对应 error_code_mappings 中的背书类错误"
    )
    violation_details = Column(
        JSON, default=list,
        comment="违规详情列表：[{code, desc, node_from, node_to}]，定位到具体哪两个节点之间有问题"
    )
    max_chain_depth = Column(
        Integer, default=0,
        comment="背书链最大深度：正常商业票据一般不超过 5 层，过深可能涉嫌虚构贸易背景"
    )
    has_cycle = Column(
        Boolean, default=False,
        comment="背书图中是否存在环：True 表示某实体既是前手又是后手，涉嫌自我背书欺诈"
    )
    blank_endorsement_count = Column(
        Integer, default=0,
        comment="空白背书次数：背书人签章但未填写被背书人，存在失控风险"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="背书链分析完成时间"
    )

    audit_task = relationship("AuditTask", back_populates="endorsement_chains")


# ══════════════════════════════════════════════════════════════════════════════
# 5. contract_reviews — 合同审核表
#    ContractReviewAgent 的输出，验证票据要素与合同的一致性
# ══════════════════════════════════════════════════════════════════════════════
class ContractReview(Base):
    __tablename__ = "contract_reviews"

    id = Column(
        String(36), primary_key=True,
        comment="合同审核记录唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务对应一条合同审核记录（一对一）"
    )
    contract_document_id = Column(
        String(36), ForeignKey("documents.id"),
        comment="参与比对的合同文档 ID，NULL 表示申请方未提交合同（降级为仅贸易背景评分）"
    )
    match_score = Column(
        Float, default=0.0,
        comment="票据与合同的综合匹配度：0.0~100.0，低于 60 触发合规预警"
    )
    trade_background_score = Column(
        Float, default=0.0,
        comment="贸易背景真实性评分：0.0~100.0，基于 trade_purpose + 合同正文判断"
    )
    amount_match = Column(
        Boolean, default=True,
        comment="金额是否匹配：票面金额是否在合同标的金额合理范围内（允许±5% 误差）"
    )
    party_match = Column(
        Boolean, default=True,
        comment="交易方是否匹配：票据出票人/收款人是否与合同甲乙方对应"
    )
    date_match = Column(
        Boolean, default=True,
        comment="日期是否匹配：票据出票日期是否在合同有效期内"
    )
    purpose_match = Column(
        Boolean, default=True,
        comment="用途是否匹配：票据 trade_purpose 与合同约定的付款条件是否一致"
    )
    mismatch_details = Column(
        JSON, default=list,
        comment="不匹配详情列表：[{field, bill_value, contract_value, severity}]，供人工核查"
    )
    review_notes = Column(
        Text,
        comment="审核备注：ContractReviewAgent 的综合判断说明，如「金额差异在允许范围内但需关注」"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="合同审核完成时间"
    )

    audit_task = relationship("AuditTask", back_populates="contract_reviews")


# ══════════════════════════════════════════════════════════════════════════════
# 6. risk_assessments — 风险评估表
#    RiskAssessmentAgent 的输出，四维加权评分后输出综合风险等级
# ══════════════════════════════════════════════════════════════════════════════
class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(
        String(36), primary_key=True,
        comment="风险评估记录唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务对应一条风险评估记录（一对一）"
    )
    # ── 四维分项评分（各维度均 0.0~100.0，分越高风险越低）──────────────────
    compliance_score = Column(
        Float, default=0.0,
        comment="合规维度得分（权重 30%）：基于 compliance_checks 中所有字段的违规率计算"
    )
    endorsement_score = Column(
        Float, default=0.0,
        comment="背书维度得分（权重 30%）：基于 is_continuous / has_cycle / violation_count 计算"
    )
    contract_score = Column(
        Float, default=0.0,
        comment="合同维度得分（权重 20%）：直接使用 contract_reviews.match_score"
    )
    fraud_score = Column(
        Float, default=0.0,
        comment="欺诈维度得分（权重 20%）：1.0 减去 fraud_detections.overall_fraud_score 后×100"
    )
    # ── 综合结果 ─────────────────────────────────────────────────────────────
    composite_score = Column(
        Float, default=0.0,
        comment="综合评分：合规×0.3 + 背书×0.3 + 合同×0.2 + 欺诈×0.2，范围 0.0~100.0"
    )
    risk_level = Column(
        SAEnum(RiskLevel), nullable=False, default=RiskLevel.MEDIUM_HIGH,
        comment="风险等级：LOW≥85 / MEDIUM_LOW 70~84 / MEDIUM_HIGH 50~69 / HIGH 30~49 / CRITICAL<30"
    )
    dimension_details = Column(
        JSON,
        comment="各维度详细评分说明：{dimension: {score, weight, contributing_factors:[...]}}，报告展示用"
    )
    missing_dimensions = Column(
        JSON, default=list,
        comment="缺失维度列表：如 ['contract'] 表示未提交合同，该维度降级为默认 50 分"
    )
    assessor_notes = Column(
        Text,
        comment="评估说明：RiskAssessmentAgent 的综合判断文本，供人工审核人员阅读"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="风险评估完成时间"
    )

    audit_task = relationship("AuditTask", back_populates="risk_assessments")


# ══════════════════════════════════════════════════════════════════════════════
# 7. audit_reports — 审核报告表
#    ReportGenerationAgent 的输出，存储 JSON 报告和 PDF 文件路径
# ══════════════════════════════════════════════════════════════════════════════
class AuditReport(Base):
    __tablename__ = "audit_reports"

    id = Column(
        String(36), primary_key=True,
        comment="审核报告唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务可能生成多版本报告（草稿→终版）"
    )
    report_json = Column(
        JSON,
        comment="结构化报告内容，9 节 JSON：summary/elements/compliance/endorsement/contract/risk/fraud/flow/conclusion"
    )
    pdf_path = Column(
        String(1000),
        comment="reportlab 生成的 PDF 文件路径（相对于 PROCESSED_DIR），NULL 表示仅有 JSON"
    )
    pdf_size_bytes = Column(
        Integer,
        comment="PDF 文件大小（字节），> 0 才认为 PDF 生成成功"
    )
    is_final = Column(
        Boolean, default=False,
        comment="是否为终版报告：False=草稿（实时更新）/ True=最终版（人工签发后不可修改）"
    )
    training_mode = Column(
        Boolean, default=False,
        comment="是否为培训模式报告：True 时附加教学注解层，供新手审核员学习"
    )
    version = Column(
        Integer, default=1,
        comment="报告版本号：同一任务可能因补充材料重新生成，版本号递增"
    )
    generated_by = Column(
        String(100),
        comment="生成者标识：agent=自动生成 / 用户 ID=人工生成"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="报告生成时间"
    )

    audit_task = relationship("AuditTask", back_populates="audit_reports")


# ══════════════════════════════════════════════════════════════════════════════
# 8. flow_tracking_tasks — 流转追踪主表
#    FlowTrackingAgent 的主记录，描述一次票据业务的 7 步报文流转全貌
# ══════════════════════════════════════════════════════════════════════════════
class FlowTrackingTask(Base):
    __tablename__ = "flow_tracking_tasks"

    id = Column(
        String(36), primary_key=True,
        comment="流转追踪任务唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"),
        comment="关联的审核任务 ID，NULL 表示该追踪任务由单独发起（不依附于全流程审核）"
    )
    business_type = Column(
        SAEnum(AuditTaskType), nullable=False,
        comment="业务类型：决定报文流转路径（如提示承兑走发起行→前置机→票交所→对手行→...）"
    )
    ticket_number = Column(
        String(100),
        comment="被追踪票据的号码，用于跨系统关联查询"
    )
    total_steps = Column(
        Integer, default=7,
        comment="该业务类型的总步骤数，固定为 7（对应报文流转协议的七步）"
    )
    completed_steps = Column(
        Integer, default=0,
        comment="已完成的步骤数：每收到一个步骤的 responded 状态则加 1"
    )
    current_node = Column(
        String(200),
        comment="当前报文所在节点名称，如「票交所-处理中」，前端实时显示"
    )
    status = Column(
        SAEnum(FlowTrackingStatus), nullable=False, default=FlowTrackingStatus.INITIATED,
        comment="追踪任务整体状态：initiated→in_flight→completed/timeout/error"
    )
    timeout_level = Column(
        String(20), default="NORMAL",
        comment="超时预警级别：NORMAL/WATCH/WARNING/URGENT/OVERDUE，根据累计耗时动态更新"
    )
    summary_text = Column(
        Text,
        comment="LLM 生成的自然语言摘要：「当前票据已到达票交所，等待对手行应答，预计 2 分钟内完成」"
    )
    initiated_at = Column(
        DateTime, server_default=func.now(),
        comment="追踪任务创建时间"
    )
    completed_at = Column(
        DateTime,
        comment="全部步骤完成时间，NULL 表示仍在流转中"
    )

    audit_task = relationship("AuditTask", back_populates="flow_tracking")
    messages   = relationship("FlowMessage", back_populates="flow_task", order_by="FlowMessage.step_index")


# ══════════════════════════════════════════════════════════════════════════════
# 9. flow_messages — 报文节点状态表
#    每条记录对应一个步骤（1~7）中的一个方向（发送/接收）的报文状态
# ══════════════════════════════════════════════════════════════════════════════
class FlowMessage(Base):
    __tablename__ = "flow_messages"

    id = Column(
        String(36), primary_key=True,
        comment="报文节点记录唯一标识"
    )
    flow_task_id = Column(
        String(36), ForeignKey("flow_tracking_tasks.id"), nullable=False,
        comment="所属流转追踪任务 ID"
    )
    step_index = Column(
        Integer, nullable=False,
        comment="步骤序号：1~7，对应七步报文流转协议中的第几步"
    )
    node_name = Column(
        String(200), nullable=False,
        comment="报文节点名称，如「发起行-发送」「票交所-接收」「对手行-处理中」"
    )
    msg_type = Column(
        String(50),
        comment="报文类型：send=发出报文 / ack=接收回执（每步有 send 和 ack 两条记录）"
    )
    msg_content = Column(
        JSON,
        comment="报文内容摘要（脱敏后），用于调试和审计，不存储完整报文体"
    )
    status = Column(
        SAEnum(FlowMessageStatus), nullable=False, default=FlowMessageStatus.SENT,
        comment="节点当前状态：sent→ack_received→processing→responded/timeout/error"
    )
    arrived_at = Column(
        DateTime,
        comment="报文到达该节点的时间戳（UTC），NULL 表示尚未到达"
    )
    processing_ms = Column(
        Integer, default=0,
        comment="节点处理耗时（毫秒）：从 arrived_at 到 responded 的时间差"
    )
    is_anomaly = Column(
        Boolean, default=False,
        comment="是否异常节点：True 表示超时/错误，在流转图上标红显示"
    )
    anomaly_reason = Column(
        Text,
        comment="异常原因说明：如「等待对手行应答超过 300 秒」，辅助人工处理"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="报文节点记录创建时间"
    )

    flow_task = relationship("FlowTrackingTask", back_populates="messages")

    __table_args__ = (
        # 联合索引：按流转任务+步骤查询（最常见的查询模式）
        Index("ix_flow_messages_task_step", "flow_task_id", "step_index"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 10. fraud_detections — 欺诈检测结果表
#     FraudDetectionAgent 的输出，存储五维并行检测的详细结果
# ══════════════════════════════════════════════════════════════════════════════
class FraudDetection(Base):
    __tablename__ = "fraud_detections"

    id = Column(
        String(36), primary_key=True,
        comment="欺诈检测记录唯一标识"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"), nullable=False,
        comment="所属审核任务 ID，一个任务对应一条欺诈检测记录（一对一）"
    )
    # ── 五维检测结果（每维存 JSON 详情 + Float 得分）──────────────────────────
    seal_score = Column(
        Float, default=0.0,
        comment="印章真伪得分：0.0=真实 / 1.0=疑似伪造，向量相似度比对黑名单特征"
    )
    seal_check_result = Column(
        JSON,
        comment="印章检测详情：{similarity, blacklist_hit, suspicious_features:[...]}"
    )
    duplicate_score = Column(
        Float, default=0.0,
        comment="重复票据得分：0.0=无重复 / 1.0=命中重复，强制置 1.0 不可降级"
    )
    duplicate_check_result = Column(
        JSON,
        comment="重复检测详情：{is_duplicate, duplicate_task_id, ticket_number_match}"
    )
    tamper_score = Column(
        Float, default=0.0,
        comment="篡改检测得分：0.0=无篡改 / 1.0=疑似篡改，OCR 置信度分布分析"
    )
    tamper_check_result = Column(
        JSON,
        comment="篡改检测详情：{suspicious_regions:[{page,bbox,confidence_drop}], method}"
    )
    network_score = Column(
        Float, default=0.0,
        comment="关联网络得分：0.0=无异常 / 1.0=高度可疑，背书链有向图闭环检测"
    )
    network_check_result = Column(
        JSON,
        comment="关联网络检测详情：{has_cycle, cycle_path:[...], suspicious_entities:[...]}"
    )
    blacklist_score = Column(
        Float, default=0.0,
        comment="历史黑名单得分：0.0=无命中 / 1.0=直接命中，blacklist_entities 模糊匹配"
    )
    blacklist_check_result = Column(
        JSON,
        comment="黑名单检测详情：{hit_entities:[{name,type,source}], match_confidence}"
    )
    # ── 综合结果 ─────────────────────────────────────────────────────────────
    overall_fraud_score = Column(
        Float, default=0.0,
        comment="综合欺诈评分：印章×0.30+重复×0.25+篡改×0.25+网络×0.15+黑名单×0.05，0.0~1.0"
    )
    fraud_level = Column(
        String(20), default="clean",
        comment="欺诈等级：clean(<0.3) / suspicious(0.3~0.6) / high_risk(0.6~0.8) / fraud(≥0.8)"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="欺诈检测完成时间"
    )

    audit_task = relationship("AuditTask", back_populates="fraud_detection")


# ══════════════════════════════════════════════════════════════════════════════
# 11. batch_tasks — 批量任务主表
#     BatchSchedulingAgent 的控制表，管理一批票据的并发审核进度
# ══════════════════════════════════════════════════════════════════════════════
class BatchTask(Base):
    __tablename__ = "batch_tasks"

    id = Column(
        String(36), primary_key=True,
        comment="批量任务唯一标识，提交后立即返回此 ID 用于进度轮询"
    )
    tenant_id = Column(
        String(36), ForeignKey("tenants.id"), nullable=False,
        comment="所属租户 ID，批量任务严格按租户隔离"
    )
    task_type = Column(
        SAEnum(AuditTaskType), nullable=False, default=AuditTaskType.FULL_AUDIT,
        comment="批次内所有子任务的业务类型（批量任务要求同质）"
    )
    total_count = Column(
        Integer, default=0,
        comment="批次内子任务总数：提交时根据上传文件数量填入"
    )
    completed_count = Column(
        Integer, default=0,
        comment="已完成子任务数（含成功和失败），每个子任务终态时原子加 1"
    )
    failed_count = Column(
        Integer, default=0,
        comment="失败子任务数，failed_count/total_count 超过阈值时整批标记为 FAILED"
    )
    concurrency = Column(
        Integer, default=10,
        comment="并发度：asyncio.Semaphore 的槽位数，默认 10，最大 100"
    )
    status = Column(
        SAEnum(BatchTaskStatus), nullable=False, default=BatchTaskStatus.PENDING,
        comment="批量任务状态：pending→running→completed/failed"
    )
    result_summary = Column(
        JSON,
        comment="批次结果汇总：{total, completed, failed, avg_score, risk_distribution:{LOW:n,...}}"
    )
    submitted_by = Column(
        String(36),
        comment="提交人的 user_id"
    )
    started_at = Column(
        DateTime,
        comment="批次开始执行时间（第一个子任务启动时写入）"
    )
    completed_at = Column(
        DateTime,
        comment="批次全部完成时间"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="批量任务提交时间"
    )

    items = relationship("BatchTaskItem", back_populates="batch_task", order_by="BatchTaskItem.item_index")


# ══════════════════════════════════════════════════════════════════════════════
# 12. batch_task_items — 批量子任务表
#     每条记录对应批量任务中的一个具体审核子任务
# ══════════════════════════════════════════════════════════════════════════════
class BatchTaskItem(Base):
    __tablename__ = "batch_task_items"

    id = Column(
        String(36), primary_key=True,
        comment="批量子任务记录唯一标识"
    )
    batch_task_id = Column(
        String(36), ForeignKey("batch_tasks.id"), nullable=False,
        comment="所属批量任务 ID，外键关联 batch_tasks.id"
    )
    audit_task_id = Column(
        String(36), ForeignKey("audit_tasks.id"),
        comment="对应的审核任务 ID，子任务被 OrchestratorAgent 受理后写入"
    )
    document_id = Column(
        String(36), ForeignKey("documents.id"),
        comment="子任务对应的文档 ID，用于追踪哪个文件是哪个子任务"
    )
    item_index = Column(
        Integer, nullable=False,
        comment="子任务在批次中的序号（0-based），用于排序和进度展示"
    )
    status = Column(
        SAEnum(AuditTaskStatus), nullable=False, default=AuditTaskStatus.PENDING,
        comment="子任务状态：独立于批量任务整体状态，便于展示每个文件的处理情况"
    )
    error_msg = Column(
        Text,
        comment="子任务失败时的错误信息"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="子任务记录创建时间"
    )

    batch_task = relationship("BatchTask", back_populates="items")


# ══════════════════════════════════════════════════════════════════════════════
# 13. error_code_mappings — 错误码字典表
#     系统所有错误码的中文说明与修复建议，前端展示友好提示时查询
# ══════════════════════════════════════════════════════════════════════════════
class ErrorCodeMapping(Base):
    __tablename__ = "error_code_mappings"

    id = Column(
        String(36), primary_key=True,
        comment="记录唯一标识"
    )
    error_code = Column(
        String(50), unique=True, nullable=False,
        comment="错误码，全系统唯一，格式：类别前缀+数字，如 E001/C002/F003"
    )
    category = Column(
        String(50), nullable=False,
        comment="错误类别：element=要素错误 / compliance=合规错误 / endorsement=背书错误 / fraud=欺诈 / system=系统异常"
    )
    severity = Column(
        SAEnum(ViolationLevel), nullable=False, default=ViolationLevel.WARNING,
        comment="错误严重程度：info=提示 / warning=警告 / severe=阻断审核"
    )
    zh_title = Column(
        String(200), nullable=False,
        comment="中文错误标题：简短说明错误类型，如「票据金额大小写不一致」"
    )
    zh_desc = Column(
        Text, nullable=False,
        comment="中文错误详细描述：说明为何是错误、影响是什么"
    )
    remediation = Column(
        Text,
        comment="修复建议：告知申请方如何修改材料以通过审核，NULL 表示系统内部错误无法修复"
    )
    is_active = Column(
        Boolean, default=True,
        comment="是否启用：False 表示该错误码已废弃，不再触发告警"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="错误码创建时间"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 14. blacklist_entities — 黑名单实体表
#     用于欺诈检测的黑名单数据，覆盖企业/个人/账户三类主体
# ══════════════════════════════════════════════════════════════════════════════
class BlacklistEntity(Base):
    __tablename__ = "blacklist_entities"

    id = Column(
        String(36), primary_key=True,
        comment="黑名单记录唯一标识"
    )
    entity_type = Column(
        SAEnum(EntityType), nullable=False,
        comment="实体类型：company=企业 / person=自然人 / account=银行账号"
    )
    entity_name = Column(
        String(500), nullable=False,
        comment="实体名称：企业全称/个人姓名/账号名称，用于模糊匹配"
    )
    entity_id = Column(
        String(200),
        comment="实体唯一标识：统一社会信用代码/身份证号/银行账号，用于精确匹配"
    )
    source = Column(
        String(200),
        comment="黑名单来源：如「中国人民银行失信名单」「司法冻结账户」「内部风控」"
    )
    reason = Column(
        Text,
        comment="列入黑名单的原因说明"
    )
    risk_score = Column(
        Float, default=1.0,
        comment="风险评分：0.0~1.0，全局黑名单默认 1.0，内部监控名单可设较低值"
    )
    is_active = Column(
        Boolean, default=True,
        comment="是否有效：False 表示已从黑名单移除（不可物理删除，保留审计记录）"
    )
    is_global = Column(
        Boolean, default=False,
        comment="是否全局黑名单：True=跨所有租户生效 / False=仅特定租户有效"
    )
    tenant_id = Column(
        String(36), ForeignKey("tenants.id"),
        comment="所属租户 ID，is_global=True 时为 NULL（全局不归属任何租户）"
    )
    expired_at = Column(
        DateTime,
        comment="黑名单到期时间：NULL 表示永久有效，到期后自动设 is_active=False"
    )
    created_at = Column(
        DateTime, server_default=func.now(),
        comment="列入黑名单时间"
    )
    updated_at = Column(
        DateTime, onupdate=func.now(),
        comment="黑名单记录最后更新时间"
    )

    __table_args__ = (
        # 联合索引：欺诈检测时频繁按类型+是否有效过滤
        Index("ix_blacklist_type_active", "entity_type", "is_active"),
    )
