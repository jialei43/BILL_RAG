# app/graph/state.py
# LangGraph 图状态定义
# 职责：
#   1. 定义 BillAuditState（TypedDict），替代原 AgentContext.shared_data 内存存储
#   2. 所有字段均可序列化为 JSON，支持 PostgresSaver 持久化到数据库
#   3. 每个 LangGraph 节点函数读取并更新此状态
#   4. 多容器部署时，任意容器副本都能从 PostgreSQL 恢复完整状态
#
# 设计原则：
#   - 状态字段全部可选（Optional），允许节点按顺序逐步填充
#   - 控制字段（failed_nodes/skipped_nodes）驱动条件边的路由逻辑
#   - trace_id 贯穿所有节点，确保多容器日志可聚合追踪

from __future__ import annotations

from typing import Optional                               # 可选字段类型注解
from typing_extensions import TypedDict                  # TypedDict：强类型字典，LangGraph 状态基础类型


class BillAuditState(TypedDict, total=False):
    """
    票据审核图状态（替代 AgentContext.shared_data 的内存存储）

    total=False 表示所有字段均为可选，允许节点只更新自身相关字段。
    LangGraph 框架会自动合并每个节点的返回值到完整状态中。

    持久化：所有字段均为 Python 原生类型（str/dict/list），
    PostgresSaver 可直接将其序列化为 JSON 存入 PostgreSQL。
    """

    # ── 任务标识（所有节点只读，不修改）────────────────────────────────────────
    audit_task_id: str          # 主任务 UUID（与 audit_tasks 表的主键对应）
    tenant_id: str              # 租户 ID（多租户隔离，确保不跨租户读写数据）
    task_type: str              # 业务类型（如 "full_audit"，决定图结构）
    trace_id: str               # 全链路追踪 ID（多容器日志聚合用，每次请求唯一）

    # ── 文件输入（由请求层填入，节点只读）─────────────────────────────────────
    file_path: Optional[str]    # 待审核的票据文件路径（DocumentParserAgent 使用）
    contract_text: Optional[str]  # 合同文本（ContractReviewAgent 使用，可选）
    contract_document_id: Optional[str]  # 合同文档 ID（与 contract_text 二选一）

    # ── 各 Agent 执行结果（每个节点执行后填入对应字段）─────────────────────────
    parsed_doc: Optional[dict]          # DocumentParserAgent 的解析输出
    bill_element: Optional[dict]        # ElementExtractionAgent 的 18 字段要素输出
    compliance_summary: Optional[dict]  # ComplianceRetrievalAgent 的合规汇总
    endorsement_result: Optional[dict]  # EndorsementChainAgent 的背书链分析
    fraud_result: Optional[dict]        # FraudDetectionAgent 的欺诈检测结果
    contract_result: Optional[dict]     # ContractReviewAgent 的合同匹配结果
    risk_result: Optional[dict]         # RiskAssessmentAgent 的综合风险评分
    report_result: Optional[dict]       # ReportGenerationAgent 的审核报告
    issuance_result: Optional[dict]     # BillIssuanceAgent 的出票预检结果
    flow_result: Optional[dict]         # FlowTrackingAgent 的流转追踪结果

    # ── 执行控制（条件边通过这两个字段决定路由）────────────────────────────────
    failed_nodes: list          # 失败节点名称列表（关键节点失败时，后继节点被跳过）
    skipped_nodes: list         # 被跳过的节点名称列表（前置关键节点失败导致）
    errors: dict                # 各节点的错误详情 {node_name: error_message}


# ──────────────────────────────────────────────────────────────────────────────
# 状态初始化工具函数
# ──────────────────────────────────────────────────────────────────────────────

def make_initial_state(
    audit_task_id: str,         # 任务 UUID（由请求层生成）
    tenant_id: str,             # 租户 ID
    task_type: str,             # 业务类型字符串（如 "full_audit"）
    trace_id: str,              # 请求级别的追踪 ID
    file_path: Optional[str] = None,           # 票据文件路径（可选）
    contract_text: Optional[str] = None,       # 合同文本（可选）
    contract_document_id: Optional[str] = None, # 合同文档 ID（可选）
    prefilled_element: Optional[dict] = None,  # 预填充的票据要素（跳过 OCR 快速路径）
) -> BillAuditState:
    """
    构造 LangGraph 图执行的初始状态

    Args:
        audit_task_id:     任务 UUID，与 audit_tasks 表对应
        tenant_id:         租户标识，确保数据隔离
        task_type:         业务类型（决定构建哪种 StateGraph）
        trace_id:          全链路追踪 ID（由 Trace ID 中间件生成）
        file_path:         待解析的票据文件路径
        contract_text:     贸易合同文本（贴现/合同审核场景必须）
        contract_document_id: 合同文档 ID（与 contract_text 二选一）
        prefilled_element: 已识别的票据要素 dict（跳过 OCR，加快审核速度）

    Returns:
        BillAuditState：已填充基础字段的初始状态，供 LangGraph 图执行
    """
    initial: BillAuditState = {
        # 任务标识字段：贯穿所有节点，不会被任何节点修改
        "audit_task_id": audit_task_id,
        "tenant_id":     tenant_id,
        "task_type":     task_type,
        "trace_id":      trace_id,

        # 输入文件字段：由请求层填入，各节点按需读取
        "file_path":             file_path,
        "contract_text":         contract_text,
        "contract_document_id":  contract_document_id,

        # 执行控制字段：初始为空列表/字典，节点失败时追加
        "failed_nodes":  [],    # 初始无失败节点
        "skipped_nodes": [],    # 初始无跳过节点
        "errors":        {},    # 初始无错误
    }

    # 如果提供了预填充要素，提前注入 bill_element 字段
    # ElementExtractionAgent 检测到此字段时走快速路径（跳过 Qwen-VL OCR）
    if prefilled_element:
        initial["bill_element"] = prefilled_element  # 注入预识别的要素，跳过 OCR 步骤

    return initial


# ──────────────────────────────────────────────────────────────────────────────
# 状态与 AgentContext 的转换工具
# ──────────────────────────────────────────────────────────────────────────────

def state_to_agent_context(state: BillAuditState):
    """
    将 BillAuditState 转换为 AgentContext，供各 Agent.run() 使用

    设计说明：
      LangGraph 节点函数接收 BillAuditState，
      但 Agent.run() 需要 AgentContext（含 shared_data）。
      本函数在两者之间做桥接，避免修改 Agent 的现有接口。

    Args:
        state: 当前 LangGraph 图状态

    Returns:
        AgentContext：可直接传入任何 Agent.run() 的上下文对象
    """
    from app.agents.base_agent import AgentContext  # 延迟导入，避免循环依赖

    # 将 state 中所有 Agent 结果字段提取到 shared_data dict
    # （对应原来各 Agent 从 ctx.shared_data 读取前序结果的方式）
    shared_data = {}
    result_fields = [                              # state 中所有存储 Agent 结果的字段名
        "parsed_doc", "bill_element", "compliance_summary",
        "endorsement_result", "fraud_result", "contract_result",
        "risk_result", "report_result", "issuance_result", "flow_result",
    ]
    for field in result_fields:
        value = state.get(field)                   # 从 state 读取字段值
        if value is not None:
            shared_data[field] = value             # 非空字段写入 shared_data

    return AgentContext(
        audit_task_id=state.get("audit_task_id", ""),   # 任务 UUID
        tenant_id=state.get("tenant_id", ""),            # 租户 ID
        shared_data=shared_data,                          # 前序 Agent 结果的汇总
    )
