# app/graph/builder.py
# LangGraph 图构建器
# 职责：
#   按业务类型（task_type）构建对应的 StateGraph 并编译，
#   替代 orchestrator_agent.py 中的 TASK_TYPE_TO_DAG + DAGExecutor 机制。
#
# 支持的业务类型（8种，与原 TASK_TYPE_TO_DAG 对应）：
#   FULL_AUDIT        全流程审核（parse→extract→parallel_cef→contract→risk→report）
#   ISSUANCE_CHECK    出票预检  （parse→extract→parallel_ci→risk→report）
#   DISCOUNT_APPLY    贴现申请  （parse→extract→parallel_cef→contract*→risk→report）
#   ACCEPTANCE_PROMPT 提示承兑  （parse→extract→parallel_ce→risk→report）
#   ENDORSEMENT       背书转让  （parse→extract→parallel_ef→risk→report）
#   PAYMENT_PROMPT    提示付款  （同 FULL_AUDIT）
#   PLEDGE            质押背书  （同 FULL_AUDIT）
#   COLLECTION        托收委托  （同 FULL_AUDIT）
#
# *注：contract 在 DISCOUNT_APPLY 中是关键节点，在 FULL_AUDIT 中是非关键节点

from __future__ import annotations

from typing import Optional

from loguru import logger                               # 日志

from langgraph.graph import StateGraph, END             # LangGraph 图类型和终止符

from app.graph.state import BillAuditState             # 图状态类型（TypedDict）

# ── 节点函数（每个节点调用对应 MCP 工具）──────────────────────────────────────
from app.graph.nodes import (
    node_document_parser,                              # 节点 1：文档解析
    node_element_extraction,                           # 节点 2：要素抽取
    node_parallel_compliance_endorse_fraud,            # 节点 3A：合规+背书+欺诈 并行层
    node_parallel_compliance_issuance,                 # 节点 3B：合规+出票预检 并行层
    node_parallel_compliance_endorse,                  # 节点 3C：合规+背书 并行层
    node_parallel_endorse_fraud,                       # 节点 3D：背书+欺诈 并行层
    node_contract_review,                              # 节点 4：合同审核（部分类型）
    node_risk_assessment,                              # 节点 5：风险评估
    node_report_generation,                            # 节点 6：报告生成
)

# ── 条件边函数（控制节点失败后的路由）──────────────────────────────────────────
from app.graph.edges import (
    after_parse,                                       # parse 后路由
    after_extract,                                     # extract 后路由
    after_risk,                                        # risk 后路由
    after_contract_optional,                           # contract 后路由（非关键版）
    after_contract_critical,                           # contract 后路由（关键版）
    after_parallel_cef_to_contract,                    # 合规+背书+欺诈 → contract（FULL_AUDIT 等）
    after_parallel_cef_discount,                       # 合规+背书+欺诈 → contract（DISCOUNT_APPLY）
    after_parallel_ci,                                 # 合规+出票预检 → risk（ISSUANCE_CHECK）
    after_parallel_ce,                                 # 合规+背书 → risk（ACCEPTANCE_PROMPT）
    after_parallel_ef,                                 # 背书+欺诈 → risk（ENDORSEMENT）
)


# ──────────────────────────────────────────────────────────────────────────────
# 内部工具：构建公共起始段（parse → extract）
# ──────────────────────────────────────────────────────────────────────────────

def _add_common_start(graph: StateGraph, parallel_node_name: str) -> None:
    """
    为所有业务类型添加公共的起始节点和边

    所有图都以 parse → extract → parallel 开头，只是并行节点名称不同。

    Args:
        graph:             待构建的 StateGraph 实例
        parallel_node_name: 并行层节点的名称（如 "parallel"），用于 after_extract 路由
    """
    graph.add_node("parse",   node_document_parser)    # 节点：文档解析
    graph.add_node("extract", node_element_extraction)  # 节点：要素抽取

    graph.set_entry_point("parse")                     # 所有图从 parse 开始

    # parse 完成后：失败 → END，成功 → extract
    graph.add_conditional_edges(
        "parse",
        after_parse,
        {"end": END, "extract": "extract"},            # 路由映射（after_parse 返回 "end" 或 "extract"）
    )

    # extract 完成后：失败 → END，成功 → 并行层
    # 注意：after_extract 返回 "parallel" 作为目标，需映射到实际节点名
    graph.add_conditional_edges(
        "extract",
        after_extract,
        {"end": END, "parallel": parallel_node_name},  # "parallel" 映射到传入的并行节点名
    )


def _add_common_end(graph: StateGraph) -> None:
    """
    为所有业务类型添加公共的结束段（risk → report → END）

    Args:
        graph: 待构建的 StateGraph 实例
    """
    graph.add_node("risk",   node_risk_assessment)     # 节点：风险评估
    graph.add_node("report", node_report_generation)   # 节点：报告生成

    # risk 完成后：失败 → END，成功 → report
    graph.add_conditional_edges(
        "risk",
        after_risk,
        {"end": END, "report": "report"},
    )

    graph.add_edge("report", END)                      # report 执行完毕 → 图结束


# ──────────────────────────────────────────────────────────────────────────────
# 各业务类型的图构建函数
# ──────────────────────────────────────────────────────────────────────────────

def _build_full_audit_graph(checkpointer) -> object:
    """
    构建全流程审核图（FULL_AUDIT / PAYMENT_PROMPT / PLEDGE / COLLECTION）

    拓扑：parse → extract → parallel(合规+背书+欺诈) → contract（非关键）→ risk → report

    合同审核在全流程中是可选步骤（is_critical=False），
    即使失败也不阻断风险评估（风险模型对缺失的合同结果使用默认分）
    """
    graph = StateGraph(BillAuditState)                 # 以 BillAuditState 为状态类型创建有向图

    _add_common_start(graph, "parallel")               # 添加 parse → extract → parallel 公共段

    graph.add_node("parallel", node_parallel_compliance_endorse_fraud)  # 并行层：合规+背书+欺诈

    # 并行层完成后：compliance 或 endorsement 失败 → END，否则 → contract
    graph.add_conditional_edges(
        "parallel",
        after_parallel_cef_to_contract,
        {"end": END, "contract": "contract"},
    )

    graph.add_node("contract", node_contract_review)   # 合同审核节点（非关键）

    # 合同审核完成后：无论成败都继续 → risk（非关键节点，失败继续）
    graph.add_conditional_edges(
        "contract",
        after_contract_optional,
        {"risk": "risk"},                              # 只有一个出口：risk
    )

    _add_common_end(graph)                             # 添加 risk → report → END 公共段

    return graph.compile(checkpointer=checkpointer)    # 编译图（注入 PostgresSaver 检查点）


def _build_issuance_graph(checkpointer) -> object:
    """
    构建出票预检图（ISSUANCE_CHECK）

    拓扑：parse → extract → parallel(合规+出票预检) → risk → report

    合规和出票预检都是关键节点（额度不足/合规不达标均无法出票）
    无合同审核步骤（出票场景不涉及合同匹配）
    """
    graph = StateGraph(BillAuditState)

    _add_common_start(graph, "parallel")               # parse → extract → parallel

    graph.add_node("parallel", node_parallel_compliance_issuance)  # 并行层：合规+出票预检

    # 并行层完成后：合规或出票预检失败 → END，否则 → risk
    graph.add_conditional_edges(
        "parallel",
        after_parallel_ci,
        {"end": END, "risk": "risk"},
    )

    _add_common_end(graph)                             # risk → report → END

    return graph.compile(checkpointer=checkpointer)


def _build_discount_graph(checkpointer) -> object:
    """
    构建贴现申请图（DISCOUNT_APPLY）

    拓扑：parse → extract → parallel(合规+背书+欺诈) → contract（关键）→ risk → report

    与 FULL_AUDIT 拓扑相同，但合同审核是关键节点（is_critical=True）：
    贴现业务必须有合同匹配，合同审核失败意味着无法贴现
    """
    graph = StateGraph(BillAuditState)

    _add_common_start(graph, "parallel")               # parse → extract → parallel

    graph.add_node("parallel", node_parallel_compliance_endorse_fraud)  # 并行层：合规+背书+欺诈

    # 并行层完成后：合规或背书失败 → END，否则 → contract
    graph.add_conditional_edges(
        "parallel",
        after_parallel_cef_discount,
        {"end": END, "contract": "contract"},
    )

    graph.add_node("contract", node_contract_review)   # 合同审核节点（关键节点！）

    # 贴现场景合同审核失败 → END（与 FULL_AUDIT 的区别）
    graph.add_conditional_edges(
        "contract",
        after_contract_critical,
        {"end": END, "risk": "risk"},
    )

    _add_common_end(graph)                             # risk → report → END

    return graph.compile(checkpointer=checkpointer)


def _build_acceptance_graph(checkpointer) -> object:
    """
    构建提示承兑图（ACCEPTANCE_PROMPT）

    拓扑：parse → extract → parallel(合规+背书) → risk → report

    承兑场景无需欺诈检测和合同审核，重点是合规性和背书链完整性
    """
    graph = StateGraph(BillAuditState)

    _add_common_start(graph, "parallel")               # parse → extract → parallel

    graph.add_node("parallel", node_parallel_compliance_endorse)  # 并行层：合规+背书

    # 并行层完成后：合规或背书失败 → END，否则 → risk
    graph.add_conditional_edges(
        "parallel",
        after_parallel_ce,
        {"end": END, "risk": "risk"},
    )

    _add_common_end(graph)                             # risk → report → END

    return graph.compile(checkpointer=checkpointer)


def _build_endorsement_graph(checkpointer) -> object:
    """
    构建背书转让图（ENDORSEMENT）

    拓扑：parse → extract → parallel(背书+欺诈) → risk → report

    背书转让场景重点是背书链分析和欺诈检测，无需合规检索
    背书链是关键节点，欺诈检测是可选的
    """
    graph = StateGraph(BillAuditState)

    _add_common_start(graph, "parallel")               # parse → extract → parallel

    graph.add_node("parallel", node_parallel_endorse_fraud)  # 并行层：背书+欺诈

    # 并行层完成后：背书链失败 → END（欺诈检测失败不阻断），否则 → risk
    graph.add_conditional_edges(
        "parallel",
        after_parallel_ef,
        {"end": END, "risk": "risk"},
    )

    _add_common_end(graph)                             # risk → report → END

    return graph.compile(checkpointer=checkpointer)


# ──────────────────────────────────────────────────────────────────────────────
# 业务类型 → 图构建函数 映射
# ──────────────────────────────────────────────────────────────────────────────

# 对应原 orchestrator_agent.py 中的 TASK_TYPE_TO_DAG
# PAYMENT_PROMPT / PLEDGE / COLLECTION 与 FULL_AUDIT 拓扑完全相同
_TASK_TYPE_TO_BUILDER = {
    "full_audit":        _build_full_audit_graph,      # 全流程审核
    "issuance_check":    _build_issuance_graph,         # 出票预检
    "discount_apply":    _build_discount_graph,         # 贴现申请
    "acceptance_prompt": _build_acceptance_graph,       # 提示承兑
    "endorsement":       _build_endorsement_graph,      # 背书转让
    "payment_prompt":    _build_full_audit_graph,       # 提示付款（同全流程）
    "pledge":            _build_full_audit_graph,       # 质押背书（同全流程）
    "collection":        _build_full_audit_graph,       # 托收委托（同全流程）
}


# ──────────────────────────────────────────────────────────────────────────────
# 公共 API：build_audit_graph
# ──────────────────────────────────────────────────────────────────────────────

def build_audit_graph(task_type: str, checkpointer: Optional[object] = None) -> object:
    """
    根据业务类型构建并编译 LangGraph 审核图

    Args:
        task_type:    业务类型字符串（如 "full_audit"，对应 AuditTaskType.value）
        checkpointer: AsyncPostgresSaver 实例，用于多容器状态持久化（可为 None）
                      为 None 时图仍可运行，但状态不会持久化（适合单次调用场景）

    Returns:
        编译后的 LangGraph 图（CompiledStateGraph），可直接调用 astream() 或 ainvoke()

    Raises:
        ValueError: task_type 不在支持列表中（此时兜底使用 full_audit 图并记录警告）
    """
    builder_fn = _TASK_TYPE_TO_BUILDER.get(task_type)  # 查找对应业务类型的构建函数

    if builder_fn is None:
        # 未知业务类型：记录警告并降级为全流程审核（与原 orchestrator 行为一致）
        logger.warning(
            f"[graph.builder] 未知 task_type={task_type!r}，降级为 full_audit"
        )
        builder_fn = _build_full_audit_graph

    logger.info(f"[graph.builder] 构建图 task_type={task_type}")
    return builder_fn(checkpointer)                    # 调用对应的构建函数并返回编译好的图


def get_supported_task_types() -> list:
    """
    返回支持的业务类型列表

    Returns:
        字符串列表，包含所有支持的 task_type 值
    """
    return list(_TASK_TYPE_TO_BUILDER.keys())
