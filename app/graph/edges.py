# app/graph/edges.py
# LangGraph 条件边函数定义
# 职责：
#   根据 BillAuditState 中的 failed_nodes 字段决定图执行的下一个节点
#   复现原 DAGExecutor 的关键节点失败传播逻辑：
#     - 关键节点（is_critical=True）失败 → 后续节点全部跳过（→ END）
#     - 非关键节点失败 → 记录失败但继续执行后续节点
#
# 设计约定：
#   - 每个边函数接收 BillAuditState，返回 str（下一节点名 或 "end"）
#   - "end" 映射到 LangGraph 的 END 常量（在 builder.py 的 add_conditional_edges 中配置）
#   - make_after_parallel() 工厂函数避免为每种业务类型重复写相同的判断逻辑

from __future__ import annotations

from typing import Set                                   # 关键节点集合类型

from app.graph.state import BillAuditState               # 图状态类型


# ──────────────────────────────────────────────────────────────────────────────
# 通用关键节点：parse 和 extract 在所有业务类型中都是关键节点
# ──────────────────────────────────────────────────────────────────────────────

"""
FULL_AUDIT        = "full_audit"         # 全流程审核：依次执行所有 12 个 Agent
    ISSUANCE_CHECK    = "issuance_check"     # 出票合规预检：仅执行出票相关子集
    DISCOUNT_APPLY    = "discount_apply"     # 贴现申请审核：侧重合同审核 + 贸易背景
    ACCEPTANCE_PROMPT = "acceptance_prompt"  # 提示承兑：侧重背书链 + 承兑行资质
    ENDORSEMENT       = "endorsement"        # 背书转让：侧重背书连续性核查
    PAYMENT_PROMPT    = "payment_prompt"     # 提示付款：到期要素 + 账户校验
    PLEDGE            = "pledge"             # 质押背书：法律要素 + 质押登记
    COLLECTION        = "collection"         # 托收委托：委托链 + 资金流向
"""

def after_parse(state: BillAuditState) -> str:
    """
    文档解析节点完成后的路由

    document_parser 是所有业务类型的第一个关键节点：
    失败时整张图无法继续（没有文本就无法做任何后续分析）
    """
    if "document_parser" in state.get("failed_nodes", []):
        return "end"                                    # 关键节点失败 → 终止图执行
    return "extract"                                    # 成功 → 进入要素抽取节点


def after_extract(state: BillAuditState) -> str:
    """
    要素抽取节点完成后的路由

    element_extraction 是所有业务类型的第二个关键节点：
    失败时无票据要素，无法进行任何合规/风险分析
    """
    if "element_extraction" in state.get("failed_nodes", []):
        return "end"                                    # 关键节点失败 → 终止图执行
    return "parallel"                                   # 成功 → 进入并行层（各业务类型不同）


def after_risk(state: BillAuditState) -> str:
    """
    风险评估节点完成后的路由

    risk_assessment 失败时无风险评分，无法生成有意义的报告，但仍尝试生成包含错误信息的报告
    """
    if "risk_assessment" in state.get("failed_nodes", []):
        return "end"                                    # 风险评估失败 → 终止（无分数无法出报告）
    return "report"                                     # 成功 → 生成报告


def after_contract_optional(state: BillAuditState) -> str:
    """
    合同审核节点完成后的路由（非关键节点版本，如 FULL_AUDIT）

    FULL_AUDIT 中合同审核是可选的（is_critical=False），
    即使失败也继续进入风险评估（合同结果为 None 时风险评估使用缺省分）
    """
    return "risk"                                       # 无论成败，都进入风险评估


def after_contract_critical(state: BillAuditState) -> str:
    """
    合同审核节点完成后的路由（关键节点版本，如 DISCOUNT_APPLY）

    DISCOUNT_APPLY 中合同审核是关键节点（is_critical=True），
    失败时直接终止图执行
    """
    if "contract_review" in state.get("failed_nodes", []):
        return "end"                                    # 关键合同节点失败 → 终止
    return "risk"                                       # 成功 → 风险评估


# ──────────────────────────────────────────────────────────────────────────────
# 并行层条件边工厂函数
# ──────────────────────────────────────────────────────────────────────────────

def make_after_parallel(critical_nodes: Set[str], success_target: str):
    """
    生成并行层完成后的条件边函数（工厂函数）

    设计说明：
      不同业务类型的并行层包含不同的关键节点组合。
      用工厂函数避免为每种业务类型重复写相同的判断逻辑，
      通过参数注入关键节点集合和成功目标节点名。

    Args:
        critical_nodes:  此业务类型中，并行层内哪些节点是关键节点（失败则 END）
        success_target:  所有关键节点均成功时路由到的节点名（通常是 "risk" 或 "contract"）

    Returns:
        条件边函数（接受 state，返回 "end" 或 success_target）

    示例：
        # ISSUANCE_CHECK：compliance 和 bill_issuance 都是关键节点，成功后进入 risk
        edge_fn = make_after_parallel({"compliance_retrieval", "bill_issuance"}, "risk")
    """
    def after_parallel(state: BillAuditState) -> str:
        """并行层完成后的条件路由（由工厂函数注入关键节点集和目标节点）"""
        failed_nodes = set(state.get("failed_nodes", []))
        if critical_nodes & failed_nodes:               # 集合交集：有关键节点失败
            return "end"                                # 终止图执行
        return success_target                           # 所有关键节点均成功 → 继续

    return after_parallel                               # 返回闭包函数


# ──────────────────────────────────────────────────────────────────────────────
# 各业务类型的并行层条件边（通过工厂函数生成）
# ──────────────────────────────────────────────────────────────────────────────

# FULL_AUDIT / PAYMENT_PROMPT / PLEDGE / COLLECTION：
#   并行层 = compliance + endorse + fraud
#   关键节点：compliance（合规是法定要求）、endorse（背书链完整性必须验证）
#   fraud 非关键（欺诈检测失败不影响合规判断，继续但记录）
#   成功后进入 contract（全流程包含合同审核）
after_parallel_cef_to_contract = make_after_parallel(
    critical_nodes={"compliance_retrieval", "endorsement_chain"},
    success_target="contract",
)

# DISCOUNT_APPLY：
#   并行层 = compliance + endorse + fraud（与 FULL_AUDIT 相同）
#   关键节点相同，但成功后同样进入 contract（贴现场景合同是关键）
after_parallel_cef_discount = make_after_parallel(
    critical_nodes={"compliance_retrieval", "endorsement_chain"},
    success_target="contract",                          # 贴现场景合同是关键节点
)

# ISSUANCE_CHECK：
#   并行层 = compliance + issuance
#   两个节点都是关键节点（出票合规和额度检查缺一不可）
#   成功后进入 risk
after_parallel_ci = make_after_parallel(
    critical_nodes={"compliance_retrieval", "bill_issuance"},
    success_target="risk",
)

# ACCEPTANCE_PROMPT：
#   并行层 = compliance + endorse
#   两个节点都是关键节点（提示承兑需要完整合规和背书链）
#   成功后进入 risk
after_parallel_ce = make_after_parallel(
    critical_nodes={"compliance_retrieval", "endorsement_chain"},
    success_target="risk",
)

# ENDORSEMENT：
#   并行层 = endorse + fraud
#   endorse 是关键节点（背书链是核心，必须验证）
#   fraud 非关键（检测失败不影响背书链判断）
#   成功后进入 risk
after_parallel_ef = make_after_parallel(
    critical_nodes={"endorsement_chain"},
    success_target="risk",
)
