# app/graph/nodes.py
# LangGraph 节点函数定义
# 职责：
#   将每个 Agent 的 MCP 工具调用封装为 LangGraph 节点函数
#   每个节点函数：
#     1. 从 BillAuditState 读取所需输入数据
#     2. 调用对应的 MCP 工具函数（进程内直接调用，无 HTTP 开销）
#     3. 返回状态增量（只包含本节点修改的字段）
#     4. 失败时将节点名写入 failed_nodes，不阻断 LangGraph 框架运行
#
# 设计约定：
#   - 节点函数签名：async def node_xxx(state: BillAuditState) -> dict
#   - 返回值只包含变更的字段（LangGraph 自动合并到完整 state）
#   - 不直接调用 Agent.run()，而是通过 MCP 工具函数（保持层次清晰）
#   - 节点失败时返回 failed_nodes/errors 增量，而非抛出异常

from __future__ import annotations

import asyncio                                            # 并行节点的 gather 调用
from loguru import logger                                # 结构化日志

from app.graph.state import BillAuditState               # 图状态类型


# ──────────────────────────────────────────────────────────────────────────────
# 私有辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def _node_failed(state: BillAuditState, node_name: str, error: str) -> dict:
    """
    构造节点失败时的状态增量

    Args:
        state:     当前完整状态（用于读取现有 failed_nodes/errors）
        node_name: 失败的节点名称
        error:     错误描述

    Returns:
        只包含 failed_nodes 和 errors 的增量 dict（不覆盖其他字段）
    """
    return {
        # 追加失败节点名称（不替换，保留之前的失败节点）
        "failed_nodes": state.get("failed_nodes", []) + [node_name],
        # 合并新的错误信息（不替换，保留之前的错误）
        "errors": {**state.get("errors", {}), node_name: error},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 节点 1：文档解析
# ──────────────────────────────────────────────────────────────────────────────

async def node_document_parser(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 parse_bill_document MCP 工具解析票据文件

    读取：state["file_path"]（票据文件路径）
    写入：state["parsed_doc"]（解析结果）
    失败：state["failed_nodes"] 追加 "document_parser"
    """
    from app.mcp.tools.document_tools import parse_bill_document  # 直接调用 MCP 工具函数

    node_name = "document_parser"                        # 节点名称（用于 failed_nodes 记录）
    file_path = state.get("file_path")                   # 从 state 读取文件路径

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={state.get('audit_task_id')} trace={state.get('trace_id')}"
    )

    if not file_path:
        # 文件路径为空：无法解析，记录失败并返回
        return _node_failed(state, node_name, "file_path 为空，无法解析文件")

    result = await parse_bill_document(                  # 调用 MCP 工具函数（进程内，无网络开销）
        file_path=file_path,
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"parsed_doc": result["data"]}                # 只返回本节点修改的字段


# ──────────────────────────────────────────────────────────────────────────────
# 节点 2：要素抽取
# ──────────────────────────────────────────────────────────────────────────────

async def node_element_extraction(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 extract_bill_elements MCP 工具抽取 18 个票据要素

    读取：state["file_path"] 或 state["bill_element"]（预填充快速路径）
    写入：state["bill_element"]（要素抽取结果）
    失败：state["failed_nodes"] 追加 "element_extraction"
    """
    from app.mcp.tools.extraction_tools import extract_bill_elements

    node_name = "element_extraction"

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    result = await extract_bill_elements(
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
        file_path=state.get("file_path"),                # 文件路径（正常路径）
        prefilled_element=state.get("bill_element"),     # 预填充要素（快速路径，已由 make_initial_state 注入）
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    # 将要素抽取结果写回 state["bill_element"]（覆盖可能存在的预填充数据，使用写库后的版本）
    return {"bill_element": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 3：合规检索（并行层成员）
# ──────────────────────────────────────────────────────────────────────────────

async def node_compliance_retrieval(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 check_compliance MCP 工具检索合规性

    读取：state["bill_element"]
    写入：state["compliance_summary"]
    """
    from app.mcp.tools.compliance_tools import check_compliance

    node_name = "compliance_retrieval"
    bill_element = state.get("bill_element")

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法检索合规性")

    result = await check_compliance(
        bill_element=bill_element,
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"compliance_summary": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 4：背书链分析（并行层成员）
# ──────────────────────────────────────────────────────────────────────────────

async def node_endorsement_chain(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 analyze_endorsement_chain MCP 工具分析背书链

    读取：state["bill_element"]
    写入：state["endorsement_result"]
    """
    from app.mcp.tools.endorsement_tools import analyze_endorsement_chain

    node_name = "endorsement_chain"
    bill_element = state.get("bill_element")

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法分析背书链")

    result = await analyze_endorsement_chain(
        bill_element=bill_element,
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"endorsement_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 5：欺诈检测（并行层成员）
# ──────────────────────────────────────────────────────────────────────────────

async def node_fraud_detection(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 detect_fraud MCP 工具检测欺诈风险

    读取：state["bill_element"]、state["endorsement_result"]（可选）
    写入：state["fraud_result"]
    """
    from app.mcp.tools.fraud_tools import detect_fraud

    node_name = "fraud_detection"
    bill_element = state.get("bill_element")

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法检测欺诈")

    result = await detect_fraud(
        bill_element=bill_element,
        endorsement_result=state.get("endorsement_result"),  # 可选：有背书链结果时传入
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"fraud_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 6：合同审核（部分业务类型使用）
# ──────────────────────────────────────────────────────────────────────────────

async def node_contract_review(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 review_contract MCP 工具审核合同

    读取：state["bill_element"]、state["contract_text"]（或 contract_document_id）
    写入：state["contract_result"]
    """
    from app.mcp.tools.contract_tools import review_contract

    node_name = "contract_review"
    bill_element = state.get("bill_element")

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法审核合同")

    result = await review_contract(
        bill_element=bill_element,
        contract_text=state.get("contract_text"),              # 合同文本（可选）
        contract_document_id=state.get("contract_document_id"), # 合同文档 ID（可选）
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"contract_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 7：出票预检（仅出票业务类型使用）
# ──────────────────────────────────────────────────────────────────────────────

async def node_bill_issuance(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 check_bill_issuance MCP 工具进行出票预检

    读取：state["bill_element"]
    写入：state["issuance_result"]
    """
    from app.mcp.tools.issuance_tools import check_bill_issuance

    node_name = "bill_issuance"
    bill_element = state.get("bill_element")

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法进行出票预检")

    result = await check_bill_issuance(
        bill_element=bill_element,
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"issuance_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 8：风险评估（所有业务类型都有）
# ──────────────────────────────────────────────────────────────────────────────

async def node_risk_assessment(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 assess_risk MCP 工具进行综合风险评估

    读取：state["compliance_summary/endorsement_result/contract_result/fraud_result"]
    写入：state["risk_result"]
    """
    from app.mcp.tools.risk_tools import assess_risk

    node_name = "risk_assessment"

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    result = await assess_risk(
        compliance_summary=state.get("compliance_summary"),   # 合规检索结果（可选，缺失时用默认分）
        endorsement_result=state.get("endorsement_result"),   # 背书链结果（可选）
        contract_result=state.get("contract_result"),         # 合同审核结果（可选）
        fraud_result=state.get("fraud_result"),               # 欺诈检测结果（可选）
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"risk_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 节点 9：报告生成（所有业务类型都有，最后执行）
# ──────────────────────────────────────────────────────────────────────────────

async def node_report_generation(state: BillAuditState) -> dict:
    """
    LangGraph 节点：调用 generate_report MCP 工具生成最终审核报告

    读取：state 中所有维度的结果字段
    写入：state["report_result"]
    """
    from app.mcp.tools.report_tools import generate_report

    node_name = "report_generation"

    logger.info(f"[node:{node_name}] 开始执行 task={state.get('audit_task_id')}")

    result = await generate_report(
        bill_element=state.get("bill_element"),
        compliance_summary=state.get("compliance_summary"),
        endorsement_result=state.get("endorsement_result"),
        contract_result=state.get("contract_result"),
        fraud_result=state.get("fraud_result"),
        risk_result=state.get("risk_result"),
        issuance_result=state.get("issuance_result"),
        audit_task_id=state.get("audit_task_id", ""),
        tenant_id=state.get("tenant_id", "mcp_caller"),
        is_final=True,                                         # LangGraph 末尾节点生成终版报告
    )

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    logger.info(f"[node:{node_name}] 执行成功 task={state.get('audit_task_id')}")
    return {"report_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 并行层聚合节点（合规+背书+欺诈 并行执行）
# ──────────────────────────────────────────────────────────────────────────────

async def node_parallel_compliance_endorse_fraud(state: BillAuditState) -> dict:
    """
    LangGraph 节点：并行执行合规检索 + 背书链分析 + 欺诈检测（三个节点同时运行）

    设计说明：
      LangGraph 的 Send API 可实现真正的并行，但需要更复杂的 Reducer 配置。
      此处使用 asyncio.gather() 在单个节点内并行调用三个 MCP 工具，
      效果与 Send API 相同（并发执行），但图结构更简单（一个节点替代三个节点的扇出/扇入）。

    写入：state["compliance_summary"]、state["endorsement_result"]、state["fraud_result"]
    """
    node_name = "parallel_compliance_endorse_fraud"

    logger.info(
        f"[node:{node_name}] 并行执行三个工具 task={state.get('audit_task_id')}"
    )

    # asyncio.gather：三个协程并发执行，总耗时 ≈ max(各自耗时) 而非 sum(各自耗时)
    compliance_task = node_compliance_retrieval(state)     # 合规检索协程
    endorse_task    = node_endorsement_chain(state)        # 背书链分析协程
    fraud_task      = node_fraud_detection(state)          # 欺诈检测协程

    results = await asyncio.gather(
        compliance_task,
        endorse_task,
        fraud_task,
        return_exceptions=False,                           # 单个工具失败不影响其他工具（已在各节点内部处理）
    )

    # 合并三个节点的返回值（每个都是增量 dict）
    merged: dict = {}
    for result in results:
        merged.update(result)                              # 合并各节点返回的字段

    # 合并 failed_nodes 和 errors（避免后一个覆盖前一个）
    all_failed_nodes = state.get("failed_nodes", [])
    all_errors = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes              # 合并所有失败节点
    merged["errors"] = all_errors                          # 合并所有错误信息

    logger.info(
        f"[node:{node_name}] 并行执行完成 "
        f"failed={len(all_failed_nodes)} task={state.get('audit_task_id')}"
    )
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 并行层变体：合规 + 出票预检（ISSUANCE_CHECK 业务类型专用）
# ──────────────────────────────────────────────────────────────────────────────

async def node_parallel_compliance_issuance(state: BillAuditState) -> dict:
    """
    LangGraph 节点：并行执行合规检索 + 出票预检（ISSUANCE_CHECK 场景）

    写入：state["compliance_summary"]、state["issuance_result"]
    """
    node_name = "parallel_compliance_issuance"

    logger.info(
        f"[node:{node_name}] 并行执行两个工具 task={state.get('audit_task_id')}"
    )

    # 并发执行合规检索和出票预检两个协程
    compliance_task = node_compliance_retrieval(state)
    issuance_task   = node_bill_issuance(state)

    results = await asyncio.gather(compliance_task, issuance_task, return_exceptions=False)

    # 合并两个节点的返回值
    merged: dict = {}
    for result in results:
        merged.update(result)

    # 合并 failed_nodes 和 errors（防止后一个覆盖前一个）
    all_failed_nodes = list(state.get("failed_nodes", []))
    all_errors       = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"]       = all_errors

    logger.info(
        f"[node:{node_name}] 并行执行完成 "
        f"failed={len(all_failed_nodes)} task={state.get('audit_task_id')}"
    )
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 并行层变体：合规 + 背书链（ACCEPTANCE_PROMPT 业务类型专用）
# ──────────────────────────────────────────────────────────────────────────────

async def node_parallel_compliance_endorse(state: BillAuditState) -> dict:
    """
    LangGraph 节点：并行执行合规检索 + 背书链分析（ACCEPTANCE_PROMPT 场景）

    写入：state["compliance_summary"]、state["endorsement_result"]
    """
    node_name = "parallel_compliance_endorse"

    logger.info(
        f"[node:{node_name}] 并行执行两个工具 task={state.get('audit_task_id')}"
    )

    # 并发执行合规检索和背书链分析
    compliance_task = node_compliance_retrieval(state)
    endorse_task    = node_endorsement_chain(state)

    results = await asyncio.gather(compliance_task, endorse_task, return_exceptions=False)

    # 合并两个节点的返回值
    merged: dict = {}
    for result in results:
        merged.update(result)

    # 合并 failed_nodes 和 errors
    all_failed_nodes = list(state.get("failed_nodes", []))
    all_errors       = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"]       = all_errors

    logger.info(
        f"[node:{node_name}] 并行执行完成 "
        f"failed={len(all_failed_nodes)} task={state.get('audit_task_id')}"
    )
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 并行层变体：背书链 + 欺诈检测（ENDORSEMENT 业务类型专用）
# ──────────────────────────────────────────────────────────────────────────────

async def node_parallel_endorse_fraud(state: BillAuditState) -> dict:
    """
    LangGraph 节点：并行执行背书链分析 + 欺诈检测（ENDORSEMENT 场景）

    写入：state["endorsement_result"]、state["fraud_result"]
    """
    node_name = "parallel_endorse_fraud"

    logger.info(
        f"[node:{node_name}] 并行执行两个工具 task={state.get('audit_task_id')}"
    )

    # 并发执行背书链分析和欺诈检测
    endorse_task = node_endorsement_chain(state)
    fraud_task   = node_fraud_detection(state)

    results = await asyncio.gather(endorse_task, fraud_task, return_exceptions=False)

    # 合并两个节点的返回值
    merged: dict = {}
    for result in results:
        merged.update(result)

    # 合并 failed_nodes 和 errors
    all_failed_nodes = list(state.get("failed_nodes", []))
    all_errors       = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"]       = all_errors

    logger.info(
        f"[node:{node_name}] 并行执行完成 "
        f"failed={len(all_failed_nodes)} task={state.get('audit_task_id')}"
    )
    return merged
