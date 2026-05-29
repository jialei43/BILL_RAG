# app/graph/nodes.py
# LangGraph 节点函数定义
# 职责：
#   将每个 Agent 的 MCP 工具调用封装为 LangGraph 节点函数
#   每个节点函数：
#     1. 从 BillAuditState 读取所需输入数据
#     2. 通过 MCPClient 发 HTTP 请求到独立 MCP Server 调用对应工具
#     3. 返回状态增量（只包含本节点修改的字段）
#     4. 失败时将节点名写入 failed_nodes，不阻断 LangGraph 框架运行
#
# 设计约定：
#   - 节点函数签名：async def node_xxx(state: BillAuditState) -> dict
#   - 返回值只包含变更的字段（LangGraph 自动合并到完整 state）
#   - 不直接 import 工具函数，统一通过 get_mcp_client().call_tool() 调用
#   - 节点失败时返回 failed_nodes/errors 增量，而非抛出异常

from __future__ import annotations

import asyncio                                            # 并行节点的 gather 调用
from loguru import logger                                # 结构化日志

from app.graph.state import BillAuditState               # 图状态类型
from app.mcp.client import get_mcp_client                # HTTP MCP 客户端单例


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
    logger.error(
        f"[node:{node_name}] 节点执行失败 "
        f"task={state.get('audit_task_id')} error={error}"
    )
    return {
        "failed_nodes": state.get("failed_nodes", []) + [node_name],
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
    node_name = "document_parser"
    file_path = state.get("file_path")
    task_id   = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} file_path={file_path} trace={state.get('trace_id')}"
    )

    if not file_path:
        return _node_failed(state, node_name, "file_path 为空，无法解析文件")

    result = await get_mcp_client().call_tool("parse_bill_document", {
        "file_path":    file_path,
        "audit_task_id": state.get("audit_task_id", ""),
        "tenant_id":    state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} elements={data.get('element_count', '?')} "
        f"pages={data.get('page_count', '?')} confidence={data.get('confidence', '?')}"
    )
    return {"parsed_doc": result["data"]}


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
    node_name   = "element_extraction"
    task_id     = state.get("audit_task_id")
    has_prefill = bool(state.get("bill_element"))

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} file_path={state.get('file_path')} prefilled={has_prefill}"
    )

    result = await get_mcp_client().call_tool("extract_bill_elements", {
        "audit_task_id":    state.get("audit_task_id", ""),
        "tenant_id":        state.get("tenant_id", "mcp_caller"),
        "file_path":        state.get("file_path"),
        "prefilled_element": state.get("bill_element"),   # 预填充路径：MCP Server 从此参数或 Redis 读取
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} ticket={data.get('ticket_number')} "
        f"confidence={data.get('confidence_score') or data.get('confidence')} "
        f"source={data.get('source')}"
    )
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
    node_name    = "compliance_retrieval"
    bill_element = state.get("bill_element")
    task_id      = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} ticket={bill_element.get('ticket_number') if bill_element else None}"
    )

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法检索合规性")

    result = await get_mcp_client().call_tool("check_compliance", {
        "bill_element":  bill_element,
        "audit_task_id": state.get("audit_task_id", ""),
        "tenant_id":     state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} total={data.get('total_fields')} "
        f"violations={data.get('violation_count')} rate={data.get('compliance_rate')}"
    )
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
    node_name    = "endorsement_chain"
    bill_element = state.get("bill_element")
    task_id      = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} ticket={bill_element.get('ticket_number') if bill_element else None} "
        f"endorsers={len(bill_element.get('endorsers') or []) if bill_element else 0}"
    )

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法分析背书链")

    result = await get_mcp_client().call_tool("analyze_endorsement_chain", {
        "bill_element":  bill_element,
        "audit_task_id": state.get("audit_task_id", ""),
        "tenant_id":     state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} continuous={data.get('is_continuous')} "
        f"violations={data.get('violation_count')} codes={data.get('violation_codes')}"
    )
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
    node_name    = "fraud_detection"
    bill_element = state.get("bill_element")
    task_id      = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} ticket={bill_element.get('ticket_number') if bill_element else None}"
    )

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法检测欺诈")

    result = await get_mcp_client().call_tool("detect_fraud", {
        "bill_element":      bill_element,
        "endorsement_result": state.get("endorsement_result"),
        "audit_task_id":     state.get("audit_task_id", ""),
        "tenant_id":         state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} fraud_score={data.get('overall_fraud_score')}"
    )
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
    node_name    = "contract_review"
    bill_element = state.get("bill_element")
    task_id      = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} ticket={bill_element.get('ticket_number') if bill_element else None} "
        f"has_contract_text={bool(state.get('contract_text'))} "
        f"contract_doc_id={state.get('contract_document_id')}"
    )

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法审核合同")

    result = await get_mcp_client().call_tool("review_contract", {
        "bill_element":        bill_element,
        "contract_text":       state.get("contract_text"),
        "contract_document_id": state.get("contract_document_id"),
        "audit_task_id":       state.get("audit_task_id", ""),
        "tenant_id":           state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} match_score={data.get('match_score')} "
        f"mismatches={data.get('mismatch_count')}"
    )
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
    node_name    = "bill_issuance"
    bill_element = state.get("bill_element")
    task_id      = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 "
        f"task={task_id} ticket={bill_element.get('ticket_number') if bill_element else None}"
    )

    if not bill_element:
        return _node_failed(state, node_name, "bill_element 为空，无法进行出票预检")

    result = await get_mcp_client().call_tool("check_bill_issuance", {
        "bill_element":  bill_element,
        "audit_task_id": state.get("audit_task_id", ""),
        "tenant_id":     state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 "
        f"task={task_id} passed={data.get('passed')} issues={data.get('issue_count')}"
    )
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
    node_name = "risk_assessment"
    task_id   = state.get("audit_task_id")

    logger.info(
        f"[node:{node_name}] 开始执行 task={task_id} "
        f"has_compliance={bool(state.get('compliance_summary'))} "
        f"has_endorsement={bool(state.get('endorsement_result'))} "
        f"has_contract={bool(state.get('contract_result'))} "
        f"has_fraud={bool(state.get('fraud_result'))}"
    )

    result = await get_mcp_client().call_tool("assess_risk", {
        "compliance_summary":  state.get("compliance_summary"),
        "endorsement_result":  state.get("endorsement_result"),
        "contract_result":     state.get("contract_result"),
        "fraud_result":        state.get("fraud_result"),
        "audit_task_id":       state.get("audit_task_id", ""),
        "tenant_id":           state.get("tenant_id", "mcp_caller"),
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 task={task_id} "
        f"score={data.get('composite_score')} level={data.get('risk_level')} "
        f"missing={data.get('missing_dimensions')}"
    )
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
    node_name = "report_generation"
    task_id   = state.get("audit_task_id")

    risk_data = state.get("risk_result") or {}
    logger.info(
        f"[node:{node_name}] 开始执行 task={task_id} "
        f"risk_level={risk_data.get('risk_level')} score={risk_data.get('composite_score')}"
    )

    result = await get_mcp_client().call_tool("generate_report", {
        "bill_element":       state.get("bill_element"),
        "compliance_summary": state.get("compliance_summary"),
        "endorsement_result": state.get("endorsement_result"),
        "contract_result":    state.get("contract_result"),
        "fraud_result":       state.get("fraud_result"),
        "risk_result":        state.get("risk_result"),
        "issuance_result":    state.get("issuance_result"),
        "audit_task_id":      state.get("audit_task_id", ""),
        "tenant_id":          state.get("tenant_id", "mcp_caller"),
        "is_final":           True,
    })

    if not result["success"]:
        return _node_failed(state, node_name, result.get("error", "未知错误"))

    data = result["data"] or {}
    logger.info(
        f"[node:{node_name}] 执行成功 task={task_id} "
        f"report_id={data.get('report_id')} has_pdf={bool(data.get('pdf_path'))}"
    )
    return {"report_result": result["data"]}


# ──────────────────────────────────────────────────────────────────────────────
# 并行层聚合节点（合规+背书+欺诈 并行执行）
# ──────────────────────────────────────────────────────────────────────────────

async def node_parallel_compliance_endorse_fraud(state: BillAuditState) -> dict:
    """
    LangGraph 节点：并行执行合规检索 + 背书链分析 + 欺诈检测（三个节点同时运行）

    设计说明：
      使用 asyncio.gather() 在单个节点内并行调用三个 MCP 工具，
      总耗时 ≈ max(三者耗时)。三个 HTTP 请求并发发往 MCP Server。

    写入：state["compliance_summary"]、state["endorsement_result"]、state["fraud_result"]
    """
    node_name = "parallel_compliance_endorse_fraud"

    logger.info(
        f"[node:{node_name}] 并行执行三个工具 task={state.get('audit_task_id')}"
    )

    results = await asyncio.gather(
        node_compliance_retrieval(state),
        node_endorsement_chain(state),
        node_fraud_detection(state),
        return_exceptions=False,
    )

    merged: dict = {}
    for result in results:
        merged.update(result)

    all_failed_nodes = state.get("failed_nodes", [])
    all_errors = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"] = all_errors

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

    results = await asyncio.gather(
        node_compliance_retrieval(state),
        node_bill_issuance(state),
        return_exceptions=False,
    )

    merged: dict = {}
    for result in results:
        merged.update(result)

    all_failed_nodes = list(state.get("failed_nodes", []))
    all_errors = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"] = all_errors

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

    results = await asyncio.gather(
        node_compliance_retrieval(state),
        node_endorsement_chain(state),
        return_exceptions=False,
    )

    merged: dict = {}
    for result in results:
        merged.update(result)

    all_failed_nodes = list(state.get("failed_nodes", []))
    all_errors = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"] = all_errors

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

    results = await asyncio.gather(
        node_endorsement_chain(state),
        node_fraud_detection(state),
        return_exceptions=False,
    )

    merged: dict = {}
    for result in results:
        merged.update(result)

    all_failed_nodes = list(state.get("failed_nodes", []))
    all_errors = dict(state.get("errors", {}))
    for result in results:
        for failed in result.get("failed_nodes", []):
            if failed not in all_failed_nodes:
                all_failed_nodes.append(failed)
        all_errors.update(result.get("errors", {}))

    merged["failed_nodes"] = all_failed_nodes
    merged["errors"] = all_errors

    logger.info(
        f"[node:{node_name}] 并行执行完成 "
        f"failed={len(all_failed_nodes)} task={state.get('audit_task_id')}"
    )
    return merged