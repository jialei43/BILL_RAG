# app/mcp/tools/contract_tools.py
# MCP 工具：合同审核
# 对应 Agent：ContractReviewAgent
# 功能：将票据要素与对应贸易合同进行比对，验证 6 项核心字段一致性
#       （金额/交易方/日期/用途/承兑条款/特殊记载），计算整体匹配度。

from typing import Optional                                # 可选参数

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def review_contract(
    bill_element: dict,                                    # 票据要素 dict
    contract_text: Optional[str] = None,                  # 合同全文（纯文本格式，与票据对比用）
    contract_document_id: Optional[str] = None,           # 合同文档 ID（与 contract_text 二选一，从库中读取）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    对比票据要素与贸易合同，验证贸易背景真实性。

    两种使用模式：
      模式 A：提供 contract_text 全文，直接进行比对
      模式 B：提供 contract_document_id，从数据库读取合同内容后比对

    比对的 6 项核心字段：
      1. 金额是否在合同约定范围内
      2. 出票人是否为合同买方
      3. 收款人是否为合同卖方
      4. 到期日是否与合同付款期一致
      5. 贸易用途是否匹配合同标的
      6. 承兑条款是否符合合同约定

    输出（成功）：
      {
        "success": true,
        "data": {
          "match_score": 0.87,              // 整体匹配度（0~1）
          "trade_background_score": 0.90,  // 贸易背景真实性评分
          "field_matches": {
            "amount": {"match": true, "detail": "票面金额在合同范围内"},
            "parties": {"match": true},
            ...
          },
          "is_trade_background_valid": true
        }
      }
    """
    from app.agents.contract_review_agent import ContractReviewAgent  # 延迟导入

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data={"bill_element": bill_element},        # 注入要素供 Agent 读取
    )

    return await run_agent_tool(
        agent_instance=ContractReviewAgent(),              # 合同审核 Agent 实例
        ctx=ctx,
        commit=True,                                       # 合同审核结果写入 contract_reviews 表
        contract_text=contract_text,                       # 透传合同文本
        contract_document_id=contract_document_id,        # 透传合同文档 ID
    )
