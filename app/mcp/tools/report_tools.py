# app/mcp/tools/report_tools.py
# MCP 工具：审核报告生成
# 对应 Agent：ReportGenerationAgent
# 功能：聚合所有维度的中间结果，生成完整的结构化审核报告（JSON + PDF），
#       包含 9 个部分：概要/要素/合规/背书/合同/欺诈/风险/建议/附件。

from typing import Optional                                # 可选参数

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def generate_report(
    bill_element: Optional[dict] = None,                   # 票据要素
    compliance_summary: Optional[dict] = None,             # 合规检索结果
    endorsement_result: Optional[dict] = None,             # 背书链分析结果
    contract_result: Optional[dict] = None,                # 合同审核结果
    fraud_result: Optional[dict] = None,                   # 欺诈检测结果
    risk_result: Optional[dict] = None,                    # 风险评估结果
    issuance_result: Optional[dict] = None,                # 出票预检结果（可选）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
    is_final: bool = True,                                 # True=终版报告，False=中间版（草稿）
) -> dict:
    """
    聚合所有审核维度结果，生成完整的 JSON 结构报告并渲染 PDF。

    报告包含 9 个章节：
      1. 执行概要（结论 + 风险等级 + 主要发现）
      2. 票据基础信息（18 字段要素明细）
      3. 合规检查结果（18 字段逐一检查）
      4. 背书链分析（图谱 + 违规列表）
      5. 贸易背景核查（合同匹配度）
      6. 欺诈风险检测（五维评分）
      7. 综合风险评分（四维加权）
      8. 审核建议（自动生成的业务建议）
      9. 附件索引（关联文档 ID 列表）

    输出（成功）：
      {
        "success": true,
        "data": {
          "report_id": "UUID",
          "conclusion": "可执行",
          "risk_level": "LOW",
          "pdf_path": "/app/data/reports/UUID.pdf",  // PDF 文件路径（可为空）
          "section_count": 9
        }
      }
    """
    from app.agents.report_generation_agent import ReportGenerationAgent  # 延迟导入

    # 将所有维度结果注入 shared_data，ReportGenerationAgent 从此处读取
    shared_data = {}
    if bill_element:
        shared_data["bill_element"] = bill_element
    if compliance_summary:
        shared_data["compliance_summary"] = compliance_summary
    if endorsement_result:
        shared_data["endorsement_result"] = endorsement_result
    if contract_result:
        shared_data["contract_result"] = contract_result
    if fraud_result:
        shared_data["fraud_result"] = fraud_result
    if risk_result:
        shared_data["risk_result"] = risk_result                   # 风险评估是报告的核心依据
    if issuance_result:
        shared_data["issuance_result"] = issuance_result           # 出票场景特有字段

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data=shared_data,
    )

    return await run_agent_tool(
        agent_instance=ReportGenerationAgent(),                    # 报告生成 Agent 实例
        ctx=ctx,
        commit=True,                                               # 报告写入 audit_reports 表
        is_final=is_final,                                         # 是否为终版报告
    )
