# app/mcp/tools/risk_tools.py
# MCP 工具：综合风险评估
# 对应 Agent：RiskAssessmentAgent
# 功能：聚合四个维度的中间结果（合规30%+背书30%+合同20%+欺诈20%），
#       计算综合风险评分，判定五档风险等级（LOW/MEDIUM_LOW/MEDIUM_HIGH/HIGH/CRITICAL）。

from typing import Optional                                # 可选参数

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def assess_risk(
    compliance_summary: Optional[dict] = None,             # 合规检索结果（来自 check_compliance）
    endorsement_result: Optional[dict] = None,             # 背书链分析结果（来自 analyze_endorsement_chain）
    contract_result: Optional[dict] = None,                # 合同审核结果（来自 review_contract）
    fraud_result: Optional[dict] = None,                   # 欺诈检测结果（来自 detect_fraud）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    聚合四维结果计算综合风险评分，返回风险等级和业务建议。

    权重配置（合计 100%）：
      - 合规维度：30%（覆盖 18 个要素字段的法规符合性）
      - 背书维度：30%（背书链完整性和违规检测）
      - 合同维度：20%（票据与贸易合同的匹配程度）
      - 欺诈维度：20%（反欺诈检测的综合得分）

    缺失维度降级处理：
      - 未传入的维度使用默认分（合规60/背书70/合同75/欺诈80），不阻断评估

    输出（成功）：
      {
        "success": true,
        "data": {
          "composite_score": 78.5,         // 综合评分（0~100，越高越安全）
          "risk_level": "MEDIUM_LOW",      // LOW/MEDIUM_LOW/MEDIUM_HIGH/HIGH/CRITICAL
          "dimension_scores": {
            "compliance": 65.0,
            "endorsement": 90.0,
            "contract": 85.0,
            "fraud": 80.0
          },
          "conclusion": "可执行（需关注）",
          "recommendation": "建议核查合规问题后继续流程"
        }
      }
    """
    from app.agents.risk_assessment_agent import RiskAssessmentAgent  # 延迟导入

    # 将四维结果全部注入 shared_data，RiskAssessmentAgent 从此处读取
    shared_data = {}
    if compliance_summary:
        shared_data["compliance_summary"] = compliance_summary    # 合规检索结果
    if endorsement_result:
        shared_data["endorsement_result"] = endorsement_result    # 背书链分析结果
    if contract_result:
        shared_data["contract_result"] = contract_result          # 合同审核结果
    if fraud_result:
        shared_data["fraud_result"] = fraud_result                # 欺诈检测结果

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data=shared_data,                                   # 注入全部四维数据
    )

    return await run_agent_tool(
        agent_instance=RiskAssessmentAgent(),                      # 风险评估 Agent 实例
        ctx=ctx,
        commit=True,                                               # 评估结果写入 risk_assessments 表
    )
