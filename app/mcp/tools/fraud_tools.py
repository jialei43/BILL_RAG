# app/mcp/tools/fraud_tools.py
# MCP 工具：欺诈检测
# 对应 Agent：FraudDetectionAgent
# 功能：五维并行检测票据欺诈风险（印章/重复票号/篡改/关联网络/黑名单），
#       各维度加权计算综合欺诈评分（0~1），超过 0.6 触发风险警告。

from typing import Optional                                # 可选参数类型注解

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def detect_fraud(
    bill_element: dict,                                    # 票据要素 dict
    endorsement_result: Optional[dict] = None,            # 背书链分析结果（可选，有助于提高欺诈检测准确率）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    五维并行检测票据欺诈风险，返回综合欺诈评分和各维度详情。

    检测维度：
      1. 印章一致性（seal_consistency）：印章位置、清晰度验证
      2. 重复票号（duplicate_check）：检查历史数据中的相同票号
      3. 金额篡改（amount_tampering）：大小写金额一致性检测
      4. 关联网络异常（network_anomaly）：交易方之间的异常关联关系
      5. 黑名单匹配（blacklist_check）：交易方与黑名单对照

    输出（成功）：
      {
        "success": true,
        "data": {
          "overall_fraud_score": 0.15,    // 综合欺诈评分（0=安全，1=高风险）
          "risk_level": "LOW",            // LOW/MEDIUM/HIGH
          "dimension_scores": {
            "seal_consistency": 0.05,
            "duplicate_check": 0.0,
            "amount_tampering": 0.1,
            "network_anomaly": 0.3,
            "blacklist_check": 0.0
          },
          "triggered_rules": []           // 触发的风险规则列表
        }
      }
    """
    from app.agents.fraud_detection_agent import FraudDetectionAgent  # 延迟导入

    # 构建共享数据：注入要素和背书链结果（欺诈检测需要两者）
    shared_data = {"bill_element": bill_element}
    if endorsement_result:
        shared_data["endorsement_result"] = endorsement_result  # 有背书链结果时注入

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data=shared_data,
    )

    return await run_agent_tool(
        agent_instance=FraudDetectionAgent(),              # 欺诈检测 Agent 实例
        ctx=ctx,
        commit=True,                                       # 欺诈检测结果写入数据库
    )
