# app/mcp/tools/compliance_tools.py
# MCP 工具：票据合规检索
# 对应 Agent：ComplianceRetrievalAgent
# 功能：对票据的 18 个要素字段进行合规性检查，通过 RAG 检索法规知识库，
#       判断每个字段是否符合《票据法》《商业汇票承兑贴现办法》等法规要求，
#       结果写入 compliance_checks 表。

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def check_compliance(
    bill_element: dict,                                    # 票据要素 dict（来自 extract_bill_elements 的输出）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    对票据 18 个要素字段进行合规性检查，并行查询 RAG 知识库。

    输入：
      - bill_element: 包含 18 个字段的票据要素 dict
        必填键：ticket_number, ticket_type, issue_date, due_date,
                amount_numeric, drawer, acceptor 等

    输出（成功）：
      {
        "success": true,
        "data": {
          "compliance_summary": {
            "ticket_number": {"is_compliant": true, "score": 0.95},
            "due_date": {"is_compliant": false, "violation_level": "error",
                         "violation_desc": "到期日超过最长期限 6 个月"},
            ...  // 每个字段一条结果
          },
          "total_fields": 18,
          "compliant_count": 15,
          "violation_count": 3
        }
      }
    """
    from app.agents.compliance_retrieval_agent import ComplianceRetrievalAgent  # 延迟导入

    # 将 bill_element 注入 shared_data，ComplianceRetrievalAgent 从此处读取
    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data={"bill_element": bill_element},        # 注入要素数据到共享上下文
    )

    return await run_agent_tool(
        agent_instance=ComplianceRetrievalAgent(),         # 合规检索 Agent 实例
        ctx=ctx,
        commit=True,                                       # 合规结果写入 compliance_checks 表，需要提交
    )
