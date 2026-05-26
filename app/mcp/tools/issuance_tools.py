# app/mcp/tools/issuance_tools.py
# MCP 工具：出票合规预检
# 对应 Agent：BillIssuanceAgent
# 功能：在出票前快速验证三项核心条件：
#       1. 必填要素完整性（8个关键字段是否全部填写）
#       2. 黑名单匹配（出票人/承兑人/收款人是否在制裁名单中）
#       3. 授信额度检查（出票金额是否超出承兑行的授信限额）
#       三项全部通过才允许出票，任一失败则拒绝并说明原因。

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def check_bill_issuance(
    bill_element: dict,                                    # 票据要素 dict（必须含关键字段）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    快速预检票据出票资格（适合出票环节，比全流程审核快 5~10 倍）。

    检查顺序（短路逻辑：前一项失败则不继续）：
      1. 必填要素完整性（fail_fast）
      2. 黑名单匹配（并行）
      3. 授信额度（需查询信用系统，最慢）

    输出（成功）：
      {
        "success": true,
        "data": {
          "issuance_allowed": true,      // true=允许出票，false=拒绝
          "reject_reason": null,         // 拒绝原因（允许时为 null）
          "checks": {
            "completeness": {"pass": true},
            "blacklist": {"pass": true, "checked_parties": 3},
            "credit_limit": {"pass": true, "available_amount": 5000000}
          }
        }
      }
    """
    from app.agents.bill_issuance_agent import BillIssuanceAgent  # 延迟导入

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data={"bill_element": bill_element},        # 注入要素供三项检查使用
    )

    return await run_agent_tool(
        agent_instance=BillIssuanceAgent(),                # 出票预检 Agent 实例
        ctx=ctx,
        commit=True,                                       # 预检结果写入数据库（审计留存）
    )
