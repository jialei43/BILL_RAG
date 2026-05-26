# app/mcp/tools/flow_tools.py
# MCP 工具：票据流转追踪
# 对应 Agent：FlowTrackingAgent
# 功能：驱动七步业务状态机，记录票据从出票到承兑/贴现/背书/付款的全生命周期报文，
#       计算每个节点的超时预警级别（正常/警告/危险/超时）。

from typing import Optional                                # 可选参数

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def track_bill_flow(
    bill_element: dict,                                    # 票据要素 dict（含 ticket_number 等关键字段）
    business_type: str = "full_audit",                    # 业务类型（决定流转路径：承兑/贴现/背书等）
    metadata: Optional[dict] = None,                      # 扩展元数据（如处理机构、渠道编号等）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    追踪票据在业务流程中的流转状态，记录各节点报文和超时情况。

    七步状态机节点（按流转顺序）：
      1. 出票申请 → 2. 承兑行受理 → 3. 承兑完成
      → 4. 背书转让（可选，多次）
      → 5. 贴现申请 → 6. 贴现审批 → 7. 资金到账

    超时预警级别：
      - 正常（green）：当前节点未超时
      - 警告（yellow）：超时 < 24 小时
      - 危险（orange）：超时 24~72 小时
      - 超时（red）：超时 > 72 小时

    输出（成功）：
      {
        "success": true,
        "data": {
          "current_step": 3,              // 当前所在步骤编号
          "current_status": "承兑完成",
          "flow_nodes": [...],            // 14 个报文节点的详情列表
          "timeout_alert": "green",       // 超时预警级别
          "elapsed_hours": 8.5            // 当前步骤已耗时（小时）
        }
      }
    """
    from app.agents.flow_tracking_agent import FlowTrackingAgent  # 延迟导入

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data={"bill_element": bill_element},        # 注入要素，Agent 从此读取 ticket_number
    )

    return await run_agent_tool(
        agent_instance=FlowTrackingAgent(),                # 流转追踪 Agent 实例
        ctx=ctx,
        commit=True,                                       # 流转状态写入 flow_tracking 表
        business_type=business_type,                       # 透传业务类型
        ticket_number=bill_element.get("ticket_number"),   # 从要素 dict 提取票号（追踪主键）
        metadata=metadata or {},                           # 透传扩展元数据
    )
