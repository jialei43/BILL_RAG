# app/mcp/tools/endorsement_tools.py
# MCP 工具：背书链分析
# 对应 Agent：EndorsementChainAgent
# 功能：构建票据背书的有向图，检测背书连续性、闭环、重复背书等 9 类违规，
#       结果写入 endorsement_chains 表。

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def analyze_endorsement_chain(
    bill_element: dict,                                    # 票据要素 dict（必须含 endorsers 字段）
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    分析票据背书链路的连续性和合法性。

    输入：
      - bill_element: 票据要素，其中 endorsers 字段为背书人列表
        示例：{"endorsers": ["甲公司", "乙公司"], "payee": "丙公司", ...}

    输出（成功）：
      {
        "success": true,
        "data": {
          "chain_length": 3,              // 背书链总长度（含出票人和收款人）
          "is_continuous": true,          // 背书是否连续
          "violation_codes": [],          // 违规码列表（空=无违规）
          "violation_count": 0,
          "endorser_nodes": [...]         // 背书图的节点列表
        }
      }
    """
    from app.agents.endorsement_chain_agent import EndorsementChainAgent  # 延迟导入

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data={"bill_element": bill_element},        # 注入要素数据，Agent 从此读取 endorsers 字段
    )

    return await run_agent_tool(
        agent_instance=EndorsementChainAgent(),            # 背书链分析 Agent 实例
        ctx=ctx,
        commit=True,                                       # 背书链结果写入 endorsement_chains 表
    )
