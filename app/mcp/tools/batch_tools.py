# app/mcp/tools/batch_tools.py
# MCP 工具：批量审核调度
# 对应 Agent：BatchSchedulingAgent
# 功能：接受多个审核任务的参数列表，通过 Semaphore 控制并发数量，
#       分发批量审核子任务并汇总结果，适合月结批量处理和大批量合规检查场景。

from typing import Optional                                # 可选参数

from app.mcp.server import mcp
from app.mcp.tools._base import build_minimal_context, run_agent_tool


@mcp.tool()
async def schedule_batch_audit(
    items: list,                                           # 批量任务列表，每项为一个任务参数 dict
    task_type: str = "full_audit",                        # 业务类型（每个子任务共用同一类型）
    concurrency: int = 10,                                 # 最大并发数（Semaphore 槽位，范围 1~100）
    audit_task_id: str = "",                               # 父任务 ID（批量任务的根任务 ID）
    tenant_id: str = "mcp_caller",                        # 租户 ID
) -> dict:
    """
    批量分发审核任务，汇总执行结果。

    输入：
      - items: 任务列表，每项格式为：
          {"file_path": "...", "ticket_number": "...", "contract_text": "...（可选）"}
      - task_type: 批量使用的审核类型（所有子任务共用）
      - concurrency: 并发控制（默认 10，建议不超过 CPU 核数 × 4）

    输出（成功）：
      {
        "success": true,
        "data": {
          "total": 100,              // 提交的总任务数
          "success_count": 95,       // 成功完成的任务数
          "failed_count": 5,         // 失败的任务数
          "elapsed_ms": 12500.0,     // 总耗时（毫秒）
          "avg_ms_per_task": 125.0,  // 平均每个任务耗时
          "failed_items": [...]      // 失败任务的详情列表
        }
      }
    """
    from app.agents.batch_scheduling_agent import BatchSchedulingAgent  # 延迟导入

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
    )

    return await run_agent_tool(
        agent_instance=BatchSchedulingAgent(),             # 批量调度 Agent 实例
        ctx=ctx,
        commit=True,                                       # 批量汇总结果写入 batch_tasks 表
        items=items,                                       # 透传任务列表
        concurrency=concurrency,                           # 透传并发控制参数
        task_type=task_type,                               # 透传业务类型
    )
