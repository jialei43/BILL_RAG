# app/mcp/tools/extraction_tools.py
# MCP 工具：票据要素抽取
# 对应 Agent：ElementExtractionAgent
# 功能：从已解析的文档内容（或直接从文件字节）中抽取 18 个标准票据要素字段，
#       写入 bill_elements 数据库表，返回票据号码、置信度等关键信息。

from typing import Optional                                # 类型注解：可选参数

from app.mcp.server import mcp                              # 全局 MCP 实例
from app.mcp.tools._base import (
    build_minimal_context,
    run_agent_tool,
)


@mcp.tool()
async def extract_bill_elements(
    audit_task_id: str = "",                               # 关联的审核任务 ID
    tenant_id: str = "mcp_caller",                        # 租户 ID
    file_path: Optional[str] = None,                      # 票据文件路径（与 prefilled_element 二选一）
    prefilled_element: Optional[dict] = None,             # 已预先识别的票据要素 dict（跳过 OCR）
) -> dict:
    """
    从票据文件或预填充要素中抽取 18 个标准字段，写入数据库。

    两种调用模式：
      模式 A（文件路径）：提供 file_path，调用 Qwen-VL 视觉模型识别
      模式 B（预填充）：提供 prefilled_element dict，跳过 OCR，直接写库（节省 3~8 秒）

    输出（成功）：
      {
        "success": true,
        "data": {
          "element_id": "UUID",
          "ticket_number": "2024XXXX",
          "confidence": 0.91,
          "low_confidence": false,
          "source": "vision_llm" | "prefilled"
        }
      }
    """
    from app.agents.element_extraction_agent import ElementExtractionAgent  # 延迟导入避免慢启动

    # 初始化共享数据：若提供了预填充要素，注入 shared_data 触发快速路径
    shared_data = {}
    if prefilled_element:
        shared_data["bill_element"] = prefilled_element    # ElementExtractionAgent 检测到此字段跳过 OCR

    ctx = build_minimal_context(
        audit_task_id=audit_task_id or None,
        tenant_id=tenant_id,
        shared_data=shared_data,                           # 注入预填充数据（可能为空）
    )

    return await run_agent_tool(
        agent_instance=ElementExtractionAgent(),           # 要素抽取 Agent 实例
        ctx=ctx,
        commit=True,                                       # 要素写入 bill_elements 表，需要提交
        file_path=file_path,                               # 透传文件路径（预填充模式下忽略此参数）
    )
