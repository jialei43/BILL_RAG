# app/mcp/tools/document_tools.py
# MCP 工具：票据文档解析
# 对应 Agent：DocumentParserAgent
# 功能：将 PDF/图片格式的票据文件解析为结构化文本和要素，
#       计算解析置信度，输出供后续要素抽取使用的中间结果。

from app.mcp.server import mcp                              # 全局 MCP 实例（@mcp.tool() 注册到此）
from app.mcp.tools._base import (                          # 共享工具函数
    build_minimal_context,                                 # 构造最小化 AgentContext
    run_agent_tool,                                        # 通用 Agent 调用包装
)


@mcp.tool()
async def parse_bill_document(
    file_path: str,                                        # 待解析的票据文件路径（绝对路径）
    audit_task_id: str = "",                               # 关联的审核任务 ID（空则自动生成）
    tenant_id: str = "mcp_caller",                        # 租户 ID（多租户隔离）
) -> dict:
    """
    解析票据文件，提取结构化文本、表格和 OCR 内容。

    输入：
      - file_path: 票据文件的本地路径（支持 PDF、PNG、JPG、TIFF）
      - audit_task_id: 关联的审核任务 UUID（可选，为空时自动生成临时 ID）

    输出（成功）：
      {
        "success": true,
        "data": {
          "file_path": "解析的文件路径",
          "element_count": 42,       // 提取到的文本片段数量
          "page_count": 2,           // 文档页数
          "confidence": 0.95,        // 解析置信度（0~1）
          "low_confidence": false    // 是否触发低置信度告警
        }
      }

    输出（失败）：
      {"success": false, "error": "错误描述", "error_code": "E_PARSE_NO_FILE"}
    """
    from app.agents.document_parser_agent import DocumentParserAgent  # 延迟导入：避免启动时加载 PaddleOCR

    ctx = build_minimal_context(                           # 构造执行上下文
        audit_task_id=audit_task_id or None,               # 空字符串转为 None，触发自动生成 UUID
        tenant_id=tenant_id,                               # 租户标识
    )

    return await run_agent_tool(                           # 调用 Agent 并返回标准格式结果
        agent_instance=DocumentParserAgent(),              # 实例化 Agent（每次调用创建新实例，线程安全）
        ctx=ctx,                                           # 执行上下文
        commit=False,                                      # DocumentParserAgent 不写库，无需提交
        file_path=file_path,                               # 透传文件路径给 Agent
    )
