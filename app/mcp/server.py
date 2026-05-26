# app/mcp/server.py
# MCP（Model Context Protocol）服务器主入口
# 职责：
#   1. 创建全局 FastMCP 实例（单例），所有工具通过 @mcp.tool() 注册到此实例
#   2. 导入所有工具模块，触发工具注册（必须在 server.py 中导入，否则工具不会被注册）
#   3. 提供 get_mcp_asgi_app()，供 FastAPI 将 MCP 挂载为子应用（/mcp 路由）
#
# MCP 协议简介：
#   - MCP Server 暴露工具列表（list_tools），供 LLM 客户端发现
#   - LLM 客户端发现工具后，调用 call_tool 执行具体工具函数
#   - 本项目中，LangGraph 节点进程内直接调用工具函数（无 HTTP 开销）
#   - 外部 LLM（Claude API / Claude Desktop）通过 /mcp HTTP 端点调用

from mcp.server.fastmcp import FastMCP   # FastMCP：快速创建 MCP Server 的高级封装类

# ── 全局 FastMCP 实例（单例）───────────────────────────────────────────────────
# 所有 @mcp.tool() 装饰的函数都注册到这个实例
# 实例名称 "bill-audit" 是 MCP Server 的标识符，供客户端识别
mcp = FastMCP(
    "bill-audit",                              # MCP Server 名称（客户端连接时显示）
    instructions=(                             # 工具集的使用说明（LLM 客户端读取）
        "票据合规智能审核平台工具集。"
        "提供票据文件解析、要素抽取、合规检索、背书链分析、"
        "欺诈检测、风险评估、报告生成等能力。"
        "所有工具均为异步执行，返回标准 dict 格式。"
    ),
)

# ── 导入所有工具模块（触发 @mcp.tool() 注册）──────────────────────────────────
# 注意：必须在 mcp 实例创建之后才能导入，否则工具找不到 mcp 实例
# 导入顺序对应 DAG 执行顺序（便于阅读，实际注册顺序不影响功能）
from app.mcp.tools import document_tools      # 文档解析工具（DocumentParserAgent）
from app.mcp.tools import extraction_tools    # 要素抽取工具（ElementExtractionAgent）
from app.mcp.tools import compliance_tools    # 合规检索工具（ComplianceRetrievalAgent）
from app.mcp.tools import endorsement_tools   # 背书链分析工具（EndorsementChainAgent）
from app.mcp.tools import fraud_tools         # 欺诈检测工具（FraudDetectionAgent）
from app.mcp.tools import contract_tools      # 合同审核工具（ContractReviewAgent）
from app.mcp.tools import risk_tools          # 风险评估工具（RiskAssessmentAgent）
from app.mcp.tools import report_tools        # 报告生成工具（ReportGenerationAgent）
from app.mcp.tools import issuance_tools      # 出票预检工具（BillIssuanceAgent）
from app.mcp.tools import flow_tools          # 流转追踪工具（FlowTrackingAgent）
from app.mcp.tools import batch_tools         # 批量调度工具（BatchSchedulingAgent）


def get_mcp_asgi_app():
    """
    获取 MCP 的 ASGI 应用对象，供 FastAPI 使用 app.mount() 挂载

    挂载后，/mcp 路径下的所有请求由 MCP Server 处理（Streamable HTTP 传输模式）：
      POST /mcp        → 处理 MCP 协议请求（initialize / list_tools / call_tool 等）
      GET  /mcp/sse    → SSE 事件流（Streamable HTTP 模式下的事件推送端点）

    选择 streamable_http_app() 而非 sse_app() 的原因：
      - Streamable HTTP 是 MCP 2025-03-26 规范的推荐传输方式
      - 支持双向流式通信，延迟比 SSE 低
      - 更适合多容器部署场景（无需维持长连接状态）

    Returns:
        ASGI 可调用对象（任何支持 ASGI 的服务器都可以运行，如 Starlette/FastAPI）
    """
    return mcp.streamable_http_app()  # FastMCP 的 Streamable HTTP 传输模式 ASGI 适配器
