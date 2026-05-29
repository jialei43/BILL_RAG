# app/mcp/client.py
# MCPClient：主应用通过 HTTP 调用独立 MCP Server 的客户端封装
#
# 使用方式：
#   # lifespan 启动时
#   await init_mcp_client(settings.MCP_SERVER_URL)
#
#   # 节点内调用工具
#   result = await get_mcp_client().call_tool("parse_bill_document", {...})
#
#   # lifespan 关闭时
#   await close_mcp_client()
#
# 实现说明：
#   每次 call_tool 创建新的 streamablehttp_client 会话（open → initialize → call → close）。
#   MCP 协议要求初始化握手，因此每次调用有两次 HTTP 往返。
#   底层 httpx 连接池会复用 TCP 连接，实际开销仅为协议握手报文，生产场景可接受。

from __future__ import annotations

import json
from typing import Optional

from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

_client: Optional["MCPClient"] = None


class MCPClient:
    """通过 MCP Streamable HTTP 协议调用独立 MCP Server 的客户端"""

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url
        logger.info(f"[mcp_client] 初始化完成 server_url={server_url}")

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        调用 MCP Server 上的工具，返回工具函数的 dict 返回值。

        Args:
            tool_name:  MCP 工具名称（即 @mcp.tool() 装饰的函数名）
            arguments:  工具参数（必须与工具函数签名匹配，None 值可省略）

        Returns:
            工具函数返回的 dict（格式为 {"success": bool, "data": ..., "error": ...}）
        """
        # 过滤掉值为 None 的参数，避免 MCP 协议校验失败
        clean_args = {k: v for k, v in arguments.items() if v is not None}

        try:
            async with streamablehttp_client(self._server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()                       # MCP 协议握手（必须）
                    call_result = await session.call_tool(tool_name, clean_args)

            if call_result.isError:
                error_text = str(call_result.content) if call_result.content else "unknown error"
                logger.error(f"[mcp_client] 工具返回错误 tool={tool_name} error={error_text}")
                return {
                    "success": False,
                    "error": f"MCP tool error: {error_text}",
                    "error_code": "MCP_TOOL_ERROR",
                }

            if not call_result.content:
                logger.error(f"[mcp_client] 工具返回空内容 tool={tool_name}")
                return {"success": False, "error": "Empty MCP response", "error_code": "MCP_EMPTY_RESPONSE"}

            content = call_result.content[0]
            if not isinstance(content, TextContent):
                logger.error(f"[mcp_client] 非文本内容 tool={tool_name} type={type(content)}")
                return {"success": False, "error": "Unexpected content type", "error_code": "MCP_BAD_CONTENT"}

            return json.loads(content.text)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[mcp_client] 调用异常 tool={tool_name} error={error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "error_code": "MCP_CLIENT_ERROR",
            }


# ── 模块级单例管理 ────────────────────────────────────────────────────────────

async def init_mcp_client(server_url: str) -> MCPClient:
    """在 FastAPI lifespan 启动时调用，初始化全局 MCPClient 单例"""
    global _client
    _client = MCPClient(server_url)
    return _client


def get_mcp_client() -> MCPClient:
    """获取已初始化的 MCPClient，节点函数调用此函数拿到客户端实例"""
    if _client is None:
        raise RuntimeError(
            "MCPClient 尚未初始化，请确认 lifespan 已调用 init_mcp_client()"
        )
    return _client


async def close_mcp_client() -> None:
    """在 FastAPI lifespan 关闭时调用"""
    global _client
    _client = None
    logger.info("[mcp_client] MCPClient 已关闭")