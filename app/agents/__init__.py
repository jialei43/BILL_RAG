# app/agents/__init__.py
# 多智能体系统包初始化文件
# 导出核心类，外部模块通过 from app.agents import BaseAgent, AgentContext 导入

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult  # 导出三个核心类

__all__ = ["BaseAgent", "AgentContext", "AgentResult"]  # 声明公开 API，避免 * 导入时引入不相关名称
