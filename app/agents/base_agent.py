# app/agents/base_agent.py
# 多智能体系统的抽象基类定义
# 所有专项 Agent 必须继承 BaseAgent，并实现 run() 方法
# BaseAgent 负责：计时、状态上报到数据库、统一异常兜底——子类专注核心业务逻辑即可

from __future__ import annotations       # 支持类型注解中的前向引用（Python 3.9 以下需要）

import time                              # 用于计算 Agent 执行耗时（毫秒）
from abc import ABC, abstractmethod      # ABC=抽象基类，abstractmethod=声明必须被子类实现的方法
from dataclasses import dataclass, field # dataclass 自动生成 __init__/__repr__，field 设置默认值
from typing import Any, Optional         # 类型注解工具

from loguru import logger                # 结构化日志库，比标准 logging 更简洁
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话类型，由 FastAPI 依赖注入传入


# ──────────────────────────────────────────────────────────────────────────────
# AgentContext — Agent 执行上下文
# 贯穿整个多 Agent 调用链路，携带任务标识和 Agent 间共享的中间结果
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AgentContext:
    """
    Agent 执行上下文数据类
    由 OrchestratorAgent 创建，通过 DAG 执行器传递给每个子 Agent
    """
    audit_task_id: str                          # 主任务 ID（UUID），写数据库时的外键引用
    tenant_id: str                              # 租户 ID，确保数据隔离不跨租户读写
    bill_record_id: Optional[str] = None        # 票据主档 ID，要素抽取成功后由 ElementExtractionAgent 回填
    document_id: Optional[str] = None           # 原始文档 ID，DocumentParserAgent 解析的起点
    shared_data: dict = field(default_factory=dict)  # Agent 间共享的中间结果，避免重复查询数据库
    # shared_data 的约定键名：
    #   "parsed_doc"   -> DocumentParserAgent 的解析结果（dict）
    #   "bill_element" -> BillElement ORM 对象（已写库）
    #   "compliance_summary" -> 合规检查汇总（dict）
    #   "endorsement_result" -> 背书链分析结果（dict）
    #   "contract_result"    -> 合同审核结果（dict）
    #   "fraud_result"       -> 欺诈检测结果（dict）


# ──────────────────────────────────────────────────────────────────────────────
# AgentResult — Agent 执行结果
# 标准化的返回格式，让 OrchestratorAgent 可以统一处理所有 Agent 的输出
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class AgentResult:
    """
    Agent 执行结果数据类
    所有 Agent 的 run() 方法必须返回此类型
    """
    agent_name: str                     # Agent 名称（与 BaseAgent.agent_name 一致），便于日志追踪
    success: bool                       # 执行是否成功：False 时 DAG 后续依赖此 Agent 的步骤不执行
    data: Any = None                    # 核心输出数据，具体结构由各 Agent 定义
    error_code: Optional[str] = None    # 失败时的错误码，对应 error_code_mappings.error_code
    error_msg: Optional[str] = None     # 失败时的详细错误信息，含异常类型和上下文
    elapsed_ms: float = 0.0             # 执行耗时（毫秒），由 BaseAgent.execute() 自动填入


# ──────────────────────────────────────────────────────────────────────────────
# BaseAgent — 所有 Agent 的抽象基类
# 子类只需实现 run() 方法，框架层面的计时/日志/状态上报由此类统一处理
# ──────────────────────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    """
    多智能体系统抽象基类

    使用方式：
        class MyAgent(BaseAgent):
            agent_name = "my_agent"

            async def run(self, ctx: AgentContext, db: AsyncSession, **kwargs) -> AgentResult:
                # 实现核心业务逻辑
                return AgentResult(agent_name=self.agent_name, success=True, data={...})
    """

    agent_name: str = "base"  # 子类必须覆盖此属性，用于日志标识和 current_agent 字段更新

    async def execute(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        **kwargs
    ) -> AgentResult:
        """
        统一执行入口（外部调用此方法，不直接调用 run）
        负责：计时 + 结构化日志 + 异常兜底 + 状态上报

        Args:
            ctx: Agent 执行上下文，含任务 ID 和共享数据
            db:  异步数据库会话，由 FastAPI 依赖注入或 DAG 执行器传入
            **kwargs: 各 Agent 的额外参数（如 document_bytes、contract_text 等）

        Returns:
            AgentResult: 标准化执行结果，success=False 时含 error_code 和 error_msg
        """
        start_time = time.monotonic()  # monotonic 时钟不受系统时间调整影响，适合计时

        # 记录 Agent 启动日志（格式统一，便于日志系统聚合分析）
        logger.info(
            f"[{self.agent_name}] start | task={ctx.audit_task_id} tenant={ctx.tenant_id}"
        )

        # 更新 audit_tasks 表中的 current_agent 字段，前端可实时展示当前执行到哪个 Agent
        await self._update_current_agent(ctx.audit_task_id, db)

        try:
            # 调用子类实现的核心业务逻辑
            result = await self.run(ctx, db, **kwargs)

        except Exception as e:
            # 异常兜底：子类 run() 中不需要 try-except，统一在此捕获
            elapsed = (time.monotonic() - start_time) * 1000  # 转换为毫秒
            error_msg = f"{type(e).__name__}: {str(e)}"       # 记录异常类型和消息

            logger.error(
                f"[{self.agent_name}] failed | task={ctx.audit_task_id} "
                f"elapsed={elapsed:.1f}ms error={error_msg}"
            )
            # 返回失败结果，让 DAG 执行器决定是否继续执行后续 Agent
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                error_code="SYS_AGENT_EXCEPTION",  # 系统级异常错误码
                error_msg=error_msg,
                elapsed_ms=elapsed,
            )

        # 计算实际执行耗时并回填到结果对象
        result.elapsed_ms = (time.monotonic() - start_time) * 1000

        # 记录 Agent 完成日志
        status_str = "done" if result.success else "failed"
        logger.info(
            f"[{self.agent_name}] {status_str} | task={ctx.audit_task_id} "
            f"elapsed={result.elapsed_ms:.1f}ms"
        )

        return result

    @abstractmethod
    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        **kwargs
    ) -> AgentResult:
        """
        子类必须实现的核心业务逻辑方法

        约定：
        - 所有数据库写操作都在此方法内完成（不需要 commit，由 execute() 的调用者负责）
        - 不需要 try-except（异常由 execute() 统一捕获）
        - 结果写入 ctx.shared_data 供后续 Agent 读取（减少重复数据库查询）
        """
        ...  # 子类实现，抽象方法占位符

    async def _update_current_agent(
        self,
        audit_task_id: str,
        db: AsyncSession
    ) -> None:
        """
        更新 audit_tasks 表的 current_agent 字段
        采用 ORM 方式更新，确保 updated_at 字段也自动刷新
        """
        try:
            from sqlalchemy import update                    # 延迟导入，避免循环依赖
            from app.models.agent_models import AuditTask   # 同上

            # 使用 ORM update 语句，只更新 current_agent 字段
            stmt = (
                update(AuditTask)
                .where(AuditTask.id == audit_task_id)       # 精确定位到当前任务
                .values(current_agent=self.agent_name)      # 写入当前 Agent 名称
            )
            await db.execute(stmt)   # 执行更新（不提交，由外层调用者统一 commit）
            # 注意：不调用 db.flush()，避免在 Agent 执行中途产生不完整的事务状态

        except Exception as e:
            # current_agent 更新失败不影响主流程（非关键操作），只记录警告
            logger.warning(
                f"[{self.agent_name}] current_agent 更新失败（非致命）: {e}"
            )
