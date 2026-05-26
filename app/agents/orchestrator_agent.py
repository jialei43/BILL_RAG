# app/agents/orchestrator_agent.py
# OrchestratorAgent：多智能体系统的"指挥中枢"（LangGraph 版本）
# 职责：
#   1. 根据业务类型调用 build_audit_graph() 构建对应的 LangGraph 图
#   2. 创建/更新 audit_tasks 表记录（状态、进度、当前 Agent）
#   3. 通过 graph.astream() 流式执行图，每个节点完成时更新 DB 进度
#   4. 执行结束后将最终状态写回 ctx.shared_data，保持与上层 API 的兼容性
#
# 与旧版区别：
#   - 删除 DAGExecutor / DAGNode / DAGEdge（全部由 LangGraph 替代）
#   - 删除 TASK_TYPE_TO_DAG 常量字典（改由 app.graph.builder 管理）
#   - 图状态外置到 PostgreSQL（多容器部署时任意副本可接手任务）
#   - 支持 graph.astream() 流式进度推送（每节点完成即更新数据库）

from __future__ import annotations

import asyncio                            # wait_for 超时保护
import uuid                               # 生成审核任务 UUID
from datetime import datetime             # 记录任务时间
from typing import Optional               # 类型注解

from loguru import logger                 # 结构化日志
from sqlalchemy import select, update     # ORM 查询和更新语句
from sqlalchemy.ext.asyncio import AsyncSession  # 异步数据库会话

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult   # 基类和数据类
from app.graph.builder import build_audit_graph                           # LangGraph 图构建器
from app.graph.state import make_initial_state                            # 初始状态工厂
from app.graph import get_checkpointer                                    # PostgresSaver 实例
from app.models.agent_models import (
    AuditTask, AuditTaskStatus, AuditTaskType,   # 审核任务 ORM 和枚举
)


# ──────────────────────────────────────────────────────────────────────────────
# OrchestratorAgent 实现
# ──────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """
    编排调度 Agent：多智能体系统的入口（LangGraph 版本）

    通过 build_audit_graph() 按业务类型构建图，
    用 graph.astream() 替代原有 DAGExecutor.execute()，
    实现流式执行、失败传播和 PostgreSQL 状态持久化。
    """

    agent_name = "orchestrator_agent"   # Agent 标识符（日志/注册表使用）

    def __init__(self):
        """
        构造函数：无需再传入 agent_registry 或 DAGExecutor
        LangGraph 图在 run() 调用时按需构建（每次构建耗时极低）
        """
        pass                            # 图构建移到 run() 内部，支持按任务类型动态选择

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        task_type: AuditTaskType = AuditTaskType.FULL_AUDIT,
        **kwargs
    ) -> AgentResult:
        """
        编排主流程（LangGraph 版本）

        Args:
            ctx:       执行上下文（audit_task_id、tenant_id、document_id 等）
            db:        异步数据库会话
            task_type: 业务类型枚举，决定构建哪条 LangGraph 图路径
            **kwargs:  可额外传入 file_path（覆盖 document_id 解析）

        Returns:
            AgentResult: 编排完成后的汇总结果
        """
        logger.info(
            f"[orchestrator] 开始编排 task={ctx.audit_task_id} type={task_type.value}"
        )

        # ── 步骤 1：将任务状态更新为 RUNNING，记录开始时间 ──────────────────
        await self._update_task_status(
            audit_task_id=ctx.audit_task_id,
            status=AuditTaskStatus.RUNNING,
            started_at=datetime.utcnow(),
            db=db,
        )

        # ── 步骤 2：解析文件路径（优先 kwargs，其次 document_id 查库）────────
        file_path = kwargs.get("file_path") or ctx.shared_data.get("file_path")
        if not file_path and ctx.document_id:
            # document_id 不为空时，从数据库 documents 表查询文件路径
            file_path = await self._resolve_file_path(ctx.document_id, db)

        # ── 步骤 3：构造 LangGraph 初始状态（替代 AgentContext.shared_data）──
        trace_id = ctx.shared_data.get("trace_id", str(uuid.uuid4())[:8])  # 从 shared_data 获取 trace_id

        initial_state = make_initial_state(
            audit_task_id=ctx.audit_task_id,
            tenant_id=ctx.tenant_id,
            task_type=task_type.value,
            trace_id=trace_id,
            file_path=file_path,
            contract_text=ctx.shared_data.get("contract_text"),
            contract_document_id=ctx.shared_data.get("contract_document_id"),
            prefilled_element=ctx.shared_data.get("bill_element"),  # 支持跳过 OCR 的快速路径
        )

        # ── 步骤 4：将图结构信息写入 audit_tasks.dag_plan（前端流程图展示）──
        graph_info = {
            "engine":    "langgraph",                # 标记使用 LangGraph 引擎
            "task_type": task_type.value,            # 业务类型字符串
            "nodes":     self._get_graph_nodes(task_type.value),  # 当前业务类型的节点列表
        }
        await self._update_dag_plan(ctx.audit_task_id, graph_info, db)

        # ── 步骤 5：构建并执行 LangGraph 图（替代 DAGExecutor.execute()）────
        checkpointer = get_checkpointer()            # 获取 PostgresSaver（应用启动时已初始化）
        graph = build_audit_graph(task_type.value, checkpointer=checkpointer)

        # thread_id = audit_task_id：LangGraph 用此 key 在 PostgreSQL 中查找/保存检查点
        # 多容器部署时，任意副本都能通过相同的 thread_id 恢复任务状态
        config = {"configurable": {"thread_id": ctx.audit_task_id}}

        completed_nodes = []                         # 已完成的节点名称列表（进度计算用）
        final_state = {}                             # 最终完整状态

        # astream() 流式执行：每完成一个节点就产生一个 {node_name: state_delta} 事件
        async for event in graph.astream(initial_state, config=config):
            node_name = list(event.keys())[0]        # 当前完成的节点名
            node_delta = event[node_name]             # 该节点产生的状态增量

            completed_nodes.append(node_name)        # 记录已完成节点
            final_state.update(node_delta)           # 合并增量到最终状态

            # 每个节点完成后立即更新数据库进度（前端可实时轮询）
            await self._update_progress_from_event(
                audit_task_id=ctx.audit_task_id,
                node_name=node_name,
                node_delta=node_delta,
                completed_nodes=completed_nodes,
                task_type=task_type,
                db=db,
            )

        # ── 步骤 6：计算最终状态 ─────────────────────────────────────────────
        failed_nodes = final_state.get("failed_nodes", [])  # 所有失败的节点名称

        # 关键节点失败 → 整体任务失败（与原 DAGExecutor 逻辑一致）
        critical_nodes = self._get_critical_nodes(task_type.value)
        critical_failed = bool(set(failed_nodes) & set(critical_nodes))

        final_status = AuditTaskStatus.FAILED if critical_failed else AuditTaskStatus.COMPLETED

        # ── 步骤 7：更新任务最终状态到 DB ────────────────────────────────────
        await self._update_task_status(
            audit_task_id=ctx.audit_task_id,
            status=final_status,
            progress_pct=100.0 if not critical_failed else round(
                len(completed_nodes) / max(len(critical_nodes) + 3, 1) * 100, 1
            ),
            completed_at=datetime.utcnow(),
            db=db,
        )

        # ── 步骤 8：将 LangGraph 最终状态同步回 ctx.shared_data（向后兼容）──
        # API 层和 _extract_chat_report() 仍通过 ctx.shared_data 读取结果
        self._sync_state_to_context(final_state, ctx)

        logger.info(
            f"[orchestrator] 编排完成 task={ctx.audit_task_id} "
            f"status={final_status.value} failed_nodes={failed_nodes}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=not critical_failed,             # 关键节点全部成功才视为编排成功
            data={
                "task_id":        ctx.audit_task_id,
                "status":         final_status.value,
                "completed_nodes": completed_nodes,
                "failed_nodes":   failed_nodes,
            },
            error_code=("ORCHESTRATOR_CRITICAL_FAILURE" if critical_failed else None),
            error_msg=("存在关键 Agent 执行失败，审核任务未完成" if critical_failed else None),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 私有辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    async def _resolve_file_path(self, document_id: str, db: AsyncSession) -> Optional[str]:
        """
        通过 document_id 查询 documents 表获取文件路径

        DocumentParserAgent 在旧版中自己处理这个查询，
        LangGraph 版本在编排层提前解析，确保 file_path 进入初始状态。

        Args:
            document_id: 文档 UUID
            db:          异步数据库会话

        Returns:
            文件绝对路径，或 None（文档不存在时）
        """
        try:
            from app.models.db_models import Document   # 延迟导入，避免循环依赖
            result = await db.execute(
                select(Document.file_path).where(Document.id == document_id)
            )
            return result.scalar_one_or_none()          # 返回路径或 None
        except Exception as e:
            logger.warning(f"[orchestrator] 解析文件路径失败 doc={document_id}: {e}")
            return None

    async def _update_task_status(
        self,
        audit_task_id: str,
        status: AuditTaskStatus,
        db: AsyncSession,
        progress_pct: Optional[float] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """更新 audit_tasks 表的状态字段（精确字段更新，避免加载整个对象）"""
        values = {"status": status}                     # 始终更新状态字段
        if progress_pct is not None:
            values["progress_pct"] = progress_pct      # 可选：更新进度百分比
        if started_at is not None:
            values["started_at"] = started_at          # 可选：记录开始时间
        if completed_at is not None:
            values["completed_at"] = completed_at      # 可选：记录完成时间

        stmt = update(AuditTask).where(AuditTask.id == audit_task_id).values(**values)
        await db.execute(stmt)                          # 执行更新（不提交，外层统一 commit）

    async def _update_dag_plan(
        self,
        audit_task_id: str,
        dag_json: dict,
        db: AsyncSession,
    ) -> None:
        """将 LangGraph 图结构信息写入 audit_tasks.dag_plan 字段（供前端渲染流程图）"""
        stmt = (
            update(AuditTask)
            .where(AuditTask.id == audit_task_id)
            .values(dag_plan=dag_json)
        )
        await db.execute(stmt)

    async def _update_progress_from_event(
        self,
        audit_task_id: str,
        node_name: str,
        node_delta: dict,
        completed_nodes: list,
        task_type: AuditTaskType,
        db: AsyncSession,
    ) -> None:
        """
        每个 LangGraph 节点完成后更新 audit_tasks 表的进度和当前节点

        替代原 DAGExecutor 的 _update_task_progress() 机制，
        LangGraph 每完成一个节点就调用一次，前端可实时轮询进度。

        Args:
            audit_task_id:   任务 UUID
            node_name:       刚完成的节点名称（如 "parse"、"extract"）
            node_delta:      该节点产生的状态增量
            completed_nodes: 已完成节点列表（用于计算进度百分比）
            task_type:       业务类型（用于确定总节点数）
            db:              异步数据库会话
        """
        total_nodes = len(self._get_graph_nodes(task_type.value))    # 当前业务类型总节点数
        progress_pct = round(len(completed_nodes) / total_nodes * 100.0, 1) if total_nodes > 0 else 0.0
        is_failed = "failed_nodes" in node_delta and node_name in node_delta.get("failed_nodes", [])

        stmt = (
            update(AuditTask)
            .where(AuditTask.id == audit_task_id)
            .values(
                progress_pct=progress_pct,              # 实时进度（百分比）
                current_agent=node_name,                # 当前/最后完成的节点名
            )
        )
        await db.execute(stmt)

        log_fn = logger.warning if is_failed else logger.debug
        log_fn(
            f"[orchestrator] 节点完成 task={audit_task_id} "
            f"node={node_name} progress={progress_pct:.1f}% failed={is_failed}"
        )

    def _sync_state_to_context(self, final_state: dict, ctx: AgentContext) -> None:
        """
        将 LangGraph 最终状态同步回 ctx.shared_data（向后兼容）

        API 层和 _extract_chat_report() 仍读取 ctx.shared_data，
        此方法确保新状态机制与现有代码无缝衔接，无需修改上层接口。
        """
        result_fields = [                               # LangGraph 状态中存储 Agent 结果的字段名
            "parsed_doc", "bill_element", "compliance_summary",
            "endorsement_result", "fraud_result", "contract_result",
            "risk_result", "report_result", "issuance_result", "flow_result",
        ]
        for field in result_fields:
            if field in final_state and final_state[field] is not None:
                ctx.shared_data[field] = final_state[field]  # 将状态字段复制到 shared_data

        # 额外同步控制字段（API 层可能需要查看失败信息）
        ctx.shared_data["failed_nodes"]  = final_state.get("failed_nodes", [])
        ctx.shared_data["skipped_nodes"] = final_state.get("skipped_nodes", [])
        ctx.shared_data["errors"]        = final_state.get("errors", {})

    @staticmethod
    def _get_graph_nodes(task_type: str) -> list:
        """
        返回指定业务类型的图节点名称列表（用于进度计算）

        返回的节点数量等于该业务类型图中的实际节点数，
        与 build_audit_graph() 中 add_node 的数量保持一致。
        """
        # parse 和 extract 是所有类型的公共节点
        common_start = ["parse", "extract"]
        common_end   = ["risk", "report"]

        # 按业务类型确定中间层节点
        if task_type in ("full_audit", "payment_prompt", "pledge", "collection", "discount_apply"):
            middle = ["parallel", "contract"]          # 并行层 + 合同审核
        elif task_type == "issuance_check":
            middle = ["parallel"]                      # 仅合规+出票预检并行层
        elif task_type == "acceptance_prompt":
            middle = ["parallel"]                      # 合规+背书并行层
        elif task_type == "endorsement":
            middle = ["parallel"]                      # 背书+欺诈并行层
        else:
            middle = ["parallel", "contract"]          # 未知类型兜底

        return common_start + middle + common_end

    # 各业务类型的关键节点集合（使用内部原子节点名，与 _node_failed() 中的 node_name 一致）
    # 这些名称来自 nodes.py 中每个节点函数内的 node_name 变量（如 "document_parser"）
    # 与图层节点名（"parse"/"extract" 等）不同，不可混用
    _CRITICAL_NODES_BY_TYPE: dict = {
        "full_audit":        {"document_parser", "element_extraction",
                              "compliance_retrieval", "endorsement_chain", "risk_assessment"},
        "issuance_check":    {"document_parser", "element_extraction",
                              "compliance_retrieval", "bill_issuance",    "risk_assessment"},
        "discount_apply":    {"document_parser", "element_extraction",
                              "compliance_retrieval", "endorsement_chain",
                              "contract_review",      "risk_assessment"},
        "acceptance_prompt": {"document_parser", "element_extraction",
                              "compliance_retrieval", "endorsement_chain", "risk_assessment"},
        "endorsement":       {"document_parser", "element_extraction",
                              "endorsement_chain", "risk_assessment"},
        "payment_prompt":    {"document_parser", "element_extraction",
                              "compliance_retrieval", "endorsement_chain", "risk_assessment"},
        "pledge":            {"document_parser", "element_extraction",
                              "compliance_retrieval", "endorsement_chain", "risk_assessment"},
        "collection":        {"document_parser", "element_extraction",
                              "compliance_retrieval", "endorsement_chain", "risk_assessment"},
    }

    @classmethod
    def _get_critical_nodes(cls, task_type: str) -> list:
        """
        返回指定业务类型的关键节点内部名称列表（关键节点失败 → 整体任务失败）

        使用内部原子节点名（与 nodes.py 的 _node_failed() 记录一致），
        而非 LangGraph 图层节点名（"parse"/"extract" 等）。
        与原 DAGExecutor 的 is_critical=True 节点逻辑等价。
        """
        return list(
            cls._CRITICAL_NODES_BY_TYPE.get(
                task_type,
                # 未知类型兜底：使用全流程的关键节点集
                cls._CRITICAL_NODES_BY_TYPE["full_audit"]
            )
        )

    async def run_for_chat(
        self,
        task_type_str: str,
        tenant_id: str,
        db: AsyncSession,
        bill_element_dict: Optional[dict] = None,  # 预提取的票据要素（跳过 OCR）
        bill_record_id: Optional[str] = None,       # 已入库票据主档 ID
        document_id: Optional[str] = None,          # 原始文件 ID（用于 Agent 解析）
        audit_label: str = "综合审核",              # 报告标题（前端展示用）
        timeout_seconds: float = 180.0,             # 执行超时保护（秒）
    ) -> dict:
        """
        聊天场景下的同步 Agent 审核入口（LangGraph 版本）

        与 run() 的区别：
          - 自动创建 AuditTask 记录（无需外部提前创建）
          - 支持注入预提取的票据要素（跳过 OCR 步骤）
          - 同步等待结果，直接返回结构化报告 dict（不返回 task_id 让前端轮询）
          - 设置超时保护，超时时返回 "审核中" 状态

        Returns:
            dict: 包含 task_id / conclusion / risk_level / issues / summary / elapsed_ms
        """
        import time
        t0 = time.perf_counter()

        # 解析 task_type 字符串为枚举（兜底为全流程审核）
        try:
            task_type = AuditTaskType(task_type_str)
        except ValueError:
            logger.warning(f"[orchestrator] 未知 task_type={task_type_str}，降级为 full_audit")
            task_type = AuditTaskType.FULL_AUDIT

        # 步骤 1：在数据库中创建 AuditTask 记录（状态为 PENDING）
        task_id = str(uuid.uuid4())
        audit_task = AuditTask(
            id=task_id,
            tenant_id=tenant_id,
            task_type=task_type,
            status=AuditTaskStatus.PENDING,
            bill_record_id=bill_record_id,
            document_id=document_id,
        )
        db.add(audit_task)
        await db.flush()                             # 让 task_id 在 DB 中可用，事务尚未提交

        # 步骤 2：构建 AgentContext，注入初始 shared_data
        initial_shared: dict = {}
        if bill_element_dict:
            # ElementExtractionAgent 检测到此字段时跳过 Qwen-VL 调用（快速路径）
            initial_shared["bill_element"] = bill_element_dict

        ctx = AgentContext(
            audit_task_id=task_id,
            tenant_id=tenant_id,
            bill_record_id=bill_record_id,
            document_id=document_id,
            shared_data=initial_shared,
        )

        # 步骤 3：在超时保护下执行 LangGraph 图
        try:
            orch_result = await asyncio.wait_for(
                self.run(ctx, db, task_type=task_type),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            # 超时：任务仍在后台运行，返回 "审核中" 状态，前端可用 task_id 轮询
            logger.warning(
                f"[orchestrator] 聊天审核超时 task={task_id} timeout={timeout_seconds}s"
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "task_id":       task_id,
                "conclusion":    "审核中",
                "risk_level":    "UNKNOWN",
                "overall_score": None,
                "issues":        [],
                "summary":       (
                    f"审核任务已创建（ID: {task_id[:8]}…），"
                    f"处理超过 {timeout_seconds:.0f} 秒，"
                    f"请通过任务查询接口获取最终结果。"
                ),
                "elapsed_ms":    elapsed_ms,
                "timed_out":     True,
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 步骤 4：从 shared_data 提取关键结论，合成聊天报告
        # （_sync_state_to_context 已在 run() 末尾将 LangGraph 状态写入 ctx.shared_data）
        report = self._extract_chat_report(ctx, task_id, task_type, audit_label, elapsed_ms)
        return report

    def _extract_chat_report(
        self,
        ctx: AgentContext,
        task_id: str,
        task_type: AuditTaskType,
        audit_label: str,
        elapsed_ms: float,
    ) -> dict:
        """
        从 ctx.shared_data 提取关键信息，合成聊天用的结构化报告。

        按优先级逐步降级：风险评估 → 合规检索 → 背书链 → 欺诈检测 → 基础要素。
        （此方法与旧版完全相同，通过 _sync_state_to_context 保持 shared_data 兼容）
        """
        issues: list[dict] = []

        # ── 从合规检索结果提取问题 ────────────────────────────────────────────
        compliance_summary = ctx.shared_data.get("compliance_summary", {})
        for field, result in compliance_summary.items():
            if isinstance(result, dict) and not result.get("is_compliant", True):
                level = result.get("violation_level", "warning")
                issues.append({
                    "field":          field,
                    "level":          level if level in ("error", "warning", "info") else "warning",
                    "description":    result.get("violation_desc", f"{field} 合规检查不通过"),
                    "recommendation": result.get("recommendation"),
                })

        # ── 从背书链结果提取问题 ──────────────────────────────────────────────
        endorsement_result = ctx.shared_data.get("endorsement_result", {})
        if endorsement_result:
            for code in endorsement_result.get("violation_codes", []):
                issues.append({
                    "field":          "endorsers",
                    "level":          "error",
                    "description":    f"背书链异常：{code}",
                    "recommendation": "请联系持票人核查背书连续性",
                })

        # ── 从欺诈检测结果提取问题 ────────────────────────────────────────────
        fraud_result = ctx.shared_data.get("fraud_result", {})
        if fraud_result:
            fraud_score = fraud_result.get("overall_fraud_score", 0.0)
            if fraud_score >= 0.6:                     # 欺诈评分超 0.6 时列为高风险问题
                issues.append({
                    "field":          "fraud_detection",
                    "level":          "error" if fraud_score >= 0.8 else "warning",
                    "description":    f"欺诈风险评分 {fraud_score:.2f}，超过安全阈值",
                    "recommendation": "建议人工复核票据真实性",
                })

        # ── 从风险评估结果提取整体结论 ───────────────────────────────────────
        risk_result   = ctx.shared_data.get("risk_result", {})
        risk_level    = risk_result.get("risk_level", "UNKNOWN")
        overall_score = risk_result.get("composite_score")

        conclusion_map = {
            "LOW":         "可执行",
            "MEDIUM_LOW":  "可执行（需关注）",
            "MEDIUM":      "需关注",
            "HIGH":        "不可执行",
            "CRITICAL":    "不可执行",
            "UNKNOWN":     "审核中",
        }
        conclusion = conclusion_map.get(risk_level, "需关注")

        # ── 生成自然语言摘要 ─────────────────────────────────────────────────
        issue_count   = len(issues)
        error_count   = sum(1 for i in issues if i.get("level") == "error")
        warning_count = sum(1 for i in issues if i.get("level") == "warning")
        bill_el       = ctx.shared_data.get("bill_element", {})
        ticket_no     = bill_el.get("ticket_number", "（未识别）")

        summary_parts = [
            f'已完成"{audit_label}"，票据号码：{ticket_no}。',
            f"综合风险等级：{risk_level}，业务结论：{conclusion}。",
        ]
        if issue_count > 0:
            summary_parts.append(
                f"共发现 {issue_count} 项问题"
                f"（{error_count} 项违规、{warning_count} 项警告）。"
            )
        else:
            summary_parts.append("未发现合规问题，票据状态正常。")

        if overall_score is not None:
            summary_parts.append(f"综合评分：{overall_score:.1f} / 100。")

        if conclusion in ("可执行", "可执行（需关注）"):
            summary_parts.append("建议：可继续发起业务流程，注意关注上述警告项。")
        else:
            summary_parts.append("建议：请在解决上述违规问题后再发起业务流程。")

        return {
            "task_id":       task_id,
            "conclusion":    conclusion,
            "risk_level":    risk_level,
            "overall_score": overall_score,
            "issues":        issues,
            "summary":       "".join(summary_parts),
            "elapsed_ms":    elapsed_ms,
            "timed_out":     False,
        }
