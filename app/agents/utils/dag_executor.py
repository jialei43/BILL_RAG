# app/agents/utils/dag_executor.py
# DAG（有向无环图）执行器：负责按照依赖关系调度 Agent 的串行/并行执行
#
# 核心思想：
#   - 拓扑排序：先找出没有前置依赖的 Agent，并行执行，完成后解锁依赖它们的 Agent
#   - 并行执行：同一层级（无相互依赖）的 Agent 用 asyncio.gather 并发执行，最大化效率
#   - 串行保障：有数据依赖的 Agent 严格等待前置 Agent 完成后才启动
#   - 失败传播：关键 Agent 失败时，所有依赖它的后续 Agent 跳过执行

from __future__ import annotations   # 支持类型注解前向引用

import asyncio                        # Python 异步编程标准库，gather 实现并行
from dataclasses import dataclass, field  # dataclass 简化数据类定义
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING  # 类型注解工具

from loguru import logger             # 日志库

if TYPE_CHECKING:
    # 仅在类型检查时导入，避免运行时循环依赖
    from sqlalchemy.ext.asyncio import AsyncSession
    from app.agents.base_agent import AgentContext, AgentResult, BaseAgent


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构定义
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    """
    DAG 中的一个节点，代表一个 Agent 的执行任务
    """
    node_id: str                          # 节点唯一标识，通常与 agent_name 相同
    agent_name: str                       # 对应的 Agent 名称，用于从注册表中查找实例
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 传给该 Agent 的额外参数
    is_critical: bool = True              # 是否为关键节点：True 时失败会传播给所有依赖节点
    timeout_seconds: float = 300.0        # 单个 Agent 的超时阈值（秒），超时视为失败


@dataclass
class DAGEdge:
    """
    DAG 中的一条有向边：from_node → to_node，表示 from_node 必须在 to_node 之前完成
    """
    from_node: str   # 前置节点 ID（依赖方）
    to_node: str     # 后置节点 ID（被依赖方）


@dataclass
class DAGPlan:
    """
    完整的 DAG 执行计划
    由 OrchestratorAgent 根据业务类型生成，传入 DAGExecutor 执行
    """
    nodes: List[DAGNode]          # 所有节点列表
    edges: List[DAGEdge]          # 所有有向边列表（定义执行顺序）
    plan_name: str = "unnamed"    # 计划名称，便于日志追踪


# ──────────────────────────────────────────────────────────────────────────────
# DAGExecutor — 核心执行引擎
# ──────────────────────────────────────────────────────────────────────────────

class DAGExecutor:
    """
    DAG 执行器：接收一个 DAGPlan，按拓扑顺序调度所有 Agent 执行

    执行策略：
    1. 计算每个节点的入度（有多少前置依赖）
    2. 将入度为 0 的节点放入「就绪队列」并行启动
    3. 任一节点完成后，减少其后继节点的入度
    4. 入度降为 0 的节点立即加入就绪队列（动态解锁）
    5. 重复直到所有节点完成
    """

    def __init__(self, agent_registry: Dict[str, "BaseAgent"]):
        """
        Args:
            agent_registry: Agent 名称 → Agent 实例的映射字典
                            { "document_parser_agent": DocumentParserAgent(), ... }
        """
        self.agent_registry = agent_registry  # Agent 注册表，执行时通过 agent_name 查找实例

    async def execute(
        self,
        plan: DAGPlan,
        ctx: "AgentContext",
        db: "AsyncSession",
    ) -> Dict[str, "AgentResult"]:
        """
        按 DAG 计划执行所有 Agent，返回每个节点的执行结果

        Args:
            plan: DAG 执行计划（节点 + 边）
            ctx:  Agent 执行上下文（含任务 ID、共享数据）
            db:   异步数据库会话

        Returns:
            Dict[node_id -> AgentResult]：所有节点的执行结果（包含跳过的节点）
        """
        logger.info(
            f"[dag_executor] 开始执行 plan={plan.plan_name} "
            f"nodes={len(plan.nodes)} edges={len(plan.edges)} "
            f"task={ctx.audit_task_id}"
        )

        # 构建邻接表和入度表（拓扑排序的数据基础）
        in_degree: Dict[str, int] = {node.node_id: 0 for node in plan.nodes}   # 各节点入度（前置依赖数）
        successors: Dict[str, List[str]] = {node.node_id: [] for node in plan.nodes}  # 各节点的后继节点
        node_map: Dict[str, DAGNode] = {node.node_id: node for node in plan.nodes}    # id → 节点对象

        for edge in plan.edges:
            in_degree[edge.to_node] += 1          # to_node 多了一个前置依赖，入度加 1
            successors[edge.from_node].append(edge.to_node)  # from_node 完成后需要通知的节点

        # 跳过节点集合：关键前置节点失败时，其所有依赖节点标记为跳过
        skipped: Set[str] = set()
        results: Dict[str, "AgentResult"] = {}    # 收集所有节点的执行结果

        # 初始化就绪队列：入度为 0 的节点可以立即执行
        ready_queue: List[str] = [
            nid for nid, deg in in_degree.items() if deg == 0
        ]

        # 主循环：持续处理就绪节点，直到所有节点完成
        while ready_queue or len(results) + len(skipped) < len(plan.nodes):

            if not ready_queue:
                # 就绪队列为空但仍有节点未处理：理论上不应发生（DAG 无环保证），记录警告
                logger.warning(
                    f"[dag_executor] 就绪队列为空但仍有 {len(plan.nodes) - len(results) - len(skipped)} 个节点未处理，"
                    f"可能存在图结构错误 task={ctx.audit_task_id}"
                )
                break  # 避免死循环

            # 取出当前批次所有就绪节点（同一批次并行执行）
            current_batch = ready_queue.copy()   # 当前批次的节点 ID 列表
            ready_queue.clear()                   # 清空就绪队列，等待本批次完成后重新填入

            logger.debug(
                f"[dag_executor] 执行批次 nodes={current_batch} task={ctx.audit_task_id}"
            )

            # 并行执行当前批次所有 Agent（asyncio.gather 并发，最大化执行效率）
            batch_results = await asyncio.gather(
                *[
                    self._execute_single(node_map[nid], ctx, db, skipped)
                    for nid in current_batch
                ],
                return_exceptions=False  # 单个 Agent 的异常已由 BaseAgent.execute() 捕获，不会传到这里
            )

            # 处理本批次执行结果，更新就绪队列
            for node_id, result in zip(current_batch, batch_results):
                results[node_id] = result  # 记录执行结果

                # 检查是否需要传播失败：关键节点失败，其所有后继节点跳过
                if not result.success and node_map[node_id].is_critical:
                    self._propagate_skip(node_id, successors, skipped, node_map)
                    logger.warning(
                        f"[dag_executor] 关键节点 {node_id} 失败，跳过其后继节点 "
                        f"successors={successors[node_id]} task={ctx.audit_task_id}"
                    )

                # 更新后继节点的入度，入度降为 0 的节点加入就绪队列
                for succ_id in successors[node_id]:
                    if succ_id in skipped:
                        continue  # 已被标记跳过的节点不需要更新入度

                    in_degree[succ_id] -= 1          # 前置依赖完成，入度减 1
                    if in_degree[succ_id] == 0:      # 所有前置依赖都已完成
                        ready_queue.append(succ_id)  # 加入就绪队列，下一批次执行

        # 为所有跳过的节点生成跳过结果（让调用方能完整遍历所有节点）
        from app.agents.base_agent import AgentResult  # 延迟导入避免循环依赖
        for skipped_id in skipped:
            if skipped_id not in results:  # 可能已在上面记录（直接跳过的节点）
                results[skipped_id] = AgentResult(
                    agent_name=node_map[skipped_id].agent_name,
                    success=False,
                    error_code="AGENT_SKIPPED",
                    error_msg="前置关键节点失败，当前节点已跳过",
                )

        logger.info(
            f"[dag_executor] 执行完成 plan={plan.plan_name} "
            f"total={len(plan.nodes)} success={sum(1 for r in results.values() if r.success)} "
            f"failed={sum(1 for r in results.values() if not r.success)} "
            f"task={ctx.audit_task_id}"
        )

        return results

    async def _execute_single(
        self,
        node: DAGNode,
        ctx: "AgentContext",
        db: "AsyncSession",
        skipped: Set[str],
    ) -> "AgentResult":
        """
        执行单个节点对应的 Agent（带超时控制）

        Args:
            node:    DAG 节点（含 agent_name 和额外参数）
            ctx:     执行上下文
            db:      数据库会话
            skipped: 已跳过节点集合（若当前节点已被标记跳过则直接返回）

        Returns:
            AgentResult: 执行结果
        """
        from app.agents.base_agent import AgentResult  # 延迟导入避免循环依赖

        # 检查节点是否已被标记为跳过（由前置关键节点失败触发）
        if node.node_id in skipped:
            return AgentResult(
                agent_name=node.agent_name,
                success=False,
                error_code="AGENT_SKIPPED",
                error_msg="前置关键节点失败，当前节点已跳过",
            )

        # 从注册表中查找 Agent 实例
        agent = self.agent_registry.get(node.agent_name)
        if agent is None:
            # 找不到 Agent：可能是 agent_name 拼写错误或未注册
            logger.error(
                f"[dag_executor] Agent 未注册: {node.agent_name} task={ctx.audit_task_id}"
            )
            return AgentResult(
                agent_name=node.agent_name,
                success=False,
                error_code="AGENT_NOT_REGISTERED",
                error_msg=f"Agent '{node.agent_name}' 未在注册表中找到，请检查 agent_registry.py",
            )

        # 使用 asyncio.wait_for 实现超时控制：超时自动取消该 Agent 的执行
        try:
            result = await asyncio.wait_for(
                agent.execute(ctx, db, **node.kwargs),  # 调用 BaseAgent.execute()（含计时+日志+异常兜底）
                timeout=node.timeout_seconds,           # 超时阈值，默认 300 秒
            )
        except asyncio.TimeoutError:
            # 超时：记录日志并返回超时失败结果
            logger.warning(
                f"[dag_executor] Agent 超时: {node.agent_name} "
                f"timeout={node.timeout_seconds}s task={ctx.audit_task_id}"
            )
            result = AgentResult(
                agent_name=node.agent_name,
                success=False,
                error_code="AGENT_TIMEOUT",
                error_msg=f"Agent 执行超时（>{node.timeout_seconds}秒）",
                elapsed_ms=node.timeout_seconds * 1000,  # 超时时耗时 = 超时阈值
            )

        return result

    def _propagate_skip(
        self,
        failed_node_id: str,
        successors: Dict[str, List[str]],
        skipped: Set[str],
        node_map: Dict[str, DAGNode],
    ) -> None:
        """
        递归传播跳过标记：失败节点的所有后继节点（包括间接后继）都标记为跳过
        使用 BFS（广度优先搜索）避免递归栈溢出

        Args:
            failed_node_id: 失败的关键节点 ID
            successors:     节点后继关系图
            skipped:        跳过节点集合（原地修改）
            node_map:       节点 ID → DAGNode 映射
        """
        queue = list(successors.get(failed_node_id, []))  # 初始化队列：直接后继节点

        while queue:
            current = queue.pop(0)          # 取出队列头部节点
            if current not in skipped:      # 避免重复处理
                skipped.add(current)        # 标记为跳过
                queue.extend(successors.get(current, []))  # 将该节点的后继节点也加入队列
