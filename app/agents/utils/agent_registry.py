# app/agents/utils/agent_registry.py
# Agent 注册表：维护 agent_name → 类 的映射关系
# 作用：OrchestratorAgent 和 API 路由按名称动态实例化 Agent，无需硬编码 import
# 使用方式：AgentRegistry.get("element_extraction_agent")() 返回 Agent 实例

from __future__ import annotations

from typing import Dict, Optional, Type  # 类型注解

from loguru import logger  # 日志


# ── 延迟导入（避免循环导入和不必要的启动开销）─────────────────────────────────
def _load_all_agents() -> Dict[str, Type]:
    """
    加载全部 Agent 类到注册表
    延迟导入：注册表首次被访问时才执行，避免启动时全量导入导致慢启动
    """
    # 每个导入对应一个专项 Agent，导入失败时单个 Agent 不影响其他
    registry: Dict[str, Type] = {}

    # M3：文档解析层
    try:
        from app.agents.document_parser_agent import DocumentParserAgent
        registry["document_parser_agent"] = DocumentParserAgent
    except Exception as e:
        logger.warning(f"[agent_registry] document_parser_agent 加载失败: {e}")

    try:
        from app.agents.element_extraction_agent import ElementExtractionAgent
        registry["element_extraction_agent"] = ElementExtractionAgent
    except Exception as e:
        logger.warning(f"[agent_registry] element_extraction_agent 加载失败: {e}")

    # M4：合规检索层
    try:
        from app.agents.compliance_retrieval_agent import ComplianceRetrievalAgent
        registry["compliance_retrieval_agent"] = ComplianceRetrievalAgent
    except Exception as e:
        logger.warning(f"[agent_registry] compliance_retrieval_agent 加载失败: {e}")

    try:
        from app.agents.endorsement_chain_agent import EndorsementChainAgent
        registry["endorsement_chain_agent"] = EndorsementChainAgent
    except Exception as e:
        logger.warning(f"[agent_registry] endorsement_chain_agent 加载失败: {e}")

    # M5：风险评估层
    try:
        from app.agents.contract_review_agent import ContractReviewAgent
        registry["contract_review_agent"] = ContractReviewAgent
    except Exception as e:
        logger.warning(f"[agent_registry] contract_review_agent 加载失败: {e}")

    try:
        from app.agents.risk_assessment_agent import RiskAssessmentAgent
        registry["risk_assessment_agent"] = RiskAssessmentAgent
    except Exception as e:
        logger.warning(f"[agent_registry] risk_assessment_agent 加载失败: {e}")

    # M6：报告生成层
    try:
        from app.agents.report_generation_agent import ReportGenerationAgent
        registry["report_generation_agent"] = ReportGenerationAgent
    except Exception as e:
        logger.warning(f"[agent_registry] report_generation_agent 加载失败: {e}")

    # M7：流转追踪层
    try:
        from app.agents.flow_tracking_agent import FlowTrackingAgent
        registry["flow_tracking_agent"] = FlowTrackingAgent
    except Exception as e:
        logger.warning(f"[agent_registry] flow_tracking_agent 加载失败: {e}")

    # M8：欺诈检测层
    try:
        from app.agents.fraud_detection_agent import FraudDetectionAgent
        registry["fraud_detection_agent"] = FraudDetectionAgent
    except Exception as e:
        logger.warning(f"[agent_registry] fraud_detection_agent 加载失败: {e}")

    # M9：批量调度层
    try:
        from app.agents.batch_scheduling_agent import BatchSchedulingAgent
        registry["batch_scheduling_agent"] = BatchSchedulingAgent
    except Exception as e:
        logger.warning(f"[agent_registry] batch_scheduling_agent 加载失败: {e}")

    # M10：出票预检层
    try:
        from app.agents.bill_issuance_agent import BillIssuanceAgent
        registry["bill_issuance_agent"] = BillIssuanceAgent
    except Exception as e:
        logger.warning(f"[agent_registry] bill_issuance_agent 加载失败: {e}")

    # M2：编排层（最后加载，依赖以上所有 Agent）
    try:
        from app.agents.orchestrator_agent import OrchestratorAgent
        registry["orchestrator_agent"] = OrchestratorAgent
    except Exception as e:
        logger.warning(f"[agent_registry] orchestrator_agent 加载失败: {e}")

    logger.info(f"[agent_registry] 已注册 {len(registry)} 个 Agent: {list(registry.keys())}")
    return registry


# ── 全局注册表单例（首次访问时懒加载）─────────────────────────────────────────
_REGISTRY: Optional[Dict[str, Type]] = None  # None 表示尚未初始化


class AgentRegistry:
    """
    Agent 注册表门面类
    提供按名称获取 Agent 类、列举所有已注册 Agent 等静态方法
    """

    @staticmethod
    def _ensure_loaded() -> Dict[str, Type]:
        """确保注册表已加载（懒加载：首次调用时才 import 所有 Agent）"""
        global _REGISTRY
        if _REGISTRY is None:
            _REGISTRY = _load_all_agents()  # 首次访问时触发全量加载
        return _REGISTRY

    @classmethod
    def get(cls, agent_name: str) -> Optional[Type]:
        """
        按名称获取 Agent 类（未注册时返回 None）

        Args:
            agent_name: Agent 标识符，如 "element_extraction_agent"

        Returns:
            Agent 类（未实例化），调用方需自行 agent_cls() 实例化
        """
        registry = cls._ensure_loaded()
        agent_cls = registry.get(agent_name)
        if agent_cls is None:
            logger.warning(f"[agent_registry] 未找到 Agent: {agent_name}")
        return agent_cls

    @classmethod
    def get_or_raise(cls, agent_name: str) -> Type:
        """
        按名称获取 Agent 类，未找到时抛出 KeyError（用于必须存在的 Agent）

        Args:
            agent_name: Agent 标识符

        Raises:
            KeyError: 指定名称的 Agent 未注册
        """
        agent_cls = cls.get(agent_name)
        if agent_cls is None:
            raise KeyError(f"Agent '{agent_name}' 未在注册表中，请检查 agent_registry.py")
        return agent_cls

    @classmethod
    def all_names(cls) -> list:
        """返回所有已注册 Agent 的名称列表"""
        return list(cls._ensure_loaded().keys())

    @classmethod
    def is_registered(cls, agent_name: str) -> bool:
        """检查指定 Agent 是否已注册"""
        return agent_name in cls._ensure_loaded()

    @classmethod
    def reload(cls) -> None:
        """强制重新加载注册表（测试和热更新场景使用）"""
        global _REGISTRY
        _REGISTRY = None  # 清空缓存，下次访问时重新加载
        cls._ensure_loaded()
        logger.info("[agent_registry] 注册表已重新加载")
