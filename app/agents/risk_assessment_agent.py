# app/agents/risk_assessment_agent.py
# RiskAssessmentAgent：四维风险评分专项 Agent
# 职责：
#   1. 从 shared_data 读取合规/背书/合同/欺诈四个维度的中间结果
#   2. 按权重（合规30%+背书30%+合同20%+欺诈20%）计算综合评分
#   3. 按五档阈值判定风险等级（LOW/MEDIUM_LOW/MEDIUM_HIGH/HIGH/CRITICAL）
#   4. 将评估结果写入 risk_assessments 表

from __future__ import annotations

import uuid                                  # 生成评估记录 UUID
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.models.agent_models import RiskAssessment, RiskLevel


# ── 四维权重配置（合计 = 1.0）─────────────────────────────────────────────────
DIMENSION_WEIGHTS: Dict[str, float] = {
    "compliance":  0.30,   # 合规维度：覆盖 18 个要素字段的合规性
    "endorsement": 0.30,   # 背书维度：背书链路完整性和违规检测
    "contract":    0.20,   # 合同维度：票据要素与合同的匹配程度
    "fraud":       0.20,   # 欺诈维度：基于欺诈检测的反欺诈得分
}

# ── 五档风险阈值（分越高风险越低，类似信用评分）────────────────────────────────
RISK_THRESHOLDS = [
    (85.0, RiskLevel.LOW),          # ≥ 85 分：低风险，自动通过
    (70.0, RiskLevel.MEDIUM_LOW),   # 70~84 分：中低风险，快速审核
    (50.0, RiskLevel.MEDIUM_HIGH),  # 50~69 分：中高风险，需人工复核
    (30.0, RiskLevel.HIGH),         # 30~49 分：高风险，建议拒绝
    (0.0,  RiskLevel.CRITICAL),     # < 30 分：极高风险，强制拒绝
]

# ── 缺失维度降级默认分（不阻断评估流程，但降低信心）────────────────────────────
DEFAULT_SCORE_MISSING_DIMENSION: Dict[str, float] = {
    "compliance":  60.0,   # 无合规检测结果：给予 60 分（中间偏低）
    "endorsement": 70.0,   # 无背书检测结果：给予 70 分（较宽松）
    "contract":    50.0,   # 无合同审核结果：给予 50 分（强制人工复核）
    "fraud":       80.0,   # 无欺诈检测结果：给予 80 分（欺诈默认无嫌疑）
}


class RiskAssessmentAgent(BaseAgent):
    """
    四维风险评估 Agent：综合各专项 Agent 的结果，输出最终风险等级
    设计原则：缺失维度降级处理（不阻断流程），所有计算可在无外部调用的情况下完成
    """

    agent_name = "risk_assessment_agent"

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        **kwargs
    ) -> AgentResult:
        """
        风险评估主逻辑

        Args:
            ctx: 上下文（读取 compliance_summary/endorsement_result/contract_result/fraud_result）
            db:  数据库会话

        Returns:
            AgentResult: 包含 composite_score / risk_level / 各维度得分
        """
        # 步骤 1：从 shared_data 读取各维度中间结果（各 Agent 运行后写入）
        compliance_summary  = ctx.shared_data.get("compliance_summary")   # ComplianceRetrievalAgent 输出
        endorsement_result  = ctx.shared_data.get("endorsement_result")   # EndorsementChainAgent 输出
        contract_result     = ctx.shared_data.get("contract_result")      # ContractReviewAgent 输出
        fraud_result        = ctx.shared_data.get("fraud_result")         # FraudDetectionAgent 输出

        # 步骤 2：将各维度中间结果转换为 0~100 的得分，缺失则降级
        missing_dimensions: List[str] = []   # 记录哪些维度缺失

        compliance_score  = self._calc_compliance_score(compliance_summary,  missing_dimensions)
        endorsement_score = self._calc_endorsement_score(endorsement_result, missing_dimensions)
        contract_score    = self._calc_contract_score(contract_result,        missing_dimensions)
        fraud_score       = self._calc_fraud_score(fraud_result,              missing_dimensions)

        # 步骤 3：四维加权求和，计算综合评分
        composite_score = (
            compliance_score  * DIMENSION_WEIGHTS["compliance"]  +
            endorsement_score * DIMENSION_WEIGHTS["endorsement"] +
            contract_score    * DIMENSION_WEIGHTS["contract"]    +
            fraud_score       * DIMENSION_WEIGHTS["fraud"]
        )
        composite_score = round(composite_score, 2)   # 保留两位小数

        # 步骤 4：按阈值判定风险等级
        risk_level = self._determine_risk_level(composite_score)

        # 步骤 5：构建各维度详细说明（供报告展示）
        dimension_details = self._build_dimension_details(
            compliance_score,  compliance_summary,
            endorsement_score, endorsement_result,
            contract_score,    contract_result,
            fraud_score,       fraud_result,
        )

        # 步骤 6：生成评估说明文本
        assessor_notes = self._gen_assessor_notes(
            composite_score, risk_level, missing_dimensions, dimension_details
        )

        # 步骤 7：写入 risk_assessments 表
        assessment_id = str(uuid.uuid4())
        assessment = RiskAssessment(
            id=assessment_id,
            audit_task_id=ctx.audit_task_id,
            compliance_score=round(compliance_score, 2),
            endorsement_score=round(endorsement_score, 2),
            contract_score=round(contract_score, 2),
            fraud_score=round(fraud_score, 2),
            composite_score=composite_score,
            risk_level=risk_level,
            dimension_details=dimension_details,
            missing_dimensions=missing_dimensions,
            assessor_notes=assessor_notes,
        )
        db.add(assessment)

        # 步骤 8：写入 shared_data 供 ReportGenerationAgent 使用
        ctx.shared_data["risk_result"] = {
            "assessment_id":   assessment_id,
            "compliance_score":  compliance_score,
            "endorsement_score": endorsement_score,
            "contract_score":    contract_score,
            "fraud_score":       fraud_score,
            "composite_score":   composite_score,
            "risk_level":        risk_level.value,
            "missing_dimensions": missing_dimensions,
        }

        logger.info(
            f"[{self.agent_name}] 风险评估完成 "
            f"composite={composite_score:.1f} level={risk_level.value} "
            f"missing={missing_dimensions} task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "compliance_score":  compliance_score,
                "endorsement_score": endorsement_score,
                "contract_score":    contract_score,
                "fraud_score":       fraud_score,
                "composite_score":   composite_score,
                "risk_level":        risk_level.value,
                "missing_dimensions": missing_dimensions,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 各维度得分计算（返回 0.0~100.0）
    # ──────────────────────────────────────────────────────────────────────────

    def _calc_compliance_score(
        self, summary: Optional[dict], missing: List[str]
    ) -> float:
        """
        合规维度得分计算
        合规率越高得分越高；存在严重违规则额外扣分
        """
        if not summary:
            missing.append("compliance")   # 记录缺失维度
            return DEFAULT_SCORE_MISSING_DIMENSION["compliance"]

        compliance_rate = summary.get("compliance_rate", 0.0)   # 0.0~1.0
        severe_count    = summary.get("severe_count", 0)         # 严重违规数

        # 基础分：合规率直接映射到 0~100（合规率=1.0 → 100分）
        base_score = compliance_rate * 100.0

        # 严重违规惩罚：每个 SEVERE 违规扣 15 分（下限 0 分）
        penalty = severe_count * 15.0
        score = max(0.0, base_score - penalty)

        return round(score, 2)

    def _calc_endorsement_score(
        self, result: Optional[dict], missing: List[str]
    ) -> float:
        """
        背书维度得分计算
        满分 100；断链扣 40，闭环扣 50，每个违规扣 10
        """
        if not result:
            missing.append("endorsement")   # 记录缺失维度
            return DEFAULT_SCORE_MISSING_DIMENSION["endorsement"]

        is_continuous   = result.get("is_continuous", True)    # 背书链是否连续
        has_cycle       = result.get("has_cycle", False)        # 是否存在闭环
        violation_count = result.get("violation_count", 0)      # 背书违规总数

        score = 100.0
        if not is_continuous:
            score -= 40.0   # 断链是严重违规，大幅扣分
        if has_cycle:
            score -= 50.0   # 闭环涉嫌欺诈，重度扣分
        # 每个其他违规扣 10 分（断链和闭环已单独处理，此处再扣剩余违规）
        other_violations = max(0, violation_count - (1 if not is_continuous else 0) - (1 if has_cycle else 0))
        score -= other_violations * 10.0

        return round(max(0.0, score), 2)   # 下限 0 分

    def _calc_contract_score(
        self, result: Optional[dict], missing: List[str]
    ) -> float:
        """
        合同维度得分计算
        直接使用 ContractReviewAgent 计算的 match_score（已在 0~100 范围）
        """
        if not result:
            missing.append("contract")   # 记录缺失维度
            return DEFAULT_SCORE_MISSING_DIMENSION["contract"]

        # match_score 已是 0~100 的综合匹配度，直接使用
        match_score = result.get("match_score", 50.0)
        return round(float(match_score), 2)

    def _calc_fraud_score(
        self, result: Optional[dict], missing: List[str]
    ) -> float:
        """
        欺诈维度得分计算
        欺诈评分为风险分（越高越危险），需转换为安全分（越高越安全）
        安全分 = (1 - overall_fraud_score) × 100
        """
        if not result:
            missing.append("fraud")   # 记录缺失维度
            return DEFAULT_SCORE_MISSING_DIMENSION["fraud"]

        overall_fraud_score = result.get("overall_fraud_score", 0.0)   # 0.0~1.0 风险分
        # 翻转：欺诈风险越高，安全得分越低
        safety_score = (1.0 - float(overall_fraud_score)) * 100.0
        return round(max(0.0, min(100.0, safety_score)), 2)

    # ──────────────────────────────────────────────────────────────────────────
    # 风险等级判定
    # ──────────────────────────────────────────────────────────────────────────

    def _determine_risk_level(self, composite_score: float) -> RiskLevel:
        """
        按五档阈值判定风险等级
        阈值从高到低遍历，第一个满足的阈值即为风险等级
        """
        for threshold, level in RISK_THRESHOLDS:
            if composite_score >= threshold:
                return level
        return RiskLevel.CRITICAL   # 兜底：所有阈值都不满足时为极高风险

    # ──────────────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    def _build_dimension_details(
        self,
        compliance_score: float,  compliance_summary: Optional[dict],
        endorsement_score: float, endorsement_result: Optional[dict],
        contract_score: float,    contract_result: Optional[dict],
        fraud_score: float,       fraud_result: Optional[dict],
    ) -> dict:
        """
        构建各维度详细说明字典（写入 dimension_details 字段，供报告展示）
        格式：{dimension: {score, weight, contributing_factors:[...]}}
        """
        details = {}

        # 合规维度详情
        if compliance_summary:
            details["compliance"] = {
                "score":  compliance_score,
                "weight": DIMENSION_WEIGHTS["compliance"],
                "contributing_factors": [
                    f"合规率: {compliance_summary.get('compliance_rate', 0):.1%}",
                    f"违规字段数: {compliance_summary.get('violation_count', 0)}",
                    f"严重违规数: {compliance_summary.get('severe_count', 0)}",
                ],
            }
        else:
            details["compliance"] = {
                "score":  compliance_score,
                "weight": DIMENSION_WEIGHTS["compliance"],
                "contributing_factors": ["合规检测结果缺失，使用降级默认分"],
            }

        # 背书维度详情
        if endorsement_result:
            details["endorsement"] = {
                "score":  endorsement_score,
                "weight": DIMENSION_WEIGHTS["endorsement"],
                "contributing_factors": [
                    f"背书链连续性: {'是' if endorsement_result.get('is_continuous') else '否'}",
                    f"存在闭环: {'是' if endorsement_result.get('has_cycle') else '否'}",
                    f"违规数量: {endorsement_result.get('violation_count', 0)}",
                    f"违规代码: {endorsement_result.get('violation_codes', [])}",
                ],
            }
        else:
            details["endorsement"] = {
                "score":  endorsement_score,
                "weight": DIMENSION_WEIGHTS["endorsement"],
                "contributing_factors": ["背书链检测结果缺失，使用降级默认分"],
            }

        # 合同维度详情
        if contract_result:
            details["contract"] = {
                "score":  contract_score,
                "weight": DIMENSION_WEIGHTS["contract"],
                "contributing_factors": [
                    f"要素匹配度: {contract_result.get('match_score', 0):.1f}",
                    f"金额匹配: {'是' if contract_result.get('amount_match') else '否'}",
                    f"交易方匹配: {'是' if contract_result.get('party_match') else '否'}",
                    f"不匹配项数: {contract_result.get('mismatch_count', 0)}",
                ],
            }
        else:
            details["contract"] = {
                "score":  contract_score,
                "weight": DIMENSION_WEIGHTS["contract"],
                "contributing_factors": ["合同审核结果缺失，使用降级默认分"],
            }

        # 欺诈维度详情
        if fraud_result:
            details["fraud"] = {
                "score":  fraud_score,
                "weight": DIMENSION_WEIGHTS["fraud"],
                "contributing_factors": [
                    f"综合欺诈风险分: {fraud_result.get('overall_fraud_score', 0):.3f}",
                    f"安全得分（翻转）: {fraud_score:.1f}",
                ],
            }
        else:
            details["fraud"] = {
                "score":  fraud_score,
                "weight": DIMENSION_WEIGHTS["fraud"],
                "contributing_factors": ["欺诈检测结果缺失，使用降级默认分"],
            }

        return details

    def _gen_assessor_notes(
        self,
        composite_score: float,
        risk_level: RiskLevel,
        missing_dimensions: List[str],
        dimension_details: dict,
    ) -> str:
        """生成供人工审核人员阅读的评估说明文本"""
        level_desc = {
            RiskLevel.LOW:         "低风险，建议自动通过",
            RiskLevel.MEDIUM_LOW:  "中低风险，建议快速人工审核后放行",
            RiskLevel.MEDIUM_HIGH: "中高风险，需要人工复核所有维度",
            RiskLevel.HIGH:        "高风险，建议拒绝，需补充材料重新提交",
            RiskLevel.CRITICAL:    "极高风险，强制拒绝，请联系合规部门",
        }

        notes = f"综合评分 {composite_score:.1f}，风险等级 {risk_level.value}：{level_desc[risk_level]}。"

        if missing_dimensions:
            notes += f" 注意：以下维度数据缺失，使用降级分处理：{', '.join(missing_dimensions)}。"

        # 指出得分最低的维度，提示重点关注
        scores = {
            "合规": dimension_details.get("compliance", {}).get("score", 100),
            "背书": dimension_details.get("endorsement", {}).get("score", 100),
            "合同": dimension_details.get("contract", {}).get("score", 100),
            "欺诈": dimension_details.get("fraud", {}).get("score", 100),
        }
        min_dim = min(scores, key=scores.get)
        min_score = scores[min_dim]
        if min_score < 70:
            notes += f" 重点关注：{min_dim}维度得分最低（{min_score:.1f}分）。"

        return notes
