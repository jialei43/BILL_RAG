# app/agents/report_generation_agent.py
# ReportGenerationAgent：审核报告生成专项 Agent
# 职责：
#   1. 聚合 shared_data 中所有 Agent 的中间结果，构建 9 节完整报告 JSON
#   2. 调用 ReportRenderer 将 JSON 渲染为 PDF 文件
#   3. 将报告记录写入 audit_reports 表
#   4. 将报告路径写入 shared_data["report_result"]

from __future__ import annotations

import asyncio       # asyncio.to_thread：将同步 PDF 渲染放到线程池执行（避免阻塞事件循环）
import os
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.agents.utils.report_renderer import ReportRenderer
from app.models.agent_models import AuditReport


class ReportGenerationAgent(BaseAgent):
    """
    报告生成 Agent：将多 Agent 协作产出的各维度结果汇总为人可读的完整报告
    同时输出结构化 JSON（供 API 返回）和 PDF 文件（供下载存档）
    """

    agent_name = "report_generation_agent"

    def __init__(self, output_dir: Optional[str] = None):
        # output_dir：PDF 文件输出目录；None 时使用系统临时目录
        self._output_dir = output_dir

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        training_mode: bool = False,   # 是否附加培训注解层
        is_final: bool = False,         # 是否标记为终版报告
        **kwargs
    ) -> AgentResult:
        """
        报告生成主逻辑

        Args:
            ctx:           上下文（读取所有 Agent 的 shared_data 输出）
            db:            数据库会话
            training_mode: 培训模式时在 PDF 中附加教学注解
            is_final:      True=终版报告（签发后不可修改）/ False=草稿

        Returns:
            AgentResult: data 包含 report_id / pdf_path / pdf_size_bytes
        """
        # 步骤 1：聚合所有维度数据，构建 9 节 JSON 结构
        report_json = self._build_report_json(ctx, training_mode)

        # 步骤 2：使用 asyncio.to_thread 在线程池中调用同步 PDF 渲染
        # （reportlab 是同步库，直接调用会阻塞 FastAPI 的异步事件循环）
        pdf_path = None
        pdf_size = 0
        try:
            renderer = ReportRenderer(output_dir=self._output_dir)
            pdf_path = await asyncio.to_thread(
                renderer.render,
                report_json,
                ctx.audit_task_id,
                training_mode,
            )
            pdf_size = os.path.getsize(pdf_path) if pdf_path and os.path.exists(pdf_path) else 0
        except Exception as e:
            # PDF 渲染失败不阻断流程：报告 JSON 仍会保存，PDF 路径为空
            logger.warning(f"[{self.agent_name}] PDF 渲染失败（JSON 仍保存）: {e}")

        # 步骤 3：写入 audit_reports 表
        report_id = str(uuid.uuid4())
        report = AuditReport(
            id=report_id,
            audit_task_id=ctx.audit_task_id,
            report_json=report_json,
            pdf_path=pdf_path,
            pdf_size_bytes=pdf_size,
            is_final=is_final,
            training_mode=training_mode,
            version=1,                          # 初始版本号
            generated_by="agent",               # 自动生成标识
        )
        db.add(report)

        # 步骤 4：写入 shared_data 供后续使用（如 API 返回下载链接）
        ctx.shared_data["report_result"] = {
            "report_id":      report_id,
            "pdf_path":       pdf_path,
            "pdf_size_bytes": pdf_size,
            "is_final":       is_final,
            "has_pdf":        pdf_size > 0,
        }

        logger.info(
            f"[{self.agent_name}] 报告生成完成 "
            f"report_id={report_id[:8]} pdf_size={pdf_size} "
            f"training={training_mode} task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "report_id":      report_id,
                "pdf_path":       pdf_path,
                "pdf_size_bytes": pdf_size,
                "has_pdf":        pdf_size > 0,
                "sections_count": len(report_json),
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 报告 JSON 构建（9 节结构）
    # ──────────────────────────────────────────────────────────────────────────

    def _build_report_json(self, ctx: AgentContext, training_mode: bool) -> dict:
        """
        聚合所有维度数据，构建完整的 9 节报告 JSON

        节编号对应的数据来源：
          summary      → risk_result 综合评分
          elements     → bill_element 18 字段
          compliance   → compliance_summary
          endorsement  → endorsement_result
          contract     → contract_result
          risk         → risk_result 四维分项
          fraud        → fraud_result
          flow         → flow_result（如有）
          conclusion   → 根据 risk_level 自动生成
        """
        risk_result        = ctx.shared_data.get("risk_result", {})
        bill_element       = ctx.shared_data.get("bill_element", {})
        compliance_summary = ctx.shared_data.get("compliance_summary", {})
        endorsement_result = ctx.shared_data.get("endorsement_result", {})
        contract_result    = ctx.shared_data.get("contract_result", {})
        fraud_result       = ctx.shared_data.get("fraud_result", {})
        flow_result        = ctx.shared_data.get("flow_result", {})

        report = {
            # 节1：审核摘要（快速概览）
            "summary": {
                "task_id":        ctx.audit_task_id,
                "tenant_id":      ctx.tenant_id,
                "composite_score": risk_result.get("composite_score"),
                "risk_level":     risk_result.get("risk_level"),
                "conclusion":     self._gen_conclusion_text(risk_result.get("risk_level")),
                "generated_at":   datetime.now().isoformat(),
                "training_mode":  training_mode,
            },
            # 节2：票据要素（18字段详情）
            "elements": {
                "ticket_number":    bill_element.get("ticket_number"),
                "ticket_type":      bill_element.get("ticket_type"),
                "issue_date":       bill_element.get("issue_date"),
                "due_date":         bill_element.get("due_date"),
                "amount_numeric":   bill_element.get("amount_numeric"),
                "amount_text":      bill_element.get("amount_text"),
                "currency":         bill_element.get("currency"),
                "drawer":           bill_element.get("drawer"),
                "drawer_account":   bill_element.get("drawer_account"),
                "drawer_bank":      bill_element.get("drawer_bank"),
                "acceptor":         bill_element.get("acceptor"),
                "payee":            bill_element.get("payee"),
                "drawee_bank":      bill_element.get("drawee_bank"),
                "endorsers":        bill_element.get("endorsers", []),
                "maturity_days":    bill_element.get("maturity_days"),
                "trade_purpose":    bill_element.get("trade_purpose"),
                "acceptance_clause": bill_element.get("acceptance_clause"),
                "special_remarks":  bill_element.get("special_remarks"),
                "confidence_score": bill_element.get("confidence_score"),
            },
            # 节3：合规检查结果
            "compliance": {
                "total_fields":    compliance_summary.get("total_fields"),
                "violation_count": compliance_summary.get("violation_count"),
                "severe_count":    compliance_summary.get("severe_count"),
                "compliance_rate": compliance_summary.get("compliance_rate"),
                "is_compliant":    compliance_summary.get("is_overall_compliant"),
                "violations":      [],   # 详细违规列表（简化：通过 compliance_checks 表查询）
            },
            # 节4：背书链分析
            "endorsement": {
                "endorser_count":  endorsement_result.get("endorser_count"),
                "is_continuous":   endorsement_result.get("is_continuous"),
                "has_cycle":       endorsement_result.get("has_cycle"),
                "violation_count": endorsement_result.get("violation_count"),
                "violation_codes": endorsement_result.get("violation_codes", []),
            },
            # 节5：合同审核
            "contract": {
                "match_score":            contract_result.get("match_score"),
                "trade_background_score": contract_result.get("trade_background_score"),
                "amount_match":           contract_result.get("amount_match"),
                "party_match":            contract_result.get("party_match"),
                "mismatch_count":         contract_result.get("mismatch_count"),
                "review_notes":           None,   # 详细备注从 contract_reviews 表查询
            },
            # 节6：风险评估（四维得分）
            "risk": {
                "compliance_score":  risk_result.get("compliance_score"),
                "endorsement_score": risk_result.get("endorsement_score"),
                "contract_score":    risk_result.get("contract_score"),
                "fraud_score":       risk_result.get("fraud_score"),
                "composite_score":   risk_result.get("composite_score"),
                "risk_level":        risk_result.get("risk_level"),
                "missing_dimensions": risk_result.get("missing_dimensions", []),
                "assessor_notes":    None,   # 详细说明从 risk_assessments 表查询
            },
            # 节7：欺诈检测
            "fraud": {
                "overall_fraud_score": fraud_result.get("overall_fraud_score"),
                "seal_score":         fraud_result.get("seal_score"),
                "duplicate_score":    fraud_result.get("duplicate_score"),
                "tamper_score":       fraud_result.get("tamper_score"),
                "network_score":      fraud_result.get("network_score"),
                "blacklist_score":    fraud_result.get("blacklist_score"),
            },
            # 节8：流转追踪状态（如已运行 FlowTrackingAgent）
            "flow": {
                "task_status":     flow_result.get("status"),
                "completed_steps": flow_result.get("completed_steps"),
                "current_node":    flow_result.get("current_node"),
                "timeout_level":   flow_result.get("timeout_level"),
            },
            # 节9：审核结论和建议
            "conclusion": {
                "decision":        self._gen_decision(risk_result.get("risk_level")),
                "risk_level":      risk_result.get("risk_level"),
                "composite_score": risk_result.get("composite_score"),
                "recommendations": self._gen_recommendations(
                    risk_result, compliance_summary, endorsement_result
                ),
                "missing_dimensions": risk_result.get("missing_dimensions", []),
            },
        }
        return report

    def _gen_conclusion_text(self, risk_level: Optional[str]) -> str:
        """根据风险等级生成一句话审核结论"""
        conclusions = {
            "LOW":         "Ticket passes compliance review. Auto-approval recommended.",
            "MEDIUM_LOW":  "Ticket is mostly compliant. Quick manual review recommended.",
            "MEDIUM_HIGH": "Ticket has notable issues. Full manual review required.",
            "HIGH":        "Ticket has serious issues. Rejection recommended.",
            "CRITICAL":    "Ticket has critical violations. Mandatory rejection.",
        }
        return conclusions.get(risk_level or "", "Risk assessment incomplete. Manual review required.")

    def _gen_decision(self, risk_level: Optional[str]) -> str:
        """根据风险等级生成审核决策"""
        decisions = {
            "LOW":         "APPROVE",
            "MEDIUM_LOW":  "APPROVE_WITH_REVIEW",
            "MEDIUM_HIGH": "MANUAL_REVIEW",
            "HIGH":        "REJECT",
            "CRITICAL":    "MANDATORY_REJECT",
        }
        return decisions.get(risk_level or "", "PENDING")

    def _gen_recommendations(
        self,
        risk_result: dict,
        compliance_summary: dict,
        endorsement_result: dict,
    ) -> list:
        """生成针对具体问题的改进建议列表"""
        recommendations = []

        # 合规类建议
        if compliance_summary.get("severe_count", 0) > 0:
            recommendations.append(
                f"Fix {compliance_summary['severe_count']} severe compliance violations before resubmission."
            )
        if compliance_summary.get("violation_count", 0) > 0:
            recommendations.append(
                "Review all flagged compliance fields with your compliance officer."
            )

        # 背书类建议
        if not endorsement_result.get("is_continuous", True):
            recommendations.append(
                "Endorsement chain is broken. Provide complete endorsement history."
            )
        if endorsement_result.get("has_cycle", False):
            recommendations.append(
                "Circular endorsement detected. This may indicate fraudulent activity."
            )

        # 缺失维度建议
        missing = risk_result.get("missing_dimensions", [])
        if "contract" in missing:
            recommendations.append(
                "Submit supporting contract documentation for contract review."
            )
        if "fraud" in missing:
            recommendations.append(
                "Fraud detection was not performed. Manual fraud review required."
            )

        # 无建议时的默认提示
        if not recommendations:
            recommendations.append(
                "No specific issues identified. Standard processing applies."
            )

        return recommendations
