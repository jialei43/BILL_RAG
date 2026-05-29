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

# 风险等级 → 中文业务结论
_RISK_LEVEL_ZH = {
    "LOW":         "低风险",
    "MEDIUM_LOW":  "中低风险",
    "MEDIUM_HIGH": "中高风险",
    "HIGH":        "高风险",
    "CRITICAL":    "极高风险",
    "UNKNOWN":     "风险未知",
}


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

        # 步骤 1.5：调用 LLM 生成中文业务解读（不阻断主流程）
        try:
            ai_text = await self._generate_ai_interpretation(report_json)
            if ai_text:
                report_json["ai_interpretation"] = ai_text
        except Exception as e:
            logger.warning(f"[{self.agent_name}] LLM 解读生成失败（不影响报告）: {e}")

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
                "violations":      compliance_summary.get("violations", []),
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
        """根据风险等级生成一句话中文审核结论"""
        conclusions = {
            "LOW":         "票据通过合规审核，建议直接办理。",
            "MEDIUM_LOW":  "票据基本合规，建议快速人工复核后办理。",
            "MEDIUM_HIGH": "票据存在明显问题，需要人工全面审核。",
            "HIGH":        "票据存在严重问题，建议拒绝受理。",
            "CRITICAL":    "票据存在重大违规，必须强制拒绝受理。",
        }
        return conclusions.get(risk_level or "", "风险评估未完成，需人工审核。")

    def _gen_decision(self, risk_level: Optional[str]) -> str:
        """根据风险等级生成中文审核决策"""
        decisions = {
            "LOW":         "审核通过",
            "MEDIUM_LOW":  "有条件通过",
            "MEDIUM_HIGH": "需人工审核",
            "HIGH":        "建议拒绝",
            "CRITICAL":    "强制拒绝",
        }
        return decisions.get(risk_level or "", "待定")

    def _gen_recommendations(
        self,
        risk_result: dict,
        compliance_summary: dict,
        endorsement_result: dict,
    ) -> list:
        """生成针对具体问题的中文改进建议列表"""
        recommendations = []

        # 合规类建议
        severe = compliance_summary.get("severe_count", 0)
        violation = compliance_summary.get("violation_count", 0)
        if severe > 0:
            recommendations.append(
                f"存在 {severe} 项严重合规违规，请联系出票人或承兑行补正后重新提交。"
            )
        if violation > 0:
            recommendations.append(
                "存在合规风险字段，请与合规部门核对后再推进业务流程。"
            )

        # 背书类建议
        if not endorsement_result.get("is_continuous", True):
            recommendations.append(
                "背书链不连续，持票人需补充完整的背书历史记录方可继续流转。"
            )
        if endorsement_result.get("has_cycle", False):
            recommendations.append(
                "检测到循环背书，存在欺诈嫌疑，建议暂停业务并上报风控部门。"
            )

        # 缺失维度建议
        missing = risk_result.get("missing_dimensions", [])
        if "contract" in missing:
            recommendations.append(
                "缺少合同匹配结果，请提供贸易合同原件或扫描件后重新审核。"
            )
        if "fraud" in missing:
            recommendations.append(
                "未完成欺诈检测，建议人工核查票据真实性。"
            )

        # 无建议时的默认提示
        if not recommendations:
            recommendations.append(
                "未发现明确问题，按常规流程处理即可。"
            )

        return recommendations

    async def _generate_ai_interpretation(self, report_json: dict) -> str:
        """
        调用 LLM 生成面向业务人员的中文审核解读

        输入报告的关键数据，输出自然语言段落，
        让不懂技术的业务人员也能快速理解审核结果和下一步操作。
        """
        from config.settings import settings

        if not settings.OPENAI_API_KEY:
            return ""  # 未配置 LLM，跳过

        summary    = report_json.get("summary", {})
        elements   = report_json.get("elements", {})
        compliance = report_json.get("compliance", {})
        endorse    = report_json.get("endorsement", {})
        risk       = report_json.get("risk", {})
        conclusion = report_json.get("conclusion", {})
        fraud      = report_json.get("fraud", {})

        risk_level_zh = _RISK_LEVEL_ZH.get(summary.get("risk_level", ""), "未知")

        prompt = f"""你是一位资深票据业务合规顾问。请根据以下票据自动审核结果，为业务人员生成一份通俗易懂的中文解读报告。

【审核数据摘要】
- 票据号码：{elements.get('ticket_number', '未识别')}
- 票据类型：{elements.get('ticket_type', '-')}
- 出票日期：{elements.get('issue_date', '-')}  到期日：{elements.get('due_date', '-')}
- 票面金额：{elements.get('amount_numeric', '-')}（{elements.get('amount_text', '-')}）
- 出票人：{elements.get('drawer', '-')}  承兑人：{elements.get('acceptor', '-')}
- 收款人：{elements.get('payee', '-')}  付款行：{elements.get('drawee_bank', '-')}
- 综合评分：{summary.get('composite_score', '-')} / 100
- 风险等级：{risk_level_zh}（{summary.get('risk_level', '')}）
- 合规检查：共 {compliance.get('total_fields', 0)} 项，违规 {compliance.get('violation_count', 0)} 项（严重 {compliance.get('severe_count', 0)} 项），合规率 {compliance.get('compliance_rate', 0):.1%}
- 背书链：共 {endorse.get('endorser_count', 0)} 手，连续性 {'正常' if endorse.get('is_continuous', True) else '断裂'}，循环背书 {'有' if endorse.get('has_cycle', False) else '无'}
- 欺诈风险评分：{fraud.get('overall_fraud_score', '-')}
- 合规维度得分：{risk.get('compliance_score', '-')}  背书维度：{risk.get('endorsement_score', '-')}  合同维度：{risk.get('contract_score', '-')}  欺诈维度：{risk.get('fraud_score', '-')}
- 审核决定：{conclusion.get('decision', '-')}

请按以下结构输出中文解读（每节 2-4 句话，合计不超过 400 字）：

**一、整体评价**
（综合评分和风险等级的业务含义，是否可以办理）

**二、主要问题**
（列出最关键的 1-3 个问题，说明其业务影响）

**三、操作建议**
（给出明确的下一步操作建议，业务人员能直接执行）

直接输出解读内容，不要重复审核数据，不要加开场白。"""

        try:
            import openai
            client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
            )
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"[{self.agent_name}] LLM 解读 API 调用失败: {e}")
            return ""
