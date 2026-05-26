# app/agents/utils/report_renderer.py
# ReportRenderer：reportlab PDF 渲染工具
# 职责：
#   1. 接收结构化的 9 节报告 JSON
#   2. 使用 reportlab 渲染为 PDF 文件
#   3. 返回 PDF 文件路径和文件大小
# 注意：使用内置 Helvetica 字体（避免中文字体依赖），中文内容通过 latin-1 编码兼容

from __future__ import annotations

import os
import tempfile
from typing import Optional

from loguru import logger

# reportlab 核心模块
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── 版面配置 ──────────────────────────────────────────────────────────────────
PAGE_WIDTH, PAGE_HEIGHT = A4   # A4 尺寸：210mm × 297mm
MARGIN_LEFT   = 20 * mm        # 左边距
MARGIN_RIGHT  = 20 * mm        # 右边距
MARGIN_TOP    = 25 * mm        # 上边距
MARGIN_BOTTOM = 20 * mm        # 下边距

# ── 风险等级颜色映射 ─────────────────────────────────────────────────────────
RISK_LEVEL_COLORS = {
    "LOW":         colors.HexColor("#27AE60"),   # 绿色
    "MEDIUM_LOW":  colors.HexColor("#F39C12"),   # 橙色
    "MEDIUM_HIGH": colors.HexColor("#E67E22"),   # 深橙
    "HIGH":        colors.HexColor("#E74C3C"),   # 红色
    "CRITICAL":    colors.HexColor("#8E44AD"),   # 紫色（极高危）
}


def _safe_str(text) -> str:
    """
    将任意值转换为 PDF 安全字符串
    reportlab Paragraph 需要合法字符串，None 值和特殊字符需处理
    """
    if text is None:
        return "-"
    return str(text).replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")


class ReportRenderer:
    """
    PDF 报告渲染器：将 9 节结构化报告 JSON 渲染为 A4 PDF 文件
    使用 reportlab 的 Platypus 高层 API，支持自动分页
    """

    def __init__(self, output_dir: Optional[str] = None):
        # 输出目录：未指定时使用系统临时目录
        self.output_dir = output_dir or tempfile.gettempdir()
        # 初始化段落样式
        self._styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        """初始化自定义段落样式（标题/正文/表格等）"""
        # 报告主标题样式
        self._styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self._styles["Title"],
            fontSize=18,
            spaceAfter=6,
            textColor=colors.HexColor("#2C3E50"),
        ))
        # 节标题样式（如「1. 审核摘要」）
        self._styles.add(ParagraphStyle(
            name="SectionTitle",
            parent=self._styles["Heading1"],
            fontSize=13,
            spaceBefore=12,
            spaceAfter=4,
            textColor=colors.HexColor("#2980B9"),
        ))
        # 普通正文样式
        self._styles.add(ParagraphStyle(
            name="BodyCN",
            parent=self._styles["Normal"],
            fontSize=10,
            spaceAfter=3,
            leading=14,   # 行距
        ))
        # 警告文本样式（红色）
        self._styles.add(ParagraphStyle(
            name="Warning",
            parent=self._styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#E74C3C"),
        ))
        # 培训注解样式（蓝色斜体）
        self._styles.add(ParagraphStyle(
            name="TrainingNote",
            parent=self._styles["Italic"],
            fontSize=9,
            textColor=colors.HexColor("#2980B9"),
            leftIndent=10,
            spaceAfter=2,
        ))

    def render(self, report_json: dict, task_id: str, training_mode: bool = False) -> str:
        """
        渲染 PDF 报告主方法

        Args:
            report_json:   9 节结构化报告 JSON
            task_id:       审核任务 ID（用于文件命名）
            training_mode: 是否附加培训注解层

        Returns:
            pdf_path: 生成的 PDF 文件绝对路径
        """
        # 确定输出文件路径
        filename = f"audit_report_{task_id[:8]}.pdf"   # 取任务 ID 前8位防止文件名过长
        pdf_path = os.path.join(self.output_dir, filename)

        # 构建 PDF 文档（A4 + 边距）
        doc = BaseDocTemplate(
            pdf_path,
            pagesize=A4,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        # 定义内容区域 Frame
        content_frame = Frame(
            MARGIN_LEFT, MARGIN_BOTTOM,
            PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="content",
        )
        doc.addPageTemplates([PageTemplate(id="main", frames=content_frame)])

        # 构建报告内容（Story = reportlab 内容元素列表）
        story = []
        story.extend(self._build_cover(report_json, task_id))
        story.extend(self._build_summary(report_json.get("summary", {})))
        story.extend(self._build_elements(report_json.get("elements", {})))
        story.extend(self._build_compliance(report_json.get("compliance", {})))
        story.extend(self._build_endorsement(report_json.get("endorsement", {})))
        story.extend(self._build_contract(report_json.get("contract", {})))
        story.extend(self._build_risk(report_json.get("risk", {})))
        story.extend(self._build_conclusion(report_json.get("conclusion", {})))

        # 培训模式附加注解
        if training_mode:
            story.extend(self._build_training_notes(report_json))

        # 生成 PDF
        doc.build(story)
        logger.info(f"[ReportRenderer] PDF 生成完成: {pdf_path}")
        return pdf_path

    # ── 各节内容构建方法 ───────────────────────────────────────────────────────

    def _build_cover(self, report_json: dict, task_id: str) -> list:
        """构建封面（标题 + 任务信息）"""
        elements = []
        elements.append(Spacer(1, 20 * mm))   # 顶部留白

        # 主标题
        elements.append(Paragraph(
            "Bill Compliance Audit Report",
            self._styles["ReportTitle"],
        ))
        elements.append(Paragraph(
            "票据合规智能审核报告",
            self._styles["ReportTitle"],
        ))
        elements.append(Spacer(1, 8 * mm))

        # 任务信息表格
        summary = report_json.get("summary", {})
        risk_level = summary.get("risk_level", "UNKNOWN")
        risk_color = RISK_LEVEL_COLORS.get(risk_level, colors.gray)

        info_data = [
            ["Task ID", _safe_str(task_id)],
            ["Risk Level", _safe_str(risk_level)],
            ["Composite Score", _safe_str(summary.get("composite_score", "-"))],
            ["Generated At", _safe_str(summary.get("generated_at", "-"))],
        ]
        info_table = Table(info_data, colWidths=[50 * mm, 100 * mm])
        info_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECF0F1")),
            ("TEXTCOLOR", (1, 1), (1, 1), risk_color),   # 风险等级着色
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(info_table)
        elements.append(PageBreak())   # 封面后分页
        return elements

    def _build_summary(self, summary: dict) -> list:
        """构建第1节：审核摘要"""
        elements = []
        elements.append(Paragraph("1. Audit Summary / 审核摘要", self._styles["SectionTitle"]))
        elements.append(Paragraph(
            f"Composite Score: {_safe_str(summary.get('composite_score'))}  |  "
            f"Risk Level: {_safe_str(summary.get('risk_level'))}",
            self._styles["BodyCN"],
        ))
        if summary.get("conclusion"):
            elements.append(Paragraph(
                f"Conclusion: {_safe_str(summary.get('conclusion'))}",
                self._styles["BodyCN"],
            ))
        elements.append(Spacer(1, 4 * mm))
        return elements

    def _build_elements(self, elements_data: dict) -> list:
        """构建第2节：票据要素（18字段表格）"""
        result = []
        result.append(Paragraph("2. Bill Elements / 票据要素", self._styles["SectionTitle"]))

        # 要素字段中英文映射
        field_labels = [
            ("ticket_number",    "Ticket Number"),
            ("ticket_type",      "Ticket Type"),
            ("issue_date",       "Issue Date"),
            ("due_date",         "Due Date"),
            ("amount_numeric",   "Amount (Numeric)"),
            ("amount_text",      "Amount (Text)"),
            ("currency",         "Currency"),
            ("drawer",           "Drawer"),
            ("acceptor",         "Acceptor"),
            ("payee",            "Payee"),
            ("drawee_bank",      "Drawee Bank"),
            ("trade_purpose",    "Trade Purpose"),
            ("confidence_score", "Confidence Score"),
        ]

        table_data = [["Field", "Value"]]   # 表头
        for field, label in field_labels:
            value = elements_data.get(field)
            table_data.append([label, _safe_str(value)])

        table = Table(table_data, colWidths=[60 * mm, 100 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980B9")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
            ("GRID",  (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        result.append(table)
        result.append(Spacer(1, 4 * mm))
        return result

    def _build_compliance(self, compliance: dict) -> list:
        """构建第3节：合规检查结果"""
        result = []
        result.append(Paragraph("3. Compliance Check / 合规检查", self._styles["SectionTitle"]))
        result.append(Paragraph(
            f"Compliance Rate: {_safe_str(compliance.get('compliance_rate'))}  |  "
            f"Violations: {_safe_str(compliance.get('violation_count'))}  |  "
            f"Severe: {_safe_str(compliance.get('severe_count'))}",
            self._styles["BodyCN"],
        ))

        # 违规详情（若有）
        violations = compliance.get("violations", [])
        if violations:
            result.append(Paragraph("Violations Found:", self._styles["BodyCN"]))
            for v in violations[:10]:   # 最多显示10条
                result.append(Paragraph(
                    f"  - [{v.get('field')}] {_safe_str(v.get('desc'))}",
                    self._styles["Warning"],
                ))

        result.append(Spacer(1, 4 * mm))
        return result

    def _build_endorsement(self, endorsement: dict) -> list:
        """构建第4节：背书链分析"""
        result = []
        result.append(Paragraph("4. Endorsement Chain / 背书链分析", self._styles["SectionTitle"]))
        result.append(Paragraph(
            f"Endorsers: {_safe_str(endorsement.get('endorser_count'))}  |  "
            f"Continuous: {_safe_str(endorsement.get('is_continuous'))}  |  "
            f"Has Cycle: {_safe_str(endorsement.get('has_cycle'))}  |  "
            f"Violations: {_safe_str(endorsement.get('violation_count'))}",
            self._styles["BodyCN"],
        ))

        violation_codes = endorsement.get("violation_codes", [])
        if violation_codes:
            result.append(Paragraph(
                f"Violation Codes: {', '.join(violation_codes)}",
                self._styles["Warning"],
            ))

        result.append(Spacer(1, 4 * mm))
        return result

    def _build_contract(self, contract: dict) -> list:
        """构建第5节：合同审核"""
        result = []
        result.append(Paragraph("5. Contract Review / 合同审核", self._styles["SectionTitle"]))
        result.append(Paragraph(
            f"Match Score: {_safe_str(contract.get('match_score'))}  |  "
            f"Amount Match: {_safe_str(contract.get('amount_match'))}  |  "
            f"Party Match: {_safe_str(contract.get('party_match'))}",
            self._styles["BodyCN"],
        ))
        if contract.get("review_notes"):
            result.append(Paragraph(
                f"Notes: {_safe_str(contract.get('review_notes'))}",
                self._styles["BodyCN"],
            ))
        result.append(Spacer(1, 4 * mm))
        return result

    def _build_risk(self, risk: dict) -> list:
        """构建第6节：风险评估（四维得分雷达）"""
        result = []
        result.append(Paragraph("6. Risk Assessment / 风险评估", self._styles["SectionTitle"]))

        risk_level = risk.get("risk_level", "UNKNOWN")
        risk_color = RISK_LEVEL_COLORS.get(risk_level, colors.gray)

        # 综合得分和风险等级（突出显示）
        result.append(Paragraph(
            f"<b>Composite Score: {_safe_str(risk.get('composite_score'))}</b>  |  "
            f"<b>Risk Level: {_safe_str(risk_level)}</b>",
            self._styles["BodyCN"],
        ))

        # 四维得分表格
        score_data = [
            ["Dimension", "Score", "Weight"],
            ["Compliance",  _safe_str(risk.get("compliance_score")),  "30%"],
            ["Endorsement", _safe_str(risk.get("endorsement_score")), "30%"],
            ["Contract",    _safe_str(risk.get("contract_score")),    "20%"],
            ["Fraud",       _safe_str(risk.get("fraud_score")),       "20%"],
        ]
        score_table = Table(score_data, colWidths=[50 * mm, 40 * mm, 30 * mm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980B9")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
            ("GRID",  (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        result.append(score_table)

        if risk.get("assessor_notes"):
            result.append(Spacer(1, 3 * mm))
            result.append(Paragraph(
                f"Notes: {_safe_str(risk.get('assessor_notes'))}",
                self._styles["BodyCN"],
            ))

        result.append(Spacer(1, 4 * mm))
        return result

    def _build_conclusion(self, conclusion: dict) -> list:
        """构建第7节：审核结论"""
        result = []
        result.append(Paragraph("7. Conclusion / 审核结论", self._styles["SectionTitle"]))
        result.append(Paragraph(
            f"Decision: {_safe_str(conclusion.get('decision'))}",
            self._styles["BodyCN"],
        ))
        if conclusion.get("recommendations"):
            result.append(Paragraph("Recommendations:", self._styles["BodyCN"]))
            for rec in conclusion.get("recommendations", []):
                result.append(Paragraph(f"  - {_safe_str(rec)}", self._styles["BodyCN"]))
        result.append(Spacer(1, 4 * mm))
        return result

    def _build_training_notes(self, report_json: dict) -> list:
        """构建培训注解层（仅培训模式附加）"""
        result = []
        result.append(PageBreak())
        result.append(Paragraph("Training Notes / 培训注解", self._styles["SectionTitle"]))
        result.append(Paragraph(
            "The following section provides explanations for training purposes.",
            self._styles["TrainingNote"],
        ))
        result.append(Paragraph(
            "Risk scoring formula: compliance(30%) + endorsement(30%) + contract(20%) + fraud(20%)",
            self._styles["TrainingNote"],
        ))
        result.append(Paragraph(
            "Risk levels: LOW(>=85) / MEDIUM_LOW(70-84) / MEDIUM_HIGH(50-69) / HIGH(30-49) / CRITICAL(<30)",
            self._styles["TrainingNote"],
        ))
        return result
