# app/agents/utils/report_renderer.py
# ReportRenderer：reportlab PDF 渲染工具（中文版）
# 职责：
#   1. 接收结构化的 9 节报告 JSON（含 LLM 生成的 ai_interpretation）
#   2. 注册中文字体（优先系统 STHeiti，备选 WenQuanYi，兜底 CID STSong-Light）
#   3. 渲染全中文 PDF，包含 AI 智能解读章节
#   4. 返回 PDF 文件路径

from __future__ import annotations

import os
import tempfile
from typing import Optional

from loguru import logger

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

# ── 版面配置 ──────────────────────────────────────────────────────────────────
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT   = 20 * mm
MARGIN_RIGHT  = 20 * mm
MARGIN_TOP    = 25 * mm
MARGIN_BOTTOM = 20 * mm

# ── 风险等级颜色 ──────────────────────────────────────────────────────────────
RISK_COLORS = {
    "LOW":         colors.HexColor("#27AE60"),
    "MEDIUM_LOW":  colors.HexColor("#F39C12"),
    "MEDIUM_HIGH": colors.HexColor("#E67E22"),
    "HIGH":        colors.HexColor("#E74C3C"),
    "CRITICAL":    colors.HexColor("#8E44AD"),
}

# ── 风险等级中文名 ─────────────────────────────────────────────────────────────
RISK_LEVEL_ZH = {
    "LOW":         "低风险",
    "MEDIUM_LOW":  "中低风险",
    "MEDIUM_HIGH": "中高风险",
    "HIGH":        "高风险",
    "CRITICAL":    "极高风险",
    "UNKNOWN":     "未知",
}

# ── 字段中文名映射 ─────────────────────────────────────────────────────────────
FIELD_LABELS_ZH = {
    "ticket_number":     "票据号码",
    "ticket_type":       "票据种类",
    "issue_date":        "出票日期",
    "due_date":          "到期日期",
    "amount_numeric":    "票面金额（数字）",
    "amount_text":       "票面金额（大写）",
    "currency":          "币种",
    "drawer":            "出票人",
    "drawer_account":    "出票人账号",
    "drawer_bank":       "出票人开户行",
    "acceptor":          "承兑人",
    "payee":             "收款人",
    "drawee_bank":       "付款行",
    "endorsers":         "背书人",
    "maturity_days":     "距到期天数",
    "trade_purpose":     "贸易背景",
    "acceptance_clause": "承兑条款",
    "special_remarks":   "特殊记载事项",
    "confidence_score":  "识别置信度",
}


# ── 中文字体注册 ──────────────────────────────────────────────────────────────

_CN_FONT_NAME = "CNFont"   # 全局注册的中文字体名

def _register_chinese_font() -> str:
    """
    注册中文字体，返回字体名称。
    优先级：macOS STHeiti → Linux WenQuanYi/Noto → reportlab 内置 CID STSong-Light
    """
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Songti.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(_CN_FONT_NAME, path))
                logger.debug(f"[ReportRenderer] 注册中文字体: {path}")
                return _CN_FONT_NAME
            except Exception as e:
                logger.debug(f"[ReportRenderer] 字体注册失败 {path}: {e}")
                continue

    # 兜底：使用 reportlab 内置 CID 宋体（无需外部字体文件）
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        logger.debug("[ReportRenderer] 使用内置 CID 字体 STSong-Light")
        return "STSong-Light"
    except Exception as e:
        logger.warning(f"[ReportRenderer] 中文字体全部不可用，中文可能显示乱码: {e}")
        return "Helvetica"


def _safe(text) -> str:
    """将任意值转为 PDF 安全字符串（None → '-'，转义 XML 特殊字符）"""
    if text is None:
        return "-"
    s = str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ReportRenderer:
    """
    中文 PDF 报告渲染器：将 9 节结构化报告 JSON 渲染为 A4 中文 PDF
    支持 AI 智能解读章节，全中文标签
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.gettempdir()
        self._font = _register_chinese_font()
        self._styles = getSampleStyleSheet()
        self._init_styles()

    def _init_styles(self):
        """初始化中文段落样式"""
        fn = self._font  # 字体名简写

        self._styles.add(ParagraphStyle(
            name="CoverTitle",
            fontName=fn, fontSize=20,
            spaceAfter=6, leading=28,
            textColor=colors.HexColor("#2C3E50"),
            alignment=1,  # 居中
        ))
        self._styles.add(ParagraphStyle(
            name="CoverSub",
            fontName=fn, fontSize=11,
            spaceAfter=4, leading=16,
            textColor=colors.HexColor("#7F8C8D"),
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            name="Section",
            fontName=fn, fontSize=13,
            spaceBefore=10, spaceAfter=4, leading=20,
            textColor=colors.HexColor("#2980B9"),
        ))
        self._styles.add(ParagraphStyle(
            name="Body",
            fontName=fn, fontSize=10,
            spaceAfter=3, leading=16,
        ))
        self._styles.add(ParagraphStyle(
            name="BodyBold",
            fontName=fn, fontSize=10,
            spaceAfter=3, leading=16,
            textColor=colors.HexColor("#2C3E50"),
        ))
        self._styles.add(ParagraphStyle(
            name="Warn",
            fontName=fn, fontSize=10,
            spaceAfter=2, leading=16,
            textColor=colors.HexColor("#E74C3C"),
        ))
        self._styles.add(ParagraphStyle(
            name="AI",
            fontName=fn, fontSize=10,
            spaceAfter=4, leading=18,
            leftIndent=4,
            textColor=colors.HexColor("#1A252F"),
            backColor=colors.HexColor("#EBF5FB"),
        ))
        self._styles.add(ParagraphStyle(
            name="AIHead",
            fontName=fn, fontSize=11,
            spaceBefore=6, spaceAfter=3, leading=18,
            textColor=colors.HexColor("#1A5276"),
        ))

    # ── 主渲染方法 ─────────────────────────────────────────────────────────────

    def render(self, report_json: dict, task_id: str, training_mode: bool = False) -> str:
        """
        渲染中文 PDF 报告

        Returns:
            生成的 PDF 文件绝对路径
        """
        filename = f"audit_report_{task_id[:8]}.pdf"
        pdf_path = os.path.join(self.output_dir, filename)

        doc = BaseDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,   bottomMargin=MARGIN_BOTTOM,
        )
        frame = Frame(
            MARGIN_LEFT, MARGIN_BOTTOM,
            PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="main",
        )
        doc.addPageTemplates([PageTemplate(id="main", frames=frame)])

        story = []
        story += self._cover(report_json, task_id)
        story += self._ai_interpretation(report_json.get("ai_interpretation", ""))
        story += self._elements_section(report_json.get("elements", {}))
        story += self._compliance_section(report_json.get("compliance", {}))
        story += self._endorsement_section(report_json.get("endorsement", {}))
        story += self._contract_section(report_json.get("contract", {}))
        story += self._risk_section(report_json.get("risk", {}))
        story += self._conclusion_section(report_json.get("conclusion", {}))

        doc.build(story)
        logger.info(f"[ReportRenderer] 中文 PDF 生成完成: {pdf_path}")
        return pdf_path

    # ── 各节构建 ───────────────────────────────────────────────────────────────

    def _cover(self, report_json: dict, task_id: str) -> list:
        """封面：标题 + 关键信息表格"""
        out = [Spacer(1, 15 * mm)]

        out.append(Paragraph("票据合规智能审核报告", self._styles["CoverTitle"]))
        out.append(Paragraph("Bill Compliance Audit Report", self._styles["CoverSub"]))
        out.append(Spacer(1, 10 * mm))

        summary    = report_json.get("summary", {})
        risk_level = summary.get("risk_level", "UNKNOWN")
        risk_color = RISK_COLORS.get(risk_level, colors.gray)
        risk_zh    = RISK_LEVEL_ZH.get(risk_level, "未知")
        score      = summary.get("composite_score")
        score_str  = f"{score:.1f}" if isinstance(score, (int, float)) else _safe(score)
        elements   = report_json.get("elements", {})

        info = [
            ["任务编号",   _safe(task_id)],
            ["票据号码",   _safe(elements.get("ticket_number"))],
            ["风险等级",   f"{risk_zh}（{risk_level}）"],
            ["综合评分",   f"{score_str} / 100"],
            ["生成时间",   _safe(summary.get("generated_at", ""))[:19].replace("T", " ")],
        ]
        t = Table(info, colWidths=[45 * mm, 115 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",   (0, 0), (-1, -1), self._font),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (0, -1),  colors.HexColor("#EBF5FB")),
            ("TEXTCOLOR",  (1, 2), (1, 2),   risk_color),   # 风险等级行着色
            ("FONTNAME",   (1, 2), (1, 2),   self._font),
            ("GRID",  (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        out.append(t)
        out.append(PageBreak())
        return out

    def _ai_interpretation(self, text: str) -> list:
        """第1节：AI 智能解读（LLM 生成的中文业务解读）"""
        out = []
        out.append(Paragraph("一、智能解读", self._styles["Section"]))
        out.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor("#AED6F1"), spaceAfter=4))

        if not text:
            out.append(Paragraph(
                "（智能解读功能未启用，请配置 OPENAI_API_KEY 后重新生成报告）",
                self._styles["Body"],
            ))
            out.append(Spacer(1, 4 * mm))
            return out

        # 按行渲染，保留 **小标题** 加粗处理
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                out.append(Spacer(1, 2 * mm))
                continue
            if line.startswith("**") and line.endswith("**"):
                out.append(Paragraph(line[2:-2], self._styles["AIHead"]))
            else:
                # 行内 **bold** 转为 <b> 标签
                line = self._md_bold(line)
                out.append(Paragraph(line, self._styles["AI"]))

        out.append(Spacer(1, 5 * mm))
        return out

    def _elements_section(self, elem: dict) -> list:
        """第2节：票据要素"""
        out = [Paragraph("二、票据要素", self._styles["Section"]),
               HRFlowable(width="100%", thickness=1,
                          color=colors.HexColor("#AED6F1"), spaceAfter=4)]

        fields = [
            "ticket_number", "ticket_type", "issue_date", "due_date",
            "amount_numeric", "amount_text", "currency",
            "drawer", "drawer_account", "drawer_bank",
            "acceptor", "payee", "drawee_bank",
            "trade_purpose", "maturity_days", "confidence_score",
        ]

        data = [["字段", "内容"]]
        for f in fields:
            val = elem.get(f)
            if f == "confidence_score" and isinstance(val, (int, float)):
                val = f"{val:.1%}"
            elif f == "maturity_days" and isinstance(val, int):
                val = f"{val} 天" if val > 0 else f"已逾期 {abs(val)} 天"
            data.append([FIELD_LABELS_ZH.get(f, f), _safe(val)])

        t = Table(data, colWidths=[55 * mm, 105 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (-1, -1), self._font),
            ("FONTSIZE",  (0, 0), (-1, -1), 9),
            ("BACKGROUND",(0, 0), (-1, 0),  colors.HexColor("#2980B9")),
            ("TEXTCOLOR", (0, 0), (-1, 0),  colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F8F9FA")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        out.append(t)
        out.append(Spacer(1, 5 * mm))
        return out

    def _compliance_section(self, comp: dict) -> list:
        """第3节：合规检查 —— 统计概要 + 违规明细表格"""
        out = [Paragraph("三、合规检查", self._styles["Section"]),
               HRFlowable(width="100%", thickness=1,
                          color=colors.HexColor("#AED6F1"), spaceAfter=4)]

        total    = comp.get("total_fields", 0)
        viols    = comp.get("violation_count", 0)
        severe   = comp.get("severe_count", 0)
        rate     = comp.get("compliance_rate", 0)
        rate_str = f"{rate:.1%}" if isinstance(rate, (int, float)) else _safe(rate)
        is_ok    = comp.get("is_compliant", True)

        summary_color = colors.HexColor("#27AE60") if is_ok else colors.HexColor("#E74C3C")
        status_text   = "整体合规" if is_ok else f"存在违规（严重 {severe} 项）"

        # ── 统计概要 ────────────────────────────────────────────────────────────
        stats = [
            ["检查字段数", str(total), "合规率", rate_str],
            ["违规总数",   str(viols), "整体状态", status_text],
        ]
        t = Table(stats, colWidths=[40 * mm, 35 * mm, 40 * mm, 45 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (-1, -1), self._font),
            ("FONTSIZE",  (0, 0), (-1, -1), 10),
            ("BACKGROUND",(0, 0), (0, -1),  colors.HexColor("#EBF5FB")),
            ("BACKGROUND",(2, 0), (2, -1),  colors.HexColor("#EBF5FB")),
            ("TEXTCOLOR", (3, 1), (3, 1),   summary_color),
            ("FONTNAME",  (3, 1), (3, 1),   self._font),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        out.append(t)

        # ── 违规明细表格 ────────────────────────────────────────────────────────
        violations = comp.get("violations", [])
        if violations:
            out.append(Spacer(1, 4 * mm))
            out.append(Paragraph("违规明细", self._styles["BodyBold"]))
            out.append(Spacer(1, 1 * mm))

            _level_zh = {"severe": "严重", "warning": "警告", "info": "提示"}
            _level_bg = {
                "severe":  colors.HexColor("#FADBD8"),   # 红色背景
                "warning": colors.HexColor("#FDEBD0"),   # 橙色背景
                "info":    colors.HexColor("#FDFEFE"),   # 默认白
            }
            _level_fg = {
                "severe":  colors.HexColor("#C0392B"),
                "warning": colors.HexColor("#D35400"),
                "info":    colors.HexColor("#7F8C8D"),
            }

            header = ["序号", "字段名称", "违规等级", "违规描述", "法规依据"]
            rows   = [header]
            level_per_row = []   # 记录每行的 level，用于后续动态着色

            fn = self._font
            for idx, v in enumerate(violations[:20], 1):
                field    = v.get("field", "")
                field_zh = FIELD_LABELS_ZH.get(field, field)
                level    = (v.get("violation_level") or "warning").lower()
                desc     = _safe(v.get("violation_desc") or f"「{field_zh}」合规检查不通过")
                ref      = _safe(v.get("regulation_ref") or "-")
                level_zh = _level_zh.get(level, level)

                rows.append([
                    str(idx),
                    Paragraph(field_zh, ParagraphStyle("cv", fontName=fn, fontSize=9, leading=13)),
                    Paragraph(level_zh, ParagraphStyle(
                        "cl", fontName=fn, fontSize=9, leading=13,
                        textColor=_level_fg.get(level, colors.black),
                    )),
                    Paragraph(desc, ParagraphStyle("cd", fontName=fn, fontSize=9, leading=13)),
                    Paragraph(ref,  ParagraphStyle("cr", fontName=fn, fontSize=9, leading=13,
                                                   textColor=colors.HexColor("#5D6D7E"))),
                ])
                level_per_row.append(level)

            col_w = [10 * mm, 28 * mm, 18 * mm, 72 * mm, 32 * mm]
            vt = Table(rows, colWidths=col_w, repeatRows=1)

            # 基础样式
            ts = [
                ("FONTNAME",  (0, 0), (-1, -1), self._font),
                ("FONTSIZE",  (0, 0), (-1, 0),  10),
                ("BACKGROUND",(0, 0), (-1, 0),  colors.HexColor("#2980B9")),
                ("TEXTCOLOR", (0, 0), (-1, 0),  colors.white),
                ("ALIGN",     (0, 0), (0, -1),  "CENTER"),
                ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
                ("GRID",      (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
            # 按违规等级动态着色每行背景
            for i, level in enumerate(level_per_row, 1):
                bg = _level_bg.get(level, colors.white)
                ts.append(("BACKGROUND", (0, i), (-1, i), bg))

            vt.setStyle(TableStyle(ts))
            out.append(vt)

        elif viols == 0:
            out.append(Spacer(1, 3 * mm))
            out.append(Paragraph(
                "✔ 所有字段合规检查通过，未发现违规问题。",
                self._styles["Body"],
            ))

        out.append(Spacer(1, 5 * mm))
        return out

    def _endorsement_section(self, endorse: dict) -> list:
        """第4节：背书链分析"""
        out = [Paragraph("四、背书链分析", self._styles["Section"]),
               HRFlowable(width="100%", thickness=1,
                          color=colors.HexColor("#AED6F1"), spaceAfter=4)]

        cnt        = endorse.get("endorser_count", 0)
        continuous = endorse.get("is_continuous", True)
        has_cycle  = endorse.get("has_cycle", False)
        v_count    = endorse.get("violation_count", 0)
        codes      = endorse.get("violation_codes", [])

        status_ok = continuous and not has_cycle and v_count == 0
        status_txt = "背书链完整无异常" if status_ok else "背书链存在异常"
        status_color = colors.HexColor("#27AE60") if status_ok else colors.HexColor("#E74C3C")

        data = [
            ["背书手数", str(cnt), "链路连续", "是" if continuous else "否"],
            ["循环背书", "有" if has_cycle else "无", "违规数量", str(v_count)],
        ]
        t = Table(data, colWidths=[40 * mm, 35 * mm, 40 * mm, 45 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (-1, -1), self._font),
            ("FONTSIZE",  (0, 0), (-1, -1), 10),
            ("BACKGROUND",(0, 0), (0, -1),  colors.HexColor("#EBF5FB")),
            ("BACKGROUND",(2, 0), (2, -1),  colors.HexColor("#EBF5FB")),
            ("TEXTCOLOR", (1, 0), (1, 0),   colors.HexColor("#2C3E50")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        out.append(t)

        out.append(Spacer(1, 3 * mm))
        style = "Body" if status_ok else "Warn"
        out.append(Paragraph(f"综合评价：{status_txt}", self._styles[style]))

        if codes:
            out.append(Paragraph(
                f"违规代码：{', '.join(codes)}",
                self._styles["Warn"],
            ))

        out.append(Spacer(1, 5 * mm))
        return out

    def _contract_section(self, contract: dict) -> list:
        """第5节：合同审核"""
        out = [Paragraph("五、合同审核", self._styles["Section"]),
               HRFlowable(width="100%", thickness=1,
                          color=colors.HexColor("#AED6F1"), spaceAfter=4)]

        score = contract.get("match_score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else _safe(score)
        amount_ok = contract.get("amount_match")
        party_ok  = contract.get("party_match")

        def yn(v):
            if v is None: return "-"
            return "匹配" if v else "不匹配"

        data = [
            ["合同匹配得分", score_str, "金额匹配", yn(amount_ok)],
            ["交易主体匹配", yn(party_ok), "不符项数",
             _safe(contract.get("mismatch_count"))],
        ]
        t = Table(data, colWidths=[50 * mm, 30 * mm, 50 * mm, 30 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (-1, -1), self._font),
            ("FONTSIZE",  (0, 0), (-1, -1), 10),
            ("BACKGROUND",(0, 0), (0, -1),  colors.HexColor("#EBF5FB")),
            ("BACKGROUND",(2, 0), (2, -1),  colors.HexColor("#EBF5FB")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        out.append(t)

        if contract.get("review_notes"):
            out.append(Spacer(1, 3 * mm))
            out.append(Paragraph(
                f"审核备注：{_safe(contract['review_notes'])}",
                self._styles["Body"],
            ))

        out.append(Spacer(1, 5 * mm))
        return out

    def _risk_section(self, risk: dict) -> list:
        """第6节：风险评估"""
        out = [Paragraph("六、风险评估", self._styles["Section"]),
               HRFlowable(width="100%", thickness=1,
                          color=colors.HexColor("#AED6F1"), spaceAfter=4)]

        risk_level = risk.get("risk_level", "UNKNOWN")
        risk_color = RISK_COLORS.get(risk_level, colors.gray)
        risk_zh    = RISK_LEVEL_ZH.get(risk_level, "未知")
        score      = risk.get("composite_score")
        score_str  = f"{score:.1f}" if isinstance(score, (int, float)) else _safe(score)

        out.append(Paragraph(
            f"综合评分：<b>{score_str} / 100</b>　　风险等级：<b>{risk_zh}</b>",
            self._styles["BodyBold"],
        ))
        out.append(Spacer(1, 3 * mm))

        def fmt_score(v):
            if isinstance(v, (int, float)):
                return f"{v:.1f}"
            return _safe(v)

        dim_data = [
            ["评估维度", "得分", "权重", "说明"],
            ["合规检查", fmt_score(risk.get("compliance_score")),  "30%", "票据字段合规性"],
            ["背书链路", fmt_score(risk.get("endorsement_score")), "30%", "背书连续性与风险"],
            ["合同匹配", fmt_score(risk.get("contract_score")),    "20%", "贸易背景真实性"],
            ["欺诈检测", fmt_score(risk.get("fraud_score")),       "20%", "票据真实性与防伪"],
        ]
        t = Table(dim_data, colWidths=[45 * mm, 30 * mm, 25 * mm, 60 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",   (0, 0), (-1, -1), self._font),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0),  colors.HexColor("#2980B9")),
            ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F8F9FA")]),
            ("ALIGN",  (1, 0), (2, -1), "CENTER"),
            ("GRID",  (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        out.append(t)

        missing = risk.get("missing_dimensions", [])
        if missing:
            out.append(Spacer(1, 3 * mm))
            missing_zh = {"contract": "合同审核", "fraud": "欺诈检测",
                          "endorsement": "背书分析", "compliance": "合规检查"}
            labels = "、".join(missing_zh.get(m, m) for m in missing)
            out.append(Paragraph(
                f"⚠ 以下维度数据缺失，评分已按默认值计算：{labels}",
                self._styles["Warn"],
            ))

        out.append(Spacer(1, 5 * mm))
        return out

    def _conclusion_section(self, conclusion: dict) -> list:
        """第7节：审核结论"""
        out = [Paragraph("七、审核结论与建议", self._styles["Section"]),
               HRFlowable(width="100%", thickness=1,
                          color=colors.HexColor("#AED6F1"), spaceAfter=4)]

        decision = conclusion.get("decision", "待定")
        score    = conclusion.get("composite_score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else _safe(score)

        decision_color = {
            "审核通过":   colors.HexColor("#27AE60"),
            "有条件通过": colors.HexColor("#F39C12"),
            "需人工审核": colors.HexColor("#E67E22"),
            "建议拒绝":   colors.HexColor("#E74C3C"),
            "强制拒绝":   colors.HexColor("#8E44AD"),
        }.get(decision, colors.HexColor("#7F8C8D"))

        out.append(Paragraph(
            f"审核决定：<b>{_safe(decision)}</b>　　综合评分：{score_str} / 100",
            self._styles["BodyBold"],
        ))
        out.append(Spacer(1, 4 * mm))

        recs = conclusion.get("recommendations", [])
        if recs:
            out.append(Paragraph("操作建议：", self._styles["BodyBold"]))
            for i, rec in enumerate(recs, 1):
                out.append(Paragraph(
                    f"  {i}. {_safe(rec)}",
                    self._styles["Body"],
                ))

        out.append(Spacer(1, 5 * mm))
        return out

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _md_bold(text: str) -> str:
        """将 Markdown **bold** 转换为 reportlab <b> 标签"""
        import re
        return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
