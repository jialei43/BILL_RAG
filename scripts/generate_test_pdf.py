"""
生成复杂票据测试PDF文件，包含多种表格、图表和图片，用于验证RAG解析效果。
"""
import io
import os
import random
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF

matplotlib.use("Agg")  # 无GUI后端

# 配置 matplotlib 中文字体（优先 STHeiti，回退 DejaVu）
_MPL_CN_FONT = "DejaVu Sans"
for _fp, _name in [
    ("/System/Library/Fonts/STHeiti Light.ttc", "STHeiti"),
    ("/Library/Fonts/Arial Unicode.ttf", "Arial Unicode MS"),
]:
    if os.path.exists(_fp):
        try:
            import matplotlib.font_manager as _fm
            _fm.fontManager.addfont(_fp)
            _MPL_CN_FONT = _name
            break
        except Exception:
            pass
matplotlib.rcParams["font.family"] = [_MPL_CN_FONT, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 中文字体注册 ────────────────────────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

CN_FONT = "Helvetica"  # 回退值
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("ChineseFont", fp))
            CN_FONT = "ChineseFont"
            break
        except Exception:
            continue

BOLD_FONT = CN_FONT  # reportlab TTFont 不带 Bold 变体，统一用同一字体

# ── 调色板 ─────────────────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor("#1a3a5c")   # 深蓝
SECONDARY = colors.HexColor("#2e86c1")   # 中蓝
ACCENT    = colors.HexColor("#e74c3c")   # 红
LIGHT_BG  = colors.HexColor("#eaf4fb")   # 浅蓝背景
GOLD      = colors.HexColor("#d4ac0d")   # 金色
GRAY      = colors.HexColor("#7f8c8d")
WHITE     = colors.white
GREEN     = colors.HexColor("#27ae60")

# ── 样式 ───────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, font=CN_FONT, size=10, color=colors.black,
               align=TA_LEFT, bold=False, leading=None):
    """工厂：创建 ParagraphStyle，如已存在则直接返回。"""  # 避免重复注册
    if name in styles:
        return styles[name]
    s = ParagraphStyle(
        name,
        fontName=font,
        fontSize=size,
        textColor=color,
        alignment=align,
        leading=leading or size * 1.4,
        spaceAfter=2,
    )
    styles.add(s)
    return s

s_title      = make_style("DocTitle",    size=22, color=PRIMARY,    align=TA_CENTER)
s_subtitle   = make_style("DocSubtitle", size=13, color=SECONDARY,  align=TA_CENTER)
s_section    = make_style("Section",     size=13, color=PRIMARY)
s_body       = make_style("Body",        size=9)
s_small      = make_style("Small",       size=8,  color=GRAY)
s_red_center = make_style("RedCenter",   size=9,  color=ACCENT,     align=TA_CENTER)
s_right      = make_style("Right",       size=9,  align=TA_RIGHT)
s_center     = make_style("Center",      size=9,  align=TA_CENTER)
s_bold       = make_style("Bold",        size=10, color=PRIMARY)
s_watermark  = make_style("Watermark",   size=40, color=colors.HexColor("#d5e8f5"), align=TA_CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def p(text, style=None):
    """快捷生成 Paragraph。"""  # 默认 Body 样式
    return Paragraph(str(text), style or s_body)


def hr(color=SECONDARY, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4, spaceBefore=4)


def spacer(h=6):
    return Spacer(1, h * mm)


def _tbl_style(header_color=PRIMARY, row_alt=LIGHT_BG):
    """通用表格样式。"""  # 含斑马纹
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, -1), CN_FONT),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, row_alt]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#b3cde0")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 图表生成（matplotlib → PIL Image → ReportLab Image）
# ══════════════════════════════════════════════════════════════════════════════

def _fig_to_rl_image(fig, width_mm=160, height_mm=80):
    """将 matplotlib Figure 转为 ReportLab Image flowable。"""  # 内存传递，不写磁盘
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_mm * mm, height=height_mm * mm)


def make_monthly_revenue_chart():
    """月度营收柱状图（含同比折线）。"""  # 双轴图表
    months = ["1月", "2月", "3月", "4月", "5月", "6月",
              "7月", "8月", "9月", "10月", "11月", "12月"]
    revenue_cur = [820, 650, 910, 1050, 980, 1200, 1380, 1250, 1450, 1600, 1520, 1800]
    revenue_prev = [700, 580, 820, 900, 850, 1050, 1180, 1100, 1280, 1400, 1350, 1620]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(len(months))
    w = 0.35
    bars1 = ax1.bar(x - w/2, revenue_cur,  w, label="本年度", color="#2e86c1", alpha=0.85)
    bars2 = ax1.bar(x + w/2, revenue_prev, w, label="上年度", color="#85c1e9", alpha=0.85)

    ax2 = ax1.twinx()
    yoy = [(c - p) / p * 100 for c, p in zip(revenue_cur, revenue_prev)]
    ax2.plot(x, yoy, "o-", color="#e74c3c", linewidth=2, markersize=5, label="同比增长率")
    ax2.set_ylabel("同比增长率 (%)", color="#e74c3c", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#e74c3c", labelsize=7)
    ax2.yaxis.set_tick_params(labelsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(months, fontsize=7)
    ax1.set_ylabel("金额（万元）", fontsize=8)
    ax1.set_title("2024年度月度营收对比分析", fontsize=10, fontweight="bold", pad=8)
    ax1.tick_params(labelsize=7)

    lines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    handles = lines_labels[0][0] + lines_labels[1][0]
    labels  = lines_labels[0][1] + lines_labels[1][1]
    ax1.legend(handles, labels, loc="upper left", fontsize=7)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _fig_to_rl_image(fig, 160, 75)


def make_category_pie_chart():
    """票据类别占比饼图。"""  # 分离式饼图
    labels = ["增值税专用发票", "普通发票", "电子发票", "运输发票", "医疗票据", "其他"]
    sizes  = [35, 25, 20, 10, 7, 3]
    explode = [0.05, 0, 0.05, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels,
        autopct="%1.1f%%", startangle=140,
        colors=["#2e86c1","#85c1e9","#27ae60","#f39c12","#e74c3c","#9b59b6"],
        textprops={"fontsize": 7},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title("票据类别分布（2024年度）", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _fig_to_rl_image(fig, 75, 65)


def make_quarterly_trend_chart():
    """季度账期趋势折线图。"""  # 多系列折线
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    series = {
        "应收账款": [320, 450, 510, 680],
        "已核销":   [280, 390, 470, 590],
        "逾期未收": [40,  60,  40,  90],
    }
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors_list = ["#2e86c1", "#27ae60", "#e74c3c"]
    markers = ["o", "s", "^"]
    for (label, vals), c, m in zip(series.items(), colors_list, markers):
        ax.plot(quarters, vals, marker=m, color=c, linewidth=2,
                markersize=6, label=label)
        for q, v in zip(quarters, vals):
            ax.annotate(str(v), (q, v), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_title("季度账款趋势（万元）", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_ylim(0, 800)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_rl_image(fig, 80, 65)


# ══════════════════════════════════════════════════════════════════════════════
# 页面模板（页眉 / 页脚）
# ══════════════════════════════════════════════════════════════════════════════

class BillDocTemplate(BaseDocTemplate):
    """自定义文档模板，带页眉/页脚。"""  # 继承 BaseDocTemplate 以支持双页模板

    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, **kw)
        W, H = A4
        margin = 18 * mm

        # 内容帧（为页眉/页脚留空）
        content_frame = Frame(
            margin, margin + 12 * mm,
            W - 2 * margin, H - 2 * margin - 25 * mm,
            id="content",
        )
        # 首页模板（无页眉）
        cover_frame = Frame(
            margin, margin + 12 * mm,
            W - 2 * margin, H - 2 * margin - 15 * mm,
            id="cover",
        )
        self.addPageTemplates([
            PageTemplate(id="Cover",   frames=[cover_frame],   onPage=self._cover_page),
            PageTemplate(id="Content", frames=[content_frame], onPage=self._content_page),
        ])

    @staticmethod
    def _cover_page(canvas, doc):
        W, H = A4
        # 顶部色块
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, H - 28 * mm, W, 28 * mm, fill=True, stroke=False)
        # 底部色块
        canvas.setFillColor(SECONDARY)
        canvas.rect(0, 0, W, 12 * mm, fill=True, stroke=False)
        canvas.setFillColor(WHITE)
        canvas.setFont(CN_FONT, 8)
        canvas.drawCentredString(W / 2, 4 * mm,
            "机密文件 · 仅供内部使用 · 华夏金融集团有限公司  ©2024")

    @staticmethod
    def _content_page(canvas, doc):
        W, H = A4
        # 页眉
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, H - 16 * mm, W, 16 * mm, fill=True, stroke=False)
        canvas.setFillColor(WHITE)
        canvas.setFont(CN_FONT, 8)
        canvas.drawString(18 * mm, H - 10 * mm, "华夏金融集团 · 票据业务综合报告")
        canvas.drawRightString(W - 18 * mm, H - 10 * mm, "CONFIDENTIAL  2024-ANNUAL")
        # 页眉下方金色线
        canvas.setStrokeColor(GOLD)
        canvas.setLineWidth(1.5)
        canvas.line(0, H - 16 * mm, W, H - 16 * mm)
        # 页脚
        canvas.setFillColor(SECONDARY)
        canvas.rect(0, 0, W, 10 * mm, fill=True, stroke=False)
        canvas.setFillColor(WHITE)
        canvas.setFont(CN_FONT, 7.5)
        canvas.drawCentredString(W / 2, 3.5 * mm,
            f"第 {doc.page} 页  |  华夏金融集团有限公司票据业务部  |  {datetime.now().strftime('%Y年%m月%d日')}")
        # 侧边装饰条
        canvas.setFillColor(LIGHT_BG)
        canvas.rect(0, 10 * mm, 5 * mm, H - 26 * mm, fill=True, stroke=False)


# ══════════════════════════════════════════════════════════════════════════════
# 文档内容构建
# ══════════════════════════════════════════════════════════════════════════════

def build_cover(story):
    """封面页内容。"""  # 大标题 + 摘要信息
    story.append(spacer(30))
    story.append(p("华夏金融集团有限公司", make_style("CoverCo", size=14, color=SECONDARY, align=TA_CENTER)))
    story.append(spacer(4))
    story.append(p("票据业务综合管理报告", make_style("CoverTitle", size=28, color=PRIMARY, align=TA_CENTER)))
    story.append(spacer(2))
    story.append(p("2024年度", make_style("CoverYear", size=16, color=GOLD, align=TA_CENTER)))
    story.append(spacer(6))
    story.append(hr(GOLD, 2))
    story.append(spacer(4))

    # 封面信息块
    info_data = [
        ["报告编号", "HX-BILL-2024-ANNUAL-001",  "密级",     "机密"],
        ["编制部门", "票据业务管理部",             "版本",     "V3.2"],
        ["编制日期", "2024年12月31日",             "页数",     "共 18 页"],
        ["审核人",   "张伟（副总裁）",             "批准人",   "李明（总裁）"],
    ]
    tbl = Table(info_data, colWidths=[30*mm, 65*mm, 25*mm, 45*mm])
    tbl.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(-1,-1), CN_FONT),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("BACKGROUND",   (0,0),(0,-1),  LIGHT_BG),
        ("BACKGROUND",   (2,0),(2,-1),  LIGHT_BG),
        ("TEXTCOLOR",    (0,0),(0,-1),  PRIMARY),
        ("TEXTCOLOR",    (2,0),(2,-1),  PRIMARY),
        ("GRID",         (0,0),(-1,-1), 0.5, colors.HexColor("#b3cde0")),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("ALIGN",        (1,0),(1,-1),  "LEFT"),
        ("ALIGN",        (3,0),(3,-1),  "LEFT"),
        ("ALIGN",        (0,0),(0,-1),  "CENTER"),
        ("ALIGN",        (2,0),(2,-1),  "CENTER"),
    ]))
    story.append(tbl)
    story.append(spacer(6))
    story.append(hr(PRIMARY))
    story.append(spacer(4))
    story.append(p(
        "本报告涵盖华夏金融集团2024年度全口径票据业务数据，包括但不限于增值税发票、"
        "银行承兑汇票、商业承兑汇票、电子票据、运输票据及医疗票据等各类票据的收发、"
        "流转、核销及风险管理情况。报告数据截止至2024年12月31日24时。",
        s_body
    ))
    story.append(NextPageTemplate("Content"))
    story.append(PageBreak())


def build_overview(story):
    """第一章：年度业务总览。"""  # KPI 卡片 + 总览表
    story.append(p("第一章  年度业务总览", s_section))
    story.append(hr())
    story.append(spacer(3))

    # KPI 卡片（用表格模拟）
    kpi_data = [
        ["指标",             "本年度",       "上年度",       "同比变动",    "完成率"],
        ["票据总金额（万元）", "158,620.00",  "132,450.00",  "▲ 19.76%",   "105.4%"],
        ["票据总张数（张）",  "24,386",       "20,112",      "▲ 21.25%",   "108.2%"],
        ["平均单张金额（元）", "6,505.82",    "6,585.96",    "▼  1.22%",   "—"],
        ["已核销金额（万元）", "142,318.00",  "118,900.00",  "▲ 19.70%",   "103.8%"],
        ["逾期未收（万元）",  "3,240.00",     "2,890.00",    "▲ 12.11%",   "—"],
        ["逾期率",           "2.04%",         "2.18%",       "▼  0.14ppt", "✓ 达标"],
        ["票据退回率",        "0.32%",         "0.45%",       "▼  0.13ppt", "✓ 达标"],
    ]
    tbl = Table(kpi_data, colWidths=[52*mm, 32*mm, 32*mm, 32*mm, 25*mm])
    ts = _tbl_style()
    ts.add("TEXTCOLOR", (3,2), (3,-1), GREEN)   # 正向变动绿色
    ts.add("TEXTCOLOR", (3,4), (3,4),  ACCENT)  # 逾期上升红色
    ts.add("FONTSIZE",  (0,0), (-1,0), 9)
    ts.add("BACKGROUND",(3,1), (3,1),  colors.HexColor("#d5f5e3"))
    ts.add("BACKGROUND",(3,4), (3,4),  colors.HexColor("#fde8e8"))
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(spacer(4))
    story.append(p("注：▲表示同比上升，▼表示同比下降；完成率 = 本年度 / 年度目标值。", s_small))
    story.append(spacer(5))


def build_revenue_chart(story):
    """月度营收图表。"""  # matplotlib 图表嵌入
    story.append(p("1.1  月度营收对比分析", s_bold))
    story.append(spacer(2))
    story.append(make_monthly_revenue_chart())
    story.append(spacer(2))
    story.append(p(
        "图1：2024年度各月营收持续增长，其中12月份达到峰值1800万元，"
        "全年同比增长率维持在13%～21%区间，第四季度增势尤为显著。",
        s_small
    ))
    story.append(spacer(5))


def build_invoice_detail(story):
    """第二章：增值税发票明细。"""  # 复杂嵌套表格
    story.append(PageBreak())
    story.append(p("第二章  增值税发票明细清单", s_section))
    story.append(hr())
    story.append(spacer(3))

    # 发票明细主表
    headers = ["序号","发票代码","发票号码","开票日期","购买方名称",
               "销售方名称","金额（元）","税率","税额（元）","价税合计","状态"]
    sample_buyers  = ["北京科技有限公司","上海贸易集团","广州制造有限公司",
                      "深圳电子股份","杭州软件科技","成都物流有限公司","武汉钢铁集团"]
    sample_sellers = ["华夏供应链管理","东方物产贸易","南洋商贸集团",
                      "中原工业制造","西部矿产开发","北方建材科技","海峡科技园区"]
    statuses = ["已认证","已认证","待认证","已作废","已认证","已认证","待认证","已作废","已认证"]
    status_colors = {"已认证": GREEN, "待认证": GOLD, "已作废": ACCENT}

    rows = [headers]
    base_date = datetime(2024, 1, 5)
    for i in range(1, 19):
        amount = round(random.uniform(5000, 200000), 2)
        rate = random.choice([0.06, 0.09, 0.13])
        tax = round(amount * rate, 2)
        total = round(amount + tax, 2)
        date = base_date + timedelta(days=i * 18)
        status = statuses[i % len(statuses)]
        rows.append([
            str(i),
            f"310{random.randint(100000,999999):09d}",
            f"{random.randint(10000000,99999999):08d}",
            date.strftime("%Y-%m-%d"),
            sample_buyers[i % len(sample_buyers)],
            sample_sellers[i % len(sample_sellers)],
            f"{amount:,.2f}",
            f"{int(rate*100)}%",
            f"{tax:,.2f}",
            f"{total:,.2f}",
            status,
        ])

    # 合计行
    total_amount = sum(float(r[6].replace(",","")) for r in rows[1:])
    total_tax    = sum(float(r[8].replace(",","")) for r in rows[1:])
    total_all    = sum(float(r[9].replace(",","")) for r in rows[1:])
    rows.append(["合计", "—", "—", "—", "—", "—",
                 f"{total_amount:,.2f}", "—", f"{total_tax:,.2f}",
                 f"{total_all:,.2f}", f"共{len(rows)-1}张"])

    col_w = [10,28,24,22,32,32,24,12,20,24,16]
    col_w = [w*mm for w in col_w]
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    ts = _tbl_style()
    # 合计行样式
    last = len(rows) - 1
    ts.add("BACKGROUND", (0, last), (-1, last), PRIMARY)
    ts.add("TEXTCOLOR",  (0, last), (-1, last), WHITE)
    ts.add("FONTSIZE",   (0, last), (-1, last), 8)
    # 状态列着色
    for idx, row in enumerate(rows[1:-1], 1):
        st = row[-1]
        c  = status_colors.get(st, colors.black)
        ts.add("TEXTCOLOR", (10, idx), (10, idx), c)
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(spacer(2))
    story.append(p("※ 以上数据均已脱敏处理，仅用于系统解析测试。", s_small))
    story.append(spacer(5))


def build_bank_acceptance(story):
    """第三章：银行承兑汇票。"""  # 票据要素表 + 分析
    story.append(PageBreak())
    story.append(p("第三章  银行承兑汇票管理", s_section))
    story.append(hr())
    story.append(spacer(3))

    story.append(p("3.1  汇票要素明细", s_bold))
    story.append(spacer(2))

    ba_headers = ["票据编号","出票日期","到期日","出票人","承兑行","收款人","票面金额（元）","承兑状态","背书次数"]
    banks = ["中国工商银行","中国建设银行","中国农业银行","中国银行",
             "交通银行","招商银行","浦发银行","民生银行"]
    corps = ["华夏物资有限公司","东方电气集团","中原煤业有限公司",
             "南方电网公司","西部油气开发","北方重工集团","海峡科技有限公司"]
    ba_statuses = ["已承兑","已贴现","未到期","已到期","已背书","贴现中","已托收"]

    ba_rows = [ba_headers]
    for i in range(1, 16):
        issue_date = datetime(2024, random.randint(1,10), random.randint(1,28))
        maturity   = issue_date + timedelta(days=random.choice([90, 180, 270, 365]))
        amount     = round(random.uniform(100000, 5000000), 2)
        ba_rows.append([
            f"BA2024{i:05d}",
            issue_date.strftime("%Y-%m-%d"),
            maturity.strftime("%Y-%m-%d"),
            corps[i % len(corps)],
            banks[i % len(banks)],
            corps[(i+2) % len(corps)],
            f"{amount:,.2f}",
            ba_statuses[i % len(ba_statuses)],
            str(random.randint(0, 5)),
        ])

    col_w = [22, 20, 20, 28, 28, 28, 26, 18, 14]
    col_w = [w*mm for w in col_w]
    tbl = Table(ba_rows, colWidths=col_w, repeatRows=1)
    tbl.setStyle(_tbl_style(SECONDARY))
    story.append(tbl)
    story.append(spacer(4))

    # 分析小结
    story.append(p("3.2  风险评估摘要", s_bold))
    story.append(spacer(2))
    risk_data = [
        ["风险类别",      "涉及票据数", "涉及金额（万元）", "风险等级", "处置建议"],
        ["集中度风险",    "3",          "450.00",           "中",       "分散出票人结构"],
        ["流动性风险",    "5",          "1,200.00",         "低",       "提前贴现安排"],
        ["信用风险",      "2",          "280.00",           "高",       "追加担保措施"],
        ["操作合规风险",  "1",          "90.00",            "低",       "加强审核流程"],
        ["汇率风险",      "0",          "0.00",             "无",       "—"],
    ]
    tbl2 = Table(risk_data, colWidths=[35*mm, 25*mm, 35*mm, 20*mm, 55*mm])
    ts2 = _tbl_style()
    ts2.add("TEXTCOLOR", (3,3), (3,3), ACCENT)   # 高风险红色
    ts2.add("TEXTCOLOR", (3,1), (3,1), GOLD)     # 中风险金色
    ts2.add("TEXTCOLOR", (3,2), (3,2), GREEN)    # 低风险绿色
    ts2.add("TEXTCOLOR", (3,4), (3,4), GREEN)
    ts2.add("TEXTCOLOR", (3,5), (3,5), GRAY)
    tbl2.setStyle(ts2)
    story.append(tbl2)
    story.append(spacer(5))


def build_charts_page(story):
    """第四章：数据分析图表页。"""  # 并排双图
    story.append(PageBreak())
    story.append(p("第四章  票据业务数据分析", s_section))
    story.append(hr())
    story.append(spacer(3))

    story.append(p("4.1  票据类别与账款趋势", s_bold))
    story.append(spacer(2))

    # 并排两图：用单行两列表格承载
    pie  = make_category_pie_chart()
    line = make_quarterly_trend_chart()
    chart_tbl = Table([[pie, line]], colWidths=[80*mm, 88*mm])
    chart_tbl.setStyle(TableStyle([
        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING",   (0,0), (-1,-1), 2),
        ("RIGHTPADDING",  (0,0), (-1,-1), 2),
    ]))
    story.append(chart_tbl)
    story.append(spacer(2))
    story.append(p(
        "图2（左）：2024年度电子发票占比持续提升，达20%，较上年增加6个百分点。"
        "  图3（右）：第四季度应收账款及已核销金额均创历史新高，逾期率控制在目标范围内。",
        s_small
    ))
    story.append(spacer(5))


def build_expense_reimbursement(story):
    """第五章：费用报销票据汇总。"""  # 多级表头 + 合并单元格
    story.append(PageBreak())
    story.append(p("第五章  费用报销票据汇总", s_section))
    story.append(hr())
    story.append(spacer(3))

    story.append(p("5.1  部门费用汇总表", s_bold))
    story.append(spacer(2))

    # 多级表头（手工合并）
    hdr1 = ["部门", "员工人数",
            "交通费", "", "住宿费", "", "餐饮费", "", "办公用品", "", "合计（元）"]
    hdr2 = ["", "", "票据数", "金额（元）", "票据数", "金额（元）",
            "票据数", "金额（元）", "票据数", "金额（元）", ""]
    depts = ["市场营销部","技术研发部","财务管理部","人力资源部",
             "法务合规部","业务运营部","IT基础设施部","战略发展部"]

    dept_rows = []
    grand_total = 0
    for dept in depts:
        headcount = random.randint(8, 60)
        transport_n = random.randint(20, 150); transport_a = round(random.uniform(5000, 80000), 2)
        hotel_n     = random.randint(5,  80);  hotel_a     = round(random.uniform(8000, 120000), 2)
        meal_n      = random.randint(30, 200); meal_a      = round(random.uniform(3000, 50000), 2)
        office_n    = random.randint(10, 60);  office_a    = round(random.uniform(2000, 30000), 2)
        total = transport_a + hotel_a + meal_a + office_a
        grand_total += total
        dept_rows.append([
            dept, str(headcount),
            str(transport_n), f"{transport_a:,.2f}",
            str(hotel_n),     f"{hotel_a:,.2f}",
            str(meal_n),      f"{meal_a:,.2f}",
            str(office_n),    f"{office_a:,.2f}",
            f"{total:,.2f}",
        ])
    # 合计行
    dept_rows.append(["合计", "—", "—", "—", "—", "—", "—", "—", "—", "—",
                       f"{grand_total:,.2f}"])

    all_rows = [hdr1, hdr2] + dept_rows
    col_w = [35, 16, 14, 22, 14, 22, 14, 22, 14, 22, 25]
    col_w = [w*mm for w in col_w]
    tbl = Table(all_rows, colWidths=col_w, repeatRows=2)
    ts = TableStyle([
        # 第一行（多级表头上层）
        ("BACKGROUND",   (0,0), (-1,0),  PRIMARY),
        ("TEXTCOLOR",    (0,0), (-1,0),  WHITE),
        ("FONTNAME",     (0,0), (-1,-1), CN_FONT),
        ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#b3cde0")),
        ("TOPPADDING",   (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
        # 第二行（子表头）
        ("BACKGROUND",   (0,1), (-1,1),  SECONDARY),
        ("TEXTCOLOR",    (0,1), (-1,1),  WHITE),
        # 斑马纹
        ("ROWBACKGROUNDS",(0,2),(-1,-2), [WHITE, LIGHT_BG]),
        # 合计行
        ("BACKGROUND",   (0,-1),(-1,-1), PRIMARY),
        ("TEXTCOLOR",    (0,-1),(-1,-1), WHITE),
        # 合并：部门列跨两行、员工人数、合计列
        ("SPAN",         (0,0), (0,1)),
        ("SPAN",         (1,0), (1,1)),
        ("SPAN",         (10,0),(10,1)),
        # 合并费用分组表头
        ("SPAN",         (2,0), (3,0)),
        ("SPAN",         (4,0), (5,0)),
        ("SPAN",         (6,0), (7,0)),
        ("SPAN",         (8,0), (9,0)),
    ])
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(spacer(2))
    story.append(p(f"全公司年度报销票据总金额：{grand_total:,.2f} 元，同比增长 14.3%。", s_body))
    story.append(spacer(5))


def build_electronic_invoice(story):
    """第六章：电子发票验证记录。"""  # 含二维码占位图和验证状态
    story.append(PageBreak())
    story.append(p("第六章  电子发票核验记录", s_section))
    story.append(hr())
    story.append(spacer(3))

    story.append(p("6.1  电子发票样本（节选）", s_bold))
    story.append(spacer(2))

    # 模拟电子发票样式框
    def fake_einvoice(idx, amount, buyer, seller, date_str, code, number):
        """用表格模拟一张电子发票外观。"""  # 仅作展示用途
        title_row = [[p(f"电子普通发票  No.{idx:03d}", make_style(
            f"EInv{idx}", size=11, color=PRIMARY, align=TA_CENTER))]]
        title_tbl = Table(title_row, colWidths=[168*mm])
        title_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,0), LIGHT_BG),
            ("TOPPADDING",    (0,0),(0,0), 6),
            ("BOTTOMPADDING", (0,0),(0,0), 6),
            ("BOX",           (0,0),(0,0), 1, SECONDARY),
        ]))

        body_data = [
            [p("发票代码：", s_small), p(code, s_body),
             p("发票号码：", s_small), p(number, s_body)],
            [p("开票日期：", s_small), p(date_str, s_body),
             p("校验码：",   s_small), p(f"{random.randint(10000,99999):05d}...{random.randint(100,999)}", s_body)],
            [p("购买方：",   s_small), p(buyer,    s_body),
             p("销售方：",   s_small), p(seller,   s_body)],
            [p("商品名称：", s_small), p("技术服务费", s_body),
             p("规格型号：", s_small), p("—", s_body)],
            [p("金额（元）：",s_small), p(f"{amount:,.2f}", make_style(
                f"AmtRed{idx}", size=10, color=ACCENT, align=TA_LEFT)),
             p("税额（元）：",s_small), p(f"{amount*0.06:,.2f}", s_body)],
            [p("价税合计：", s_small),
             p(f"¥ {amount*1.06:,.2f}", make_style(
                 f"TotalRed{idx}", size=11, color=ACCENT, align=TA_LEFT)),
             p("核验状态：", s_small),
             p("✔ 已验真", make_style(f"Veri{idx}", size=10, color=GREEN, align=TA_LEFT))],
        ]
        body_tbl = Table(body_data, colWidths=[28*mm, 60*mm, 28*mm, 52*mm])
        body_tbl.setStyle(TableStyle([
            ("FONTNAME",     (0,0),(-1,-1), CN_FONT),
            ("FONTSIZE",     (0,0),(-1,-1), 8.5),
            ("GRID",         (0,0),(-1,-1), 0.3, colors.HexColor("#cdd8e3")),
            ("TOPPADDING",   (0,0),(-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        return [title_tbl, Spacer(1, 1*mm), body_tbl, Spacer(1, 3*mm)]

    sample_corps = [
        ("杭州智联科技有限公司",     "华夏云服务平台"),
        ("北京数字信通股份有限公司",  "东方网络技术"),
        ("上海浦江数字科技集团",      "南方软件服务"),
        ("广州穗华信息技术有限公司",  "西部云计算中心"),
    ]
    for i, (buyer, seller) in enumerate(sample_corps, 1):
        amount = round(random.uniform(3000, 80000), 2)
        date_str = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        code   = f"310{random.randint(1000,9999):04d}{random.randint(10,99):02d}"
        number = f"{random.randint(10000000,99999999):08d}"
        for elem in fake_einvoice(i, amount, buyer, seller, date_str, code, number):
            story.append(elem)

    story.append(spacer(3))


def build_audit_trail(story):
    """第七章：审计追踪与合规日志。"""  # 操作日志表
    story.append(PageBreak())
    story.append(p("第七章  票据操作审计追踪", s_section))
    story.append(hr())
    story.append(spacer(3))

    log_headers = ["序号","操作时间","操作人","工号","操作类型","票据编号","涉及金额（元）","IP地址","操作结果"]
    op_types = ["录入","审核通过","退回修改","作废","背书","贴现申请","核销","打印","导出"]
    operators = ["张伟","李静","王磊","赵敏","陈建国","刘芳","杨帆","吴晓红","郑明"]
    results   = ["成功","成功","成功","成功","失败-权限不足","成功","成功","成功","成功"]

    log_rows = [log_headers]
    for i in range(1, 25):
        dt = datetime(2024, random.randint(1,12), random.randint(1,28),
                      random.randint(8,18), random.randint(0,59), random.randint(0,59))
        op = operators[i % len(operators)]
        ot = op_types[i % len(op_types)]
        res = results[i % len(results)]
        amount = round(random.uniform(1000, 500000), 2)
        log_rows.append([
            str(i),
            dt.strftime("%Y-%m-%d %H:%M:%S"),
            op,
            f"EMP{random.randint(1000,9999):04d}",
            ot,
            f"TK{random.randint(100000,999999):06d}",
            f"{amount:,.2f}",
            f"192.168.{random.randint(1,10)}.{random.randint(1,254)}",
            res,
        ])

    col_w = [10, 32, 14, 16, 18, 22, 24, 28, 24]
    col_w = [w*mm for w in col_w]
    tbl = Table(log_rows, colWidths=col_w, repeatRows=1)
    ts = _tbl_style(colors.HexColor("#2c3e50"))
    for i, row in enumerate(log_rows[1:], 1):
        if "失败" in row[-1]:
            ts.add("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fde8e8"))
            ts.add("TEXTCOLOR",  (8,i), (8,i),  ACCENT)
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(spacer(3))
    story.append(p("以上日志记录均由系统自动生成，任何手工篡改均属违规行为，将触发实时告警。", s_small))
    story.append(spacer(5))


def build_summary(story):
    """第八章：结论与建议。"""  # 结尾文字 + 签章区
    story.append(PageBreak())
    story.append(p("第八章  综合结论与改进建议", s_section))
    story.append(hr())
    story.append(spacer(3))

    conclusions = [
        ("总体评价",
         "2024年度票据业务整体运行良好，各项核心指标均超额完成年度目标，"
         "票据管理数字化转型稳步推进，电子票据占比大幅提升。"),
        ("亮点成果",
         "全年新增电子票据占比首次突破20%；逾期率降至2.04%，为近五年最低值；"
         "票据退回率降至0.32%，系统自动审核准确率达99.1%。"),
        ("主要风险",
         "个别部门仍存在票据集中度偏高问题；部分信用风险票据需追加担保；"
         "跨区域票据流转效率有待进一步提升。"),
        ("2025年建议",
         "（1）推进区块链票据平台建设，目标2025年Q2上线；"
         "（2）优化AI自动审核模型，目标准确率提升至99.5%；"
         "（3）建立动态风险预警机制，实现实时监控全覆盖；"
         "（4）完善跨行票据交换标准，推动行业互联互通。"),
    ]
    for title, content in conclusions:
        row = [[p(title, make_style(f"CTitle{title}", size=9, color=PRIMARY)),
                p(content, s_body)]]
        tbl = Table(row, colWidths=[30*mm, 140*mm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",     (0,0),(-1,-1), CN_FONT),
            ("BACKGROUND",   (0,0),(0,0),   LIGHT_BG),
            ("GRID",         (0,0),(-1,-1), 0.3, colors.HexColor("#b3cde0")),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
            ("VALIGN",       (0,0),(-1,-1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 2*mm))

    story.append(spacer(6))
    story.append(hr(GOLD))
    story.append(spacer(4))

    # 签章区
    sig_data = [
        ["编制人签章", "审核人签章", "批准人签章"],
        ["\n\n\n（签章处）\n", "\n\n\n（签章处）\n", "\n\n\n（签章处）\n"],
        ["票据业务管理部", "风险管理委员会", "集团总裁办公室"],
        [datetime.now().strftime("%Y年%m月%d日"),
         datetime.now().strftime("%Y年%m月%d日"),
         datetime.now().strftime("%Y年%m月%d日")],
    ]
    sig_tbl = Table(sig_data, colWidths=[56*mm, 56*mm, 56*mm])
    sig_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(-1,-1), CN_FONT),
        ("FONTSIZE",     (0,0),(-1,-1), 9),
        ("ALIGN",        (0,0),(-1,-1), "CENTER"),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("BACKGROUND",   (0,0),(-1,0),  PRIMARY),
        ("TEXTCOLOR",    (0,0),(-1,0),  WHITE),
        ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#b3cde0")),
        ("INNERGRID",    (0,0),(-1,-1), 0.3, colors.HexColor("#b3cde0")),
        ("TOPPADDING",   (0,1),(-1,1),  18),
        ("BOTTOMPADDING",(0,1),(-1,1),  18),
        ("FONTSIZE",     (0,2),(-1,-1), 8),
        ("TEXTCOLOR",    (0,2),(-1,2),  GRAY),
    ]))
    story.append(sig_tbl)
    story.append(spacer(4))
    story.append(p(
        "本报告由华夏金融集团票据业务管理系统（HXBMS v4.2）自动汇编，"
        "经人工复核后正式发布。报告中所有数据均以系统数据库为准。"
        "如有疑问，请联系票据业务管理部（内线：8888）。",
        s_small
    ))


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)  # 固定随机种子，保证结果可复现
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "华夏金融集团票据业务综合报告2024.pdf")

    doc = BillDocTemplate(output_path, title="华夏金融集团票据业务综合报告2024",
                          author="华夏金融集团票据业务管理部",
                          subject="年度票据业务综合分析")
    story = []

    # 封面（Cover 模板）
    story.append(NextPageTemplate("Cover"))
    build_cover(story)

    # 正文各章（Content 模板，封面末尾已切换）
    build_overview(story)
    build_revenue_chart(story)
    build_invoice_detail(story)
    build_bank_acceptance(story)
    build_charts_page(story)
    build_expense_reimbursement(story)
    build_electronic_invoice(story)
    build_audit_trail(story)
    build_summary(story)

    doc.build(story)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"PDF 已生成：{output_path}")
    print(f"文件大小：{size_kb:.1f} KB")


if __name__ == "__main__":
    main()
