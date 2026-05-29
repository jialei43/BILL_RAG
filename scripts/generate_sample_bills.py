#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sample_bills.py
生成多场景票据样本图片，用于测试票据审核系统
运行：python scripts/generate_sample_bills.py
输出：data/sample_bills/*.png
"""

import sys
import os
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample_bills"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

def get_font(size: int, bold: bool = False):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()

# ── 颜色常量 ─────────────────────────────────────────────────────────────────
C_BG        = (252, 248, 235)   # 票据底色：米黄
C_BORDER    = (120,  80,  20)   # 边框：深棕
C_TITLE     = (140,  20,  20)   # 标题：暗红
C_LABEL     = ( 60,  60,  80)   # 标签文字：深蓝灰
C_VALUE     = ( 10,  10,  10)   # 值文字：近黑
C_AMOUNT    = (180,  20,  20)   # 金额：红色
C_SEAL_RED  = (200,  30,  30)   # 印章：红
C_SEAL_BLUE = ( 30,  60, 180)   # 印章：蓝
C_GRID      = (180, 160, 120)   # 表格线
C_WARN      = (220,  80,   0)   # 警告橙
C_WATERMARK = (220, 200, 170)   # 水印


# ── 基础绘制工具 ──────────────────────────────────────────────────────────────
def new_bill(w=1200, h=800) -> tuple:
    img = Image.new("RGB", (w, h), C_BG)
    draw = ImageDraw.Draw(img)
    # 双边框
    draw.rectangle([4, 4, w-5, h-5], outline=C_BORDER, width=3)
    draw.rectangle([10, 10, w-11, h-11], outline=C_BORDER, width=1)
    return img, draw


def hline(draw, y, x0, x1, color=C_GRID, w=1):
    draw.line([(x0, y), (x1, y)], fill=color, width=w)


def vline(draw, x, y0, y1, color=C_GRID, w=1):
    draw.line([(x, y0), (x, y1)], fill=color, width=w)


def text(draw, xy, s, size=16, color=C_VALUE, bold=False, anchor="la"):
    f = get_font(size, bold)
    draw.text(xy, s, font=f, fill=color, anchor=anchor)


def field(draw, x, y, label, value, lw=160, vw=300, h=32, lsize=14, vsize=16):
    """绘制一个 label: value 字段行"""
    draw.rectangle([x, y, x+lw, y+h], outline=C_GRID, width=1)
    draw.rectangle([x+lw, y, x+lw+vw, y+h], outline=C_GRID, width=1)
    text(draw, (x+6, y+h//2), label, size=lsize, color=C_LABEL, anchor="lm")
    text(draw, (x+lw+6, y+h//2), value, size=vsize, color=C_VALUE, anchor="lm")


def seal_round(draw, cx, cy, r, label, color=C_SEAL_RED, star=True):
    """绘制圆形公章"""
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=3)
    draw.ellipse([cx-r+8, cy-r+8, cx+r-8, cy+r-8], outline=color, width=1)
    if star:
        # 五角星（简化版）
        pts = []
        for i in range(5):
            a = math.radians(-90 + i * 72)
            pts.append((cx + 14*math.cos(a), cy + 14*math.sin(a)))
        star_pts = []
        for i in range(5):
            star_pts.append(pts[i])
            a2 = math.radians(-90 + i*72 + 36)
            star_pts.append((cx + 6*math.cos(a2), cy + 6*math.sin(a2)))
        draw.polygon(star_pts, fill=color)
    # 章内文字
    f = get_font(13)
    for i, ch in enumerate(label):
        a = math.radians(-90 + i * (360/max(len(label),1)) - (360/max(len(label),1))*(len(label)-1)/2)
        tx = cx + (r-18) * math.cos(a)
        ty = cy + (r-18) * math.sin(a)
        draw.text((tx, ty), ch, font=f, fill=color, anchor="mm")


def seal_oval(draw, cx, cy, label_top, label_bot, color=C_SEAL_RED):
    """绘制椭圆骑缝章"""
    rx, ry = 55, 38
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=color, width=2)
    f = get_font(13)
    draw.text((cx, cy-14), label_top, font=f, fill=color, anchor="mm")
    draw.text((cx, cy+4),  label_bot, font=f, fill=color, anchor="mm")


def endorsement_block(draw, x, y, w, h, idx, from_name, to_name, date_str,
                      broken=False, color=C_LABEL):
    """绘制一个背书格"""
    draw.rectangle([x, y, x+w, y+h], outline=C_GRID, width=1)
    f14 = get_font(14)
    f13 = get_font(13)
    draw.text((x+6, y+6),  f"第{idx}手背书", font=f14, fill=color)
    draw.text((x+6, y+26), f"背书人：{from_name}", font=f13, fill=C_VALUE)
    if broken:
        draw.text((x+6, y+46), "被背书人：（空白）", font=f13, fill=C_WARN)
        draw.text((x+6, y+66), f"日期：{date_str}", font=f13, fill=C_VALUE)
        draw.line([(x+20, y+42), (x+w-20, y+42)], fill=C_WARN, width=2)
    else:
        draw.text((x+6, y+46), f"被背书人：{to_name}", font=f13, fill=C_VALUE)
        draw.text((x+6, y+66), f"日期：{date_str}", font=f13, fill=C_VALUE)


# ══════════════════════════════════════════════════════════════════════════════
# 场景 1：合规银行承兑汇票
# ══════════════════════════════════════════════════════════════════════════════
def gen_01_compliant_bah():
    img, draw = new_bill(1200, 820)

    # 标题
    text(draw, (600, 38), "中国人民银行  银行承兑汇票", size=26, color=C_TITLE, bold=True, anchor="mm")
    text(draw, (600, 70), "BANK ACCEPTANCE BILL", size=14, color=C_LABEL, anchor="mm")
    hline(draw, 85, 20, 1180, C_BORDER, 2)

    # 票号区
    text(draw, (30, 100), "票据号码：", size=14, color=C_LABEL)
    text(draw, (120, 100), "4400 2024 0315 0000 0128", size=16, color=C_VALUE, bold=True)
    text(draw, (900, 100), "出票日期：2024-03-15", size=14, color=C_LABEL)
    hline(draw, 120, 20, 1180)

    # 金额区（突出显示）
    draw.rectangle([20, 128, 1180, 185], fill=(255, 245, 220), outline=C_BORDER, width=1)
    text(draw, (30, 142), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 138), "壹佰伍拾万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (800, 142), "小写：¥", size=14, color=C_LABEL)
    text(draw, (860, 138), "1,500,000.00", size=24, color=C_AMOUNT, bold=True)
    hline(draw, 185, 20, 1180, C_BORDER, 1)

    # 各要素字段
    fields_left = [
        ("出  票  人", "深圳市鑫远贸易有限公司"),
        ("出票人账号", "4400 9801 2345 6789 012"),
        ("出票人开户行", "中国工商银行深圳南山支行"),
        ("收  款  人", "广州市博远供应链管理有限公司"),
    ]
    fields_right = [
        ("承  兑  人", "中国工商银行深圳南山支行"),
        ("承兑人账号", "4400 0100 0900 1234 567"),
        ("付 款 行 号", "102584001234"),
        ("到  期  日", "2024-09-15"),
    ]

    y = 193
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=420, h=36, lsize=14, vsize=15)
        y += 36
    y = 193
    for label, val in fields_right:
        field(draw, 590, y, label, val, lw=130, vw=440, h=36, lsize=14, vsize=15)
        y += 36

    hline(draw, 193+36*4, 20, 1180, C_BORDER, 1)

    # 承兑信息
    y2 = 193 + 36*4 + 6
    text(draw, (30, y2), "承兑信息：本汇票已经承兑，到期无条件付款。", size=15, color=C_VALUE)
    text(draw, (30, y2+24), "承兑日期：2024-03-15　　承兑到期日：2024-09-15", size=14, color=C_LABEL)

    # 背书区
    bx, by = 20, 420
    text(draw, (bx, by-24), "背书记录", size=15, color=C_LABEL, bold=True)
    draw.rectangle([bx, by, 1180, by+95], outline=C_GRID, width=1)
    endorsement_block(draw, bx, by, 380, 95, "一",
                      "广州市博远供应链管理有限公司",
                      "上海联华商贸有限公司", "2024-05-20")
    endorsement_block(draw, bx+380, by, 380, 95, "二",
                      "上海联华商贸有限公司",
                      "北京恒信科技集团有限公司", "2024-07-08")
    draw.rectangle([bx+760, by, 1180, by+95], outline=C_GRID, width=1)
    text(draw, (bx+770, by+40), "（背书栏未使用）", size=13, color=C_WATERMARK)

    # 印章
    seal_round(draw, 420, 260, 42, "深圳市鑫远贸易有限公司财务专用章", C_SEAL_RED)
    seal_round(draw, 820, 260, 42, "中国工商银行深圳南山支行承兑专用章", C_SEAL_BLUE)
    seal_oval(draw, 1100, 540, "广州市博远", "供应链管理有限公司", C_SEAL_RED)

    # 状态标签
    draw.rectangle([980, 20, 1180, 55], fill=(220, 255, 220), outline=(0, 150, 0), width=2)
    text(draw, (1080, 37), "✓ 合规票据", size=16, color=(0, 130, 0), bold=True, anchor="mm")

    # 底部说明
    hline(draw, 560, 20, 1180, C_BORDER, 1)
    text(draw, (30, 568), "票据用途：货物贸易结算", size=13, color=C_LABEL)
    text(draw, (400, 568), "货物描述：电子元器件采购款", size=13, color=C_LABEL)
    text(draw, (750, 568), "币  种：人民币", size=13, color=C_LABEL)
    text(draw, (30, 592), "备注：本票据由中国工商银行承兑，信用等级AAA，可在全国银行间市场流通。", size=13, color=C_LABEL)

    img.save(OUTPUT_DIR / "01_合规_银行承兑汇票.png", dpi=(150, 150))
    print("✓ 01_合规_银行承兑汇票.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 2：金额大小写不一致（风险票据）
# ══════════════════════════════════════════════════════════════════════════════
def gen_02_amount_mismatch():
    img, draw = new_bill(1200, 700)

    text(draw, (600, 38), "银行承兑汇票", size=26, color=C_TITLE, bold=True, anchor="mm")
    hline(draw, 65, 20, 1180, C_BORDER, 2)

    text(draw, (30, 78), "票据号码：4400 2024 0520 0000 0237", size=14, color=C_VALUE, bold=True)
    text(draw, (850, 78), "出票日期：2024-05-20", size=14, color=C_LABEL)

    # 金额区 — 大小写故意不一致
    draw.rectangle([20, 105, 1180, 170], fill=(255, 235, 230), outline=(200, 50, 50), width=2)
    text(draw, (30, 115), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 110), "贰佰万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (700, 115), "小写：¥", size=14, color=C_LABEL)
    text(draw, (760, 110), "1,800,000.00", size=24, color=(180, 20, 20), bold=True)

    # 警告标注
    draw.rectangle([600, 108, 900, 168], outline=C_WARN, width=3)
    text(draw, (910, 130), "⚠ 大写200万 ≠ 小写180万", size=15, color=C_WARN, bold=True)
    text(draw, (910, 152), "  金额大小写不符，存在欺诈风险", size=13, color=C_WARN)

    fields_left = [
        ("出  票  人", "杭州市天悦进出口有限公司"),
        ("出票人账号", "3301 0123 4567 8901 234"),
        ("出票人开户行", "招商银行杭州西湖支行"),
        ("收  款  人", "宁波盛达物流集团有限公司"),
    ]
    fields_right = [
        ("承  兑  人", "招商银行杭州西湖支行"),
        ("承兑人账号", "3301 9801 0000 1122 334"),
        ("付 款 行 号", "308331000012"),
        ("到  期  日", "2024-11-20"),
    ]
    y = 178
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=420, h=34); y += 34
    y = 178
    for label, val in fields_right:
        field(draw, 590, y, label, val, lw=130, vw=440, h=34); y += 34

    hline(draw, 178+34*4, 20, 1180, C_BORDER, 1)

    # 印章
    seal_round(draw, 400, 248, 40, "杭州市天悦进出口有限公司财务章", C_SEAL_RED)
    seal_round(draw, 820, 248, 40, "招商银行杭州西湖支行承兑专用章", C_SEAL_BLUE)

    # 背书区
    by = 360
    text(draw, (20, by-22), "背书记录", size=14, color=C_LABEL, bold=True)
    draw.rectangle([20, by, 1180, by+90], outline=C_GRID, width=1)
    endorsement_block(draw, 20, by, 580, 90, "一",
                      "宁波盛达物流集团有限公司",
                      "成都汇通供应链有限公司", "2024-08-15")
    draw.rectangle([600, by, 1180, by+90], outline=C_GRID)
    text(draw, (610, by+38), "（背书栏未使用）", size=13, color=C_WATERMARK)

    seal_oval(draw, 580, by+45, "宁波盛达物流", "集团有限公司", C_SEAL_RED)

    # 风险标签
    draw.rectangle([980, 20, 1180, 55], fill=(255, 230, 220), outline=(200, 50, 50), width=2)
    text(draw, (1080, 37), "⚠ 金额异常", size=15, color=(180, 20, 20), bold=True, anchor="mm")

    hline(draw, 465, 20, 1180, C_BORDER, 1)
    text(draw, (30, 474), "风险提示：票面金额大写「贰佰万元整」与小写「1,800,000.00」不一致，差额20万元，请核实原始票据。", size=13, color=C_WARN)

    img.save(OUTPUT_DIR / "02_风险_金额大小写不符.png", dpi=(150, 150))
    print("✓ 02_风险_金额大小写不符.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 3：票据已过期
# ══════════════════════════════════════════════════════════════════════════════
def gen_03_expired():
    img, draw = new_bill(1200, 680)

    text(draw, (600, 38), "商业承兑汇票", size=26, color=C_TITLE, bold=True, anchor="mm")
    hline(draw, 65, 20, 1180, C_BORDER, 2)

    text(draw, (30, 78), "票据号码：6100 2023 0610 0000 0056", size=14, color=C_VALUE, bold=True)
    text(draw, (850, 78), "出票日期：2023-06-10", size=14, color=C_LABEL)

    draw.rectangle([20, 105, 1180, 165], fill=(252, 248, 235), outline=C_BORDER, width=1)
    text(draw, (30, 115), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 110), "伍拾万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (700, 115), "小写：¥", size=14, color=C_LABEL)
    text(draw, (760, 110), "500,000.00", size=24, color=C_AMOUNT, bold=True)

    fields_left = [
        ("出  票  人", "西安宏达机械设备有限公司"),
        ("出票人账号", "6101 8802 3456 7890 123"),
        ("出票人开户行", "中国银行西安高新支行"),
        ("收  款  人", "郑州联合零部件供应有限公司"),
    ]
    fields_right = [
        ("承  兑  人", "西安宏达机械设备有限公司（商承）"),
        ("承兑人账号", "6101 8802 3456 7890 123"),
        ("付 款 行 号", "104791001023"),
        ("到  期  日", "2023-12-10"),
    ]
    y = 172
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=420, h=34); y += 34
    y = 172
    for label, val in fields_right:
        field(draw, 590, y, label, val, lw=130, vw=440, h=34); y += 34

    # 过期水印
    f_big = get_font(80, bold=True)
    draw.text((200, 200), "已  过  期", font=f_big, fill=(220, 50, 50, 80))

    hline(draw, 172+34*4, 20, 1180, C_BORDER, 1)

    # 到期日标红
    draw.rectangle([596, 172+34*3, 1180, 172+34*4], fill=(255, 220, 220))
    text(draw, (726, 172+34*3+10), "2023-12-10  【已逾期 365+ 天】", size=15, color=(180, 20, 20), bold=True)

    seal_round(draw, 400, 248, 40, "西安宏达机械设备有限公司", C_SEAL_RED)
    seal_round(draw, 820, 248, 40, "西安宏达机械设备有限公司承兑章", C_SEAL_RED)

    by = 348
    text(draw, (20, by-22), "背书记录", size=14, color=C_LABEL, bold=True)
    draw.rectangle([20, by, 1180, by+90], outline=C_GRID, width=1)
    endorsement_block(draw, 20, by, 580, 90, "一",
                      "郑州联合零部件供应有限公司",
                      "武汉盛泰贸易有限公司", "2023-09-18")
    draw.rectangle([600, by, 1180, by+90], outline=C_GRID)
    text(draw, (610, by+38), "（背书栏未使用）", size=13, color=C_WATERMARK)

    # 过期标签
    draw.rectangle([940, 20, 1180, 55], fill=(255, 220, 220), outline=(180, 20, 20), width=2)
    text(draw, (1060, 37), "✗ 票据已过期", size=15, color=(180, 20, 20), bold=True, anchor="mm")

    hline(draw, 454, 20, 1180, C_BORDER, 1)
    text(draw, (30, 463), "审核结论：票据到期日 2023-12-10 已过，逾期超过12个月，商业承兑汇票兑付风险极高，建议拒绝受理。", size=13, color=C_WARN)

    img.save(OUTPUT_DIR / "03_拒绝_票据已过期.png", dpi=(150, 150))
    print("✓ 03_拒绝_票据已过期.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 4：背书链断裂（被背书人空白）
# ══════════════════════════════════════════════════════════════════════════════
def gen_04_broken_endorsement():
    img, draw = new_bill(1200, 760)

    text(draw, (600, 38), "银行承兑汇票", size=26, color=C_TITLE, bold=True, anchor="mm")
    hline(draw, 65, 20, 1180, C_BORDER, 2)

    text(draw, (30, 78), "票据号码：3100 2024 0102 0000 0399", size=14, color=C_VALUE, bold=True)
    text(draw, (850, 78), "出票日期：2024-01-02", size=14, color=C_LABEL)

    draw.rectangle([20, 105, 1180, 162], fill=(252, 248, 235), outline=C_BORDER, width=1)
    text(draw, (30, 112), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 108), "叁佰贰拾万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (740, 112), "小写：¥", size=14, color=C_LABEL)
    text(draw, (800, 108), "3,200,000.00", size=24, color=C_AMOUNT, bold=True)

    fields_left = [
        ("出  票  人", "上海德昌电气集团有限公司"),
        ("出票人账号", "3100 5501 2233 4455 667"),
        ("出票人开户行", "中国建设银行上海浦东新区支行"),
        ("收  款  人", "苏州精密制造有限公司"),
    ]
    fields_right = [
        ("承  兑  人", "中国建设银行上海浦东新区支行"),
        ("承兑人账号", "3100 9050 0100 2345 678"),
        ("付 款 行 号", "105291010010"),
        ("到  期  日", "2024-07-02"),
    ]
    y = 170
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=420, h=34); y += 34
    y = 170
    for label, val in fields_right:
        field(draw, 590, y, label, val, lw=130, vw=440, h=34); y += 34

    hline(draw, 170+34*4, 20, 1180, C_BORDER, 1)

    seal_round(draw, 400, 240, 40, "上海德昌电气集团有限公司财务章", C_SEAL_RED)
    seal_round(draw, 820, 240, 40, "中国建设银行上海浦东新区支行承兑章", C_SEAL_BLUE)

    # 背书区（3格，第2格断裂）
    by = 350
    text(draw, (20, by-22), "背书记录（第二手背书链断裂）", size=14, color=C_WARN, bold=True)
    bw = 385
    draw.rectangle([20, by, 20+bw*3, by+100], outline=C_GRID, width=1)

    # 第一手：正常
    endorsement_block(draw, 20, by, bw, 100, "一",
                      "苏州精密制造有限公司",
                      "南京华兴物资有限公司", "2024-03-10")
    seal_oval(draw, 20+bw//2, by+72, "苏州精密", "制造有限公司", C_SEAL_RED)

    # 第二手：背书人未填 = 断裂
    draw.rectangle([20+bw, by, 20+bw*2, by+100], fill=(255, 245, 220), outline=C_WARN, width=2)
    endorsement_block(draw, 20+bw, by, bw, 100, "二",
                      "南京华兴物资有限公司",
                      "（被背书人未填写）", "2024-04-22",
                      broken=True, color=C_WARN)

    # 第三手：有被背书人但上手断裂
    draw.rectangle([20+bw*2, by, 20+bw*3, by+100], fill=(255, 245, 220), outline=C_WARN, width=1)
    endorsement_block(draw, 20+bw*2, by, bw, 100, "三",
                      "???（持票方不明）",
                      "深圳汇通金融服务有限公司", "2024-05-30",
                      color=C_WARN)

    # 箭头说明链路
    draw.line([(20+bw, by+50), (20+bw*2, by+50)], fill=C_WARN, width=2)
    text(draw, (20+bw+bw//2, by+50), "⚡断裂", size=13, color=C_WARN, anchor="mm")

    # 标签
    draw.rectangle([900, 20, 1180, 55], fill=(255, 245, 210), outline=C_WARN, width=2)
    text(draw, (1040, 37), "⚡ 背书链不完整", size=14, color=C_WARN, bold=True, anchor="mm")

    hline(draw, 462, 20, 1180, C_BORDER, 1)
    text(draw, (30, 472), "风险说明：第二手背书被背书人栏为空白，导致持票权归属不明，第三手来源存疑，建议退回补全背书手续。", size=13, color=C_WARN)

    img.save(OUTPUT_DIR / "04_风险_背书链断裂.png", dpi=(150, 150))
    print("✓ 04_风险_背书链断裂.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 5：伪造票据嫌疑（承兑行不存在 + 印章模糊）
# ══════════════════════════════════════════════════════════════════════════════
def gen_05_suspected_fraud():
    img, draw = new_bill(1200, 720)

    text(draw, (600, 38), "银行承兑汇票", size=26, color=C_TITLE, bold=True, anchor="mm")
    hline(draw, 65, 20, 1180, C_BORDER, 2)

    text(draw, (30, 78), "票据号码：9900 2024 0801 0000 9999", size=14, color=C_VALUE, bold=True)
    text(draw, (850, 78), "出票日期：2024-08-01", size=14, color=C_LABEL)

    draw.rectangle([20, 105, 1180, 162], fill=(255, 240, 240), outline=(200, 50, 50), width=2)
    text(draw, (30, 112), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 108), "壹千万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (750, 112), "小写：¥", size=14, color=C_LABEL)
    text(draw, (810, 108), "10,000,000.00", size=24, color=C_AMOUNT, bold=True)

    fields_left = [
        ("出  票  人", "深圳市恒盛达投资管理有限公司"),
        ("出票人账号", "9988 0011 2233 4455 667"),
        ("出票人开户行", "深圳市南山融汇商业银行"),       # 虚假银行
        ("收  款  人", "香港汇融国际贸易有限公司"),
    ]
    fields_right = [
        ("承  兑  人", "深圳市南山融汇商业银行"),         # 虚假银行
        ("承兑人账号", "9988 9900 1111 2222 333"),
        ("付 款 行 号", "999000000001"),                  # 不合规行号
        ("到  期  日", "2025-02-01"),
    ]
    y = 170
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=420, h=34); y += 34
    y = 170
    for label, val in fields_right:
        field(draw, 590, y, label, val, lw=130, vw=440, h=34); y += 34

    # 标注虚假信息
    draw.rectangle([726, 170+34*0, 1180, 170+34*1], fill=(255, 220, 220))
    draw.rectangle([726, 170+34*3, 1180, 170+34*4], fill=(255, 220, 220))
    text(draw, (900, 170+34*0+10), "⚠ 查无此银行机构", size=13, color=(180, 20, 20))
    text(draw, (900, 170+34*3+10), "⚠ 行号格式非法", size=13, color=(180, 20, 20))

    hline(draw, 170+34*4, 20, 1180, C_BORDER, 1)

    # 模糊印章（用低透明度圆环模拟）
    for r_extra in [0, 4, 8]:
        draw.ellipse([360-42-r_extra, 226-42-r_extra, 360+42+r_extra, 226+42+r_extra],
                     outline=(200, 50, 50, 30), width=1)
    text(draw, (360, 230), "印章模糊\n无法识别", size=13, color=(160, 160, 160), anchor="mm")
    draw.rectangle([318, 184, 402, 268], outline=C_WARN, width=2)
    text(draw, (410, 224), "↑ 印章模糊，\n  真伪存疑", size=12, color=C_WARN)

    seal_round(draw, 820, 248, 40, "深圳市南山融汇商业银行（存疑）", (160, 100, 100))

    # 背书
    by = 370
    text(draw, (20, by-22), "背书记录", size=14, color=C_LABEL, bold=True)
    draw.rectangle([20, by, 1180, by+90], outline=C_GRID)
    endorsement_block(draw, 20, by, 580, 90, "一",
                      "香港汇融国际贸易有限公司",
                      "（被背书人未填写）", "2024-09-15", broken=True)
    draw.rectangle([600, by, 1180, by+90], outline=C_GRID)
    text(draw, (610, by+38), "（背书栏未使用）", size=13, color=C_WATERMARK)

    # 欺诈标签
    draw.rectangle([900, 20, 1180, 55], fill=(255, 200, 200), outline=(180, 20, 20), width=2)
    text(draw, (1040, 37), "✗ 疑似伪造票据", size=14, color=(180, 20, 20), bold=True, anchor="mm")

    hline(draw, 476, 20, 1180, C_BORDER, 1)
    text(draw, (30, 485), "风险项：①承兑行「深圳市南山融汇商业银行」在央行金融机构名单中查无；", size=13, color=C_WARN)
    text(draw, (30, 505), "        ②付款行号「999000000001」不符合12位联行号规范；③出票人印章模糊，疑似伪造。", size=13, color=C_WARN)
    text(draw, (30, 525), "建议：立即移交风控合规部门，向公安机关报案核查。", size=13, color=(180, 20, 20), bold=True)

    img.save(OUTPUT_DIR / "05_欺诈_伪造票据嫌疑.png", dpi=(150, 150))
    print("✓ 05_欺诈_伪造票据嫌疑.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 6：多次背书正常流转（4次背书）
# ══════════════════════════════════════════════════════════════════════════════
def gen_06_multi_endorsement():
    img, draw = new_bill(1400, 860)

    text(draw, (700, 38), "银行承兑汇票（多次背书流转）", size=24, color=C_TITLE, bold=True, anchor="mm")
    hline(draw, 65, 20, 1380, C_BORDER, 2)

    text(draw, (30, 78), "票据号码：2100 2024 0220 0000 0512", size=14, color=C_VALUE, bold=True)
    text(draw, (1050, 78), "出票日期：2024-02-20", size=14, color=C_LABEL)

    draw.rectangle([20, 105, 1380, 160], fill=(252, 248, 235), outline=C_BORDER, width=1)
    text(draw, (30, 112), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 108), "捌拾万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (840, 112), "小写：¥", size=14, color=C_LABEL)
    text(draw, (900, 108), "800,000.00", size=24, color=C_AMOUNT, bold=True)

    fields_left = [
        ("出  票  人", "天津德信工贸有限公司"),
        ("出票人账号", "1201 6601 2345 6789 012"),
        ("出票人开户行", "中国农业银行天津滨海新区支行"),
        ("收  款  人", "大连港务物资供应有限公司"),
    ]
    fields_right = [
        ("承  兑  人", "中国农业银行天津滨海新区支行"),
        ("承兑人账号", "1201 9030 1000 1234 567"),
        ("付 款 行 号", "103601000019"),
        ("到  期  日", "2024-08-20"),
    ]
    y = 168
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=380, h=34); y += 34
    y = 168
    for label, val in fields_right:
        field(draw, 560, y, label, val, lw=130, vw=460, h=34); y += 34

    hline(draw, 168+34*4, 20, 1380, C_BORDER, 1)

    seal_round(draw, 380, 238, 38, "天津德信工贸有限公司财务章", C_SEAL_RED)
    seal_round(draw, 900, 238, 38, "中国农业银行天津滨海新区支行承兑章", C_SEAL_BLUE)

    # 4次背书
    by = 360
    text(draw, (20, by-24), "背书流转记录（共4手）", size=15, color=C_LABEL, bold=True)
    bw = 335
    bh = 110

    endorsers = [
        ("大连港务物资供应有限公司", "沈阳国联铁路物流有限公司", "2024-04-05"),
        ("沈阳国联铁路物流有限公司", "哈尔滨冰城商贸集团有限公司", "2024-05-18"),
        ("哈尔滨冰城商贸集团有限公司", "长春吉丰农业发展有限公司", "2024-06-30"),
        ("长春吉丰农业发展有限公司", "（当前持票方，尚未背书）", "—"),
    ]
    seal_colors = [C_SEAL_RED, (150, 50, 150), (50, 120, 50), (30, 80, 180)]

    for i, (frm, to, dt) in enumerate(endorsers):
        x = 20 + i * bw
        draw.rectangle([x, by, x+bw, by+bh], outline=C_GRID, width=1)
        f14 = get_font(14)
        f13 = get_font(13)
        draw.text((x+6, by+6),   f"第{['一','二','三','四'][i]}手背书", font=f14, fill=C_LABEL)
        draw.text((x+6, by+28),  f"背书人：{frm}", font=f13, fill=C_VALUE)
        if i < 3:
            draw.text((x+6, by+50), f"被背书人：{to}", font=f13, fill=C_VALUE)
            draw.text((x+6, by+72), f"日期：{dt}", font=f13, fill=C_VALUE)
            seal_oval(draw, x+bw//2, by+bh+16, frm[:4], frm[4:8], seal_colors[i])
        else:
            draw.text((x+6, by+50), f"被背书人：（待背书）", font=f13, fill=C_LABEL)
            draw.text((x+6, by+72), f"当前持票方", font=f13, fill=C_LABEL)

    # 流转箭头
    for i in range(3):
        ax = 20 + (i+1)*bw - 1
        draw.polygon([(ax, by+bh//2-8), (ax+16, by+bh//2), (ax, by+bh//2+8)], fill=(80, 120, 180))

    draw.rectangle([1160, 20, 1380, 55], fill=(220, 255, 220), outline=(0, 150, 0), width=2)
    text(draw, (1270, 37), "✓ 背书链完整", size=15, color=(0, 130, 0), bold=True, anchor="mm")

    hline(draw, by+bh+50, 20, 1380, C_BORDER, 1)
    text(draw, (30, by+bh+60), "审核结论：票据背书链完整，4次流转记录清晰，各手背书印章齐全，持票权归属明确，可正常受理。", size=13, color=(0, 120, 0))

    img.save(OUTPUT_DIR / "06_合规_多次背书流转.png", dpi=(150, 150))
    print("✓ 06_合规_多次背书流转.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 7：商业承兑汇票 + 承兑人风险
# ══════════════════════════════════════════════════════════════════════════════
def gen_07_commercial_high_risk():
    img, draw = new_bill(1200, 720)

    text(draw, (600, 38), "商业承兑汇票", size=26, color=C_TITLE, bold=True, anchor="mm")
    text(draw, (600, 64), "COMMERCIAL ACCEPTANCE BILL", size=12, color=C_LABEL, anchor="mm")
    hline(draw, 78, 20, 1180, C_BORDER, 2)

    text(draw, (30, 90), "票据号码：5000 2024 0410 0000 0188", size=14, color=C_VALUE, bold=True)
    text(draw, (850, 90), "出票日期：2024-04-10", size=14, color=C_LABEL)

    draw.rectangle([20, 112, 1180, 168], fill=(252, 248, 235), outline=C_BORDER, width=1)
    text(draw, (30, 120), "票面金额（大写）：", size=14, color=C_LABEL)
    text(draw, (190, 116), "贰佰万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (700, 120), "小写：¥", size=14, color=C_LABEL)
    text(draw, (760, 116), "2,000,000.00", size=24, color=C_AMOUNT, bold=True)

    fields_left = [
        ("出  票  人", "重庆华成建筑材料有限公司"),
        ("出票人账号", "5000 4401 9988 7766 554"),
        ("出票人开户行", "重庆银行江北支行"),
        ("收  款  人", "贵州黔南钢铁实业有限公司"),
    ]
    fields_right = [
        ("承  兑  人（付款方）", "重庆华成建筑材料有限公司"),
        ("承兑人账号", "5000 4401 9988 7766 554"),
        ("付 款 行 号", "402653000018"),
        ("到  期  日", "2024-10-10"),
    ]
    y = 176
    for label, val in fields_left:
        field(draw, 20, y, label, val, lw=130, vw=420, h=34); y += 34
    y = 176
    for label, val in fields_right:
        field(draw, 590, y, label, val, lw=150, vw=420, h=34); y += 34

    # 商承风险提示
    draw.rectangle([590, 176, 1180, 210], fill=(255, 240, 200))
    text(draw, (596, 181), "⚠ 商承：承兑人即出票人，兑付依赖企业信用，无银行兜底", size=13, color=C_WARN)

    # 风险信息框
    draw.rectangle([20, 316, 560, 390], fill=(255, 248, 220), outline=C_WARN, width=2)
    text(draw, (30, 322), "承兑人信用评估", size=14, color=C_WARN, bold=True)
    text(draw, (30, 342), "企业信用评级：BB（中等偏低）", size=13, color=C_VALUE)
    text(draw, (30, 360), "近12月逾期记录：2次", size=13, color=(180, 20, 20))
    text(draw, (30, 378), "经营状态：存续（有异常）", size=13, color=C_WARN)

    draw.rectangle([590, 316, 1180, 390], fill=(255, 248, 220), outline=C_WARN, width=2)
    text(draw, (600, 322), "合规核查项", size=14, color=C_WARN, bold=True)
    text(draw, (600, 342), "是否在黑名单：否", size=13, color=(0, 120, 0))
    text(draw, (600, 360), "是否在法院失信名单：是 ⚠", size=13, color=(180, 20, 20))
    text(draw, (600, 378), "税务状态：正常", size=13, color=(0, 120, 0))

    seal_round(draw, 380, 258, 40, "重庆华成建筑材料有限公司财务专用章", C_SEAL_RED)
    seal_round(draw, 850, 258, 40, "重庆华成建筑材料有限公司承兑专用章", C_SEAL_RED)

    by = 404
    text(draw, (20, by-22), "背书记录", size=14, color=C_LABEL, bold=True)
    draw.rectangle([20, by, 1180, by+90], outline=C_GRID, width=1)
    endorsement_block(draw, 20, by, 580, 90, "一",
                      "贵州黔南钢铁实业有限公司",
                      "成都西部材料交易中心有限公司", "2024-06-20")
    draw.rectangle([600, by, 1180, by+90], outline=C_GRID)
    text(draw, (610, by+38), "（背书栏未使用）", size=13, color=C_WATERMARK)

    draw.rectangle([870, 20, 1180, 55], fill=(255, 245, 210), outline=C_WARN, width=2)
    text(draw, (1025, 37), "⚠ 商承高风险", size=15, color=C_WARN, bold=True, anchor="mm")

    hline(draw, 506, 20, 1180, C_BORDER, 1)
    text(draw, (30, 515), "审核建议：商承票据承兑人列入法院失信被执行人名单，建议追加保证金或要求银行保函后方可受理。", size=13, color=C_WARN)

    img.save(OUTPUT_DIR / "07_高风险_商承承兑人失信.png", dpi=(150, 150))
    print("✓ 07_高风险_商承承兑人失信.png")


# ══════════════════════════════════════════════════════════════════════════════
# 场景 8：支票（超期提示 + 要素缺失）
# ══════════════════════════════════════════════════════════════════════════════
def gen_08_cheque_missing_fields():
    img, draw = new_bill(1000, 620)

    text(draw, (500, 38), "转账支票", size=26, color=C_TITLE, bold=True, anchor="mm")
    text(draw, (500, 62), "TRANSFER CHEQUE", size=12, color=C_LABEL, anchor="mm")
    hline(draw, 76, 20, 980, C_BORDER, 2)

    text(draw, (30, 88), "支票号码：0012 3456 78", size=14, color=C_VALUE, bold=True)
    text(draw, (650, 88), "出票日期：2024-09-01", size=14, color=C_LABEL)
    hline(draw, 108, 20, 980)

    # 金额
    draw.rectangle([20, 114, 980, 165], fill=(252, 248, 235), outline=C_BORDER, width=1)
    text(draw, (30, 122), "金额（大写）：", size=14, color=C_LABEL)
    text(draw, (165, 118), "拾贰万元整", size=28, color=C_AMOUNT, bold=True)
    text(draw, (650, 122), "¥ 120,000.00", size=22, color=C_AMOUNT, bold=True)
    hline(draw, 165, 20, 980, C_BORDER, 1)

    # 字段（部分缺失）
    rows = [
        ("收  款  人", "（未填写）", True),
        ("开户银行", "中国光大银行广州天河支行", False),
        ("出  票  人", "广州市越秀区瑞丰商行", False),
        ("付款期限", "见票即付，10日内有效", False),
    ]
    y = 172
    for label, val, missing in rows:
        bg = (255, 235, 225) if missing else C_BG
        draw.rectangle([20, y, 20+140, y+36], fill=bg, outline=C_GRID, width=1)
        draw.rectangle([160, y, 980, y+36], fill=bg, outline=C_GRID, width=1)
        text(draw, (26, y+18), label, size=14, color=C_LABEL, anchor="lm")
        clr = C_WARN if missing else C_VALUE
        text(draw, (168, y+18), val, size=15, color=clr, anchor="lm")
        if missing:
            text(draw, (600, y+18), "← 必填项缺失", size=13, color=(180, 20, 20), anchor="lm")
        y += 36

    hline(draw, y, 20, 980, C_BORDER, 1)

    # 10日有效期提示
    draw.rectangle([20, y+6, 980, y+50], fill=(255, 240, 220), outline=C_WARN, width=1)
    text(draw, (30, y+16), "⚠ 支票提示付款期限：自出票日起10日内有效（2024-09-01 ~ 2024-09-10）", size=13, color=C_WARN)
    text(draw, (30, y+34), "   当前日期 2024-11-01 已超过提示付款期，该支票已失效，不可再提示付款。", size=13, color=(180, 20, 20))

    # 印章（正常）
    seal_round(draw, 750, 290, 38, "广州市越秀区瑞丰商行预留印鉴", C_SEAL_RED)

    # 标签
    draw.rectangle([730, 20, 980, 55], fill=(255, 220, 220), outline=(180, 20, 20), width=2)
    text(draw, (855, 37), "✗ 超期+要素缺失", size=13, color=(180, 20, 20), bold=True, anchor="mm")

    hline(draw, y+60, 20, 980, C_BORDER, 1)
    text(draw, (30, y+68), "拒绝原因：①收款人字段为空（必填项）；②支票已超过10日提示付款期，依法失效。", size=13, color=C_WARN)

    img.save(OUTPUT_DIR / "08_拒绝_支票超期要素缺失.png", dpi=(150, 150))
    print("✓ 08_拒绝_支票超期要素缺失.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"输出目录：{OUTPUT_DIR}\n")
    gen_01_compliant_bah()
    gen_02_amount_mismatch()
    gen_03_expired()
    gen_04_broken_endorsement()
    gen_05_suspected_fraud()
    gen_06_multi_endorsement()
    gen_07_commercial_high_risk()
    gen_08_cheque_missing_fields()
    print(f"\n完成！共生成 8 张票据图片。")
    print(f"路径：{OUTPUT_DIR}")
