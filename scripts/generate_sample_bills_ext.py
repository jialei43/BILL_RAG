#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sample_bills_ext.py
新增 10 个补全场景的票据图片（场景09-18）
覆盖：EN02/EN03背书闭环重复、FR001重复票号、FR003篡改、
      质押融资、不得转让、合同不匹配、黑名单、出票预检、银行本票
"""
import math, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample_bills"

FONT_PATHS = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
def get_font(size):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

C_BG      = (252,248,235); C_BORDER=(120,80,20);  C_TITLE=(140,20,20)
C_LABEL   = (60,60,80);    C_VALUE=(10,10,10);    C_AMOUNT=(180,20,20)
C_SEAL_R  = (200,30,30);   C_SEAL_B=(30,60,180);  C_GRID=(180,160,120)
C_WARN    = (220,80,0);    C_WMK=(220,200,170);   C_OK=(0,130,0)
C_ERR     = (180,20,20);   C_PURPLE=(120,0,160)

def new_bill(w=1200, h=780):
    img = Image.new("RGB",(w,h),C_BG); d=ImageDraw.Draw(img)
    d.rectangle([4,4,w-5,h-5],outline=C_BORDER,width=3)
    d.rectangle([10,10,w-11,h-11],outline=C_BORDER,width=1)
    return img,d

def txt(d,xy,s,sz=15,c=C_VALUE,a="la"):
    d.text(xy,s,font=get_font(sz),fill=c,anchor=a)

def hl(d,y,x0,x1,c=C_GRID,w=1): d.line([(x0,y),(x1,y)],fill=c,width=w)
def vl(d,x,y0,y1,c=C_GRID,w=1): d.line([(x,y0),(x,y1)],fill=c,width=w)

def hdr(d,w,title,sub=""):
    txt(d,(w//2,38),title,sz=25,c=C_TITLE,a="mm")
    if sub: txt(d,(w//2,64),sub,sz=12,c=C_LABEL,a="mm")
    hl(d,78 if sub else 60,20,w-20,C_BORDER,2)

def amount_row(d,y,big,small,w=1200,warn=False):
    bg=(255,240,235) if warn else (252,248,235)
    d.rectangle([20,y,w-20,y+58],fill=bg,outline=C_BORDER if not warn else C_ERR,width=1+(2 if warn else 0))
    txt(d,(30,y+8),"票面金额（大写）：",sz=13,c=C_LABEL)
    txt(d,(195,y+4),big,sz=27,c=C_AMOUNT)
    txt(d,(w-430,y+8),"小写：¥",sz=13,c=C_LABEL)
    txt(d,(w-360,y+4),small,sz=23,c=C_AMOUNT)

def fields(d,rows,x,y,lw=130,vw=390,h=34):
    for label,val,*opts in rows:
        bg=opts[0] if opts else C_BG
        clr=opts[1] if len(opts)>1 else C_VALUE
        d.rectangle([x,y,x+lw,y+h],fill=bg,outline=C_GRID,width=1)
        d.rectangle([x+lw,y,x+lw+vw,y+h],fill=bg,outline=C_GRID,width=1)
        txt(d,(x+6,y+h//2),label,sz=13,c=C_LABEL,a="lm")
        txt(d,(x+lw+6,y+h//2),val,sz=14,c=clr,a="lm")
        y+=h
    return y

def seal(d,cx,cy,r,label,c=C_SEAL_R):
    d.ellipse([cx-r,cy-r,cx+r,cy+r],outline=c,width=3)
    d.ellipse([cx-r+7,cy-r+7,cx+r-7,cy+r-7],outline=c,width=1)
    pts=[]
    for i in range(5):
        a=math.radians(-90+i*72); b=math.radians(-90+i*72+36)
        pts+= [(cx+13*math.cos(a),cy+13*math.sin(a)),
               (cx+6*math.cos(b), cy+6*math.sin(b))]
    d.polygon(pts,fill=c)
    f=get_font(12)
    n=len(label)
    for i,ch in enumerate(label):
        a=math.radians(-90+(i-(n-1)/2)*(340/max(n,1)))
        d.text((cx+(r-16)*math.cos(a),cy+(r-16)*math.sin(a)),ch,font=f,fill=c,anchor="mm")

def badge(d,w,h,x,y,label,fg,bg):
    d.rectangle([x,y,x+w,y+h],fill=bg,outline=fg,width=2)
    txt(d,(x+w//2,y+h//2),label,sz=14,c=fg,a="mm")

def endb(d,x,y,bw,bh,idx,frm,to,dt,broken=False,dup=False,loop=False):
    bg=C_BG
    fc=C_LABEL
    if broken: bg=(255,245,220); fc=C_WARN
    if dup:    bg=(255,228,228); fc=C_ERR
    if loop:   bg=(255,228,228); fc=C_PURPLE
    d.rectangle([x,y,x+bw,y+bh],fill=bg,outline=C_GRID if not (broken or dup or loop) else fc,width=1+(1 if (broken or dup or loop) else 0))
    f14=get_font(14); f13=get_font(13)
    chs=["一","二","三","四","五","六","七","八","九","十"]
    d.text((x+6,y+6), f"第{chs[idx]}手背书",font=f14,fill=fc)
    d.text((x+6,y+26),f"背书人：{frm}",font=f13,fill=C_VALUE)
    d.text((x+6,y+48),f"被背书人：{to}",font=f13,fill=C_VALUE if not broken else C_WARN)
    d.text((x+6,y+70),f"日期：{dt}",font=f13,fill=C_VALUE)
    if dup:
        d.text((x+6,y+88),"⚠ 该主体已在链中出现过！",font=f13,fill=C_ERR)
    if loop:
        d.text((x+6,y+88),"⚠ 闭环！背书人即原始收款人",font=f13,fill=C_PURPLE)


# ══════════════════════════════════════════════════════════════════════════════
# 09  [背书校验-EN03] 循环背书闭环  A→B→C→A
# ══════════════════════════════════════════════════════════════════════════════
def gen_09():
    W,H=1300,800; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票","BANK ACCEPTANCE BILL")
    txt(d,(30,90),"票据号码：5200 2024 0318 0000 0711",sz=14,c=C_VALUE)
    txt(d,(900,90),"出票日期：2024-03-18",sz=13,c=C_LABEL)
    amount_row(d,104,"壹佰万元整","1,000,000.00",W)
    y=fields(d,[
        ("出  票  人","成都华信科技集团有限公司"),
        ("出票人账号","5101 6601 2233 4455 667"),
        ("出票人开户行","中国民生银行成都高新支行"),
        ("收  款  人","重庆联合贸易发展有限公司（A）"),
    ],20,170,lw=130,vw=440)
    fields(d,[
        ("承  兑  人","中国民生银行成都高新支行"),
        ("承兑人账号","5101 9950 0011 2233 445"),
        ("付 款 行 号","305651000062"),
        ("到  期  日","2024-09-18"),
    ],620,170,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)
    seal(d,390,248,40,"成都华信科技集团有限公司财务章",C_SEAL_R)
    seal(d,870,248,40,"中国民生银行成都高新支行承兑章",C_SEAL_B)

    # 背书区 — 4格，最后一格闭环回A
    by=y+6; bw=(W-40)//4; bh=112
    txt(d,(20,by-22),"背书记录 ——【EN03】背书人在链中形成闭环 A→B→C→A，涉嫌循环融资欺诈",sz=14,c=C_PURPLE)
    parties=["重庆联合贸易发展有限公司","西安鑫源供应链有限公司","武汉恒达物资贸易有限公司","重庆联合贸易发展有限公司"]
    dates  =["2024-05-10","2024-06-20","2024-07-30","2024-08-25"]
    for i in range(4):
        is_loop=(i==3)
        endb(d,20+i*bw,by,bw,bh,i,parties[i],parties[(i+1)%4],dates[i],loop=is_loop)
    # 闭环箭头
    ax=20+4*bw-10; ay=by+bh//2
    d.line([(ax,ay),(ax+18,ay)],fill=C_PURPLE,width=3)
    d.polygon([(ax+18,ay-8),(ax+28,ay),(ax+18,ay+8)],fill=C_PURPLE)
    d.arc([20-28,by-28,20+28,by+bh+28],-60,60,fill=C_PURPLE,width=3)
    txt(d,(W-60,by+bh+8),"↑ 闭环",sz=13,c=C_PURPLE)

    badge(d,220,36,W-240,20,"✗ EN03 背书闭环 冻结上报",C_PURPLE,(240,225,255))
    hl(d,by+bh+50,20,W-20,C_BORDER)
    txt(d,(30,by+bh+58),"错误码：EN03 | 风险等级：CRITICAL（< 30分）| 处置：立即冻结票据，保全证据，移交合规部门，同步上报人行。",sz=13,c=C_PURPLE)
    img.save(OUTPUT_DIR/"09_[背书校验-EN03]_银承_循环背书A→B→C→A闭环_冻结上报.png",dpi=(150,150))
    print("✓ 09")


# ══════════════════════════════════════════════════════════════════════════════
# 10  [背书校验-EN02] 重复背书 — 同一主体在链中二次出现
# ══════════════════════════════════════════════════════════════════════════════
def gen_10():
    W,H=1300,800; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票")
    txt(d,(30,68),"票据号码：3100 2024 0425 0000 0823",sz=14,c=C_VALUE)
    txt(d,(900,68),"出票日期：2024-04-25",sz=13,c=C_LABEL)
    amount_row(d,82,"贰佰伍拾万元整","2,500,000.00",W)
    y=fields(d,[
        ("出  票  人","南京嘉华电子制造有限公司"),
        ("出票人账号","3201 4401 9900 8877 665"),
        ("出票人开户行","兴业银行南京建邺支行"),
        ("收  款  人","上海联华商贸集团有限公司"),
    ],20,148,lw=130,vw=430)
    fields(d,[
        ("承  兑  人","兴业银行南京建邺支行"),
        ("承兑人账号","3201 9920 0011 3344 556"),
        ("付 款 行 号","309332003010"),
        ("到  期  日","2024-10-25"),
    ],620,148,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)
    seal(d,380,228,38,"南京嘉华电子制造有限公司财务章",C_SEAL_R)
    seal(d,860,228,38,"兴业银行南京建邺支行承兑专用章",C_SEAL_B)

    by=y+6; bw=(W-40)//4; bh=115
    txt(d,(20,by-22),"背书记录 ——【EN02】「上海联华商贸」第1手背书后，第3手被背书人再次指向同一主体，涉嫌重复融资",sz=13,c=C_ERR)
    rows=[
        ("上海联华商贸集团有限公司","北京恒信科技发展有限公司","2024-06-12",False,False),
        ("北京恒信科技发展有限公司","上海联华商贸集团有限公司","2024-07-18",False,True),# dup
        ("上海联华商贸集团有限公司","广州汇通金融服务有限公司","2024-08-05",False,True),# dup
        ("广州汇通金融服务有限公司","（当前持票方）","—",False,False),
    ]
    for i,(frm,to,dt,broken,dup) in enumerate(rows):
        endb(d,20+i*bw,by,bw,bh,i,frm,to,dt,broken,dup)
    # 标注第2手
    d.rectangle([20+bw,by-2,20+2*bw,by+bh+2],outline=C_ERR,width=3)
    txt(d,(20+bw+bw//2,by+bh+8),"↑ 同一主体二次出现",sz=12,c=C_ERR,a="mm")

    badge(d,200,36,W-220,20,"⚠ EN02 重复背书 退回核查",C_ERR,(255,230,230))
    hl(d,by+bh+50,20,W-20,C_BORDER)
    txt(d,(30,by+bh+58),"错误码：EN02 | 同一主体「上海联华商贸」在背书链第1、3手两次出现，疑似虚假贸易背景循环融资，需彻查资金流向。",sz=13,c=C_ERR)
    img.save(OUTPUT_DIR/"10_[背书校验-EN02]_银承_重复背书同一主体二次出现_退回核查.png",dpi=(150,150))
    print("✓ 10")


# ══════════════════════════════════════════════════════════════════════════════
# 11  [欺诈检测-FR001] 重复票号 双重贴现嫌疑
# ══════════════════════════════════════════════════════════════════════════════
def gen_11():
    W,H=1300,820; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票  ⚠ 重复票据警报","DUPLICATE BILL ALERT")
    txt(d,(30,90),"票据号码：4400 2024 0601 0000 0456",sz=14,c=C_VALUE)
    txt(d,(900,90),"出票日期：2024-06-01",sz=13,c=C_LABEL)
    # 红色警告框覆盖票号
    d.rectangle([20,82,780,108],outline=C_ERR,width=3)
    txt(d,(790,90),"← FR001：该票号已存在！",sz=14,c=C_ERR)
    amount_row(d,114,"叁佰万元整","3,000,000.00",W)

    # 两列对比：本次提交 vs 已入库记录
    txt(d,(W//4,180),"【本次提交票据】",sz=15,c=C_LABEL,a="mm")
    txt(d,(3*W//4,180),"【系统已入库记录（2024-06-05 贴现）】",sz=15,c=C_ERR,a="mm")
    vl(d,W//2,170,500,C_ERR,2)
    hl(d,195,20,W-20,C_GRID)
    left_rows=[
        ("出  票  人","广州银泰贸易有限公司"),
        ("出票人开户行","中信银行广州花都支行"),
        ("收  款  人","深圳华裕供应链管理有限公司"),
        ("到  期  日","2024-12-01"),
        ("申请业务","贴现申请，贴现率 3.2%"),
    ]
    right_rows=[
        ("出  票  人","广州银泰贸易有限公司"),
        ("出票人开户行","中信银行广州花都支行"),
        ("收  款  人","深圳华裕供应链管理有限公司"),
        ("到  期  日","2024-12-01"),
        ("历史状态","2024-06-05 已办理贴现 ✓"),
    ]
    y=202
    for (ll,lv),(rl,rv) in zip(left_rows,right_rows):
        d.rectangle([20,y,W//2-5,y+34],outline=C_GRID,width=1)
        d.rectangle([W//2+5,y,W-20,y+34],fill=(255,225,225),outline=C_ERR,width=1)
        txt(d,(26,y+17),f"{ll}：{lv}",sz=13,c=C_VALUE,a="lm")
        txt(d,(W//2+11,y+17),f"{rl}：{rv}",sz=13,c=C_ERR,a="lm")
        y+=34
    hl(d,y+4,20,W-20,C_BORDER)

    # 欺诈评分
    d.rectangle([20,y+12,W-20,y+72],fill=(255,230,230),outline=C_ERR,width=2)
    txt(d,(30,y+18),"FR001 欺诈检测结论：",sz=15,c=C_ERR)
    txt(d,(30,y+38),"• 重复票据命中，强制欺诈评分 = 1.0（最高级别）",sz=14,c=C_ERR)
    txt(d,(30,y+56),"• 该票号于 2024-06-05 已在本行办理贴现，疑似双重贴现欺诈，立即冻结并上报合规部。",sz=13,c=C_ERR)

    badge(d,220,36,W-240,20,"✗ FR001 重复票据 立即冻结",C_ERR,(255,210,210))
    img.save(OUTPUT_DIR/"11_[欺诈检测-FR001]_银承_重复票号双重贴现嫌疑_冻结上报.png",dpi=(150,150))
    print("✓ 11")


# ══════════════════════════════════════════════════════════════════════════════
# 12  [质押融资] 质押背书 — 未在人行系统登记
# ══════════════════════════════════════════════════════════════════════════════
def gen_12():
    W,H=1200,780; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票（质押融资申请）")
    txt(d,(30,68),"票据号码：3300 2024 0712 0000 0334",sz=14,c=C_VALUE)
    txt(d,(850,68),"出票日期：2024-07-12",sz=13,c=C_LABEL)
    amount_row(d,82,"伍佰万元整","5,000,000.00",W)
    y=fields(d,[
        ("出  票  人","苏州精工机械集团有限公司"),
        ("出票人账号","3205 8801 2234 5566 778"),
        ("出票人开户行","农业银行苏州工业园区支行"),
        ("收  款  人","无锡新兴铸管有限公司"),
    ],20,148,lw=130,vw=410)
    fields(d,[
        ("承  兑  人","农业银行苏州工业园区支行"),
        ("承兑人账号","3205 9030 0012 3456 789"),
        ("付 款 行 号","103305002055"),
        ("到  期  日","2025-01-12"),
    ],590,148,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)
    seal(d,370,228,38,"苏州精工机械集团有限公司财务章",C_SEAL_R)
    seal(d,840,228,38,"农业银行苏州工业园区支行承兑章",C_SEAL_B)

    # 质押背书区
    by=y+8
    txt(d,(20,by-22),"质押背书记录",sz=15,c=C_LABEL)
    bw=(W-40)//2; bh=110
    # 第1手：正常转让背书
    endb(d,20,by,bw,bh,0,"无锡新兴铸管有限公司","无锡科创供应链金融有限公司","2024-09-01")
    seal(d,20+bw//2,by+bh+20,36,"无锡新兴铸管有限公司财务章",C_SEAL_R)
    # 第2手：质押背书（有问题）
    d.rectangle([20+bw,by,20+2*bw,by+bh],fill=(255,245,210),outline=C_WARN,width=2)
    f14=get_font(14); f13=get_font(13)
    d.text((20+bw+6,by+6),"质押背书（第一手）",font=f14,fill=C_WARN)
    d.text((20+bw+6,by+28),f"出质人：无锡科创供应链金融有限公司",font=f13,fill=C_VALUE)
    d.text((20+bw+6,by+48),f"质权人：招商银行无锡分行（质押贷款担保）",font=f13,fill=C_VALUE)
    d.text((20+bw+6,by+68),f"质押金额：500万元  贷款期限：2024-12-31",font=f13,fill=C_VALUE)
    d.text((20+bw+6,by+88),'注记："质押"',font=f13,fill=C_WARN)

    # 核查结论框
    cy=by+bh+45
    d.rectangle([20,cy,W-20,cy+95],fill=(255,248,210),outline=C_WARN,width=2)
    txt(d,(28,cy+8),"质押融资合规核查",sz=15,c=C_WARN)
    items=[
        ("✓","质押背书格式","已注明【质押】字样，格式合规","(0,130,0)"),
        ("✗","人行质押登记","未在中国人民银行票据质押登记系统登记，质押权不完善",'C_ERR'),
        ("✓","到期日校验","票据到期日2025-01-12 > 贷款到期日2024-12-31，合规","(0,130,0)"),
        ("⚠","质权人信息","质权人与贷款合同一致，但需补充质押登记回执",'C_WARN'),
    ]
    ry=cy+28
    for icon,name,desc,clr_str in items:
        clr=eval(clr_str) if clr_str.startswith("(") else globals()[clr_str]
        txt(d,(28,ry),f"{icon} {name}：{desc}",sz=13,c=clr)
        ry+=17

    badge(d,200,36,W-220,20,"⚠ 质押未登记 拒绝融资",C_WARN,(255,248,200))
    img.save(OUTPUT_DIR/"12_[质押融资]_银承_质押背书未在人行系统登记_拒绝融资.png",dpi=(150,150))
    print("✓ 12")


# ══════════════════════════════════════════════════════════════════════════════
# 13  [背书转让-E018] 票面注明"不得转让" 但后续仍有背书
# ══════════════════════════════════════════════════════════════════════════════
def gen_13():
    W,H=1200,780; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票")
    txt(d,(30,68),"票据号码：1100 2024 0308 0000 0188",sz=14,c=C_VALUE)
    txt(d,(850,68),"出票日期：2024-03-08",sz=13,c=C_LABEL)
    amount_row(d,82,"捌拾万元整","800,000.00",W)
    # 不得转让标记框
    d.rectangle([20,82,680,140],outline=C_ERR,width=3)
    d.rectangle([685,82,W-20,140],fill=(255,220,220),outline=C_ERR,width=2)
    txt(d,(695,95),"⚠ E018：出票人已注明",sz=13,c=C_ERR)
    txt(d,(695,115),'   "不得转让"，后续全部背书无效！',sz=13,c=C_ERR)

    y=fields(d,[
        ("出  票  人","北京融汇投资管理有限公司"),
        ("出票人账号","1101 0100 2233 4455 667"),
        ("出票人开户行","光大银行北京朝阳支行"),
        ("收  款  人","天津优质钢铁贸易有限公司"),
        ("特殊记载事项",'【不 得 转 让】',C_BG,C_ERR),
    ],20,148,lw=130,vw=410)
    fields(d,[
        ("承  兑  人","光大银行北京朝阳支行"),
        ("承兑人账号","1101 9650 0011 2233 445"),
        ("付 款 行 号","303100010014"),
        ("到  期  日","2024-09-08"),
        ("备注","票据法第11条：注明不得转让后"),
    ],590,148,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)

    # 大字水印
    fw=get_font(60)
    d.text((W//2,280),"不  得  转  让",font=fw,fill=(230,180,180,60),anchor="mm")

    seal(d,370,248,38,"北京融汇投资管理有限公司财务章",C_SEAL_R)
    seal(d,840,248,38,"光大银行北京朝阳支行承兑专用章",C_SEAL_B)

    # 背书区 — 仍有2次背书（无效）
    by=y+6; bw=(W-40)//3; bh=100
    txt(d,(20,by-22),"背书记录 ——【全部无效】出票人注明「不得转让」，以下所有背书均无法律效力",sz=13,c=C_ERR)
    invalid_rows=[
        ("天津优质钢铁贸易有限公司","河北承德矿业物资有限公司","2024-05-20"),
        ("河北承德矿业物资有限公司","内蒙古包钢集团贸易有限公司","2024-07-10"),
    ]
    for i,(frm,to,dt) in enumerate(invalid_rows):
        d.rectangle([20+i*bw,by,20+(i+1)*bw,by+bh],fill=(255,225,225),outline=C_ERR,width=2)
        f14=get_font(14); f13=get_font(13)
        d.text((20+i*bw+6,by+6),f"第{['一','二'][i]}手背书【无效】",font=f14,fill=C_ERR)
        d.text((20+i*bw+6,by+26),f"背书人：{frm}",font=f13,fill=C_VALUE)
        d.text((20+i*bw+6,by+46),f"被背书人：{to}",font=f13,fill=C_VALUE)
        d.text((20+i*bw+6,by+66),f"日期：{dt}",font=f13,fill=C_VALUE)
        d.text((20+i*bw+6,by+86),"× 依票据法第11条，此背书无效",font=f13,fill=C_ERR)
    d.rectangle([20+2*bw,by,W-20,by+bh],outline=C_GRID)
    txt(d,(20+2*bw+8,by+40),"（未使用）",sz=13,c=C_WMK)

    badge(d,220,36,W-240,20,"✗ E018 不得转让背书无效",C_ERR,(255,220,220))
    hl(d,by+bh+30,20,W-20,C_BORDER)
    txt(d,(30,by+bh+38),"法规依据：《票据法》第11条 — 出票人在汇票上记载「不得转让」字样，则汇票不得转让，后手背书对被背书人不产生效力。",sz=13,c=C_ERR)
    img.save(OUTPUT_DIR/"13_[背书转让-E018]_银承_不得转让标记后仍有两次背书_背书无效.png",dpi=(150,150))
    print("✓ 13")


# ══════════════════════════════════════════════════════════════════════════════
# 14  [合同审核-CT] 票据金额超合同标的 32%（远超5%容差）
# ══════════════════════════════════════════════════════════════════════════════
def gen_14():
    W,H=1200,800; img,d=new_bill(W,H)
    hdr(d,W,"商业承兑汇票（含贸易背景核查）")
    txt(d,(30,68),"票据号码：5000 2024 0510 0000 0267",sz=14,c=C_VALUE)
    txt(d,(850,68),"出票日期：2024-05-10",sz=13,c=C_LABEL)
    amount_row(d,82,"叁佰玖拾捌万元整","3,980,000.00",W)
    y=fields(d,[
        ("出  票  人","重庆龙腾建材供应有限公司"),
        ("出票人账号","5001 4401 7788 9900 112"),
        ("出票人开户行","渝农商行重庆江北支行"),
        ("收  款  人","贵州黔建钢材贸易有限公司"),
    ],20,148,lw=130,vw=410)
    fields(d,[
        ("承  兑  人（商承）","重庆龙腾建材供应有限公司"),
        ("承兑人账号","5001 4401 7788 9900 112"),
        ("付 款 行 号","402653000088"),
        ("到  期  日","2024-11-10"),
    ],590,148,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)
    seal(d,370,228,38,"重庆龙腾建材供应有限公司财务章",C_SEAL_R)
    seal(d,830,228,38,"重庆龙腾建材供应有限公司承兑专用章",C_SEAL_R)

    # 合同比对区
    cy=y+8
    txt(d,(20,cy-22),"合同要素比对（CT系列校验）",sz=15,c=C_LABEL)
    d.rectangle([20,cy,W-20,cy+220],outline=C_GRID,width=1)
    # 表头
    cols=[("校验项",160),("票据要素",280),("合同要素",280),("偏差",180),("结论",230)]
    cx=20
    for name,cw in cols:
        d.rectangle([cx,cy,cx+cw,cy+30],fill=(230,230,250),outline=C_GRID,width=1)
        txt(d,(cx+cw//2,cy+15),name,sz=13,c=C_LABEL,a="mm")
        cx+=cw
    rows_ct=[
        ("合同编号","（无合同编号）","LTHG-2024-0508","必须提供合同","⚠ WARNING"),
        ("合同金额","¥3,980,000","¥3,018,000","超出 +32%（>5%）","✗ SEVERE"),
        ("买方/出票人","重庆龙腾建材供应有限公司","重庆龙腾建材供应有限公司","完全一致","✓ PASS"),
        ("卖方/收款人","贵州黔建钢材贸易有限公司","贵州黔建钢材贸易有限公司","完全一致","✓ PASS"),
        ("付款期限","2024-11-10","2024-11-20","票据10日提前","✓ PASS"),
        ("贸易货物","（未填写）","螺纹钢筋 HRB400E","贸易背景描述缺失","⚠ WARNING"),
    ]
    ry=cy+30
    for item in rows_ct:
        cx=20
        bgs=[C_BG,C_BG,C_BG,
             (255,220,220) if "✗" in item[4] else (255,248,220) if "⚠" in item[4] else (220,255,220),
             (255,220,220) if "✗" in item[4] else (255,248,220) if "⚠" in item[4] else (220,255,220)]
        clrs=[C_VALUE,C_VALUE,C_VALUE,
              C_ERR if "✗" in item[4] else C_WARN if "⚠" in item[4] else C_OK,
              C_ERR if "✗" in item[4] else C_WARN if "⚠" in item[4] else C_OK]
        for i,(val,(name,cw)) in enumerate(zip(item,cols)):
            d.rectangle([cx,ry,cx+cw,ry+30],fill=bgs[i],outline=C_GRID,width=1)
            txt(d,(cx+4,ry+15),val,sz=12,c=clrs[i],a="lm")
            cx+=cw
        ry+=30

    badge(d,220,36,W-240,20,"✗ CT 合同金额不匹配 退回核实",C_ERR,(255,225,225))
    hl(d,cy+220+10,20,W-20,C_BORDER)
    txt(d,(30,cy+230),"CT校验结论：票据金额（398万）超出合同标的金额（301.8万）32.2%，远超±5%容差，疑似虚增金额套现，需提供真实贸易合同并补充增值税发票。",sz=13,c=C_ERR)
    img.save(OUTPUT_DIR/"14_[合同审核-CT]_商承_票据金额超合同标的32%无贸易背景_退回核实.png",dpi=(150,150))
    print("✓ 14")


# ══════════════════════════════════════════════════════════════════════════════
# 15  [欺诈检测-FR003] 图像篡改 — 金额区域涂改痕迹
# ══════════════════════════════════════════════════════════════════════════════
def gen_15():
    W,H=1200,780; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票")
    txt(d,(30,68),"票据号码：6200 2024 0622 0000 0901",sz=14,c=C_VALUE)
    txt(d,(850,68),"出票日期：2024-06-22",sz=13,c=C_LABEL)

    # 金额区 — 涂改痕迹展示
    d.rectangle([20,82,W-20,148],fill=(252,248,235),outline=C_BORDER,width=1)
    txt(d,(30,90),"票面金额（大写）：",sz=13,c=C_LABEL)
    # 原始金额（被涂改）
    txt(d,(195,86),"伍拾万元整",sz=26,c=(200,190,180))  # 浅色表示原始
    # 涂改覆盖区域
    d.rectangle([190,84,490,130],fill=(245,238,220),outline=C_WARN,width=0)
    for i in range(6):  # 模拟涂改线条
        d.line([(195+i*12,88),(195+i*12,126)],fill=(180,160,130,120),width=2)
    # 改写的新金额
    txt(d,(195,86),"伍佰万元整",sz=26,c=C_VALUE)
    d.rectangle([190,82,495,132],outline=C_ERR,width=2)
    txt(d,(500,96),"← 涂改区域",sz=13,c=C_ERR)
    txt(d,(500,114),"   FR003：图像分析检测到叠写痕迹",sz=12,c=C_WARN)
    txt(d,(820,90),"小写：¥",sz=13,c=C_LABEL)
    txt(d,(880,86),"5,000,000.00",sz=23,c=C_AMOUNT)
    # 小写也标记异常
    d.rectangle([876,84,W-25,132],outline=C_ERR,width=2)
    txt(d,(W-200,116),"← 与原始不符",sz=12,c=C_ERR)

    y=fields(d,[
        ("出  票  人","西安博远能源技术有限公司"),
        ("出票人账号","6101 6601 2233 4455 667"),
        ("出票人开户行","华夏银行西安经开区支行"),
        ("收  款  人","兰州新区装备制造有限公司"),
    ],20,156,lw=130,vw=410)
    fields(d,[
        ("承  兑  人","华夏银行西安经开区支行"),
        ("承兑人账号","6101 9920 0010 9988 776"),
        ("付 款 行 号","304791001044"),
        ("到  期  日","2024-12-22"),
    ],590,156,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)
    seal(d,370,238,38,"西安博远能源技术有限公司财务章",C_SEAL_R)
    seal(d,840,238,38,"华夏银行西安经开区支行承兑章",C_SEAL_B)

    # FR003 检测结果
    fy=y+8
    d.rectangle([20,fy,W-20,fy+130],fill=(255,235,225),outline=C_ERR,width=2)
    txt(d,(28,fy+8),"FR003 图像篡改检测报告",sz=15,c=C_ERR)
    detect_items=[
        ("墨迹分析","金额区域检测到多层墨迹叠加，存在覆写痕迹","FAIL"),
        ("笔迹压痕","正面金额区背面压痕与正面金额不一致（50万 vs 500万）","FAIL"),
        ("字体一致性","金额数字字体与其他区域不一致（疑似后期添加）","WARN"),
        ("EXIF元数据","原件扫描时间与提交时间差异7天，中间存在PS编辑痕迹","FAIL"),
    ]
    for i,(name,desc,level) in enumerate(detect_items):
        c=C_ERR if level=="FAIL" else C_WARN
        sym="✗" if level=="FAIL" else "⚠"
        txt(d,(28,fy+28+i*22),f"{sym} {name}：{desc}",sz=13,c=c)

    badge(d,220,36,W-240,20,"✗ FR003 图像篡改 冻结上报",C_ERR,(255,210,210))
    img.save(OUTPUT_DIR/"15_[欺诈检测-FR003]_银承_金额区域涂改篡改痕迹_冻结上报.png",dpi=(150,150))
    print("✓ 15")


# ══════════════════════════════════════════════════════════════════════════════
# 16  [黑名单-BL001] 出票人命中监管黑名单
# ══════════════════════════════════════════════════════════════════════════════
def gen_16():
    W,H=1200,760; img,d=new_bill(W,H)
    hdr(d,W,"商业承兑汇票")
    txt(d,(30,68),"票据号码：4100 2024 0715 0000 0149",sz=14,c=C_VALUE)
    txt(d,(850,68),"出票日期：2024-07-15",sz=13,c=C_LABEL)
    amount_row(d,82,"壹佰贰拾万元整","1,200,000.00",W)

    # 黑名单命中警报
    d.rectangle([20,82,W-20,82+58],fill=(255,210,210),outline=C_ERR,width=3)
    txt(d,(W//2,82+14),"【 BL001 黑名单命中警报 】出票人已列入金融监管黑名单，本票据请求被强制拒绝",sz=14,c=C_ERR,a="mm")
    txt(d,(W//2,82+36),"拦截时间：2024-07-15 14:32:08  |  匹配置信度：98.7%  |  匹配类型：精确匹配",sz=12,c=(140,40,40),a="mm")

    y=fields(d,[
        ("出  票  人","郑州鑫瑞矿产资源开发有限公司",(255,220,220),C_ERR),
        ("出票人账号","4101 5501 8899 7766 554"),
        ("出票人开户行","中原银行郑州金水支行"),
        ("收  款  人","洛阳汇通物资贸易有限公司"),
    ],20,148,lw=130,vw=410)
    fields(d,[
        ("承  兑  人（商承）","郑州鑫瑞矿产资源开发有限公司",(255,220,220),C_ERR),
        ("承兑人账号","4101 5501 8899 7766 554"),
        ("付 款 行 号","402491001099"),
        ("到  期  日","2025-01-15"),
    ],590,148,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)

    # 黑名单详情
    by=y+8
    d.rectangle([20,by,W-20,by+185],fill=(255,228,228),outline=C_ERR,width=2)
    txt(d,(28,by+8),"BL001 黑名单主体信息",sz=15,c=C_ERR)
    bl_rows=[
        ("主体名称","郑州鑫瑞矿产资源开发有限公司","统一社会信用代码","91410105XXXXXXXX25"),
        ("加入时间","2023-11-20","来源机构","中国人民银行郑州支行"),
        ("违规类型","票据欺诈（历史双重贴现2次）","处罚决定号","豫银罚决字[2023]第088号"),
        ("历史拒付","近2年拒付记录：3次，累计金额580万元","当前状态","黑名单有效（永久）"),
        ("关联主体","法定代表人王某某同时控制另外2家黑名单企业","操作建议","立即拒绝，上报人行"),
    ]
    ry=by+28
    for k1,v1,k2,v2 in bl_rows:
        txt(d,(28,ry),f"【{k1}】{v1}",sz=13,c=C_ERR)
        txt(d,(W//2+10,ry),f"【{k2}】{v2}",sz=13,c=C_ERR)
        ry+=28

    badge(d,200,36,W-220,20,"✗ BL001 黑名单 拒绝受理",C_ERR,(255,200,200))
    hl(d,by+185+6,20,W-20,C_BORDER)
    txt(d,(30,by+192),"系统处置：票据请求已自动拦截，无法进入人工审核环节。如有异议，请联系合规部门（内线8888）进行申诉核实。",sz=13,c=C_ERR)
    img.save(OUTPUT_DIR/"16_[黑名单-BL001]_商承_出票人命中金融监管黑名单_拒绝受理.png",dpi=(150,150))
    print("✓ 16")


# ══════════════════════════════════════════════════════════════════════════════
# 17  [出票预检] 营业执照被吊销 + 授信额度不足
# ══════════════════════════════════════════════════════════════════════════════
def gen_17():
    W,H=1200,800; img,d=new_bill(W,H)
    hdr(d,W,"银行承兑汇票  出票合规前置预检报告","BILL ISSUANCE PRE-CHECK REPORT")
    txt(d,(30,90),"申请票号：（待生成）",sz=13,c=C_LABEL)
    txt(d,(400,90),"申请时间：2024-08-20 09:15:32",sz=13,c=C_LABEL)
    txt(d,(800,90),"预检SLA：≤20秒  实耗：4.2秒",sz=13,c=C_OK)
    amount_row(d,104,"贰千万元整","20,000,000.00",W)
    txt(d,(W-360,112),"← 超出授信额度！",sz=14,c=C_ERR)

    y=fields(d,[
        ("申请出票人","合肥领创半导体科技有限公司",(255,220,220),C_ERR),
        ("统一信用代码","91340104XXXXXXXX78"),
        ("出票人开户行","徽商银行合肥高新支行"),
        ("拟收款方","南京英伟达元器件代理有限公司"),
    ],20,170,lw=130,vw=410)
    fields(d,[
        ("申请承兑行","徽商银行合肥高新支行"),
        ("拟出票日期","2024-08-20"),
        ("拟到期日期","2025-02-20"),
        ("申请金额","¥20,000,000.00",(255,220,220),C_ERR),
    ],590,170,lw=130,vw=440)
    hl(d,y,20,W-20,C_BORDER)

    # 18项预检结果
    py=y+8
    txt(d,(20,py-22),"出票要素预检结果（18项）",sz=15,c=C_LABEL)
    items=[
        ("营业执照有效性","查询工商接口","营业执照已于2024-07-01被吊销","✗","FAIL"),
        ("法定代表人状态","行内KYC核查","法人李某某未见异常","✓","PASS"),
        ("授信额度核查","行内核心授信系统","可用授信额度：¥800万，申请¥2000万，超出¥1200万","✗","FAIL"),
        ("单张金额上限","监管规定","银票单张无上限，金额合规","✓","PASS"),
        ("历史拒付记录","人行征信+行内黑名单","近2年无拒付记录","✓","PASS"),
        ("出票日期格式","规则引擎","格式正确，非未来日期","✓","PASS"),
        ("到期日期合规","规则引擎","期限184天，≤365天合规","✓","PASS"),
        ("金额大小写预验","规则引擎","大写「贰千万」与小写「20,000,000」一致","✓","PASS"),
        ("承兑行资质","金融机构名单","徽商银行为合法持牌金融机构","✓","PASS"),
        ("收款方工商状态","工商接口","南京英伟达元器件代理有限公司正常存续","✓","PASS"),
    ]
    ry=py; col_w=[200,200,440,40,80]
    # 表头
    cx=20
    for title,cw in zip(["检查项","数据来源","检查结果","","结论"],col_w):
        d.rectangle([cx,ry,cx+cw,ry+28],fill=(230,230,250),outline=C_GRID,width=1)
        txt(d,(cx+4,ry+14),title,sz=13,c=C_LABEL,a="lm")
        cx+=cw
    ry+=28
    for row in items:
        cx=20
        c=C_ERR if row[4]=="FAIL" else C_OK
        bg_row=(255,228,228) if row[4]=="FAIL" else C_BG
        for i,(val,cw) in enumerate(zip(row,col_w)):
            d.rectangle([cx,ry,cx+cw,ry+26],fill=bg_row,outline=C_GRID,width=1)
            txt(d,(cx+4,ry+13),val,sz=12,c=c if i>=3 else C_VALUE,a="lm")
            cx+=cw
        ry+=26

    badge(d,240,36,W-260,20,"✗ 预检不通过 禁止出票",C_ERR,(255,210,210))
    hl(d,ry+4,20,W-20,C_BORDER)
    txt(d,(30,ry+10),"预检结论：2项FAIL阻断出票。整改要求：①提供重新登记的有效营业执照；②降低申请金额至800万以内，或申请临时授信增额。",sz=13,c=C_ERR)
    img.save(OUTPUT_DIR/"17_[出票预检]_银承_营业执照吊销授信额度不足_预检不通过.png",dpi=(150,150))
    print("✓ 17")


# ══════════════════════════════════════════════════════════════════════════════
# 18  [贴现审核] 银行本票 — 全要素合规 见票即付
# ══════════════════════════════════════════════════════════════════════════════
def gen_18():
    W,H=1100,700; img,d=new_bill(W,H)
    hdr(d,W,"银行本票","BANKER'S NOTE / CASHIER'S CHECK")
    txt(d,(30,90),"本票号码：0110 2024 0901 0000 0066",sz=14,c=C_VALUE)
    txt(d,(750,90),"出票日期：2024-09-01",sz=13,c=C_LABEL)
    hl(d,108,20,W-20,C_GRID)

    # 金额
    d.rectangle([20,114,W-20,168],fill=(252,248,235),outline=C_BORDER,width=1)
    txt(d,(30,122),"出票金额（大写）：",sz=13,c=C_LABEL)
    txt(d,(195,118),"捌拾伍万元整",sz=27,c=C_AMOUNT)
    txt(d,(720,122),"¥ 850,000.00",sz=23,c=C_AMOUNT)

    rows_l=[
        ("出  票  行","中国工商银行上海浦东支行（出票人）"),
        ("出票行账号","1000 1234 5678 9012 345"),
        ("出 款 地","上海市浦东新区"),
        ("收  款  人","上海宝尊电商服务有限公司"),
        ("收款人账号","3100 5599 8877 6655 443"),
    ]
    rows_r=[
        ("票据类型","银行本票（无条件付款承诺）"),
        ("付款期限","见票即付（出票日起2个月内有效）"),
        ("有效截止日","2024-11-01"),
        ("币    种","人民币"),
        ("贴现说明","可向任意银行申请贴现"),
    ]
    y=fields(d,rows_l,20,176,lw=130,vw=420,h=34)
    fields(d,rows_r,580,176,lw=130,vw=360,h=34)
    hl(d,y,20,W-20,C_BORDER)

    # 银行本票特有条款
    txt(d,(30,y+8),"无条件付款承诺：",sz=14,c=C_LABEL)
    txt(d,(30,y+28),"本行于出票日起见票后无条件向收款人支付本票所记载金额，本承诺不可撤销。",sz=14,c=C_VALUE)
    txt(d,(30,y+50),"本票由出票行（中国工商银行）承担付款责任，信用等级最高，可直接在同城任意工行网点提示付款。",sz=13,c=C_LABEL)
    hl(d,y+70,20,W-20,C_GRID)

    # 工行大印章
    seal(d,W//2,y+120,52,"中国工商银行上海浦东支行出票专用章",C_SEAL_R)

    # 本票特征说明框
    fy=y+178
    d.rectangle([20,fy,W-20,fy+100],fill=(240,255,240),outline=C_OK,width=2)
    txt(d,(28,fy+8),"银行本票核查要点（vs 银行承兑汇票差异）",sz=14,c=C_OK)
    features=[
        "① 出票人即付款人（均为银行），无需单独承兑手续",
        "② 付款方式：见票即付，无到期日概念，有效期2个月",
        "③ 信用最高：等同于现金，无违约风险，可在同城任意网点兑付",
        "④ 不可背书转让（本票仅限收款人提示付款，不进入流转市场）",
    ]
    for i,f in enumerate(features):
        txt(d,(28,fy+28+i*18),f,sz=13,c=C_OK)

    badge(d,180,36,W-200,20,"✓ 银行本票 合规通过",C_OK,(220,255,220))
    img.save(OUTPUT_DIR/"18_[贴现审核]_银行本票_见票即付全要素合规_通过.png",dpi=(150,150))
    print("✓ 18")


if __name__=="__main__":
    print(f"输出目录：{OUTPUT_DIR}\n")
    gen_09(); gen_10(); gen_11(); gen_12(); gen_13()
    gen_14(); gen_15(); gen_16(); gen_17(); gen_18()
    print(f"\n完成！新增 10 张，共 18 张测试票据。")
