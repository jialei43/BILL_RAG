# app/services/scene_handlers.py
# 8 个专项检索场景处理器（需求.md 场景一~八）
#
# 每个处理器的职责：
#   1. 接收 BillElement 字段（bill_context）和用户原始 query
#   2. 构造 1~N 条专项 RAG 查询（比通用查询更精准）
#   3. 返回：{ "queries": [...], "context_hint": "..." }
#      context_hint 注入 LLM prompt，告诉模型当前是什么专项场景
#
# 实际的向量检索和 LLM 生成统一由 RAGService.query_v2() 调用，
# 此模块只负责"如何构造查询"这一层，保持职责单一。

from datetime import date, datetime
from typing import Optional


def _days_remaining(due_date_str: Optional[str]) -> Optional[int]:
    """计算距离到期日的天数（负数表示已逾期）。"""
    if not due_date_str:
        return None
    try:
        due = datetime.strptime(due_date_str[:10], "%Y-%m-%d").date()
        return (due - date.today()).days
    except ValueError:
        return None


def _period_label(days: int) -> str:
    """将天数转换为期限描述（用于检索 query 精确匹配分档）。"""
    if days <= 30:
        return "1个月以内"
    elif days <= 90:
        return "3个月以内"
    elif days <= 180:
        return "6个月以内"
    elif days <= 365:
        return "1年以内"
    else:
        return "1年以上"


def _amount_tier(amount: Optional[float]) -> str:
    """将金额转换为档位描述（用于检索合规报告门槛）。"""
    if not amount:
        return ""
    if amount >= 10_000_000:
        return "千万以上"
    elif amount >= 5_000_000:
        return "500万以上"
    elif amount >= 1_000_000:
        return "百万以上"
    return "百万以下"


# ── 场景一：承兑人资信联查 × 贴现定价辅助 ────────────────────────────────────
def build_pricing_queries(ctx: dict) -> dict:
    """
    INTENT_PRICING：根据承兑人、期限、金额构造贴现定价专项查询。
    并行 2 条：资信查询 + 市场利率查询。
    """
    acceptor = ctx.get("acceptor", "")
    ticket_type = ctx.get("ticket_type", "")
    days = _days_remaining(ctx.get("due_date"))
    amount = ctx.get("amount_numeric")

    period = _period_label(days) if days is not None else ""
    tier = _amount_tier(amount)

    queries = []
    if acceptor:
        queries.append(f"{acceptor} 承兑额度 授信评级 风险等级 票据承兑")
    queries.append(
        f"{ticket_type} {period} 贴现利率 定价基准 转贴现 市场行情 {tier}".strip()
    )
    context_hint = (
        f"用户查询票据贴现定价。当前票据：{ticket_type}，承兑人：{acceptor}，"
        f"剩余期限约 {days} 天，票面金额约 {amount} 元。"
        "请结合检索到的利率文件和内部授信政策，给出精确的定价区间和依据。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景二：背书链穿透 × 失信名单核查 ─────────────────────────────────────────
def build_endorser_check_queries(ctx: dict) -> dict:
    """
    INTENT_ENDORSER_CHECK：对每个背书人/出票人分别构造风险查询，并行多条。
    """
    endorsers = ctx.get("endorsers", [])
    drawer = ctx.get("drawer", "")
    subjects = []
    if drawer:
        subjects.append((drawer, "出票人"))
    for e in endorsers:
        subjects.append((e, "背书人"))

    queries = [
        f"{name} {role} 失信被执行 票据拒付 黑名单 风险记录 制裁"
        for name, role in subjects
    ]
    if not queries:
        queries = ["票据背书链 风险核查 合规要求"]

    context_hint = (
        f"用户请求背书链风险核查。共 {len(subjects)} 个主体需要核查："
        f"{', '.join(n for n, _ in subjects)}。"
        "请逐主体说明风险发现情况，无命中记录也须明确说明。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景三：到期日追索权时效核查 ──────────────────────────────────────────────
def build_expiry_queries(ctx: dict) -> dict:
    """
    INTENT_EXPIRY：注入逾期天数和票据类型，构造追索权时效专项查询。
    """
    ticket_type = ctx.get("ticket_type", "票据")
    days = _days_remaining(ctx.get("due_date"))

    if days is None:
        overdue_info = "到期日未知"
    elif days < 0:
        overdue_info = f"已逾期 {abs(days)} 天"
    else:
        overdue_info = f"距到期还有 {days} 天（尚未到期）"

    queries = [
        f"{ticket_type} 持票人追索权 诉讼时效 到期后多少天 票据法第17条",
        f"票据拒付 追索权行使程序 被追索人 法律救济",
    ]
    context_hint = (
        f"用户查询票据追索权时效。当前票据：{ticket_type}，{overdue_info}。"
        "请结合票据法和司法解释，精确计算各追索对象的时效是否届满，给出明确结论。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景四：大额票据合规义务自动触发 ──────────────────────────────────────────
def build_aml_queries(ctx: dict) -> dict:
    """
    INTENT_AML：根据金额触发大额合规查询；金额 < 500万时也查常规尽职调查要求。
    """
    amount = ctx.get("amount_numeric", 0)
    drawer = ctx.get("drawer", "")
    drawer_bank = ctx.get("drawer_bank", "")
    ticket_type = ctx.get("ticket_type", "")
    tier = _amount_tier(amount)

    queries = [
        f"大额交易报告 {tier} 票据业务 反洗钱 义务 时限 人民银行",
        f"贴现业务 客户尽职调查 出票人 开户行核验 {ticket_type}",
    ]
    context_hint = (
        f"用户查询大额票据合规义务。票面金额 {amount:,.0f} 元（{tier}），"
        f"出票人：{drawer}，开户行：{drawer_bank}。"
        "请列出需履行的报告义务、时限要求和核验清单，每条注明监管依据。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景五：票据要素矛盾检测 × 监管条文定位 ────────────────────────────────────
def build_element_check_queries(ctx: dict, risk_flags: list) -> dict:
    """
    INTENT_ELEMENT_CHECK：根据已检测到的 risk_flags 构造精准条文查询。
    每条风险对应一条查询，最大化定位到具体法规段落。
    """
    CODE_QUERIES = {
        "DATE_LOGIC_ERROR":    "票据到期日早于出票日 法律效力 如何认定 票据法第22条必要记载事项",
        "ACCEPTOR_NOT_BANK":   "银行承兑汇票承兑人必须是银行 商业主体承兑是否有效 票据法第38条",
        "AMOUNT_MISMATCH":     "票据大写金额与数字金额不符 如何认定效力 票据法记载错误",
        "TENOR_TOO_LONG":      "商业汇票承兑期限最长6个月 人行规定 超期如何处理",
    }
    queries = [
        CODE_QUERIES[f["code"]]
        for f in risk_flags
        if f["code"] in CODE_QUERIES
    ]
    if not queries:
        queries = ["票据要素合规性 必要记载事项 票据法第22条 要素效力"]

    flag_desc = "；".join(f["desc"] for f in risk_flags) if risk_flags else "无"
    context_hint = (
        f"用户请求票据要素合规审查。已检测到 {len(risk_flags)} 处风险：{flag_desc}。"
        "请逐条引用监管条文，给出每处风险的法律定性和处置建议。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景六：票据流转历史溯源 ──────────────────────────────────────────────────
def build_history_queries(ctx: dict, new_endorsers: list) -> dict:
    """
    INTENT_HISTORY：核查新增背书人风险 + 背书连续性合规要求。
    """
    queries = [
        f"{name} 风险 黑名单 票据纠纷 经营异常"
        for name in new_endorsers
    ]
    queries.append("票据背书转让 背书连续性 规范要求 票据法第31条")
    context_hint = (
        f"用户查询票据流转历史。本次新增背书人：{new_endorsers}。"
        "请核查新增背书人风险记录，并验证背书链连续性是否符合票据法要求。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景七：跨票据批量风险聚合 ────────────────────────────────────────────────
def build_batch_risk_queries(ctx: dict, total_amount: float, bill_count: int) -> dict:
    """
    INTENT_BATCH_RISK：基于同承兑人批量持仓金额构造集中度风险查询。
    """
    acceptor = ctx.get("acceptor", "")
    tier = _amount_tier(total_amount)

    queries = [
        f"{acceptor} 承兑额度上限 集中度风险 内部授信限额",
        f"单一承兑行 票据承兑敞口 风险集中 银保监规定 {tier}",
    ]
    context_hint = (
        f"用户查询批量票据集中度风险。承兑人：{acceptor}，"
        f"本批 {bill_count} 张，累计金额约 {total_amount:,.0f} 元（{tier}）。"
        "请检索内部集中度限额政策，判断是否超限并给出处置建议。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 场景八：票据质押融资可行性评估 ────────────────────────────────────────────
def build_pledge_queries(ctx: dict) -> dict:
    """
    INTENT_PLEDGE：基于承兑人、期限、金额构造质押融资方案查询。
    """
    ticket_type = ctx.get("ticket_type", "")
    acceptor = ctx.get("acceptor", "")
    days = _days_remaining(ctx.get("due_date"))
    amount = ctx.get("amount_numeric", 0)
    period = _period_label(days) if days is not None else ""

    queries = [
        f"{ticket_type} {acceptor} 质押率 折扣率 质押融资",
        f"票据质押融资 {period} 利率 质押贷款条件 融资成本",
        "票据质押背书 格式规范 票据法第35条 质押字样",
    ]
    context_hint = (
        f"用户咨询票据质押融资方案。票据：{ticket_type}，承兑人：{acceptor}，"
        f"剩余期限约 {days} 天，票面金额 {amount:,.0f} 元。"
        "请给出质押率区间、可融资金额、参考利率和背书规范要求，每项注明依据。"
    )
    return {"queries": queries, "context_hint": context_hint}


# ── 统一路由入口 ──────────────────────────────────────────────────────────────
def get_scene_queries(
    intent_id: str,
    ctx: dict,
    *,
    risk_flags: list = None,
    new_endorsers: list = None,
    total_amount: float = 0,
    bill_count: int = 1,
) -> dict:
    """
    统一入口：根据 intent_id 分发到对应的场景处理器。
    返回 {"queries": [...], "context_hint": "..."}。
    """
    risk_flags = risk_flags or []
    new_endorsers = new_endorsers or []

    handlers = {
        "INTENT_PRICING":        lambda: build_pricing_queries(ctx),
        "INTENT_ENDORSER_CHECK": lambda: build_endorser_check_queries(ctx),
        "INTENT_EXPIRY":         lambda: build_expiry_queries(ctx),
        "INTENT_AML":            lambda: build_aml_queries(ctx),
        "INTENT_ELEMENT_CHECK":  lambda: build_element_check_queries(ctx, risk_flags),
        "INTENT_HISTORY":        lambda: build_history_queries(ctx, new_endorsers),
        "INTENT_BATCH_RISK":     lambda: build_batch_risk_queries(ctx, total_amount, bill_count),
        "INTENT_PLEDGE":         lambda: build_pledge_queries(ctx),
    }
    handler = handlers.get(intent_id)
    if handler:
        return handler()
    return {"queries": [], "context_hint": ""}
