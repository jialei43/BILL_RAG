# app/services/intent_router.py
# 意图识别与专项检索路由引擎（P2）
#
# 两阶段串联：
#   阶段1 关键词快速通道（本地，< 1ms）：命中直接返回，confidence=0.95
#   阶段2 qwen-max 语义识别（仅阶段1未命中时调用，200~500ms）
#
# 意图映射到 8 个专项场景（见需求.md P2），INTENT_UNKNOWN → 模糊检索兜底

import json
import time
from typing import Optional

import openai
from loguru import logger

from config.settings import settings
from app.core.cache import bill_cache, compute_query_fingerprint  # 缓存层

# ── 意图定义 ──────────────────────────────────────────────────────────────────
INTENT_UNKNOWN = "INTENT_UNKNOWN"

# 每个意图的关键词表（快速通道用）：词条越精确越好，避免误触发
# ★ 注意：Agent 意图关键词必须排在 RAG 意图之前，优先级更高
INTENT_KEYWORDS: dict[str, list[str]] = {
    # ── Agent 类意图（路由到专项 Agent，返回结构化报告） ──────────────────────
    # 出票预检：业务员发起出票前，验证票据要素合法性和出票资质
    "INTENT_AGENT_ISSUANCE":    ["出票预检", "能否出票", "可以出票吗", "出票合规", "出票资质",
                                  "出票申请审核", "出票前审核"],
    # 贴现申请：验证贴现资质、贸易背景真实性、背书连续性
    "INTENT_AGENT_DISCOUNT":    ["能否贴现", "可以贴现吗", "贴现审核", "贴现申请审核",
                                  "贴现可行性", "能做贴现", "申请贴现审核"],
    # 背书链分析：验证背书连续性、各背书人资质、背书合规性
    "INTENT_AGENT_ENDORSEMENT": ["背书链分析", "背书连续性审核", "背书合规审核", "背书链合规",
                                  "审核背书", "背书链报告", "背书是否合规",
                                  "背书申请", "能否背书", "可以背书", "背书校验", "校验背书",
                                  "背书转让", "申请背书"],
    # 欺诈筛查：验证印章真伪、重复票据、图像篡改、黑名单
    "INTENT_AGENT_FRAUD":       ["欺诈检测", "欺诈筛查", "票据真伪", "是否造假", "印章真伪",
                                  "是否重复票据", "篡改检测", "黑名单查询"],
    # 合规审核：针对全部18个要素字段做法规合规检索
    "INTENT_AGENT_COMPLIANCE":  ["合规审核", "合规检查", "要素合规", "票据合规报告",
                                  "合规性报告", "全要素合规", "出票合规报告"],
    # 全流程审核：完整的多 Agent 审核，包含合规+背书+欺诈+风险+报告
    "INTENT_AGENT_FULL_AUDIT":  ["全流程审核", "完整审核", "全面审核", "综合审核报告",
                                  "全套审核", "完整合规报告", "审核报告"],

    # ── RAG 类意图（检索知识库，返回知识性答案） ─────────────────────────────
    "INTENT_PRICING":        ["利率", "定价", "贴现率", "报价", "折扣率", "转贴现", "再贴现"],
    "INTENT_ENDORSER_CHECK": ["背书", "背书人", "背书链", "连续背书", "签章", "被背书人"],
    "INTENT_EXPIRY":         ["到期", "追索权", "时效", "逾期", "拒付", "付款期", "追索"],
    "INTENT_AML":            ["大额", "反洗钱", "报告义务", "AML", "尽职调查", "可疑交易"],
    "INTENT_ELEMENT_CHECK":  ["要素", "矛盾", "核验", "审查", "不规范", "无效票", "缺失"],
    "INTENT_HISTORY":        ["流转", "历史", "版本", "背书记录", "溯源", "转让记录"],
    "INTENT_BATCH_RISK":     ["批量", "集中度", "敞口", "风险汇总", "同一承兑", "同一出票"],
    "INTENT_PLEDGE":         ["质押", "质押率", "抵押", "融资", "担保", "质押融资"],
}

# qwen-max 中每个意图的语义描述（供 Prompt 构造）
INTENT_DESCRIPTIONS: dict[str, str] = {
    # Agent 类意图描述
    "INTENT_AGENT_ISSUANCE":    "用户希望对一张具体票据做出票资质预检，判断能否出票",
    "INTENT_AGENT_DISCOUNT":    "用户希望对一张具体票据做贴现可行性审核，判断能否贴现",
    "INTENT_AGENT_ENDORSEMENT": "用户希望对一张具体票据的背书链做合规分析报告",
    "INTENT_AGENT_FRAUD":       "用户希望对一张具体票据做欺诈风险筛查，检测真伪和黑名单",
    "INTENT_AGENT_COMPLIANCE":  "用户希望对一张具体票据做全要素合规审核，获得合规报告",
    "INTENT_AGENT_FULL_AUDIT":  "用户希望对一张具体票据做全流程综合审核，获得完整审核报告",
    # RAG 类意图描述
    "INTENT_PRICING":        "用户询问票据贴现利率、转贴现报价、定价基准等",
    "INTENT_ENDORSER_CHECK": "用户询问背书链合规性、背书人资质、背书连续性、签章规范等",
    "INTENT_EXPIRY":         "用户询问票据到期处理、追索权时效计算、拒付应对等",
    "INTENT_AML":            "用户询问大额交易合规、反洗钱报告义务、可疑交易等",
    "INTENT_ELEMENT_CHECK":  "用户询问票据要素是否合规、有无矛盾或缺失、票据效力等",
    "INTENT_HISTORY":        "用户询问票据流转历史、背书记录溯源、版本对比等",
    "INTENT_BATCH_RISK":     "用户询问多张票据风险汇总、承兑人集中度、持仓敞口等",
    "INTENT_PLEDGE":         "用户询问票据质押融资方案、质押率、融资利率等",
    INTENT_UNKNOWN:          "无法归入以上任何类别",
}

# ── Agent 意图 → task_type 映射 ───────────────────────────────────────────────
# 说明：Agent 意图在系统内部映射到 OrchestratorAgent 的 task_type，业务调用方无需知晓
AGENT_INTENT_TO_TASK_TYPE: dict[str, str] = {
    "INTENT_AGENT_ISSUANCE":    "issuance_check",    # 出票预检 DAG
    "INTENT_AGENT_DISCOUNT":    "discount_apply",    # 贴现申请 DAG
    "INTENT_AGENT_ENDORSEMENT": "endorsement",       # 背书转让 DAG
    "INTENT_AGENT_FRAUD":       "full_audit",        # 全流程（含欺诈检测）
    "INTENT_AGENT_COMPLIANCE":  "issuance_check",    # 合规审核复用出票预检路径
    "INTENT_AGENT_FULL_AUDIT":  "full_audit",        # 全流程审核 DAG
}

# Agent 意图的中文名称（用于报告标题）
AGENT_INTENT_LABELS: dict[str, str] = {
    "INTENT_AGENT_ISSUANCE":    "出票合规预检",
    "INTENT_AGENT_DISCOUNT":    "贴现申请审核",
    "INTENT_AGENT_ENDORSEMENT": "背书链合规分析",
    "INTENT_AGENT_FRAUD":       "欺诈风险筛查",
    "INTENT_AGENT_COMPLIANCE":  "全要素合规审核",
    "INTENT_AGENT_FULL_AUDIT":  "全流程综合审核",
}

# 置信度阈值：高于此值才走专项检索，否则降级为模糊检索
SPECIALIZED_CONFIDENCE_THRESHOLD = 0.75

# 票据领域宽域词汇表：命中任意一个即视为票据相关，走检索而非拒绝
# 覆盖票据全生命周期、所有业务操作、参与方、法律、风险、监管等场景
BILL_DOMAIN_KEYWORDS = [

    # ── 票据基本类型 ───────────────────────────────────────────────────────────
    "票据", "汇票", "本票", "支票",
    "银票", "商票", "电票", "纸票",
    "银行承兑汇票", "商业承兑汇票", "财务公司承兑汇票",
    "银行本票", "电子商业汇票", "标准化票据",
    "数字票据", "区块链票据",
    "票号", "票面", "票据池", "票据包",

    # ── ECDS / 新一代票据系统 业务操作 ────────────────────────────────────────
    # 出票登记环节
    "出票", "出票登记", "出票申请", "票据签发", "签发",
    "出票人申请", "票据开具",
    # 签收环节
    "签收", "收票", "提示收票", "拒签", "待签收",
    "提示签收",
    # 承兑环节
    "承兑", "提示承兑", "承兑申请", "承兑确认",
    "商承承兑确认", "银承", "商承",
    "承兑到期", "承兑行", "承兑保证",
    # 背书转让环节
    "背书", "背书转让", "背书链", "背书记录", "背书人",
    "被背书人", "背书连续", "连续背书", "背书不连续",
    "空白背书", "限制性背书", "委托收款背书",
    "质押背书", "回头背书", "背书撤回",
    # 贴现环节
    "贴现", "申请贴现", "贴现申请", "贴现行",
    "转贴现", "再贴现", "买断式贴现", "回购式贴现",
    "直贴", "转贴", "再贴",
    # 质押环节
    "质押", "质押登记", "质押申请", "质押融资",
    "质押解除", "解质押", "质押率", "质押比例",
    "出质人", "质权人",
    # 托收 / 提示付款环节
    "托收", "委托收款", "提示付款", "提示付款期",
    "付款行", "代理付款", "付款确认",
    # 冻结 / 解冻
    "冻结", "解冻", "票据冻结", "司法冻结",
    "行政冻结", "冻结申请", "冻结解除",
    # 撤回 / 撤销
    "撤回", "撤销", "票据撤回", "出票撤回",
    "背书撤回", "贴现撤回", "申请撤回",
    # 不得转让
    "不得转让", "禁止转让", "不得背书转让",
    "不可转让标志", "不可转让撤销", "转让限制",
    # 报文重发 / 系统操作
    "报文重发", "重发报文", "报文超时", "报文状态",
    "业务撤销", "交易撤销", "交易冲正",
    # 挂失 / 补办
    "挂失", "票据挂失", "挂失止付", "公示催告",
    "除权判决", "票据补办", "票据换票",

    # ── 票据参与方 ─────────────────────────────────────────────────────────────
    "出票人", "收款人", "持票人", "承兑人",
    "背书人", "被背书人", "保证人",
    "付款人", "贴现申请人", "贴现行",
    "转贴现行", "再贴现行", "托收行",
    "参加承兑人", "参加付款人", "预备付款人",
    "出质人", "质权人", "代理行",

    # ── 票据要素与条款 ─────────────────────────────────────────────────────────
    "票面金额", "大写金额", "小写金额",
    "出票日期", "承兑日期", "到期日", "到期日期",
    "付款期限", "见票即付", "定日付款",
    "出票后定期", "见票后定期",
    "付款地", "出票地", "承兑地",
    "开户行", "付款行全称",
    "票据要素", "要素审查", "要素缺失", "要素矛盾",

    # ── 票据状态 ──────────────────────────────────────────────────────────────
    "已出票", "已承兑", "已收票", "已背书",
    "已贴现", "已质押", "已托收", "已付款",
    "已到期", "已锁定", "待收票", "可流通",
    "待承兑", "待背书签收", "待贴现签收",
    "追索中", "已结清", "已终止",

    # ── 拒付与追索 ─────────────────────────────────────────────────────────────
    "拒付", "拒绝付款", "拒绝承兑",
    "拒付通知", "拒付理由书", "拒绝证明",
    "追索", "追索权", "行使追索权",
    "追索通知", "追索金额", "追索费用",
    "追索时效", "追索期限", "付款请求权",
    "被追索人", "追索义务人",
    "保全", "权利保全", "财产保全", "证据保全",

    # ── 票据风险管理 ───────────────────────────────────────────────────────────
    "信用风险", "承兑风险", "付款风险",
    "市场风险", "利率风险", "流动性风险",
    "操作风险", "法律风险", "欺诈风险",
    "票据真伪", "票据鉴别", "验票", "审票",
    "票据风险", "风险敞口", "集中度",
    "风险监测", "风险预警", "风险分类",
    "承兑余额", "保证金比例", "风险加权",
    "同一承兑人", "同一出票人", "集中敞口",
    "批量风险", "持仓风险", "整体风险",

    # ── 票据定价 ──────────────────────────────────────────────────────────────
    "贴现率", "转贴现率", "再贴现率",
    "利率", "报价", "折扣率", "收益率",
    "贴现利息", "贴现价格", "票据价格",
    "定价基准", "利率报价", "市场利率",
    "票交所报价", "利率区间",

    # ── 融资工具与产品 ────────────────────────────────────────────────────────
    "票据融资", "票据贷款", "票据理财",
    "正回购", "逆回购", "买入返售", "卖出回购",
    "票据资管", "票据基金",
    "标准化票据", "票据ABS", "ABCP", "票据资产支持",
    "票据保理", "保理融资", "反向保理",
    "供应链金融", "应收账款", "应收票据",
    "票据池融资", "池融资", "票据直融",

    # ── 监管合规 ──────────────────────────────────────────────────────────────
    "反洗钱", "AML", "大额交易", "可疑交易",
    "大额支付", "资金监测", "穿透核查",
    "真实性审查", "交易背景", "贸易背景",
    "产业政策", "合规审查", "监管要求",
    "尽职调查", "审慎开展", "内部控制",
    "信息披露", "CIPS", "票据法",
    "电子商业汇票管理办法", "票据交易管理办法",
    "票交所", "上海票据交易所", "ECDS",
    "人民银行", "央行", "银保监会", "金融监管",

    # ── 司法 / 法律 ───────────────────────────────────────────────────────────
    "票据纠纷", "票据诉讼", "票据权利",
    "无因性", "票据独立性", "文义性",
    "公示催告", "除权判决", "止付通知",
    "保全措施", "行为保全", "票据权利救济",
    "前手", "后手", "直接当事人", "间接当事人",
    "善意取得", "恶意取得",

    # ── 市场基础设施与平台 ────────────────────────────────────────────────────
    "票据交易所", "票交所", "票据平台",
    "票据系统", "票据综合服务平台",
    "票据托管", "票据登记", "票据结算",
    "票据托管行", "清算结算", "净额清算",

    # ── 担保与保证 ────────────────────────────────────────────────────────────
    "票据保证", "担保承兑", "保证人",
    "抵押", "保证金", "履约保证",
    "担保", "增信",

    # ── 其他常见票据咨询词 ────────────────────────────────────────────────────
    "票据管理", "票据操作", "票据流程",
    "票据查询", "票据状态", "票据信息",
    "票据期限", "票据有效期", "票据金额",
    "票据手续费", "资金归集", "兑付资金",
    "短期融资券", "商业信用", "银行信用",
]


class IntentRouter:
    """
    两阶段意图路由器。
    classify() 返回 (intent_id, confidence, method)：
      method = "keyword"（关键词命中）或 "llm"（qwen-max 命中）
    """

    def __init__(self):
        self._client: Optional[openai.AsyncOpenAI] = None  # 懒加载，首次调用时创建

    def _get_client(self) -> Optional[openai.AsyncOpenAI]:
        """懒加载 OpenAI 兼容客户端（通义千问）。没有 API Key 时返回 None。"""
        if self._client is None and settings.OPENAI_API_KEY:
            self._client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or None,
            )
        return self._client

    def _keyword_classify(self, query: str) -> Optional[tuple[str, float]]:
        """
        阶段1：关键词快速通道。
        遍历 INTENT_KEYWORDS，任意关键词命中即返回 (intent_id, 0.95)。
        优先级：字典顺序（越专业的意图关键词应越靠前）。
        """
        for intent_id, keywords in INTENT_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                return intent_id, 0.95
        return None

    async def _llm_classify(self, query: str) -> tuple[str, float]:
        """
        阶段2：qwen-max 语义意图识别。
        发送结构化 Prompt，强制 JSON 输出，解析 intent_id 和 confidence。
        任何异常均降级为 INTENT_UNKNOWN。
        """
        client = self._get_client()
        if client is None:
            logger.warning("[intent_router] 未配置 OPENAI_API_KEY，LLM 分类降级为 UNKNOWN")
            return INTENT_UNKNOWN, 0.0

        intent_list = "\n".join(
            f"- {iid}：{desc}"
            for iid, desc in INTENT_DESCRIPTIONS.items()
            if iid != INTENT_UNKNOWN
        )

        prompt = (
            f"你是票据业务意图识别器。根据用户输入，从以下意图中选择最匹配的一个，"
            f"并给出 0~1 的置信度。若无法匹配任何意图，返回 INTENT_UNKNOWN。\n\n"
            f"可选意图：\n{intent_list}\n- {INTENT_UNKNOWN}：{INTENT_DESCRIPTIONS[INTENT_UNKNOWN]}\n\n"
            f"用户输入：「{query}」\n\n"
            f"严格按 JSON 格式返回，不要有任何额外说明：\n"
            f'{{\"intent_id\": \"INTENT_XXX\", \"confidence\": 0.85, \"reason\": \"简短说明\"}}'
        )

        try:
            t0 = time.perf_counter()
            response = await client.chat.completions.create(
                model="qwen-max",       # 意图识别固定用 qwen-max，与 RAG 生成模型解耦
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,        # 确定性任务，不需要创造性
                max_tokens=120,         # 只需简短 JSON
                response_format={"type": "json_object"},
            )
            elapsed = (time.perf_counter() - t0) * 1000
            content = response.choices[0].message.content
            result = json.loads(content)
            intent_id = result.get("intent_id", INTENT_UNKNOWN)
            confidence = float(result.get("confidence", 0.0))

            # 校验返回的 intent_id 是否合法
            valid_intents = set(INTENT_KEYWORDS.keys()) | {INTENT_UNKNOWN}
            if intent_id not in valid_intents:
                logger.warning(f"[intent_router] LLM 返回未知意图 {intent_id}，重置为 UNKNOWN")
                intent_id = INTENT_UNKNOWN

            logger.debug(
                f"[intent_router] LLM 分类完成 intent={intent_id} "
                f"confidence={confidence:.2f} elapsed={elapsed:.0f}ms"
            )
            return intent_id, confidence

        except Exception as e:
            logger.warning(f"[intent_router] LLM 分类失败: {e}，降级为 UNKNOWN")
            return INTENT_UNKNOWN, 0.0

    async def classify(self, query: str) -> tuple[str, float, str]:
        """
        主入口：两阶段串联 + Redis 缓存。
        返回 (intent_id, confidence, method)
          method: "keyword" | "llm" | "cache"（缓存命中）

        缓存策略：
          - 关键词通道（本地 < 1ms）：不缓存，命中后直接返回，几乎无成本
          - LLM 通道（qwen-max，200~500ms）：缓存 24h，相同问题只调用一次
        """
        # 阶段1：关键词通道（本地匹配，无需缓存）
        kw_result = self._keyword_classify(query)
        if kw_result:
            logger.debug(
                f"[intent_router] 关键词命中 intent={kw_result[0]} query='{query[:30]}'"
            )
            return kw_result[0], kw_result[1], "keyword"

        # 阶段2：先查缓存，命中则跳过 LLM 调用
        query_fp = compute_query_fingerprint(query)  # 基于问题文本的哈希指纹
        cached = await bill_cache.get_intent(query_fp)
        if cached is not None:
            logger.debug(
                f"[intent_router] 意图缓存命中 intent={cached['intent_id']} "
                f"fp={query_fp[:8]} query='{query[:30]}'"
            )
            return cached["intent_id"], cached["confidence"], "cache"

        # 阶段3：调用 qwen-max 语义识别（昂贵操作）
        intent_id, confidence = await self._llm_classify(query)

        # 写入缓存（不论命中与否都缓存，避免下次重复调用 LLM）
        await bill_cache.set_intent(query_fp, intent_id, confidence, "llm")

        return intent_id, confidence, "llm"

    def should_use_specialized(self, intent_id: str, confidence: float) -> bool:
        """判断是否应走专项检索（True）还是模糊检索（False）。"""
        return intent_id != INTENT_UNKNOWN and confidence >= SPECIALIZED_CONFIDENCE_THRESHOLD

    def is_off_topic(self, intent_id: str, query: str) -> bool:
        """
        判断是否与票据业务完全无关。
        规则：意图为 UNKNOWN 且查询内容不含任何票据领域词汇。
        含领域词的宽泛问题仍走模糊检索，而非拒绝。
        """
        if intent_id != INTENT_UNKNOWN:
            return False
        return not any(kw in query for kw in BILL_DOMAIN_KEYWORDS)

    def is_agent_intent(self, intent_id: str) -> bool:
        """
        判断该意图是否需要调用多智能体（Agent）审核，而非 RAG 知识问答。
        Agent 意图以 INTENT_AGENT_ 前缀标识。
        """
        return intent_id in AGENT_INTENT_TO_TASK_TYPE

    def get_agent_task_type(self, intent_id: str) -> str:
        """
        根据 Agent 意图获取对应的 OrchestratorAgent task_type 字符串。
        业务调用方通过 intent_id 触发，无需了解内部 task_type 命名。

        Returns:
            task_type 字符串（如 "issuance_check"），若不是 Agent 意图则返回 "full_audit"
        """
        return AGENT_INTENT_TO_TASK_TYPE.get(intent_id, "full_audit")

    def get_agent_label(self, intent_id: str) -> str:
        """返回 Agent 意图的中文标签（用于报告标题展示）。"""
        return AGENT_INTENT_LABELS.get(intent_id, "综合审核")


intent_router = IntentRouter()
