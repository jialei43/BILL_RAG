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

# ── 意图定义 ──────────────────────────────────────────────────────────────────
INTENT_UNKNOWN = "INTENT_UNKNOWN"

# 每个意图的关键词表（快速通道用）：词条越精确越好，避免误触发
INTENT_KEYWORDS: dict[str, list[str]] = {
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

# 置信度阈值：高于此值才走专项检索，否则降级为模糊检索
SPECIALIZED_CONFIDENCE_THRESHOLD = 0.75


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
        主入口：两阶段串联。
        返回 (intent_id, confidence, method)
          method: "keyword" | "llm"
        """
        # 阶段1：关键词通道
        kw_result = self._keyword_classify(query)
        if kw_result:
            logger.debug(
                f"[intent_router] 关键词命中 intent={kw_result[0]} query='{query[:30]}'"
            )
            return kw_result[0], kw_result[1], "keyword"

        # 阶段2：qwen-max 语义识别
        intent_id, confidence = await self._llm_classify(query)
        return intent_id, confidence, "llm"

    def should_use_specialized(self, intent_id: str, confidence: float) -> bool:
        """判断是否应走专项检索（True）还是模糊检索（False）。"""
        return intent_id != INTENT_UNKNOWN and confidence >= SPECIALIZED_CONFIDENCE_THRESHOLD


intent_router = IntentRouter()
