# app/services/retrieval_quality.py
# 检索质量评估 + 转人工兜底机制（P3）
#
# 职责：
#   1. 对向量检索结果打质量分（SUFFICIENT / PARTIAL / INSUFFICIENT）
#   2. INSUFFICIENT 时构建转人工响应，记录工单号
#   3. 提供 PARTIAL 场景下的免责提示前缀

import uuid
from dataclasses import dataclass
from typing import Optional

# ── 质量等级常量 ──────────────────────────────────────────────────────────────
QUALITY_SUFFICIENT = "SUFFICIENT"    # 结果充足，直接生成答案
QUALITY_PARTIAL = "PARTIAL"          # 结果有限，生成答案但附免责提示
QUALITY_INSUFFICIENT = "INSUFFICIENT"  # 结果不足，触发转人工

# ── 评分阈值（可通过配置调整）────────────────────────────────────────────────
TOP1_SCORE_SUFFICIENT = 0.75   # top1 分数 ≥ 此值 + hit_count ≥ 2 → SUFFICIENT
TOP1_SCORE_PARTIAL = 0.55      # top1 分数 ≥ 此值 + hit_count ≥ 1 → PARTIAL
MIN_HIT_COUNT_SUFFICIENT = 2   # SUFFICIENT 要求的最少有效命中数
MIN_HIT_COUNT_PARTIAL = 1      # PARTIAL 要求的最少有效命中数
HIT_SCORE_THRESHOLD = 0.40     # 单条结果分数 ≥ 此值才算"有效命中"

# 转人工联系信息（实际部署时从配置读取）
HUMAN_HOTLINE = "400-XXX-XXXX"
HUMAN_SERVICE_HOURS = "工作日 9:00-18:00"


@dataclass
class QualityResult:
    """检索质量评估结果"""
    level: str               # SUFFICIENT / PARTIAL / INSUFFICIENT
    top1_score: float        # 最高分
    avg_score: float         # 前 N 条平均分
    hit_count: int           # 有效命中数（score ≥ HIT_SCORE_THRESHOLD）
    total_retrieved: int     # 共检索到多少条


def evaluate(chunks: list[dict]) -> QualityResult:
    """
    对 hybrid_search 返回的 chunks 列表进行质量评估。
    chunks 元素期望含 rerank_score 字段（Reranker 输出）。
    """
    if not chunks:
        return QualityResult(
            level=QUALITY_INSUFFICIENT,
            top1_score=0.0,
            avg_score=0.0,
            hit_count=0,
            total_retrieved=0,
        )

    scores = [float(c.get("rerank_score", 0.0)) for c in chunks]
    top1_score = max(scores)
    avg_score = sum(scores) / len(scores)
    hit_count = sum(1 for s in scores if s >= HIT_SCORE_THRESHOLD)

    if top1_score >= TOP1_SCORE_SUFFICIENT and hit_count >= MIN_HIT_COUNT_SUFFICIENT:
        level = QUALITY_SUFFICIENT
    elif top1_score >= TOP1_SCORE_PARTIAL and hit_count >= MIN_HIT_COUNT_PARTIAL:
        level = QUALITY_PARTIAL
    else:
        level = QUALITY_INSUFFICIENT

    return QualityResult(
        level=level,
        top1_score=top1_score,
        avg_score=avg_score,
        hit_count=hit_count,
        total_retrieved=len(chunks),
    )


def build_transfer_response(query: str, intent_id: Optional[str] = None) -> dict:
    """
    构造转人工响应体。
    ticket_id 唯一标识本次未命中工单，前端可用于跟踪处理进度。
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    intent_hint = f"（您的问题类别：{intent_id}）" if intent_id and intent_id != "INTENT_UNKNOWN" else ""
    return {
        "answer_type": "transfer_human",
        "answer": (
            f"抱歉，当前知识库中暂无足够的内容来准确回答您的问题{intent_hint}。\n"
            "为保证答复的准确性，建议您联系专业票据顾问获得进一步支持。"
        ),
        "transfer_to_human": True,
        "contact_info": {
            "hotline": HUMAN_HOTLINE,
            "online_service": HUMAN_SERVICE_HOURS,
            "ticket_id": ticket_id,
        },
        "query_saved": True,  # 通知用户：此问题已记录，将用于知识库优化
        "sources": [],
        "retrieval_quality": QUALITY_INSUFFICIENT,
    }


PARTIAL_DISCLAIMER = (
    "\n\n> ⚠️ 注意：本回答基于有限的参考资料生成，仅供参考。"
    "如涉及重要业务决策，请进一步核实或咨询专业人员。"
)
