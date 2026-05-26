# app/agents/fraud_detection_agent.py
# FraudDetectionAgent：五维并行欺诈检测专项 Agent
# 职责：
#   1. 五维并行检测（asyncio.gather，每维 2 秒超时）：
#      印章真伪(0.30) + 重复票据(0.25) + 图像篡改(0.25) + 关联网络(0.15) + 历史黑名单(0.05)
#   2. 计算综合欺诈评分（0.0~1.0），重复票据命中时强制置 1.0
#   3. 按四档阈值判定欺诈等级（clean/suspicious/high_risk/fraud）
#   4. 将检测结果写入 fraud_detections 表

from __future__ import annotations

import asyncio          # asyncio.gather：五维并行检测核心
import uuid
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.models.agent_models import FraudDetection
from app.core.cache import bill_cache, compute_bill_fingerprint  # 缓存层


# ── 五维权重配置（合计 = 1.0）─────────────────────────────────────────────────
FRAUD_DIMENSION_WEIGHTS: Dict[str, float] = {
    "seal":       0.30,   # 印章真伪：最重要，直接影响票据法律效力
    "duplicate":  0.25,   # 重复票据：严重欺诈行为，命中后强制置最高分
    "tamper":     0.25,   # 图像篡改：关键数字被修改是常见欺诈手段
    "network":    0.15,   # 关联网络：背书链闭环涉嫌循环欺诈
    "blacklist":  0.05,   # 历史黑名单：补充维度，置信度较低
}

# ── 欺诈等级阈值（分越高风险越大）────────────────────────────────────────────
FRAUD_LEVEL_THRESHOLDS = [
    (0.8,  "fraud"),       # ≥ 0.8：直接判定欺诈
    (0.6,  "high_risk"),   # 0.6~0.8：高风险，需人工核实
    (0.3,  "suspicious"),  # 0.3~0.6：可疑，建议人工复核
    (0.0,  "clean"),       # < 0.3：清洁，正常处理
]

# ── 每维检测的超时时间（秒）──────────────────────────────────────────────────
DIMENSION_TIMEOUT_SECONDS = 2.0   # 每维最多等待 2 秒，超时则记录 0 分（不阻断）


class FraudDetectionAgent(BaseAgent):
    """
    欺诈检测 Agent：通过五维并行分析综合判断票据是否存在欺诈嫌疑
    重复票据维度为硬性规则：一旦命中，无论其他维度得分如何，综合分强制为 1.0
    """

    agent_name = "fraud_detection_agent"

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        **kwargs
    ) -> AgentResult:
        """
        欺诈检测主逻辑

        Args:
            ctx: 上下文（读取 bill_element 和 endorsement_result）
            db:  数据库会话

        Returns:
            AgentResult: 包含 overall_fraud_score / fraud_level / 各维度得分
        """
        # 步骤 1：从 shared_data 读取所需数据
        bill_element       = ctx.shared_data.get("bill_element", {})
        endorsement_result = ctx.shared_data.get("endorsement_result", {})

        # 步骤 2：查询欺诈检测缓存（五维并行检测较耗时，相同票据结果可复用）
        bill_fp = compute_bill_fingerprint(bill_element)
        cached_fraud = await bill_cache.get_fraud(ctx.tenant_id, bill_fp)
        if cached_fraud is not None:
            logger.info(
                f"[{self.agent_name}] 欺诈检测缓存命中 "
                f"bill_fp={bill_fp[:8]} level={cached_fraud.get('fraud_level')} "
                f"task={ctx.audit_task_id}"
            )
            # 从缓存恢复，写入本次任务的 DB 记录
            detection_id = str(uuid.uuid4())
            db.add(FraudDetection(
                id=detection_id,
                audit_task_id=ctx.audit_task_id,
                seal_score=cached_fraud.get("seal_score", 0),
                seal_check_result=cached_fraud.get("seal_check_result"),
                duplicate_score=cached_fraud.get("duplicate_score", 0),
                duplicate_check_result=cached_fraud.get("duplicate_check_result"),
                tamper_score=cached_fraud.get("tamper_score", 0),
                tamper_check_result=cached_fraud.get("tamper_check_result"),
                network_score=cached_fraud.get("network_score", 0),
                network_check_result=cached_fraud.get("network_check_result"),
                blacklist_score=cached_fraud.get("blacklist_score", 0),
                blacklist_check_result=cached_fraud.get("blacklist_check_result"),
                overall_fraud_score=cached_fraud.get("overall_fraud_score", 0),
                fraud_level=cached_fraud.get("fraud_level", "clean"),
            ))
            ctx.shared_data["fraud_result"] = {**cached_fraud, "detection_id": detection_id, "from_cache": True}
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data={**cached_fraud, "detection_id": detection_id, "from_cache": True},
            )

        # 步骤 3：五维并行检测（asyncio.gather + 超时保护）
        logger.info(
            f"[{self.agent_name}] 开始五维并行欺诈检测 task={ctx.audit_task_id}"
        )

        # 每维封装为 asyncio.wait_for（2秒超时），超时返回 0 分不阻断
        tasks = [
            self._run_with_timeout("seal",      self._check_seal(bill_element, ctx.audit_task_id)),
            self._run_with_timeout("duplicate", self._check_duplicate(bill_element, ctx.audit_task_id)),
            self._run_with_timeout("tamper",    self._check_tamper(bill_element, ctx.audit_task_id)),
            self._run_with_timeout("network",   self._check_network(endorsement_result, ctx.audit_task_id)),
            self._run_with_timeout("blacklist", self._check_blacklist(bill_element, ctx.audit_task_id)),
        ]

        # asyncio.gather：五维完全并行，等所有完成后才继续
        results = await asyncio.gather(*tasks)

        # 解包五维检测结果
        (seal_score,      seal_result,      _) = results[0]
        (duplicate_score, duplicate_result, _) = results[1]
        (tamper_score,    tamper_result,    _) = results[2]
        (network_score,   network_result,   _) = results[3]
        (blacklist_score, blacklist_result, _) = results[4]

        # 步骤 3：重复票据硬性规则：命中时综合分强制为 1.0
        if duplicate_score >= 1.0:
            overall_fraud_score = 1.0
            logger.warning(
                f"[{self.agent_name}] 重复票据命中！综合欺诈分强制为 1.0 "
                f"task={ctx.audit_task_id}"
            )
        else:
            # 加权求和
            overall_fraud_score = (
                seal_score      * FRAUD_DIMENSION_WEIGHTS["seal"]      +
                duplicate_score * FRAUD_DIMENSION_WEIGHTS["duplicate"] +
                tamper_score    * FRAUD_DIMENSION_WEIGHTS["tamper"]    +
                network_score   * FRAUD_DIMENSION_WEIGHTS["network"]   +
                blacklist_score * FRAUD_DIMENSION_WEIGHTS["blacklist"]
            )
        overall_fraud_score = round(min(1.0, max(0.0, overall_fraud_score)), 4)

        # 步骤 4：判定欺诈等级
        fraud_level = self._determine_fraud_level(overall_fraud_score)

        # 步骤 5：写入 fraud_detections 表
        detection_id = str(uuid.uuid4())
        detection = FraudDetection(
            id=detection_id,
            audit_task_id=ctx.audit_task_id,
            seal_score=round(seal_score, 4),
            seal_check_result=seal_result,
            duplicate_score=round(duplicate_score, 4),
            duplicate_check_result=duplicate_result,
            tamper_score=round(tamper_score, 4),
            tamper_check_result=tamper_result,
            network_score=round(network_score, 4),
            network_check_result=network_result,
            blacklist_score=round(blacklist_score, 4),
            blacklist_check_result=blacklist_result,
            overall_fraud_score=overall_fraud_score,
            fraud_level=fraud_level,
        )
        db.add(detection)

        # 步骤 6：写入 shared_data 供 RiskAssessmentAgent 使用
        fraud_result = {
            "detection_id":        detection_id,
            "overall_fraud_score": overall_fraud_score,
            "fraud_level":         fraud_level,
            "seal_score":          seal_score,
            "seal_check_result":   seal_result,
            "duplicate_score":     duplicate_score,
            "duplicate_check_result": duplicate_result,
            "tamper_score":        tamper_score,
            "tamper_check_result": tamper_result,
            "network_score":       network_score,
            "network_check_result": network_result,
            "blacklist_score":     blacklist_score,
            "blacklist_check_result": blacklist_result,
        }
        ctx.shared_data["fraud_result"] = fraud_result

        # 步骤 7：写入欺诈检测缓存（不缓存 detection_id，每次审核生成新 ID）
        await bill_cache.set_fraud(ctx.tenant_id, bill_fp, {
            k: v for k, v in fraud_result.items() if k != "detection_id"
        })

        logger.info(
            f"[{self.agent_name}] 欺诈检测完成 "
            f"overall={overall_fraud_score:.4f} level={fraud_level} "
            f"task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "detection_id":      detection_id,
                "overall_fraud_score": overall_fraud_score,
                "fraud_level":         fraud_level,
                "seal_score":          seal_score,
                "duplicate_score":     duplicate_score,
                "tamper_score":        tamper_score,
                "network_score":       network_score,
                "blacklist_score":     blacklist_score,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 超时包装器
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_with_timeout(
        self, dimension: str, coro
    ) -> Tuple[float, dict, bool]:
        """
        为每个检测维度添加超时保护（2秒），超时或异常时返回 0 分

        Returns:
            (score, detail_dict, timed_out)
        """
        try:
            result = await asyncio.wait_for(coro, timeout=DIMENSION_TIMEOUT_SECONDS)
            return result[0], result[1], False   # score, detail, not_timed_out
        except asyncio.TimeoutError:
            logger.warning(f"[{self.agent_name}] 维度 {dimension} 检测超时，得分归零")
            return 0.0, {"error": "timeout", "dimension": dimension}, True
        except Exception as e:
            logger.warning(f"[{self.agent_name}] 维度 {dimension} 检测异常: {e}")
            return 0.0, {"error": str(e), "dimension": dimension}, True

    # ──────────────────────────────────────────────────────────────────────────
    # 五维检测方法（返回 Tuple[score, detail_dict]）
    # ──────────────────────────────────────────────────────────────────────────

    async def _check_seal(self, bill_element: dict, task_id: str) -> Tuple[float, dict]:
        """
        印章真伪检测
        模拟：通过图像向量特征与黑名单印章特征库比对
        生产环境替换为：CNN 特征提取 + Milvus 向量相似度查询
        """
        # 模拟实现：从 OCR 置信度推断印章可信度
        confidence = bill_element.get("confidence_score", 0.95)

        if confidence >= 0.90:
            score = 0.02    # 高置信度 = 印章可信
            suspicious_features = []
        elif confidence >= 0.75:
            score = 0.10    # 中置信度 = 轻微存疑
            suspicious_features = ["ink_consistency_low"]
        else:
            score = 0.30    # 低置信度 = 印章存疑
            suspicious_features = ["ink_consistency_low", "border_irregularity"]

        return score, {
            "similarity":          1.0 - score,          # 与真实印章的相似度
            "blacklist_hit":       False,                  # 是否命中黑名单印章
            "suspicious_features": suspicious_features,
            "confidence_used":     confidence,             # 使用的置信度值
        }

    async def _check_duplicate(self, bill_element: dict, task_id: str) -> Tuple[float, dict]:
        """
        重复票据检测
        模拟：查询 bill_records 表中是否存在相同票据号码的历史记录
        生产环境替换为：PostgreSQL 精确查询（需连接数据库）
        注意：命中时返回 1.0，此维度是硬性规则
        """
        ticket_number = bill_element.get("ticket_number")

        # 模拟：票据号码以 "DUP-" 开头时认为是重复票据（测试用）
        if ticket_number and ticket_number.startswith("DUP-"):
            return 1.0, {
                "is_duplicate":      True,
                "duplicate_task_id": "PREV-TASK-001",    # 历史任务 ID（模拟）
                "ticket_number_match": ticket_number,
            }
        else:
            return 0.0, {
                "is_duplicate":      False,
                "duplicate_task_id": None,
                "ticket_number_match": None,
            }

    async def _check_tamper(self, bill_element: dict, task_id: str) -> Tuple[float, dict]:
        """
        图像篡改检测
        模拟：分析 OCR 置信度分布，置信度方差异常大则疑似篡改
        生产环境替换为：CNN 图像分析（检测关键数字区域的像素连续性）
        """
        confidence = bill_element.get("confidence_score", 0.95)

        # 金额字段置信度特别低时疑似关键数字被篡改
        amount_numeric = bill_element.get("amount_numeric", 0)

        suspicious_regions = []
        if confidence < 0.70:
            suspicious_regions.append({
                "page": 1,
                "region": "amount_area",
                "confidence_drop": round(0.95 - confidence, 3),
            })

        score = max(0.0, (0.85 - confidence) * 2.0) if confidence < 0.85 else 0.0

        return round(score, 4), {
            "suspicious_regions": suspicious_regions,
            "method":             "ocr_confidence_distribution",
            "confidence_score":   confidence,
        }

    async def _check_network(
        self, endorsement_result: dict, task_id: str
    ) -> Tuple[float, dict]:
        """
        关联网络检测
        模拟：基于背书链分析结果，检测有向图闭环（已由 EndorsementChainAgent 完成）
        生产环境可扩展为：跨任务背书关联分析（同一主体参与多张票据的网络图）
        """
        if not endorsement_result:
            return 0.0, {
                "has_cycle":           False,
                "cycle_path":          [],
                "suspicious_entities": [],
                "data_source":         "no_endorsement_data",
            }

        has_cycle     = endorsement_result.get("has_cycle", False)
        violation_count = endorsement_result.get("violation_count", 0)

        # 有闭环：严重欺诈信号（得分 0.8）；无闭环但有违规：轻微可疑
        if has_cycle:
            score = 0.80
            suspicious_entities = ["cycle_participants"]
        elif violation_count >= 3:
            score = 0.20
            suspicious_entities = ["multiple_violations"]
        else:
            score = 0.0
            suspicious_entities = []

        return round(score, 4), {
            "has_cycle":           has_cycle,
            "cycle_path":          [],   # 具体路径需从图中提取（此处简化）
            "suspicious_entities": suspicious_entities,
            "violation_count":     violation_count,
        }

    async def _check_blacklist(
        self, bill_element: dict, task_id: str
    ) -> Tuple[float, dict]:
        """
        历史黑名单检测
        模拟：查询 blacklist_entities 表，与出票人/收款人/背书人模糊匹配
        生产环境替换为：数据库查询 + 企业名称模糊匹配算法
        """
        drawer = bill_element.get("drawer", "")
        payee  = bill_element.get("payee", "")

        # 模拟：名称包含 "BLACKLIST" 时命中黑名单（测试用）
        hit_entities = []
        if drawer and "BLACKLIST" in drawer.upper():
            hit_entities.append({"name": drawer, "type": "company", "source": "internal"})
        if payee and "BLACKLIST" in payee.upper():
            hit_entities.append({"name": payee, "type": "company", "source": "internal"})

        score = min(1.0, len(hit_entities) * 0.5)   # 每个命中主体增加 0.5 分，上限 1.0

        return round(score, 4), {
            "hit_entities":      hit_entities,
            "match_confidence":  0.95 if hit_entities else 0.0,
            "checked_entities":  [drawer, payee],   # 检查了哪些主体
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 欺诈等级判定
    # ──────────────────────────────────────────────────────────────────────────

    def _determine_fraud_level(self, overall_score: float) -> str:
        """
        按四档阈值判定欺诈等级
        阈值从高到低遍历，第一个满足的阈值即为等级
        """
        for threshold, level in FRAUD_LEVEL_THRESHOLDS:
            if overall_score >= threshold:
                return level
        return "clean"   # 兜底
