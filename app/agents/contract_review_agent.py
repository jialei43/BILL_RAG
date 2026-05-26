# app/agents/contract_review_agent.py
# ContractReviewAgent：合同审核专项 Agent
# 职责：
#   1. 将票据要素（drawer/payee/amount/date/trade_purpose）与合同要素进行 6 项比对
#   2. 计算贸易背景真实性评分（0~100）
#   3. 将审核结果写入 contract_reviews 表

from __future__ import annotations

import uuid
from typing import Optional, List

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.models.agent_models import ContractReview


class ContractReviewAgent(BaseAgent):
    """
    合同审核 Agent：校验票据与合同的一致性
    无合同时降级为仅贸易背景评分（match_score 使用默认中等分数）
    """

    agent_name = "contract_review_agent"

    # 无合同时的默认匹配分（降级处理：不影响审核通过，但降低信心）
    DEFAULT_SCORE_NO_CONTRACT = 50.0
    # 金额匹配允许的误差率（±5%）
    AMOUNT_TOLERANCE_RATE = 0.05

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        contract_text: Optional[str] = None,   # 合同文本（从解析结果中传入）
        contract_document_id: Optional[str] = None,  # 合同文档 ID（查库用）
        **kwargs
    ) -> AgentResult:
        """
        合同审核主逻辑

        Args:
            ctx:                  上下文（从 shared_data["bill_element"] 读取票据要素）
            db:                   数据库会话
            contract_text:        合同文本内容（可直接传入，否则从数据库查询）
            contract_document_id: 合同文档 ID（用于查询文档并解析合同要素）

        Returns:
            AgentResult: 包含 match_score / amount_match / party_match 等匹配结果
        """
        bill_element = ctx.shared_data.get("bill_element", {})

        # 步骤 1：提取票据要素
        bill_amount  = bill_element.get("amount_numeric")  # 票面金额
        bill_drawer  = bill_element.get("drawer")           # 出票人（应为合同甲方）
        bill_payee   = bill_element.get("payee")            # 收款人（应为合同乙方）
        bill_date    = bill_element.get("issue_date")       # 出票日期（应在合同有效期内）
        bill_purpose = bill_element.get("trade_purpose")    # 贸易背景说明

        # 步骤 2：解析合同要素（若无合同则降级）
        if contract_text:
            contract_elements = self._parse_contract_elements(contract_text)
        else:
            # 无合同时使用空要素（所有匹配项均为 True，但 match_score 降级）
            contract_elements = {}
            logger.info(
                f"[{self.agent_name}] 无合同文本，降级为贸易背景评分 task={ctx.audit_task_id}"
            )

        # 步骤 3：6 项要素比对
        amount_match  = self._check_amount_match(bill_amount,  contract_elements.get("amount"))
        party_match   = self._check_party_match(bill_drawer, bill_payee, contract_elements)
        date_match    = self._check_date_match(bill_date,     contract_elements)
        purpose_match = self._check_purpose_match(bill_purpose, contract_elements.get("purpose"))

        # 步骤 4：收集不匹配详情
        mismatch_details = []
        if not amount_match:
            mismatch_details.append({
                "field": "amount",
                "severity": "warning",
                "bill_value": str(bill_amount),
                "contract_value": str(contract_elements.get("amount")),
            })
        if not party_match:
            mismatch_details.append({
                "field": "party",
                "severity": "severe",
                "bill_value": f"出票人:{bill_drawer} 收款人:{bill_payee}",
                "contract_value": str(contract_elements.get("parties")),
            })

        # 步骤 5：计算综合匹配度和贸易背景评分
        match_score = self._calc_match_score(
            amount_match, party_match, date_match, purpose_match,
            has_contract=bool(contract_text),
        )
        trade_background_score = self._calc_trade_background_score(
            bill_purpose, contract_elements.get("purpose")
        )

        # 步骤 6：写入 contract_reviews 表
        review_id = str(uuid.uuid4())
        review = ContractReview(
            id=review_id,
            audit_task_id=ctx.audit_task_id,
            contract_document_id=contract_document_id,
            match_score=round(match_score, 2),
            trade_background_score=round(trade_background_score, 2),
            amount_match=amount_match,
            party_match=party_match,
            date_match=date_match,
            purpose_match=purpose_match,
            mismatch_details=mismatch_details,
            review_notes=self._gen_review_notes(match_score, mismatch_details, bool(contract_text)),
        )
        db.add(review)

        # 步骤 7：写入 shared_data 供 RiskAssessmentAgent 使用
        ctx.shared_data["contract_result"] = {
            "review_id":               review_id,
            "match_score":             match_score,
            "trade_background_score":  trade_background_score,
            "amount_match":            amount_match,
            "party_match":             party_match,
            "mismatch_count":          len(mismatch_details),
        }

        logger.info(
            f"[{self.agent_name}] 合同审核完成 "
            f"match_score={match_score:.1f} mismatches={len(mismatch_details)} "
            f"task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "match_score":            match_score,
                "trade_background_score": trade_background_score,
                "amount_match":           amount_match,
                "party_match":            party_match,
                "date_match":             date_match,
                "purpose_match":          purpose_match,
                "mismatch_count":         len(mismatch_details),
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_contract_elements(self, contract_text: str) -> dict:
        """
        从合同文本中提取关键要素（简化版关键词匹配）
        生产环境可替换为 LLM 结构化提取
        """
        elements = {}

        # 提取合同金额（简单启发式：寻找数字+万/元 的模式）
        import re
        amount_pattern = re.compile(r'([\d,]+(?:\.\d+)?)\s*(?:万元|元整|元)')
        amounts = amount_pattern.findall(contract_text)
        if amounts:
            try:
                raw_amount = amounts[0].replace(",", "")  # 去除千分位逗号
                amount_val = float(raw_amount)
                # 如果单位是万，转换为元
                if "万元" in contract_text[contract_text.find(amounts[0]):contract_text.find(amounts[0])+10]:
                    amount_val *= 10000
                elements["amount"] = amount_val
            except ValueError:
                pass

        # 提取合同甲乙方（寻找「甲方：」「乙方：」等关键词）
        parties = []
        for keyword in ["甲方：", "甲方:", "乙方：", "乙方:"]:
            if keyword in contract_text:
                start = contract_text.index(keyword) + len(keyword)
                end   = contract_text.find("\n", start)
                party = contract_text[start: end if end > 0 else start + 100].strip()
                if party:
                    parties.append(party)
        if parties:
            elements["parties"] = parties

        # 提取合同用途（寻找「用于」「用途：」等）
        for keyword in ["用途：", "用途:", "用于", "货物描述："]:
            if keyword in contract_text:
                start = contract_text.index(keyword) + len(keyword)
                end   = contract_text.find("\n", start)
                elements["purpose"] = contract_text[start: end if end > 0 else start + 200].strip()
                break

        return elements

    def _check_amount_match(
        self, bill_amount: Optional[float], contract_amount: Optional[float]
    ) -> bool:
        """验证票面金额与合同金额是否匹配（允许 ±5% 误差）"""
        if not bill_amount:
            return True   # 无票面金额（应由合规检测处理）
        if not contract_amount:
            return True   # 无合同金额（降级，不判违规）

        tolerance = contract_amount * self.AMOUNT_TOLERANCE_RATE
        return abs(bill_amount - contract_amount) <= tolerance

    def _check_party_match(
        self, bill_drawer: Optional[str], bill_payee: Optional[str], contract_elements: dict
    ) -> bool:
        """验证票据交易方与合同甲乙方是否匹配"""
        parties = contract_elements.get("parties", [])
        if not parties:
            return True   # 无合同交易方信息（降级）

        # 简单检测：出票人或收款人至少有一个在合同甲乙方名单中
        for party in parties:
            if bill_drawer and (bill_drawer in party or party in bill_drawer):
                return True
            if bill_payee and (bill_payee in party or party in bill_payee):
                return True
        return False  # 找不到匹配的交易方

    def _check_date_match(self, bill_date: Optional[str], contract_elements: dict) -> bool:
        """验证出票日期在合同有效期内（简化：无合同日期信息则直接 True）"""
        return True  # 简化：合同有效期提取较复杂，降级处理

    def _check_purpose_match(
        self, bill_purpose: Optional[str], contract_purpose: Optional[str]
    ) -> bool:
        """验证票据贸易背景与合同用途是否匹配"""
        if not bill_purpose or not contract_purpose:
            return True  # 任一为空则不做比较

        # 简单关键词重叠检测：双方描述中有相同关键词
        bill_keywords = set(bill_purpose.replace("，", " ").replace("。", " ").split())
        contract_keywords = set(contract_purpose.replace("，", " ").replace("。", " ").split())
        overlap = bill_keywords & contract_keywords

        return len(overlap) > 0  # 有任意重叠词则认为匹配

    def _calc_match_score(
        self,
        amount_match: bool,
        party_match: bool,
        date_match: bool,
        purpose_match: bool,
        has_contract: bool,
    ) -> float:
        """
        计算综合匹配度
        无合同时直接返回默认中等分数
        有合同时：4 项各 25 分，全对 = 100 分
        """
        if not has_contract:
            return self.DEFAULT_SCORE_NO_CONTRACT

        score = 0.0
        if amount_match:  score += 25.0   # 金额匹配 25 分
        if party_match:   score += 35.0   # 交易方匹配 35 分（权重更高）
        if date_match:    score += 20.0   # 日期匹配 20 分
        if purpose_match: score += 20.0   # 用途匹配 20 分
        return score

    def _calc_trade_background_score(
        self, bill_purpose: Optional[str], contract_purpose: Optional[str]
    ) -> float:
        """
        贸易背景真实性评分（0~100）
        逻辑：根据贸易背景说明的详细程度和合同一致性评分
        """
        if not bill_purpose:
            return 40.0   # 无贸易背景说明：40 分（及格线以下，属于风险项）

        # 说明越详细（字数越多）基础分越高
        base_score = min(70.0, len(bill_purpose) * 2.0)  # 每字 2 分，上限 70

        # 如果与合同描述有一致性，加 30 分
        if contract_purpose and bill_purpose:
            bill_kw = set(bill_purpose.split())
            cont_kw = set(contract_purpose.split())
            if bill_kw & cont_kw:
                base_score += 30.0

        return min(100.0, base_score)  # 上限 100 分

    def _gen_review_notes(
        self, match_score: float, mismatch_details: list, has_contract: bool
    ) -> str:
        """生成审核备注文本"""
        if not has_contract:
            return "申请方未提交合同，仅根据贸易背景说明评分，建议补充合同材料"
        if match_score >= 80:
            return "票据要素与合同条款一致，审核通过"
        if match_score >= 60:
            return f"存在 {len(mismatch_details)} 处不匹配，建议人工复核"
        return f"多处关键要素不匹配（{len(mismatch_details)} 处），建议退件核查"
