# app/agents/bill_issuance_agent.py
# BillIssuanceAgent：出票资格预检专项 Agent
# 职责：
#   1. 调用 ElementExtractionAgent（M3）进行票据要素预验证
#   2. 调用 ComplianceRetrievalAgent（M4）合规子集（仅检查出票相关规则）
#   3. 查询黑名单（blacklist_entities）验证出票人/承兑人/收款人
#   4. 模拟外部授信系统接口（查询出票人的授信额度）
#   5. 综合判断是否允许出票，输出预检结论

from __future__ import annotations

import asyncio          # asyncio.gather：并行执行合规检查和黑名单查询
import uuid
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult


# ── 出票预检所需的必填字段（子集：只检查出票阶段必须有的字段）──────────────────
ISSUANCE_REQUIRED_FIELDS = {
    "ticket_number",   # 票据号码
    "issue_date",      # 出票日期
    "due_date",        # 到期日
    "amount_numeric",  # 票面金额
    "drawer",          # 出票人
    "acceptor",        # 承兑人
    "payee",           # 收款人
    "drawee_bank",     # 付款行
}

# ── 出票合规规则（出票阶段专用，比完整合规检查更严格）──────────────────────────
ISSUANCE_RULES = {
    "max_maturity_days": 365,       # 最长期限：365天（商业汇票承兑贴现办法第8条）
    "min_amount":        10000.0,   # 最低金额：1万元（业务惯例）
    "max_amount":        500000000.0,  # 最高金额：5亿元（单张票据限额）
}

# ── 模拟授信系统是否启用（生产环境从配置文件读取）────────────────────────────
CREDIT_SYSTEM_MOCK = True   # True=使用 Mock / False=调用真实接口


class BillIssuanceAgent(BaseAgent):
    """
    出票预检 Agent：在票据正式提交前完成快速资格验证
    预检通过后才允许进入完整的全流程审核（节约计算资源）
    """

    agent_name = "bill_issuance_agent"

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        **kwargs
    ) -> AgentResult:
        """
        出票预检主逻辑

        Args:
            ctx: 上下文（从 shared_data["bill_element"] 读取要素）
            db:  数据库会话

        Returns:
            AgentResult: data 包含 is_eligible / failed_checks / recommendation
        """
        bill_element = ctx.shared_data.get("bill_element", {})

        # 步骤 1：并行执行三项预检（要素完整性 + 黑名单 + 授信）
        check_elements_task  = self._check_required_elements(bill_element)
        check_blacklist_task = self._check_blacklist(bill_element, db)
        check_credit_task    = self._check_credit_system(bill_element)

        elements_result, blacklist_result, credit_result = await asyncio.gather(
            check_elements_task,
            check_blacklist_task,
            check_credit_task,
        )

        # 步骤 2：汇总失败项
        failed_checks: List[dict] = []
        failed_checks.extend(elements_result.get("failures", []))
        failed_checks.extend(blacklist_result.get("failures", []))
        failed_checks.extend(credit_result.get("failures", []))

        # 步骤 3：判断是否允许出票
        is_eligible = len(failed_checks) == 0
        recommendation = self._gen_recommendation(is_eligible, failed_checks)

        # 步骤 4：写入 shared_data 供 OrchestratorAgent 使用
        ctx.shared_data["issuance_result"] = {
            "is_eligible":         is_eligible,
            "failed_checks":       failed_checks,
            "failed_count":        len(failed_checks),
            "elements_passed":     elements_result.get("passed", True),
            "blacklist_passed":    blacklist_result.get("passed", True),
            "credit_passed":       credit_result.get("passed", True),
            "recommendation":      recommendation,
        }

        logger.info(
            f"[{self.agent_name}] 出票预检完成 "
            f"eligible={is_eligible} failed={len(failed_checks)} "
            f"task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "is_eligible":    is_eligible,
                "failed_checks":  failed_checks,
                "failed_count":   len(failed_checks),
                "recommendation": recommendation,
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 三项预检方法
    # ──────────────────────────────────────────────────────────────────────────

    async def _check_required_elements(self, bill_element: dict) -> dict:
        """
        检查出票必填字段完整性和业务规则
        包含：必填字段非空 + 金额范围 + 期限合规
        """
        failures = []

        # 必填字段检查
        for field in ISSUANCE_REQUIRED_FIELDS:
            if not bill_element.get(field):
                failures.append({
                    "check":    "required_field",
                    "field":    field,
                    "severity": "severe",
                    "message":  f"出票必填字段「{field}」为空",
                })

        # 金额范围检查
        amount = bill_element.get("amount_numeric")
        if amount is not None:
            if amount < ISSUANCE_RULES["min_amount"]:
                failures.append({
                    "check":    "amount_range",
                    "field":    "amount_numeric",
                    "severity": "severe",
                    "message":  f"票面金额 {amount} 低于最低限额 {ISSUANCE_RULES['min_amount']}",
                })
            elif amount > ISSUANCE_RULES["max_amount"]:
                failures.append({
                    "check":    "amount_range",
                    "field":    "amount_numeric",
                    "severity": "severe",
                    "message":  f"票面金额 {amount} 超过单张上限 {ISSUANCE_RULES['max_amount']}",
                })

        # 期限检查
        maturity_days = bill_element.get("maturity_days")
        if maturity_days and maturity_days > ISSUANCE_RULES["max_maturity_days"]:
            failures.append({
                "check":    "maturity_days",
                "field":    "maturity_days",
                "severity": "severe",
                "message":  f"票据期限 {maturity_days} 天超过法定最长期限 {ISSUANCE_RULES['max_maturity_days']} 天",
            })

        return {"passed": len(failures) == 0, "failures": failures}

    async def _check_blacklist(self, bill_element: dict, db: AsyncSession) -> dict:
        """
        查询黑名单：出票人、承兑人、收款人是否在 blacklist_entities 表中
        模拟实现：名称含 "BLACKLIST" 时命中
        生产环境替换为：数据库查询 blacklist_entities 表
        """
        failures = []
        entities_to_check = [
            ("drawer",   bill_element.get("drawer"),   "出票人"),
            ("acceptor", bill_element.get("acceptor"), "承兑人"),
            ("payee",    bill_element.get("payee"),    "收款人"),
        ]

        for field, name, label in entities_to_check:
            if name and "BLACKLIST" in str(name).upper():
                failures.append({
                    "check":    "blacklist",
                    "field":    field,
                    "severity": "severe",
                    "message":  f"{label}「{name}」在黑名单中，禁止出票",
                    "entity":   name,
                })

        return {"passed": len(failures) == 0, "failures": failures}

    async def _check_credit_system(self, bill_element: dict) -> dict:
        """
        查询授信系统：验证出票人是否有足够的授信额度
        模拟实现：金额 > 1亿时超额
        生产环境替换为：HTTP 调用授信系统 API
        """
        failures = []

        if not CREDIT_SYSTEM_MOCK:
            # 生产模式：调用真实接口（此处跳过，返回通过）
            return {"passed": True, "failures": [], "credit_available": None}

        # Mock 实现：金额超过 1 亿时模拟授信不足
        amount = bill_element.get("amount_numeric", 0)
        drawer = bill_element.get("drawer", "")
        credit_limit = 100_000_000.0   # 模拟授信额度 1 亿

        if amount > credit_limit:
            failures.append({
                "check":    "credit_limit",
                "field":    "amount_numeric",
                "severity": "warning",
                "message":  f"出票金额 {amount} 超过「{drawer}」的授信额度 {credit_limit}，需人工审批",
            })

        return {
            "passed":           len(failures) == 0,
            "failures":         failures,
            "credit_available": credit_limit,
            "credit_system":    "mock" if CREDIT_SYSTEM_MOCK else "production",
        }

    def _gen_recommendation(self, is_eligible: bool, failed_checks: list) -> str:
        """生成预检结论和建议"""
        if is_eligible:
            return "出票资格预检通过，可进入全流程审核"

        severe_count  = sum(1 for c in failed_checks if c.get("severity") == "severe")
        warning_count = sum(1 for c in failed_checks if c.get("severity") == "warning")

        if severe_count > 0:
            return (
                f"出票资格预检未通过：{severe_count} 项严重问题需修正，"
                f"请补充必要信息后重新提交"
            )
        return (
            f"出票资格预检存在 {warning_count} 项警告，"
            f"建议人工确认后决定是否继续"
        )
