# app/agents/compliance_retrieval_agent.py
# ComplianceRetrievalAgent：合规 RAG 检索专项 Agent
# 职责：
#   1. 从 shared_data["bill_element"] 读取 18 个要素字段的值
#   2. 为每个字段构建合规性查询问句，通过 asyncio.gather 并行向知识库发起检索
#   3. 根据 RAG 返回的合规依据判断每个字段是否合规
#   4. 将检查结果写入 compliance_checks 表（每字段一行）
#   5. 将合规汇总写入 shared_data["compliance_summary"]

from __future__ import annotations

import asyncio                             # gather 并行发起 18 个 RAG 查询
import uuid                                # 生成记录 UUID
from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.models.agent_models import ComplianceCheck, ViolationLevel
from app.core.cache import bill_cache, compute_bill_fingerprint  # 缓存层


# 18 个要素字段的合规查询模板
# 格式：{field_name: (查询模板, 相关法规引用)}
FIELD_QUERY_TEMPLATES = {
    "ticket_number":    ("票据号码格式和唯一性要求是什么？",        "票据法第22条"),
    "ticket_type":      ("银行承兑汇票与商业承兑汇票的区别和适用条件是什么？", "票据法第26条"),
    "issue_date":       ("出票日期的填写规范和格式要求是什么？",    "票据法第22条"),
    "due_date":         ("票据到期日的规定，最长期限是多少？",       "商业汇票承兑贴现办法第8条"),
    "amount_numeric":   ("票面金额填写规范，大小写一致性要求是什么？", "票据法第22条"),
    "amount_text":      ("票面金额大写格式规范是什么？",             "票据法第22条"),
    "currency":         ("票据币种的规定，人民币以外币种的限制是什么？", "外汇管理条例第11条"),
    "drawer":           ("出票人资质要求和名称填写规范是什么？",     "票据法第22条"),
    "drawer_account":   ("出票人银行账号填写规范和格式要求是什么？", "电子商业汇票系统管理办法第18条"),
    "drawer_bank":      ("出票人开户行填写规范，名称准确性要求是什么？", "银行账户管理办法第15条"),
    "acceptor":         ("承兑人资质要求，哪些机构可以作为承兑人？",  "商业汇票承兑贴现办法第5条"),
    "payee":            ("收款人填写规范，是否可以为持票人即付是什么？", "票据法第26条"),
    "drawee_bank":      ("付款行要求，承兑行与付款行的关系是什么？",   "商业汇票承兑贴现办法第6条"),
    "endorsers":        ("背书连续性要求，背书人与前手收款人的关系规定是什么？", "票据法第33条"),
    "maturity_days":    ("票据期限的最长规定，超期票据如何处理？",    "商业汇票承兑贴现办法第8条"),
    "trade_purpose":    ("贸易背景真实性要求，贴现时如何证明贸易背景？", "中国人民银行令2016第3号第15条"),
    "acceptance_clause": ("承兑条款的法律效力和规范要求是什么？",     "票据法第44条"),
    "special_remarks":  ("票据记载事项的限制，哪些内容不得记载？",    "票据法第8条"),
}

# RAG 合规判断阈值：RAG 分数高于此值认为找到了明确的合规规定
RAG_COMPLIANCE_THRESHOLD = 0.70


class ComplianceRetrievalAgent(BaseAgent):
    """
    合规 RAG 检索 Agent：并行查询 18 个要素字段的合规依据
    每个字段对应一条 compliance_checks 记录
    """

    agent_name = "compliance_retrieval_agent"

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        **kwargs
    ) -> AgentResult:
        """
        并行合规检索主逻辑

        Args:
            ctx: 上下文（从 shared_data["bill_element"] 读取 18 个字段）
            db:  数据库会话

        Returns:
            AgentResult: data 包含违规字段数量和整体合规结论
        """
        # 步骤 1：从 shared_data 读取要素（由 ElementExtractionAgent 预先写入）
        bill_element = ctx.shared_data.get("bill_element")

        logger.info(
            f"[{self.agent_name}] 开始合规检索 task={ctx.audit_task_id} "
            f"ticket={bill_element.get('ticket_number') if bill_element else None} "
            f"has_element={bool(bill_element)}"
        )

        if not bill_element:
            logger.error(
                f"[{self.agent_name}] 缺少 bill_element，合规检索无法执行 "
                f"task={ctx.audit_task_id}"
            )
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                error_code="C_NO_ELEMENT",
                error_msg="shared_data 中缺少 bill_element，请确认 ElementExtractionAgent 已先执行",
            )

        # 步骤 2：计算票据指纹，查询合规检索缓存（18字段RAG 最贵，优先复用）
        bill_fp = compute_bill_fingerprint(bill_element)
        cached_checks = await bill_cache.get_compliance(ctx.tenant_id, bill_fp)
        if cached_checks is not None:
            # 缓存命中：直接从缓存恢复合规检查结果，写入 DB 并更新 shared_data
            logger.info(
                f"[{self.agent_name}] 合规检索缓存命中 "
                f"bill_fp={bill_fp[:8]} fields={len(cached_checks)} task={ctx.audit_task_id}"
            )
            violation_count = sum(1 for c in cached_checks if not c.get("is_compliant"))
            severe_count = sum(
                1 for c in cached_checks
                if c.get("violation_level") == ViolationLevel.SEVERE.value
            )
            # 将缓存数据写入本次审核任务（每次审核都需要独立的 DB 记录）
            for c in cached_checks:
                db.add(ComplianceCheck(
                    id=str(uuid.uuid4()),
                    audit_task_id=ctx.audit_task_id,
                    element_field=c.get("element_field"),
                    element_value=c.get("element_value"),
                    regulation_ref=c.get("regulation_ref"),
                    rag_query=c.get("rag_query"),
                    rag_answer=c.get("rag_answer"),
                    is_compliant=c.get("is_compliant", True),
                    violation_level=ViolationLevel(c["violation_level"]) if c.get("violation_level") else None,
                    violation_desc=c.get("violation_desc"),
                ))
            total_fields = len(cached_checks)
            compliance_rate = round((total_fields - violation_count) / total_fields, 3)
            violations = [
                {
                    "field":          c["element_field"],
                    "violation_level": c.get("violation_level"),
                    "violation_desc":  c.get("violation_desc"),
                    "regulation_ref":  c.get("regulation_ref"),
                }
                for c in cached_checks
                if not c.get("is_compliant") and c.get("violation_level") != ViolationLevel.INFO.value
            ]
            ctx.shared_data["compliance_summary"] = {
                "total_fields":       total_fields,
                "violation_count":    violation_count,
                "severe_count":       severe_count,
                "compliance_rate":    compliance_rate,
                "is_overall_compliant": severe_count == 0,
                "violations":         violations,   # 每字段违规明细（供聊天报告展示）
                "from_cache":         True,
            }
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                data={
                    "violation_count": violation_count,
                    "severe_count":    severe_count,
                    "compliance_rate": compliance_rate,
                    "total_fields":    total_fields,
                    "violations":      violations,   # 每字段违规明细，供 LangGraph state 传递
                    "is_overall_compliant": severe_count == 0,
                    "from_cache":      True,
                },
            )

        # 步骤 3：并行查询 18 个字段的合规规定（asyncio.gather 并发，最大化效率）
        logger.info(
            f"[{self.agent_name}] 开始并行合规检索 "
            f"fields={len(FIELD_QUERY_TEMPLATES)} task={ctx.audit_task_id}"
        )

        ticket_type = bill_element.get("ticket_type") or ""  # 票据类型，用于区分必填规则

        # 构建并发任务列表：每个字段对应一个 RAG 查询任务
        tasks = [
            self._check_field(
                field_name=field_name,
                field_value=bill_element.get(field_name),
                query_template=query_tpl,
                regulation_ref=regulation,
                audit_task_id=ctx.audit_task_id,
                ticket_type=ticket_type,
            )
            for field_name, (query_tpl, regulation) in FIELD_QUERY_TEMPLATES.items()
        ]

        # asyncio.gather：并行执行所有任务（等所有完成后才继续）
        check_results = await asyncio.gather(*tasks)

        # 步骤 4：批量写入 compliance_checks 表
        violation_count = 0
        severe_count = 0
        for check in check_results:
            db.add(check)  # 加入会话（批量提交，性能优于逐条提交）
            if not check.is_compliant:
                violation_count += 1
                if check.violation_level == ViolationLevel.SEVERE:
                    severe_count += 1

        # 步骤 5：汇总写入 shared_data，供 RiskAssessmentAgent 计算合规维度得分
        total_fields = len(check_results)
        compliance_rate = round((total_fields - violation_count) / total_fields, 3)

        violations = [
            {
                "field":          c.element_field,
                "violation_level": c.violation_level.value if c.violation_level else None,
                "violation_desc":  c.violation_desc,
                "regulation_ref":  c.regulation_ref,
            }
            for c in check_results
            if not c.is_compliant and c.violation_level != ViolationLevel.INFO
        ]

        ctx.shared_data["compliance_summary"] = {
            "total_fields":    total_fields,
            "violation_count": violation_count,
            "severe_count":    severe_count,
            "compliance_rate": compliance_rate,     # 合规率：0.0~1.0
            "is_overall_compliant": severe_count == 0,
            "violations":      violations,          # 每字段违规明细（供聊天报告展示）
        }

        # 步骤 6：写入合规检索缓存（只缓存可序列化的字段，不含 ORM 对象）
        cache_payload = [
            {
                "element_field":   c.element_field,
                "element_value":   c.element_value,
                "regulation_ref":  c.regulation_ref,
                "rag_query":       c.rag_query,
                "rag_answer":      c.rag_answer,
                "is_compliant":    c.is_compliant,
                "violation_level": c.violation_level.value if c.violation_level else None,
                "violation_desc":  c.violation_desc,
            }
            for c in check_results
        ]
        await bill_cache.set_compliance(ctx.tenant_id, bill_fp, cache_payload)

        logger.info(
            f"[{self.agent_name}] 合规检索完成 "
            f"violations={violation_count} severe={severe_count} "
            f"compliance_rate={compliance_rate:.1%} task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "total_fields":    total_fields,
                "violation_count": violation_count,
                "severe_count":    severe_count,
                "compliance_rate": compliance_rate,
                "is_overall_compliant": severe_count == 0,
                "violations":      violations,   # 每字段违规明细，供 LangGraph state 传递
            },
        )

    async def _check_field(
        self,
        field_name: str,
        field_value,
        query_template: str,
        regulation_ref: str,
        audit_task_id: str,
        ticket_type: str = "",
    ) -> ComplianceCheck:
        """
        单字段合规检查：向知识库查询该字段的合规要求，并判断字段值是否符合

        Args:
            field_name:     字段名（如 "ticket_number"）
            field_value:    字段值（来自 bill_element）
            query_template: 合规查询问句
            regulation_ref: 相关法规引用
            audit_task_id:  任务 ID（用于外键）
            ticket_type:    票据类型（用于区分银行本票/银行承兑汇票等必填规则差异）

        Returns:
            ComplianceCheck: 可直接 db.add() 的 ORM 对象
        """
        # 构造完整查询问句（加入字段值上下文，提高检索相关性）
        if field_value:
            rag_query = f"{query_template}（当前值：{field_value}）"
        else:
            rag_query = f"{query_template}（当前值：未填写）"

        # 调用 RAG 服务进行检索（复用现有 rag_service），捕获检索层面的任何异常
        try:
            rag_answer, rag_score = await self._call_rag(rag_query)
        except Exception as e:
            # _call_rag 内部应处理异常，此处作为最终兜底：异常不阻断单字段检查
            logger.warning(f"[{self.agent_name}] 字段 {field_name} RAG 检索异常: {e}")
            rag_answer, rag_score = "知识库检索异常，请人工核查", 0.0

        # 根据字段值和 RAG 结果判断合规性（传入 ticket_type 区分必填规则）
        is_compliant, violation_level, violation_desc, suggestion = (
            self._judge_compliance(field_name, field_value, rag_answer, rag_score, ticket_type)
        )

        return ComplianceCheck(
            id=str(uuid.uuid4()),
            audit_task_id=audit_task_id,
            element_field=field_name,
            element_value=str(field_value) if field_value is not None else None,
            regulation_ref=regulation_ref,
            rag_query=rag_query,
            rag_answer=rag_answer,
            rag_score=rag_score,
            is_compliant=is_compliant,
            violation_level=violation_level,
            violation_desc=violation_desc,
            suggestion=suggestion,
        )

    async def _call_rag(self, query: str) -> tuple:
        """
        调用现有 RAG 服务进行知识库检索
        复用 app/services/rag_service.py 中的检索逻辑，但只检索不生成答案

        Returns:
            (rag_answer: str, rag_score: float)
        """
        try:
            from app.services.rag_service import RAGService  # 延迟导入

            service = RAGService()
            # 使用检索模式（不生成 LLM 答案，只返回最相关的合规规定片段）
            result = await asyncio.to_thread(
                service.retrieve_only, query, top_k=3
            )

            if result and result.chunks:
                # 取最高分的合规规定片段作为答案
                best_chunk = result.chunks[0]
                return best_chunk.content, float(best_chunk.score)
            else:
                return "未在知识库中找到相关合规规定", 0.0

        except Exception as e:
            # RAG 检索失败不阻断流程，返回默认值
            logger.warning(f"[{self.agent_name}] RAG 检索异常: {e}")
            return "知识库检索异常，请人工核查", 0.0

    # 字段名到中文显示名的映射（用于生成可读的违规描述）
    _FIELD_ZH = {
        "ticket_number":    "票据号码",
        "ticket_type":      "票据种类",
        "issue_date":       "出票日期",
        "due_date":         "到期日期",
        "amount_numeric":   "票面金额（数字）",
        "amount_text":      "票面金额（大写）",
        "currency":         "币种",
        "drawer":           "出票人",
        "drawer_account":   "出票人账号",
        "drawer_bank":      "出票人开户行",
        "acceptor":         "承兑人",
        "payee":            "收款人",
        "drawee_bank":      "付款行",
        "endorsers":        "背书人",
        "maturity_days":    "距到期天数",
        "trade_purpose":    "贸易背景",
        "acceptance_clause": "承兑条款",
        "special_remarks":  "特殊记载事项",
    }

    def _judge_compliance(
        self,
        field_name: str,
        field_value,
        rag_answer: str,
        rag_score: float,
        ticket_type: str = "",
    ) -> tuple:
        """
        根据字段值、票据类型和 RAG 答案判断合规性

        必填规则按票据类型区分：
          - 银行本票：出票人银行自身即为承兑方，acceptor / drawee_bank 不单独要求
          - 银行承兑汇票 / 商业承兑汇票：acceptor 和 drawee_bank 均为必填

        Returns:
            (is_compliant, violation_level, violation_desc, suggestion)
        """
        field_zh = self._FIELD_ZH.get(field_name, field_name)   # 始终用中文字段名

        # ── 各票据类型的必填字段集合 ──────────────────────────────────────────
        # 所有类型共有的必填字段
        _common_required = {
            "ticket_number", "ticket_type", "issue_date", "due_date",
            "amount_numeric", "drawer", "payee",
        }
        # 银行承兑汇票 / 商业承兑汇票：承兑人和付款行也是必填
        _acceptance_required = _common_required | {"acceptor", "drawee_bank"}
        # 银行本票：银行本身即为出票兼承兑方，acceptor 和 drawee_bank 不单独要求
        _promissory_required = _common_required

        is_bank_promissory  = "本票" in ticket_type                      # 银行本票
        is_acceptance_bill  = "承兑汇票" in ticket_type or not ticket_type  # 汇票（含未知类型）

        if is_bank_promissory:
            required_fields = _promissory_required
        else:
            required_fields = _acceptance_required   # 汇票及未知类型都要求承兑人

        # ── 必填字段为空：严重违规 ────────────────────────────────────────────
        if field_name in required_fields and not field_value:
            return (
                False,
                ViolationLevel.SEVERE,
                f"必填字段「{field_zh}」为空",
                f"请补充填写票据的「{field_zh}」字段",
            )

        # ── 承兑人为空但票据类型为银行本票：提示级（不违规，说明原因）─────────
        if field_name == "acceptor" and not field_value and is_bank_promissory:
            return (
                True,
                ViolationLevel.INFO,
                None,
                None,
            )

        # ── 贸易背景为空：警告级（贴现时必须填写）────────────────────────────
        if field_name == "trade_purpose" and not field_value:
            return (
                False,
                ViolationLevel.WARNING,
                "贸易背景说明为空，贴现申请时必须填写",
                "如申请贴现，请补充填写贸易背景说明",
            )

        # ── RAG 检索到明确规定且字段有值：合规 ───────────────────────────────
        if rag_score >= RAG_COMPLIANCE_THRESHOLD and field_value:
            return True, ViolationLevel.INFO, None, None

        # ── 其他情况（RAG 未找到明确规定）：默认合规，不阻断业务 ───────────
        return True, ViolationLevel.INFO, None, None
