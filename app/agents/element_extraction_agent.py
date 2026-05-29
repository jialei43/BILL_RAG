# app/agents/element_extraction_agent.py
# ElementExtractionAgent：票据要素抽取专项 Agent
# 职责：
#   1. 从 shared_data["parsed_doc"] 中获取解析后的文档内容
#   2. 调用现有 BillRecognitionService 进行 18 字段要素抽取（15基础+3扩展）
#   3. 将抽取结果写入 bill_elements 数据库表
#   4. 回填 ctx.bill_record_id 和 shared_data["bill_element"]

from __future__ import annotations

import uuid                               # 生成记录 UUID
from datetime import datetime             # 计算到期天数
from typing import Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base_agent import BaseAgent, AgentContext, AgentResult
from app.models.agent_models import BillElement  # 要素表 ORM
from app.models.db_models import Document        # 文档表（获取文件字节）


class ElementExtractionAgent(BaseAgent):
    """
    要素抽取 Agent：将解析后的文档转换为结构化的 18 字段票据要素
    包装现有 BillRecognitionService，扩展 3 个新字段，写入 bill_elements 表
    """

    agent_name = "element_extraction_agent"  # 与 DAG 节点名一致

    # 置信度告警阈值：低于此值记录 WARNING 并标记 bill_elements 为低置信
    CONFIDENCE_WARNING_THRESHOLD = 0.70

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        file_bytes: Optional[bytes] = None,  # 文件字节（优先使用，避免重复读磁盘）
        filename: Optional[str] = None,      # 文件名（用于 BillRecognitionService）
        file_path: Optional[str] = None,     # 文件路径（fallback：节点从 BillAuditState 传入）
        **kwargs
    ) -> AgentResult:
        """
        核心要素抽取逻辑

        Args:
            ctx:        执行上下文（含 document_id，结果回填 bill_record_id）
            db:         数据库会话
            file_bytes: 文件字节内容（若不传则从 document_id 读取数据库）
            filename:   文件名（供模型推断文件类型）

        Returns:
            AgentResult: 抽取结果，data 字段包含 bill_element_id 和 confidence
        """
        # ── 预填充快速路径：聊天咨询流程已通过 bill_recognition_service 提取要素 ──
        # 调用方通过 ctx.shared_data["bill_element"] 注入已识别的票据要素，
        # 此时无需再调用视觉模型，直接将要素写入 bill_elements 表后返回
        if ctx.shared_data.get("bill_element"):
            return await self._handle_prefilled_element(ctx, db)

        # 步骤 1：获取文件字节（优先使用传入参数，否则查库）
        resolved_bytes, resolved_filename = await self._resolve_file_bytes(
            ctx, db, file_bytes, filename, file_path
        )
        if resolved_bytes is None:
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                error_code="E_EXTRACT_NO_FILE",
                error_msg="无法获取文件内容，document_id 不存在或文件路径无效",
            )

        # 步骤 2：调用 BillRecognitionService 进行要素识别（同步调用，线程池包装）
        import asyncio
        recognition_result = await asyncio.to_thread(
            self._sync_recognize, resolved_bytes, resolved_filename
        )

        if not recognition_result.bills:
            # 识别到 0 张票据：可能是图像质量极差或不是票据文件
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                error_code="E_EXTRACT_NO_BILL_FOUND",
                error_msg="文件中未识别到任何票据要素，请检查文件质量",
            )

        # 步骤 3：取第一张票据的识别结果（一个文件通常只有一张票据）
        bill = recognition_result.bills[0]

        # 步骤 4：计算综合置信度（各字段置信度的加权平均）
        confidence, field_confidences = self._calc_confidence(recognition_result)

        # 步骤 5：计算距到期天数（due_date - 今天，便于快速判断期限风险）
        maturity_days = self._calc_maturity_days(bill.due_date)

        # 步骤 6：从解析结果中提取 3 个扩展字段（原 BillElement 不含这些字段）
        # BillRecognitionService 的 raw_texts 中可能包含这些信息，从中解析
        extended = self._extract_extended_fields(recognition_result.raw_texts)

        # 步骤 7：写入 bill_elements 表
        element_id = str(uuid.uuid4())
        bill_element = BillElement(
            id=element_id,
            audit_task_id=ctx.audit_task_id,
            # ── 15 个基础字段（直接映射 BillElement dataclass）──────────────
            ticket_number   = bill.ticket_number,
            ticket_type     = bill.ticket_type,
            issue_date      = bill.issue_date,
            due_date        = bill.due_date,
            amount_numeric  = bill.amount_numeric,
            amount_text     = bill.amount_text,
            currency        = bill.currency or "人民币",  # 默认人民币
            drawer          = bill.drawer,
            drawer_account  = bill.drawer_account,
            drawer_bank     = bill.drawer_bank,
            acceptor        = bill.acceptor,
            payee           = bill.payee,
            drawee_bank     = bill.drawee_bank,
            endorsers       = bill.endorsers or [],       # 空列表表示无背书
            maturity_days   = maturity_days,
            # ── 3 个扩展字段 ────────────────────────────────────────────────
            trade_purpose       = extended.get("trade_purpose"),
            acceptance_clause   = extended.get("acceptance_clause"),
            special_remarks     = extended.get("special_remarks"),
            # ── 质量控制字段 ──────────────────────────────────────────────────
            confidence_score    = confidence,
            field_confidences   = field_confidences,
            raw_ocr_result      = {"bills": recognition_result.raw_texts},  # 原始识别文本
            extraction_method   = "vision_llm",  # BillRecognitionService 默认使用视觉大模型
        )
        db.add(bill_element)   # 加入数据库会话（不提交，由 BaseAgent.execute() 的调用者提交）

        # 步骤 8：将要素写入 shared_data，供 ComplianceRetrievalAgent 等后续 Agent 读取
        ctx.shared_data["bill_element"] = {
            "id":            element_id,
            "ticket_number": bill.ticket_number,
            "ticket_type":   bill.ticket_type,
            "issue_date":    bill.issue_date,
            "due_date":      bill.due_date,
            "amount_numeric": bill.amount_numeric,
            "amount_text":   bill.amount_text,
            "currency":      bill.currency or "人民币",
            "drawer":        bill.drawer,
            "acceptor":      bill.acceptor,
            "payee":         bill.payee,
            "drawee_bank":   bill.drawee_bank,
            "endorsers":     bill.endorsers or [],
            "maturity_days": maturity_days,
            "trade_purpose": extended.get("trade_purpose"),
            "confidence_score": confidence,
        }

        # 步骤 9：告警低置信度
        if confidence < self.CONFIDENCE_WARNING_THRESHOLD:
            logger.warning(
                f"[{self.agent_name}] 要素抽取置信度偏低 "
                f"confidence={confidence:.3f} < {self.CONFIDENCE_WARNING_THRESHOLD} "
                f"task={ctx.audit_task_id}"
            )

        logger.info(
            f"[{self.agent_name}] 要素抽取完成 "
            f"ticket={bill.ticket_number} confidence={confidence:.3f} "
            f"task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                **ctx.shared_data["bill_element"],   # 完整 18 字段，供 LangGraph node 写回 state
                "element_id":    element_id,
                "low_confidence": confidence < self.CONFIDENCE_WARNING_THRESHOLD,
                "source":        "vision_llm",
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 私有辅助方法
    # ──────────────────────────────────────────────────────────────────────────

    async def _resolve_file_bytes(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        file_bytes: Optional[bytes],
        filename: Optional[str],
        file_path: Optional[str] = None,
    ):
        """
        获取文件字节内容
        优先级：file_bytes > file_path kwarg > parsed_doc["file_path"] > document_id 查库
        """
        if file_bytes is not None:
            return file_bytes, filename or "document.pdf"

        resolved_path = file_path  # 直接使用传入的路径（LangGraph 节点传入）

        # 次选：从 shared_data 的 parsed_doc 读取（DocumentParserAgent 已先执行的场景）
        if resolved_path is None:
            parsed_doc = ctx.shared_data.get("parsed_doc")
            if parsed_doc and "file_path" in parsed_doc:
                resolved_path = parsed_doc["file_path"]

        # 从数据库 documents 表查询文件路径
        if resolved_path is None and ctx.document_id:
            result = await db.execute(
                select(Document.file_path, Document.filename)
                .where(Document.id == ctx.document_id)
            )
            row = result.first()
            if row:
                resolved_path, filename = row[0], row[1]

        if resolved_path is None:
            return None, None  # 找不到文件，触发失败流程

        # 从磁盘读取文件字节
        try:
            with open(resolved_path, "rb") as f:  # rb = 二进制读取模式
                file_bytes = f.read()
            return file_bytes, filename or os.path.basename(resolved_path)
        except OSError as e:
            logger.error(f"[{self.agent_name}] 读取文件失败: {e}")
            return None, None

    def _sync_recognize(self, file_bytes: bytes, filename: str):
        """
        同步调用 BillRecognitionService（在线程池中执行，不阻塞事件循环）
        """
        from app.services.bill_recognition import BillRecognitionService  # 延迟导入

        service = BillRecognitionService()  # 创建新实例（线程安全）
        return service.recognize_file(file_bytes, filename)

    def _calc_confidence(self, recognition_result) -> tuple:
        """
        计算要素抽取的综合置信度
        逻辑：统计非空字段占总字段数的比例作为简单置信度

        Returns:
            (overall_confidence, field_confidences_dict)
        """
        if not recognition_result.bills:
            return 0.0, {}

        bill = recognition_result.bills[0]
        # 15 个基础字段的值
        field_values = {
            "ticket_number":  bill.ticket_number,
            "ticket_type":    bill.ticket_type,
            "issue_date":     bill.issue_date,
            "due_date":       bill.due_date,
            "amount_numeric": bill.amount_numeric,
            "amount_text":    bill.amount_text,
            "currency":       bill.currency,
            "drawer":         bill.drawer,
            "drawer_account": bill.drawer_account,
            "drawer_bank":    bill.drawer_bank,
            "acceptor":       bill.acceptor,
            "payee":          bill.payee,
            "drawee_bank":    bill.drawee_bank,
        }

        # 为每个字段计算二值置信度：有值=1.0，无值=0.0
        # 真实场景中视觉模型会返回精确置信度，此处作简化估算
        field_confidences = {
            k: 1.0 if v else 0.0
            for k, v in field_values.items()
        }

        # 综合置信度 = 非空字段数 / 总字段数
        non_empty_count = sum(1 for v in field_confidences.values() if v > 0)
        overall = non_empty_count / len(field_confidences) if field_confidences else 0.0

        return round(overall, 3), field_confidences

    def _calc_maturity_days(self, due_date_str: Optional[str]) -> int:
        """
        计算距到期天数（due_date - 今天）
        Args:
            due_date_str: "YYYY-MM-DD" 格式的到期日字符串
        Returns:
            到期天数（负数表示已过期）
        """
        if not due_date_str:
            return 0  # 无到期日时返回 0

        try:
            due_date = datetime.strptime(due_date_str, "%Y-%m-%d")  # 解析日期字符串
            delta = due_date - datetime.now()                         # 计算时间差
            return delta.days                                          # 返回天数部分
        except ValueError:
            # 日期格式不对（如 "20240115" 缺少分隔符），返回 0 并记录日志
            logger.warning(f"[{self.agent_name}] 到期日格式无法解析: {due_date_str}")
            return 0

    def _extract_extended_fields(self, raw_texts: list) -> dict:
        """
        从原始 OCR 文本中提取 3 个扩展字段：
        trade_purpose / acceptance_clause / special_remarks
        这些字段在票据的「其他记载事项」栏填写，BillRecognitionService 未单独提取

        Args:
            raw_texts: 各页的原始 OCR 文本列表

        Returns:
            dict: {trade_purpose, acceptance_clause, special_remarks}
        """
        # 合并所有页面的 OCR 文本，便于全文检索
        full_text = "\n".join(raw_texts) if raw_texts else ""

        # 简单关键词匹配（生产环境可替换为 LLM 抽取，此处保持轻量化）
        trade_purpose = None
        acceptance_clause = None
        special_remarks = None

        # 贸易背景提取：常见填写格式「贸易背景：货款结算」或「用途：设备采购」
        for keyword in ["贸易背景：", "贸易背景:", "用途：", "用途:", "资金用途："]:
            if keyword in full_text:
                # 提取关键词之后的文本（直到换行符）
                start_idx = full_text.index(keyword) + len(keyword)
                end_idx = full_text.find("\n", start_idx)
                trade_purpose = full_text[start_idx: end_idx if end_idx > 0 else start_idx + 100].strip()
                break

        # 承兑条款提取：常见格式「承兑条件：」「特别承兑条款：」
        for keyword in ["承兑条款：", "承兑条款:", "承兑条件：", "特别承兑条款："]:
            if keyword in full_text:
                start_idx = full_text.index(keyword) + len(keyword)
                end_idx = full_text.find("\n", start_idx)
                acceptance_clause = full_text[start_idx: end_idx if end_idx > 0 else start_idx + 200].strip()
                break

        # 特殊记载事项：常见格式「其他记载事项：」「特殊事项：」「备注：」
        for keyword in ["其他记载事项：", "特殊事项：", "备注：", "备注:"]:
            if keyword in full_text:
                start_idx = full_text.index(keyword) + len(keyword)
                end_idx = full_text.find("\n", start_idx)
                special_remarks = full_text[start_idx: end_idx if end_idx > 0 else start_idx + 200].strip()
                break

        return {
            "trade_purpose":    trade_purpose,
            "acceptance_clause": acceptance_clause,
            "special_remarks":   special_remarks,
        }

    async def _handle_prefilled_element(
        self,
        ctx: AgentContext,
        db: AsyncSession,
    ) -> AgentResult:
        """
        预填充快速路径：shared_data["bill_element"] 已由聊天端点注入
        直接将要素写入 bill_elements 表，无需调用视觉模型（节省 3~8s 和 1000~2000 token）
        """
        pre = ctx.shared_data["bill_element"]  # 从上下文取出预填充的要素 dict
        element_id = pre.get("id") or str(uuid.uuid4())  # 复用已有 ID 或生成新 ID

        # 将预填充的 dict 写入 bill_elements 表（与正常路径写入内容一致）
        bill_element = BillElement(
            id=element_id,
            audit_task_id=ctx.audit_task_id,
            ticket_number     = pre.get("ticket_number"),
            ticket_type       = pre.get("ticket_type"),
            issue_date        = pre.get("issue_date"),
            due_date          = pre.get("due_date"),
            amount_numeric    = pre.get("amount_numeric"),
            amount_text       = pre.get("amount_text"),
            currency          = pre.get("currency", "人民币"),
            drawer            = pre.get("drawer"),
            drawer_account    = pre.get("drawer_account"),
            drawer_bank       = pre.get("drawer_bank"),
            acceptor          = pre.get("acceptor"),
            payee             = pre.get("payee"),
            drawee_bank       = pre.get("drawee_bank"),
            endorsers         = pre.get("endorsers", []),
            maturity_days     = pre.get("maturity_days", 0),
            trade_purpose     = pre.get("trade_purpose"),
            acceptance_clause = pre.get("acceptance_clause"),
            special_remarks   = pre.get("special_remarks"),
            confidence_score  = pre.get("confidence_score", 1.0),
            field_confidences = pre.get("field_confidences", {}),
            raw_ocr_result    = pre.get("raw_ocr_result", {}),
            extraction_method = "prefilled",  # 标记来源：由聊天端点预填充，非视觉模型
        )
        db.add(bill_element)

        # 确保 shared_data["bill_element"] 中的 id 与入库 ID 一致
        ctx.shared_data["bill_element"]["id"] = element_id

        confidence = pre.get("confidence_score", 1.0)

        logger.info(
            f"[{self.agent_name}] 预填充路径：跳过视觉模型，直接写库 "
            f"ticket={pre.get('ticket_number')} task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                **ctx.shared_data["bill_element"],   # 完整 18 字段，供 LangGraph node 写回 state
                "element_id":     element_id,
                "low_confidence": confidence < self.CONFIDENCE_WARNING_THRESHOLD,
                "source":         "prefilled",
            },
        )


import os  # os.path.basename 在 _resolve_file_bytes 中使用（放文件末尾避免顶层重复）
