# app/agents/document_parser_agent.py
# DocumentParserAgent：文档解析专项 Agent
# 职责：
#   1. 接收原始文件路径（PDF/图片），调用现有 BillPDFParser 进行解析
#   2. 计算解析置信度（基于 OCR 置信度分布），低于 0.85 触发 WARNING 日志
#   3. 将解析结果写入 ctx.shared_data["parsed_doc"]，供后续 Agent（要素抽取）使用
#   4. 更新 audit_tasks 的 current_agent 字段（由 BaseAgent.execute() 自动完成）

from __future__ import annotations

import os                             # 判断文件是否存在
from typing import Optional

from loguru import logger             # 结构化日志

from sqlalchemy.ext.asyncio import AsyncSession
from app.agents.base_agent import BaseAgent, AgentContext, AgentResult


class DocumentParserAgent(BaseAgent):
    """
    文档解析 Agent：将上传的原始文件解析为结构化文本和要素
    包装现有 BillPDFParser 服务，增加置信度监控和上下文传递
    """

    agent_name = "document_parser_agent"  # 与 orchestrator_agent.py 中的 DAG 节点名一致

    # 置信度告警阈值：低于此值记录 WARNING，触发人工复核流程
    CONFIDENCE_WARNING_THRESHOLD = 0.85

    async def run(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        file_path: Optional[str] = None,    # 文件路径（绝对路径或相对项目根目录的路径）
        **kwargs
    ) -> AgentResult:
        """
        核心解析逻辑

        Args:
            ctx:       执行上下文（含 document_id，解析结果写入 shared_data）
            db:        数据库会话（此 Agent 目前不写数据库，预留供将来扩展）
            file_path: 待解析的文件路径（优先使用此参数，否则从 document_id 查库获取）

        Returns:
            AgentResult: 解析结果，data 字段包含 ParsedDocument 的字典化表示
        """
        # 步骤 1：确定文件路径
        resolved_path = await self._resolve_file_path(ctx, db, file_path)
        if resolved_path is None:
            # 找不到文件路径时返回失败（不应继续要素抽取）
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                error_code="E_PARSE_NO_FILE",
                error_msg=f"无法确定文件路径：document_id={ctx.document_id}, file_path={file_path}",
            )

        # 步骤 2：检查文件是否实际存在
        if not os.path.exists(resolved_path):
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                error_code="E_PARSE_FILE_NOT_FOUND",
                error_msg=f"文件不存在: {resolved_path}",
            )

        # 步骤 3：调用 BillPDFParser 进行解析（解析逻辑封装在现有服务中，不重复实现）
        logger.info(f"[{self.agent_name}] 开始解析文件 path={resolved_path}")
        parsed_doc = await self._parse_file(resolved_path)

        # 步骤 4：计算置信度（解析统计中的 OCR 置信度均值，无则默认 1.0）
        confidence = self._calculate_confidence(parsed_doc)

        # 步骤 5：置信度低于阈值时记录告警（不阻断流程，由下游决定是否降级处理）
        if confidence < self.CONFIDENCE_WARNING_THRESHOLD:
            logger.warning(
                f"[{self.agent_name}] 文档解析置信度偏低 "
                f"confidence={confidence:.3f} < {self.CONFIDENCE_WARNING_THRESHOLD} "
                f"task={ctx.audit_task_id} path={resolved_path}"
            )

        # 步骤 6：将解析结果写入 shared_data，后续的 ElementExtractionAgent 从此读取
        doc_dict = self._to_dict(parsed_doc, file_path=resolved_path)  # 携带文件路径
        doc_dict["confidence"] = confidence      # 附加置信度字段（并非原始 ParsedDocument 的字段）
        ctx.shared_data["parsed_doc"] = doc_dict  # 写入共享上下文

        logger.info(
            f"[{self.agent_name}] 解析完成 elements={len(parsed_doc.elements)} "
            f"confidence={confidence:.3f} task={ctx.audit_task_id}"
        )

        return AgentResult(
            agent_name=self.agent_name,
            success=True,
            data={
                "file_path":    resolved_path,
                "element_count": len(parsed_doc.elements),  # 提取到的文本片段数量
                "page_count":   parsed_doc.parse_stats.get("page_count", 0),
                "confidence":   confidence,
                "low_confidence": confidence < self.CONFIDENCE_WARNING_THRESHOLD,
            },
        )

    async def _resolve_file_path(
        self,
        ctx: AgentContext,
        db: AsyncSession,
        file_path: Optional[str],
    ) -> Optional[str]:
        """
        确定实际文件路径：
        1. 如果 kwargs 中直接传了 file_path，优先使用
        2. 否则从 document_id 查询 documents 表获取 file_path
        """
        if file_path:
            return file_path  # 直接使用传入的路径（测试和 API 层直接调用时常用）

        if ctx.document_id:
            # 从数据库查询文档记录，获取存储路径
            from sqlalchemy import select
            from app.models.db_models import Document  # 延迟导入避免顶层循环依赖

            result = await db.execute(
                select(Document.file_path).where(Document.id == ctx.document_id)
            )
            row = result.first()
            if row and row[0]:
                return row[0]  # 返回数据库中存储的文件路径

        return None  # 既无直传路径又无数据库记录，返回 None 触发失败流程

    async def _parse_file(self, file_path: str):
        """
        调用 BillPDFParser 解析文件
        使用 asyncio.to_thread 将同步解析操作放入线程池，不阻塞事件循环
        """
        import asyncio
        from app.services.pdf_parser import BillPDFParser  # 延迟导入：避免启动时加载 PaddleOCR

        def _sync_parse():
            """同步解析函数，在线程池中执行"""
            parser = BillPDFParser()    # 每次创建新实例（线程安全，避免共享状态）
            return parser.parse(file_path)  # 调用现有解析逻辑

        # asyncio.to_thread：Python 3.9+，将阻塞操作放入默认线程池执行
        parsed_doc = await asyncio.to_thread(_sync_parse)
        return parsed_doc

    def _calculate_confidence(self, parsed_doc) -> float:
        """
        计算整体解析置信度
        逻辑：从 parse_stats 读取 OCR 置信度，无则从元素列表推算，最终兜底为 1.0
        """
        # 优先从解析统计中读取（BillPDFParser 会写入 ocr_avg_confidence）
        ocr_confidence = parsed_doc.parse_stats.get("ocr_avg_confidence")
        if ocr_confidence is not None:
            return float(ocr_confidence)

        # 次选：计算所有元素的置信度均值（元素 metadata 中可能包含 confidence 字段）
        confidences = []
        for elem in parsed_doc.elements:
            if hasattr(elem, "metadata") and "confidence" in elem.metadata:
                confidences.append(float(elem.metadata["confidence"]))
        if confidences:
            return sum(confidences) / len(confidences)

        # 兜底：无置信度信息时返回 1.0（假设解析完全成功）
        return 1.0

    def _to_dict(self, parsed_doc, file_path: str = None) -> dict:
        """
        将 ParsedDocument 转换为可序列化的字典格式
        供 shared_data 存储和后续 Agent（ElementExtractionAgent）读取
        """
        return {
            "file_path":   file_path,   # 供 ElementExtractionAgent._resolve_file_bytes() 使用
            "elements": [
                {
                    "text":    elem.content,
                    "type":    elem.element_type,
                    "metadata": elem.metadata,
                }
                for elem in parsed_doc.elements
            ],
            "parse_stats": parsed_doc.parse_stats,  # 包含页数、OCR 页数、表格数等统计
        }
