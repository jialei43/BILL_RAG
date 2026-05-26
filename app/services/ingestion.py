# app/services/ingestion.py
# 文档入库流水线编排器
# 把上传的文件变成可检索的向量数据，是整个系统最核心的流程
#
# 流水线步骤（类似工厂流水线）：
# 文件 → [Step1: PDF解析] → [Step2: 语义分块] → [Step3: 向量化+存储] → 完成
#
# 性能目标：单文档全流程 P95 耗时 < 45 秒

import time      # 计时
import uuid      # 生成唯一 ID
from pathlib import Path    # 路径处理
from loguru import logger   # 日志

from config.settings import settings                         # 配置
from app.services.pdf_parser import BillPDFParser, compute_md5  # PDF 解析器和 MD5 计算
from app.services.chunker import ChineseSemanticChunker      # 中文语义分块器
from app.services.vector_store import vector_store           # 向量存储服务
from app.services.rate_limiter import rate_limiter           # 限流服务（配额检查）
from app.services.metrics import metrics                     # 监控指标


class IngestionPipeline:
    """
    文档入库流水线
    编排解析→分块→向量化→存储的完整流程
    """

    def __init__(self):
        self.parser = BillPDFParser()          # PDF/Word/Excel/图像 解析器
        self.chunker = ChineseSemanticChunker()  # 中文语义分块器

    def ingest(
        self,
        file_path: str,          # 文件在服务器上的路径
        tenant_id: str,          # 哪个租户的文档
        document_id: str = None, # 文档 ID（None 则自动生成）
        metadata: dict = None,   # 额外元数据（预留扩展）
    ) -> dict:
        """
        执行完整的文档入库流程
        返回字典包含：document_id, chunk_count, page_count, parse_stats, elapsed_ms
        """
        document_id = document_id or str(uuid.uuid4())  # 没有传 ID 就生成一个 UUID
        metadata = metadata or {}                        # 没有传元数据就用空字典
        t_start = time.perf_counter()                    # 记录整体开始时间

        # checkpoint 文件路径：解析成功后写入此文件，全流程成功后删除
        # 命名规则：{document_id}_parsed.json，与文档 ID 一一对应
        checkpoint_path = Path(settings.PROCESSED_DIR) / f"{document_id}_parsed.json"

        # ── 配额检查：先确认该租户还有剩余文档配额 ──────────────────────────────
        current_count = rate_limiter.get_doc_count(tenant_id)       # 从 Redis 读取当前文档数
        if not rate_limiter.check_doc_quota(tenant_id, current_count):
            logger.warning(  # 配额满时记录当前用量，方便运营判断是否需要扩容
                f"[ingestion] 租户 {tenant_id} 文档配额已满 "
                f"current={current_count} quota={settings.TENANT_DOC_QUOTA}"
            )
            raise ValueError(f"租户 {tenant_id} 文档配额已满 (当前: {current_count})")

        # ── MD5 幂等检查：计算文件哈希值，后续用于检测跨 document_id 的重复文件 ───
        md5_hash = compute_md5(file_path)   # 32 位 MD5 字符串
        logger.info(f"[{tenant_id}] 开始入库: {Path(file_path).name} md5={md5_hash}")

        # ── Step 1: 文档解析（支持断点续传：有 checkpoint 直接加载，跳过 OCR/Vision）──
        t0 = time.perf_counter()
        if checkpoint_path.exists():
            # checkpoint 存在，说明上次解析已完成但后续步骤失败（如 Milvus 断连）
            try:
                from app.services.pdf_parser import ParsedDocument  # 局部导入避免循环依赖
                parsed_doc = ParsedDocument.load_checkpoint(str(checkpoint_path))
                logger.info(
                    f"[{tenant_id}] 命中解析 checkpoint: {parsed_doc.total_pages}页 "
                    f"{len(parsed_doc.elements)}元素，跳过重新解析"
                )
            except Exception as e:
                # checkpoint 文件损坏（如写入一半时进程 kill），删掉重新解析
                logger.warning(f"[{tenant_id}] checkpoint 加载失败 ({e})，重新解析")
                checkpoint_path.unlink(missing_ok=True)    # 删除损坏的 checkpoint
                parsed_doc = self.parser.parse(file_path)  # 重新完整解析
                parsed_doc.save_checkpoint(str(checkpoint_path))  # 解析成功后重新落盘
        else:
            # 首次入库：正常解析，然后立即将解析结果持久化
            parsed_doc = self.parser.parse(file_path)
            parsed_doc.save_checkpoint(str(checkpoint_path))   # 解析完成立即落盘
            # 落盘后，即使后续 Milvus 写入失败，重试时可直接跳过耗时的 OCR/Vision 步骤

        parse_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[{tenant_id}] 解析完成: {parsed_doc.total_pages}页 "
            f"{len(parsed_doc.elements)}元素 耗时={parse_ms:.0f}ms"
        )
        metrics.record_parse_time(parse_ms, tenant_id)

        # ── Step 2: 语义分块 ──────────────────────────────────────────────────
        # 把解析出的元素按语义边界切成适合向量化的小块
        t0 = time.perf_counter()
        chunks = self.chunker.chunk_document(parsed_doc)  # 执行分块
        chunk_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[{tenant_id}] 分块完成: {len(chunks)} chunks 耗时={chunk_ms:.0f}ms"
        )

        # 特殊情况：文档为空（如全是图片且 OCR 完全失败，或文件内容为空）
        if not chunks:
            logger.warning(f"[{tenant_id}] 文档 {document_id} 无有效 chunk")
            checkpoint_path.unlink(missing_ok=True)   # 空文档没有可续传的内容，清理 checkpoint
            return {
                "document_id": document_id,
                "chunk_count": 0,
                "page_count": parsed_doc.total_pages,
                "parse_stats": parsed_doc.parse_stats,
                "elapsed_ms": (time.perf_counter() - t_start) * 1000,
                "status": "empty",
            }

        # ── Step 3: 向量化 + Milvus 写入 + BM25 增量 fit ─────────────────────
        # 这是最耗时的步骤：把每个文本块转成 1024 维向量，然后写入 Milvus
        t0 = time.perf_counter()
        inserted = vector_store.upsert_chunks(
            tenant_id=tenant_id,
            document_id=document_id,
            chunks=chunks,
            md5_hash=md5_hash,      # 用于幂等检查（相同 MD5 的文件不重复写入）
        )
        vector_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[{tenant_id}] 向量写入完成: {inserted} vectors 耗时={vector_ms:.0f}ms"
        )
        metrics.record_embedding_time(vector_ms, tenant_id)  # 上报向量化耗时

        # ── 更新配额计数 + Prometheus 文档数 Gauge ────────────────────────────
        if inserted > 0:                          # 有实际写入（不是重复文档）
            new_count = rate_limiter.increment_doc_count(tenant_id)  # Redis 中的文档计数 +1
            metrics.set_doc_count(tenant_id, new_count)              # 更新文档数 Gauge
            logger.info(f"[ingestion] 配额计数更新 tenant={tenant_id} new_count={new_count}")
        else:
            logger.info(f"[ingestion] 重复文档跳过计数更新 tenant={tenant_id} doc={document_id}")

        # ── 计算全程耗时并打印汇总日志 ────────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[{tenant_id}] 入库完成: doc={document_id} "
            f"chunks={len(chunks)} total={total_ms:.0f}ms"
        )

        # 统计各类型表格提取成功的总数量（供 Prometheus 监控）
        stats = parsed_doc.parse_stats  # 解析统计信息字典
        total_tables = sum([
            stats.get("tables_lattice", 0),    # 有框线表格（Camelot lattice 方法）
            stats.get("tables_stream", 0),     # 无框线表格（Camelot stream 方法）
            stats.get("tables_pdfplumber", 0), # pdfplumber 兜底提取的表格
            stats.get("tables_vision", 0),     # 视觉模型兜底提取的表格
        ])
        metrics.record_table_extraction(total_tables, tenant_id)  # 上报表格数量

        # 统计 OCR 处理页数
        ocr_pages = stats.get("ocr_pages", 0) or stats.get("scanned_pages", 0)
        if ocr_pages > 0:
            metrics.record_ocr_processed(ocr_pages, tenant_id)

        # 更新 chunk 总数 Gauge（本次成功写入的 chunk 数）
        if inserted > 0:
            metrics.set_chunk_count(tenant_id, len(chunks))

        # ── 入库成功：删除 checkpoint 文件，释放磁盘空间 ─────────────────────────
        # 只有全流程（解析 + 向量化 + Milvus 写入）都成功后才删除
        # 如果此行之前任何步骤失败（抛出异常），checkpoint 会被保留，下次重试可复用
        checkpoint_path.unlink(missing_ok=True)                              # missing_ok=True：文件不存在时不报错
        logger.debug(f"[{tenant_id}] checkpoint 已清理: {checkpoint_path.name}")

        # 返回入库结果摘要
        return {
            "document_id": document_id,
            "md5_hash": md5_hash,
            "chunk_count": len(chunks),
            "page_count": parsed_doc.total_pages,
            "parse_stats": parsed_doc.parse_stats,
            "elapsed_ms": round(total_ms, 1),
            "status": "completed" if inserted > 0 else "duplicate",
            # _chunks 供 _update_doc_status 写入 PostgreSQL document_chunks 表
            # 仅新文档写入（inserted>0），重复文档已有记录，跳过
            "_chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "section_path": c.section_path,
                    "page_num": c.page_num,
                    "chunk_type": c.chunk_type,
                    "token_count": c.token_count,
                }
                for c in chunks
            ] if inserted > 0 else [],
        }

    def delete_document(self, tenant_id: str, document_id: str):
        """
        删除文档：从 Milvus 中删除该文档的所有向量片段
        注意：PostgreSQL 中的记录由路由层负责删除，这里只处理向量层
        """
        logger.info(f"[ingestion] 开始删除文档向量 tenant={tenant_id} doc={document_id}")  # 删除前记录，方便排查部分删除问题
        vector_store.delete_document(tenant_id, document_id)  # 从 Milvus 删除
        logger.info(f"[ingestion] 文档删除完成 tenant={tenant_id} doc={document_id}")  # 删除后确认


# 全局流水线实例（整个应用共享）
ingestion_pipeline = IngestionPipeline()
