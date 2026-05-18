# app/services/bill_lifecycle.py
# 票据生命周期管理服务（P1）
#
# 职责：将视觉识别的 BillElement 持久化到 PostgreSQL（BillRecord/BillVersion）
# 并将票据要素向量化写入 Milvus，支持票据流转的增量更新。
#
# 关键设计：
# - 以 ticket_number + tenant_id 为唯一键，同一张票据的多次上传只更新，不重复创建
# - 流转时仅追加新增背书人的向量块（P1-B），不删除历史向量
# - 矛盾检测在写入前完成，risk_flags 写入 BillRecord

import uuid
import time
from dataclasses import asdict
from datetime import date, datetime
from typing import Optional

from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import BillRecord, BillVersion, Document
from app.services.bill_recognition import BillElement
from app.services.vector_store import vector_store


# ── 要素矛盾检测规则 ──────────────────────────────────────────────────────────
def detect_risk_flags(element: BillElement) -> list[dict]:
    """
    对 BillElement 的字段做交叉校验，返回发现的矛盾/风险列表。
    每条风险格式：{"level": "severe/warning", "code": "DATE_LOGIC", "desc": "说明"}
    """
    flags = []

    # 规则 1：到期日不能早于出票日
    if element.issue_date and element.due_date:
        try:
            issue = datetime.strptime(element.issue_date[:10], "%Y-%m-%d").date()
            due = datetime.strptime(element.due_date[:10], "%Y-%m-%d").date()
            if due <= issue:
                flags.append({
                    "level": "severe",
                    "code": "DATE_LOGIC_ERROR",
                    "desc": f"到期日({element.due_date})不得早于或等于出票日({element.issue_date})，依《票据法》第22条"
                })
        except ValueError:
            pass  # 日期格式无法解析时跳过，不误报

    # 规则 2：银行承兑汇票的承兑人必须是银行类机构
    BANK_KEYWORDS = ["银行", "bank", "储蓄", "农信", "农商", "信用合作"]
    if element.ticket_type and "银行承兑" in element.ticket_type:
        if element.acceptor:
            is_bank = any(kw in element.acceptor.lower() for kw in BANK_KEYWORDS)
            if not is_bank:
                flags.append({
                    "level": "severe",
                    "code": "ACCEPTOR_NOT_BANK",
                    "desc": f"银行承兑汇票承兑人「{element.acceptor}」非银行机构，依《票据法》第38条"
                })

    # 规则 3：大写金额与数字金额不一致（简单量级核验）
    CHINESE_DIGITS = {
        "零": 0, "壹": 1, "贰": 2, "叁": 3, "肆": 4,
        "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9,
        "拾": 10, "佰": 100, "仟": 1000,
        "万": 10000, "亿": 100000000,
    }
    if element.amount_numeric and element.amount_text:
        amt = element.amount_numeric
        # 粗略判断：大写中有"百万"量级词但数字 < 100万，则不一致
        if "百万" in element.amount_text and amt < 1_000_000:
            flags.append({
                "level": "severe",
                "code": "AMOUNT_MISMATCH",
                "desc": f"大写金额含「百万」但数字金额为{amt}，两者量级不一致"
            })
        elif "千万" in element.amount_text and amt < 10_000_000:
            flags.append({
                "level": "severe",
                "code": "AMOUNT_MISMATCH",
                "desc": f"大写金额含「千万」但数字金额为{amt}，两者量级不一致"
            })

    # 规则 4：票据期限超过 6 个月（人行规定商业汇票最长承兑期限）
    if element.issue_date and element.due_date:
        try:
            issue = datetime.strptime(element.issue_date[:10], "%Y-%m-%d").date()
            due = datetime.strptime(element.due_date[:10], "%Y-%m-%d").date()
            days = (due - issue).days
            if days > 366:  # 超过约1年告警，超过182天提示
                flags.append({
                    "level": "warning",
                    "code": "TENOR_TOO_LONG",
                    "desc": f"票据期限 {days} 天，超过人行规定商业汇票最长承兑期限（6个月）"
                })
        except ValueError:
            pass

    return flags


# ── 向量块文本构造 ─────────────────────────────────────────────────────────────
def _build_bill_vector_text(element: BillElement, ticket_number: str) -> str:
    """将 BillElement 字段拼接成自然语言，用于向量化入库（P1-A）。"""
    endorser_str = "、".join(element.endorsers) if element.endorsers else "无"
    return (
        f"票据号码 {ticket_number} 的{element.ticket_type or '票据'}，"
        f"出票人：{element.drawer or '未知'}（账号：{element.drawer_account or '未知'}，"
        f"开户行：{element.drawer_bank or '未知'}），"
        f"承兑人：{element.acceptor or '未知'}，付款行：{element.drawee_bank or '未知'}，"
        f"出票日期：{element.issue_date or '未知'}，到期日：{element.due_date or '未知'}，"
        f"票面金额：{element.currency or '人民币'}{element.amount_text or '未知'}"
        f"（¥{element.amount_numeric or 0:,.2f}），"
        f"收款人：{element.payee or '未知'}，背书人：[{endorser_str}]"
    )


def _build_endorser_vector_text(ticket_number: str, version: int, new_endorsers: list[str],
                                 all_endorsers: list[str], upload_date: str) -> str:
    """构造流转增量背书人的向量块文本（P1-B）。"""
    chain = " → ".join(all_endorsers) if all_endorsers else "无"
    new_str = "、".join(new_endorsers)
    return (
        f"票据号码 {ticket_number} 第{version}次流转背书（{upload_date}），"
        f"新增背书人：{new_str}，"
        f"当前完整背书链：[{chain}]"
    )


# ── 主服务类 ──────────────────────────────────────────────────────────────────
class BillLifecycleService:
    """
    票据生命周期服务：
    - upsert_bill()：识别结果落库 + 向量化，处理新票/流转两种情形
    - get_bill_record()：按票据号码查询主档
    - get_bill_versions()：查询全部流转历史
    """

    async def upsert_bill(
        self,
        db: AsyncSession,
        element: BillElement,
        document_id: str,
        tenant_id: str,
        user_id: str,
    ) -> dict:
        """
        核心入口：将一次票据识别结果写入生命周期表并向量化。
        返回：{ bill_record_id, ticket_number, version, is_new_bill,
                new_endorsers, risk_flags, document_id, elapsed_ms }
        """
        t_start = time.perf_counter()

        ticket_number = element.ticket_number or f"UNKNOWN-{uuid.uuid4().hex[:8]}"
        risk_flags = detect_risk_flags(element)

        # ── 查询是否已存在该票据 ──────────────────────────────────────────────
        stmt = select(BillRecord).where(
            and_(
                BillRecord.tenant_id == tenant_id,
                BillRecord.ticket_number == ticket_number,
            )
        )
        result = await db.execute(stmt)
        existing: Optional[BillRecord] = result.scalar_one_or_none()

        if existing is None:
            # ── 新票据：创建主档 + v1 版本 + 完整向量块 ───────────────────────
            bill_record = BillRecord(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                ticket_number=ticket_number,
                ticket_type=element.ticket_type,
                issue_date=element.issue_date,
                due_date=element.due_date,
                amount_numeric=element.amount_numeric,
                amount_text=element.amount_text,
                currency=element.currency or "人民币",
                drawer=element.drawer,
                drawer_account=element.drawer_account,
                drawer_bank=element.drawer_bank,
                acceptor=element.acceptor,
                payee=element.payee,
                drawee_bank=element.drawee_bank,
                risk_flags=risk_flags,
                latest_version=1,
            )
            db.add(bill_record)

            bill_version = BillVersion(
                id=str(uuid.uuid4()),
                bill_record_id=bill_record.id,
                version=1,
                document_id=document_id,
                endorsers=element.endorsers,
                new_endorsers=element.endorsers,  # 首次全部都是"新增"
                uploaded_by=user_id,
                recognition_raw=asdict(element),
            )
            db.add(bill_version)
            await db.flush()  # 写入但不提交，让 ID 可用

            # P1-A：完整票据要素向量化
            await self._vectorize_bill_element(tenant_id, bill_record.id, ticket_number, element)

            logger.info(
                f"[bill_lifecycle] 新票据入库 ticket={ticket_number} "
                f"bill_id={bill_record.id} risks={len(risk_flags)}"
            )
            return {
                "bill_record_id": bill_record.id,
                "ticket_number": ticket_number,
                "version": 1,
                "is_new_bill": True,
                "new_endorsers": element.endorsers,
                "risk_flags": risk_flags,
                "document_id": document_id,
                "elapsed_ms": (time.perf_counter() - t_start) * 1000,
            }

        else:
            # ── 流转更新：找出新增背书人，追加版本 + 增量向量块 ──────────────
            prev_version_stmt = (
                select(BillVersion)
                .where(BillVersion.bill_record_id == existing.id)
                .order_by(BillVersion.version.desc())
            )
            prev_result = await db.execute(prev_version_stmt)
            prev_version: Optional[BillVersion] = prev_result.scalars().first()

            prev_endorsers = prev_version.endorsers if prev_version else []
            prev_set = set(prev_endorsers)
            new_endorsers = [e for e in element.endorsers if e not in prev_set]

            next_ver = existing.latest_version + 1
            bill_version = BillVersion(
                id=str(uuid.uuid4()),
                bill_record_id=existing.id,
                version=next_ver,
                document_id=document_id,
                endorsers=element.endorsers,
                new_endorsers=new_endorsers,
                uploaded_by=user_id,
                recognition_raw=asdict(element),
            )
            db.add(bill_version)

            # 更新主档版本号和风险标记（合并新旧 risk_flags 去重）
            existing_codes = {f["code"] for f in (existing.risk_flags or [])}
            merged_flags = list(existing.risk_flags or [])
            for flag in risk_flags:
                if flag["code"] not in existing_codes:
                    merged_flags.append(flag)
            existing.latest_version = next_ver
            existing.risk_flags = merged_flags
            await db.flush()

            # P1-B：仅追加新增背书人向量块
            if new_endorsers:
                upload_date = datetime.utcnow().strftime("%Y-%m-%d")
                await self._vectorize_new_endorsers(
                    tenant_id, existing.id, ticket_number,
                    next_ver, new_endorsers, element.endorsers, upload_date,
                )

            logger.info(
                f"[bill_lifecycle] 票据流转更新 ticket={ticket_number} "
                f"v{existing.latest_version} new_endorsers={new_endorsers}"
            )
            return {
                "bill_record_id": existing.id,
                "ticket_number": ticket_number,
                "version": next_ver,
                "is_new_bill": False,
                "new_endorsers": new_endorsers,
                "risk_flags": merged_flags,
                "document_id": document_id,
                "elapsed_ms": (time.perf_counter() - t_start) * 1000,
            }

    async def _vectorize_bill_element(
        self, tenant_id: str, bill_record_id: str, ticket_number: str, element: BillElement
    ) -> None:
        """P1-A：将完整票据要素写入 Milvus，chunk_type=bill_element。"""
        text = _build_bill_vector_text(element, ticket_number)
        try:
            vector_store.add_chunks(
                tenant_id=tenant_id,
                chunks=[{
                    "id": str(uuid.uuid4()),
                    "document_id": bill_record_id,
                    "chunk_index": 0,
                    "content": text,
                    "section_path": f"票据/{ticket_number}",
                    "page_num": 1,
                    "chunk_type": "bill_element",
                    "md5_hash": "",
                }],
            )
            logger.debug(f"[bill_lifecycle] 票据要素向量化完成 ticket={ticket_number}")
        except Exception as e:
            # 向量化失败不阻断主流程，记录日志后继续
            logger.warning(f"[bill_lifecycle] 向量化失败（票据要素）: {e}")

    async def _vectorize_new_endorsers(
        self, tenant_id: str, bill_record_id: str, ticket_number: str,
        version: int, new_endorsers: list[str], all_endorsers: list[str], upload_date: str
    ) -> None:
        """P1-B：追加新增背书人的向量块，不删除历史。"""
        text = _build_endorser_vector_text(
            ticket_number, version, new_endorsers, all_endorsers, upload_date
        )
        try:
            vector_store.add_chunks(
                tenant_id=tenant_id,
                chunks=[{
                    "id": str(uuid.uuid4()),
                    "document_id": bill_record_id,
                    "chunk_index": version,
                    "content": text,
                    "section_path": f"票据/{ticket_number}/背书链",
                    "page_num": version,
                    "chunk_type": "bill_endorser",
                    "md5_hash": "",
                }],
            )
            logger.debug(f"[bill_lifecycle] 背书人增量向量化完成 ticket={ticket_number} v{version}")
        except Exception as e:
            logger.warning(f"[bill_lifecycle] 向量化失败（背书人增量）: {e}")

    async def get_bill_record(
        self, db: AsyncSession, tenant_id: str, ticket_number: str
    ) -> Optional[BillRecord]:
        """按票据号码查询主档（含全部版本）。"""
        stmt = (
            select(BillRecord)
            .where(and_(BillRecord.tenant_id == tenant_id, BillRecord.ticket_number == ticket_number))
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_records(
        self, db: AsyncSession, tenant_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[BillRecord], int]:
        """分页查询租户下所有票据主档。"""
        from sqlalchemy import func
        count_stmt = select(func.count(BillRecord.id)).where(BillRecord.tenant_id == tenant_id)
        total = (await db.execute(count_stmt)).scalar_one()

        stmt = (
            select(BillRecord)
            .where(BillRecord.tenant_id == tenant_id)
            .order_by(BillRecord.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        records = (await db.execute(stmt)).scalars().all()
        return list(records), total


bill_lifecycle_service = BillLifecycleService()
