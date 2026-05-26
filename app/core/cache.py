# app/core/cache.py
# 业务缓存核心层
# 解决的问题：
#   票据识别（视觉模型）、合规 RAG 检索、欺诈检测、整体审核这四类操作
#   成本高（LLM/向量模型 token 消耗）且对同一张票据结果稳定，
#   通过 Redis 缓存避免重复计算，显著降低响应时间和 API 费用。
#
# 缓存 Key 设计（按层次从高到低）：
#   audit_cache:{tenant_id}:{bill_fp}:{task_type}   → 审核任务级（最高层，4h）
#   compliance:{tenant_id}:{bill_fp}                → 合规检索（18字段，6h）
#   fraud:{tenant_id}:{bill_fp}                     → 欺诈检测（5维，2h）
#   rag:{tenant_id}:{query_fp}                      → RAG 问答（检索+LLM，2h）
#   intent:{query_fp}                               → 意图识别（LLM，24h）
#   recognition:{file_md5}                          → 票据图片识别（视觉模型，24h）
#
# 票据指纹（Bill Fingerprint）设计：
#   基于所有 18 个业务要素字段的 SHA-256 截断哈希（前24位16进制）
#   只有当票据所有字段完全相同时，才命中同一个缓存条目
#   这比仅用 ticket_number 更严格，防止同号码但要素不同的伪造票据命中缓存

import hashlib                    # SHA-256 哈希计算
import json                       # JSON 序列化/反序列化
from typing import Any, Optional  # 类型注解
from loguru import logger         # 结构化日志

try:
    import redis.asyncio as aioredis  # 异步 Redis 客户端（推荐）
    REDIS_ASYNC_AVAILABLE = True
except ImportError:
    try:
        import aioredis             # 旧版 aioredis 兜底
        REDIS_ASYNC_AVAILABLE = True
    except ImportError:
        REDIS_ASYNC_AVAILABLE = False  # Redis 不可用，所有缓存操作静默跳过

from config.settings import settings  # 全局配置（REDIS_URL、各 TTL）


# ── 票据指纹计算 ──────────────────────────────────────────────────────────────

# 参与指纹计算的全部 18 个要素字段（顺序固定，保证哈希确定性）
_BILL_FINGERPRINT_FIELDS = [
    "ticket_type",       # 票据类型（银行承兑汇票 / 商业承兑汇票）
    "ticket_number",     # 票据号码
    "issue_date",        # 出票日期
    "due_date",          # 到期日
    "amount_numeric",    # 金额（数字，float → str 参与哈希）
    "amount_text",       # 大写金额
    "currency",          # 币种
    "drawer",            # 出票人
    "drawer_account",    # 出票人账号
    "drawer_bank",       # 出票人开户行
    "acceptor",          # 承兑人
    "payee",             # 收款人
    "drawee_bank",       # 付款行
    "endorsers",         # 背书人列表（list，JSON 序列化）
    "maturity_days",     # 期限天数（扩展字段）
    "trade_purpose",     # 贸易用途（扩展字段）
    "acceptance_clause", # 承兑条款（扩展字段）
    "special_remarks",   # 特殊记载事项（扩展字段）
]


def compute_bill_fingerprint(bill_element: dict) -> str:
    """
    计算票据唯一指纹（基于全部 18 个要素字段的 SHA-256 哈希）

    设计原则：
    1. 字段列表固定排序，避免字典插入顺序影响哈希结果
    2. None 值统一转为 null 参与序列化，确保相同缺省的票据得到相同指纹
    3. 取前 24 位十六进制（96 bit 熵），碰撞概率 < 10^-28，工程上可忽略
    4. 不使用 ticket_number 单字段，防止号码相同但要素不同的票据命中缓存

    Args:
        bill_element: 票据要素字典（可来自 BillElement.dict() 或 Agent shared_data）

    Returns:
        24 位小写十六进制字符串，如 "a3f7c2e1b9d4051f8e3a7c62"
    """
    canonical = {
        field: bill_element.get(field)   # 字段缺失时取 None
        for field in _BILL_FINGERPRINT_FIELDS
    }
    # sort_keys=True + ensure_ascii=False 确保同内容得到同字符串
    # default=str 兜底处理 float('nan') 等不可 JSON 序列化值
    serial = json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()[:24]


def compute_query_fingerprint(query: str, bill_context: Optional[dict] = None) -> str:
    """
    计算 RAG 查询指纹（问题文本 + 可选票据上下文）

    当 bill_context 不为空时，将其指纹也纳入哈希，
    确保相同问题但不同票据上下文得到不同缓存条目。
    """
    payload = query.strip()
    if bill_context:
        payload += "|" + compute_bill_fingerprint(bill_context)  # 拼接票据指纹
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def compute_file_fingerprint(file_bytes: bytes) -> str:
    """
    计算文件内容 MD5（用于票据图片识别缓存的 Key）

    相同物理文件（同一张票据图片），MD5 完全一致，
    视觉模型识别结果可以安全复用。
    使用 MD5 而非 SHA-256 是因为速度更快，且文件场景下碰撞无安全威胁。
    """
    return hashlib.md5(file_bytes).hexdigest()  # 32 位十六进制


# ── Redis 异步缓存操作 ────────────────────────────────────────────────────────

class BillCache:
    """
    票据业务缓存客户端（异步 Redis）

    所有方法在 Redis 不可用时静默降级（返回 None / False），
    确保缓存失效不影响业务正常运行。

    使用方式：
        cache = BillCache()
        value = await cache.get("mykey")
        await cache.set("mykey", {"data": 123}, ttl=3600)
    """

    def __init__(self):
        self._redis = None  # 异步 Redis 连接（懒加载）

    async def _get_redis(self):
        """懒加载异步 Redis 连接，首次调用时建立，之后复用。"""
        if self._redis is None and REDIS_ASYNC_AVAILABLE:
            try:
                # 从连接 URL 创建连接池，decode_responses=True 自动解码字节为字符串
                self._redis = aioredis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    max_connections=20,    # 限制连接池大小，防止 Redis 连接耗尽
                    socket_connect_timeout=2,   # 连接超时 2s，失败快速降级
                )
                logger.info(f"[cache] Redis 异步连接建立 url={settings.REDIS_URL}")
            except Exception as e:
                logger.warning(f"[cache] Redis 连接失败，缓存降级: {e}")
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """
        读取缓存，返回反序列化后的 Python 对象。
        Key 不存在或 Redis 不可用时返回 None。
        """
        if not settings.CACHE_ENABLED:  # 全局开关关闭时立即返回
            return None
        r = await self._get_redis()
        if r is None:
            return None
        try:
            raw = await r.get(key)       # Redis GET，返回字符串或 None
            if raw is None:
                return None              # 缓存未命中
            return json.loads(raw)       # JSON 反序列化为 Python 对象
        except Exception as e:
            logger.debug(f"[cache] get 失败 key={key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int) -> bool:
        """
        写入缓存，自动 JSON 序列化，设置过期时间（秒）。
        写入成功返回 True，失败（包括 Redis 不可用）返回 False。
        """
        if not settings.CACHE_ENABLED:
            return False
        r = await self._get_redis()
        if r is None:
            return False
        try:
            raw = json.dumps(value, ensure_ascii=False, default=str)
            await r.set(key, raw, ex=ttl)   # SET key value EX ttl
            return True
        except Exception as e:
            logger.debug(f"[cache] set 失败 key={key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除指定缓存条目（用于精确失效）。"""
        r = await self._get_redis()
        if r is None:
            return False
        try:
            await r.delete(key)
            return True
        except Exception as e:
            logger.debug(f"[cache] delete 失败 key={key}: {e}")
            return False

    async def scan_delete(self, pattern: str) -> int:
        """
        模糊删除：扫描匹配 pattern 的所有 Key 并删除（用于票据更新后批量失效）。
        例如：scan_delete("compliance:tenant1:a3f7c2*") 删除该票据所有合规缓存。
        返回删除的条目数量。

        注意：SCAN 在大 Key 空间下有性能开销，仅在票据更新时调用，不在热路径上。
        """
        r = await self._get_redis()
        if r is None:
            return 0
        deleted = 0
        try:
            # 使用 SCAN 迭代器避免阻塞 Redis（比 KEYS 命令更安全）
            async for key in r.scan_iter(pattern, count=100):
                await r.delete(key)
                deleted += 1
            if deleted > 0:
                logger.info(f"[cache] 批量失效 pattern={pattern} deleted={deleted}")
        except Exception as e:
            logger.debug(f"[cache] scan_delete 失败 pattern={pattern}: {e}")
        return deleted

    async def exists(self, key: str) -> bool:
        """检查 Key 是否存在（不读取内容，用于条件判断）。"""
        r = await self._get_redis()
        if r is None:
            return False
        try:
            return bool(await r.exists(key))
        except Exception:
            return False

    async def ttl(self, key: str) -> int:
        """
        返回 Key 剩余过期秒数。
        -1 表示 Key 存在但无过期时间，-2 表示 Key 不存在。
        """
        r = await self._get_redis()
        if r is None:
            return -2
        try:
            return await r.ttl(key)
        except Exception:
            return -2

    # ── 语义化缓存接口（封装 Key 前缀规则）─────────────────────────────────────

    async def get_recognition(self, file_md5: str) -> Optional[dict]:
        """读取票据图片识别缓存（视觉模型结果）"""
        return await self.get(f"recognition:{file_md5}")

    async def set_recognition(self, file_md5: str, result: dict) -> bool:
        """写入票据图片识别缓存，TTL=24h（同一文件识别结果稳定）"""
        ok = await self.set(f"recognition:{file_md5}", result, settings.CACHE_RECOGNITION_TTL)
        if ok:
            logger.debug(f"[cache] 识别结果已缓存 md5={file_md5[:8]} ttl={settings.CACHE_RECOGNITION_TTL}s")
        return ok

    async def get_intent(self, query_fp: str) -> Optional[dict]:
        """读取意图识别缓存"""
        return await self.get(f"intent:{query_fp}")

    async def set_intent(self, query_fp: str, intent_id: str, confidence: float,
                         method: str) -> bool:
        """写入意图识别缓存，TTL=24h（相同问题意图稳定）"""
        return await self.set(
            f"intent:{query_fp}",
            {"intent_id": intent_id, "confidence": confidence, "method": method},
            settings.CACHE_INTENT_TTL,
        )

    async def get_rag(self, tenant_id: str, query_fp: str) -> Optional[dict]:
        """读取 RAG 问答缓存（含检索结果和 LLM 答案）"""
        return await self.get(f"rag:{tenant_id}:{query_fp}")

    async def set_rag(self, tenant_id: str, query_fp: str, result: dict) -> bool:
        """写入 RAG 问答缓存，TTL=2h（知识库短期稳定，但不宜过长）"""
        ok = await self.set(f"rag:{tenant_id}:{query_fp}", result, settings.CACHE_RAG_TTL)
        if ok:
            logger.debug(f"[cache] RAG 结果已缓存 tenant={tenant_id} fp={query_fp[:8]} ttl={settings.CACHE_RAG_TTL}s")
        return ok

    async def get_compliance(self, tenant_id: str, bill_fp: str) -> Optional[list]:
        """读取合规检索缓存（18字段 × 合规检查结果列表）"""
        return await self.get(f"compliance:{tenant_id}:{bill_fp}")

    async def set_compliance(self, tenant_id: str, bill_fp: str,
                             checks: list) -> bool:
        """写入合规检索缓存，TTL=6h（法规一天内通常不变）"""
        ok = await self.set(f"compliance:{tenant_id}:{bill_fp}", checks, settings.CACHE_COMPLIANCE_TTL)
        if ok:
            logger.info(
                f"[cache] 合规检索已缓存 tenant={tenant_id} bill_fp={bill_fp[:8]} "
                f"checks={len(checks)} ttl={settings.CACHE_COMPLIANCE_TTL}s"
            )
        return ok

    async def get_fraud(self, tenant_id: str, bill_fp: str) -> Optional[dict]:
        """读取欺诈检测缓存（五维评分结果）"""
        return await self.get(f"fraud:{tenant_id}:{bill_fp}")

    async def set_fraud(self, tenant_id: str, bill_fp: str, result: dict) -> bool:
        """写入欺诈检测缓存，TTL=2h（黑名单可能更新，不宜过长）"""
        ok = await self.set(f"fraud:{tenant_id}:{bill_fp}", result, settings.CACHE_FRAUD_TTL)
        if ok:
            logger.info(
                f"[cache] 欺诈检测已缓存 tenant={tenant_id} bill_fp={bill_fp[:8]} "
                f"ttl={settings.CACHE_FRAUD_TTL}s"
            )
        return ok

    async def get_audit(self, tenant_id: str, bill_fp: str, task_type: str) -> Optional[dict]:
        """
        读取审核任务级缓存。
        命中时返回 {"task_id": ..., "cached_at": ...}，
        调用方可直接告知用户已有已完成的审核任务，无需重新运行。
        """
        return await self.get(f"audit_cache:{tenant_id}:{bill_fp}:{task_type}")

    async def set_audit(self, tenant_id: str, bill_fp: str, task_type: str,
                        task_id: str) -> bool:
        """
        写入审核任务级缓存，TTL=4h（工作日内审核结果可复用）。
        存储完成的 task_id，下次提交同票据同类型时直接指向历史任务。
        """
        import time
        ok = await self.set(
            f"audit_cache:{tenant_id}:{bill_fp}:{task_type}",
            {"task_id": task_id, "cached_at": int(time.time())},
            settings.CACHE_AUDIT_TTL,
        )
        if ok:
            logger.info(
                f"[cache] 审核任务已缓存 tenant={tenant_id} bill_fp={bill_fp[:8]} "
                f"type={task_type} task_id={task_id[:8]} ttl={settings.CACHE_AUDIT_TTL}s"
            )
        return ok

    async def invalidate_bill(self, tenant_id: str, bill_fp: str) -> int:
        """
        票据更新后（如背书操作完成），失效该票据相关的所有中间层缓存。
        注意：不失效识别缓存（文件内容未变）和 RAG 缓存（问答与票据状态无关）。

        Args:
            tenant_id: 租户 ID
            bill_fp:   要失效的票据指纹

        Returns:
            删除的缓存条目总数
        """
        total = 0
        # 精确删除合规缓存（每张票据只有一条合规缓存）
        total += await self.scan_delete(f"compliance:{tenant_id}:{bill_fp}")
        # 精确删除欺诈缓存
        total += await self.scan_delete(f"fraud:{tenant_id}:{bill_fp}")
        # 模糊删除所有业务类型的审核任务缓存（同一票据可能有多种类型）
        total += await self.scan_delete(f"audit_cache:{tenant_id}:{bill_fp}:*")
        if total > 0:
            logger.info(
                f"[cache] 票据缓存已失效 tenant={tenant_id} bill_fp={bill_fp[:8]} total={total}"
            )
        return total

    async def close(self):
        """关闭 Redis 连接（应用关闭时调用）。"""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("[cache] Redis 连接已关闭")


# 全局缓存实例（整个应用共享，避免重复建立 Redis 连接）
bill_cache = BillCache()
