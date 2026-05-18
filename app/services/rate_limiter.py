# app/services/rate_limiter.py
# 租户级 QPS 限流与文档配额管控
# 解决两个问题：
# 1. QPS 限流：防止某个租户每秒发太多请求，把服务器打崩
# 2. 文档配额：防止某个租户无限上传文档，把存储空间耗尽
#
# 算法：滑动窗口（Sliding Window）
# 比固定窗口更准确，不会出现"窗口边界瞬间爆发"的问题

import time          # 获取当前时间戳
from loguru import logger  # 日志

# 尝试导入 Redis 库
try:
    import redis
    REDIS_AVAILABLE = True   # Redis 库已安装
except ImportError:
    REDIS_AVAILABLE = False  # 没安装，限流功能降级（放行所有请求）

from config.settings import settings  # 导入配置


class RateLimiter:
    """
    基于 Redis 的滑动窗口 QPS 限流器
    原理：用 Redis 的有序集合（Sorted Set）记录请求时间戳
    - key = "rl:租户ID"，value = 时间戳，score = 时间戳
    - 每次请求：删除窗口之外的旧记录，统计窗口内请求数，判断是否超限
    """

    def __init__(self):
        self._redis = None   # Redis 连接（懒加载，第一次用时才建连接）

    def _get_redis(self):
        """懒加载 Redis 连接"""
        if self._redis is None and REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
                logger.info(f"[rate_limiter] Redis 连接成功 url={settings.REDIS_URL}")  # 连接成功确认
            except Exception as e:
                logger.warning(f"[rate_limiter] Redis 连接失败，限流降级放行: {e}")  # 失败时的影响说明
        return self._redis

    def check_rate_limit(self, tenant_id: str, qps_limit: int = None) -> tuple[bool, dict]:
        """
        检查某个租户是否超过 QPS 限制
        参数：
          tenant_id：租户 ID
          qps_limit：每秒最多请求数（None 用配置默认值）
        返回：(是否放行, 限流信息字典)
          - True：允许本次请求
          - False：拒绝，需要等待一段时间
        """
        qps_limit = qps_limit or settings.TENANT_QPS_LIMIT  # 默认 20 次/秒
        r = self._get_redis()                                # 获取 Redis 连接

        if r is None:
            # Redis 不可用时，直接放行（宁可不限流，不能影响正常使用）
            return True, {"remaining": qps_limit, "reset_in": settings.TENANT_QPS_WINDOW}

        key = f"rl:{tenant_id}"           # 该租户的限流 Redis key
        now = time.time()                 # 当前时间戳（精确到小数秒）
        window_start = now - settings.TENANT_QPS_WINDOW
        # 窗口起始时间 = 现在 - 窗口大小（60秒），窗口外的记录视为"过期"

        # 使用 Redis Pipeline：把多个命令打包一次发送，减少网络往返次数（提高性能）
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        # zremrangebyscore：删除有序集合中 score 在 [0, window_start] 范围的记录
        # 即：删除当前时间窗口之前的所有旧请求记录
        pipe.zcard(key)
        # zcard：统计有序集合的元素数量（= 窗口内的请求数）
        pipe.zadd(key, {str(now): now})
        # zadd：把当前时间戳加入集合（记录本次请求）
        pipe.expire(key, settings.TENANT_QPS_WINDOW * 2)
        # expire：设置 key 的过期时间（窗口的 2 倍），避免 Redis 内存泄漏
        results = pipe.execute()          # 执行所有命令，返回各命令结果列表

        current_count = results[1]        # 取 zcard 的结果（窗口内请求数，不含刚加入的那条）

        # 判断是否超限：窗口内请求数 >= QPS限制 × 窗口秒数
        # 例如：QPS=20，窗口=60秒，则60秒内最多 20×60=1200 次请求
        if current_count >= qps_limit * settings.TENANT_QPS_WINDOW:
            r.zrem(key, str(now))         # 超限！撤回刚刚记录的这条请求（不计入统计）
            oldest = r.zrange(key, 0, 0, withscores=True)
            # zrange：取有序集合中第 0 个元素（score 最小的=最早的请求）
            # withscores=True：同时返回 score（时间戳）
            reset_in = settings.TENANT_QPS_WINDOW - (now - oldest[0][1]) if oldest else settings.TENANT_QPS_WINDOW
            # 计算还需等多久：窗口大小 - (现在 - 最早请求的时间) = 最早请求什么时候滑出窗口
            logger.warning(  # 超限告警：记录租户和窗口内请求数，方便判断是攻击还是正常高峰
                f"[rate_limiter] 租户 {tenant_id} QPS 超限 "
                f"count={current_count} limit={qps_limit * settings.TENANT_QPS_WINDOW} "
                f"reset_in={reset_in:.1f}s"
            )
            return False, {
                "remaining": 0,                 # 剩余可用次数为 0
                "reset_in": round(reset_in, 1), # 还需等待的秒数（保留1位小数）
                "limit": qps_limit,             # 限制值
            }

        # 未超限，放行
        remaining = max(0, qps_limit * settings.TENANT_QPS_WINDOW - current_count - 1)
        # 剩余次数 = 总配额 - 已用 - 1（减去本次请求）
        return True, {
            "remaining": remaining,
            "reset_in": settings.TENANT_QPS_WINDOW,  # 窗口重置时间
            "limit": qps_limit,
        }

    def check_doc_quota(self, tenant_id: str, current_count: int, quota: int = None) -> bool:
        """
        检查文档数量是否超过配额
        current_count：当前已有的文档数
        quota：最大允许数量（None 用配置默认值 10000）
        返回 True=未超配额，False=已超配额
        """
        quota = quota or settings.TENANT_DOC_QUOTA  # 默认 10000 个文档
        allowed = current_count < quota             # 严格小于（= 时也不允许继续上传）
        if not allowed:
            logger.warning(  # 配额耗尽告警：记录当前用量和上限，方便运营决策
                f"[rate_limiter] 租户 {tenant_id} 文档配额耗尽 "
                f"current={current_count} quota={quota}"
            )
        return allowed

    def increment_doc_count(self, tenant_id: str, delta: int = 1) -> int:
        """
        文档成功入库后，给该租户的文档计数 +1（或 +delta）
        返回更新后的文档数
        """
        r = self._get_redis()
        if r is None:
            return 0           # Redis 不可用，计数功能失效

        key = f"doc_count:{tenant_id}"  # 该租户文档计数的 Redis key
        return r.incrby(key, delta)     # Redis 原子性自增（线程安全，多个请求同时操作不会出错）

    def get_doc_count(self, tenant_id: str) -> int:
        """
        获取某个租户当前的文档数量
        """
        r = self._get_redis()
        if r is None:
            return 0                    # Redis 不可用，返回 0（乐观估计）

        key = f"doc_count:{tenant_id}"
        val = r.get(key)                # 从 Redis 读取计数值（字符串格式）
        return int(val) if val else 0   # 转为整数，如果 key 不存在则返回 0


# 全局限流器实例（整个应用共享）
rate_limiter = RateLimiter()
