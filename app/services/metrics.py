# app/services/metrics.py
# Prometheus 监控埋点服务
# 作用：在代码的关键位置"埋点"，收集运行时指标（耗时、次数等）
# 这些指标会被 Prometheus 定期抓取，然后在 Grafana 上可视化展示
#
# 监控的维度：
# - 查询次数（按租户、按状态）
# - 各环节耗时（embedding推理、Milvus检索、Reranker精排、LLM生成）
# - 业务指标（表格提取成功率、OCR处理量、限流次数）

from loguru import logger  # 日志

# 尝试导入 Prometheus 客户端库
try:
    from prometheus_client import (
        Counter,            # 计数器：只增不减（如：总请求数）
        Histogram,          # 直方图：记录值的分布（如：耗时分布，可计算 P95）
        Gauge,              # 仪表盘：可增可减的当前值（如：当前活跃租户数）
        Summary,            # 摘要：类似 Histogram，但计算方式不同
        CollectorRegistry,  # 指标注册器
        generate_latest,    # 生成 Prometheus 格式的文本（供抓取接口使用）
        CONTENT_TYPE_LATEST,# Prometheus 文本的 Content-Type 头
    )
    PROMETHEUS_AVAILABLE = True   # Prometheus 库已安装
except ImportError:
    PROMETHEUS_AVAILABLE = False  # 没装，监控功能禁用

from config.settings import settings  # 配置（读取是否启用 Prometheus）


class MetricsService:
    """
    Prometheus 指标服务
    在整个应用的关键路径上收集性能和业务数据
    """

    def __init__(self):
        # 检查是否能启用监控（需要库已安装 + 配置开启）
        if not PROMETHEUS_AVAILABLE or not settings.PROMETHEUS_ENABLED:
            self._enabled = False   # 监控禁用，所有 record_* 方法都会直接返回
            return
        self._enabled = True        # 监控启用

        # ── 计数器（Counter）：只增不减，记录累计次数 ────────────────────────────
        self.query_total = Counter(
            "bill_rag_query_total",           # 指标名称（Prometheus 中的唯一标识）
            "总查询次数",                      # 指标描述
            ["tenant_id", "status"],           # 标签维度：按租户ID和状态分类统计
            # 例如：bill_rag_query_total{tenant_id="abc",status="success"} 42
        )

        self.ingest_total = Counter(
            "bill_rag_ingest_total",
            "文档入库次数",
            ["tenant_id", "status"],           # status="success" 或 "failed"
        )

        # ── 直方图（Histogram）：记录耗时分布，用于计算 P95、P99 等百分位延迟 ──────
        # 延迟分桶：0.1ms, 0.25ms, 0.5ms, 1ms, 2.5ms, 5s, 10s, 30s, 60s
        # 系统会记录每个桶（区间）有多少个请求，从而计算百分位
        latency_buckets = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

        self.embedding_latency = Histogram(
            "bill_rag_embedding_latency_ms",
            "Embedding 推理耗时（毫秒）",     # BGE-M3 把文字转向量的耗时
            ["tenant_id"],
            buckets=latency_buckets,
        )

        self.retrieval_latency = Histogram(
            "bill_rag_retrieval_latency_ms",
            "Milvus 检索耗时（毫秒）",        # 从向量库搜索相似文档的耗时
            ["tenant_id"],
            buckets=latency_buckets,
        )

        self.rerank_latency = Histogram(
            "bill_rag_rerank_latency_ms",
            "Reranker 精排耗时（毫秒）",      # BGE-Reranker 重新打分排序的耗时
            ["tenant_id"],
            buckets=latency_buckets,
        )

        self.llm_latency = Histogram(
            "bill_rag_llm_latency_ms",
            "LLM 生成耗时（毫秒）",           # 大语言模型生成答案的耗时
            ["tenant_id", "provider"],        # 额外按 LLM 提供商分类（openai/anthropic）
            buckets=latency_buckets,
        )

        self.total_latency = Histogram(
            "bill_rag_total_latency_ms",
            "全链路耗时（毫秒）",             # 从用户发问到返回答案的总耗时
            ["tenant_id"],
            buckets=latency_buckets,
        )

        self.parse_latency = Histogram(
            "bill_rag_parse_latency_ms",
            "PDF 解析耗时（毫秒）",           # 解析文档（OCR+表格提取）的耗时
            ["tenant_id"],
            buckets=latency_buckets,
        )

        # ── 业务指标计数器 ────────────────────────────────────────────────────
        self.top5_hit = Counter(
            "bill_rag_top5_hit_total",
            "Top-5 检索命中次数",             # 前5个检索结果包含正确答案的次数（评估检索质量）
            ["tenant_id"],
        )

        self.table_extracted = Counter(
            "bill_rag_table_extracted_total",
            "表格提取成功次数",               # 成功从 PDF 中提取出表格的次数
            ["tenant_id"],
        )

        self.ocr_processed = Counter(
            "bill_rag_ocr_processed_total",
            "OCR 处理页数",                   # 通过 PaddleOCR 识别的扫描页面数
            ["tenant_id"],
        )

        # ── 仪表盘（Gauge）：实时状态值，可增可减 ────────────────────────────
        self.active_tenants = Gauge(
            "bill_rag_active_tenants",
            "活跃租户数",                     # 当前有多少个租户在使用系统
        )

        self.doc_count = Gauge(
            "bill_rag_doc_count",
            "各租户文档总数",                 # 每个租户当前存储的文档数量
            ["tenant_id"],
        )

        self.chunk_count = Gauge(
            "bill_rag_chunk_count",
            "各租户 chunk 总数",              # 每个租户的文档片段总数
            ["tenant_id"],
        )

        # ── 限流计数器 ────────────────────────────────────────────────────────
        self.rate_limited = Counter(
            "bill_rag_rate_limited_total",
            "限流拒绝次数",                   # 被限流拒绝的请求数（高说明某租户频繁超限）
            ["tenant_id"],
        )

        logger.info("Prometheus metrics initialized")   # 初始化成功日志

    def _check(self) -> bool:
        """检查监控是否已启用（每个 record_* 方法都先调用这个）"""
        return self._enabled

    # ── 各类指标记录方法 ──────────────────────────────────────────────────────
    def record_query(self, tenant_id: str, status: str = "success"):
        """记录一次查询（status="success" 或 "error"）"""
        if not self._check(): return  # 监控未启用，直接返回
        self.query_total.labels(tenant_id=tenant_id, status=status).inc()
        # labels()：设置标签值；inc()：计数 +1

    def record_ingest(self, tenant_id: str, status: str = "success"):
        """记录一次文档入库"""
        if not self._check(): return
        self.ingest_total.labels(tenant_id=tenant_id, status=status).inc()

    def record_embedding_time(self, ms: float, tenant_id: str = "global"):
        """记录 Embedding 推理耗时（毫秒）"""
        if not self._check(): return
        self.embedding_latency.labels(tenant_id=tenant_id).observe(ms)
        # observe()：向直方图提交一个观测值

    def record_retrieval_time(self, ms: float, tenant_id: str = "global"):
        """记录 Milvus 检索耗时"""
        if not self._check(): return
        self.retrieval_latency.labels(tenant_id=tenant_id).observe(ms)

    def record_rerank_time(self, ms: float, tenant_id: str = "global"):
        """记录 Reranker 精排耗时"""
        if not self._check(): return
        self.rerank_latency.labels(tenant_id=tenant_id).observe(ms)

    def record_llm_time(self, ms: float, tenant_id: str = "global", provider: str = "openai"):
        """记录 LLM 生成耗时"""
        if not self._check(): return
        self.llm_latency.labels(tenant_id=tenant_id, provider=provider).observe(ms)

    def record_total_time(self, ms: float, tenant_id: str = "global"):
        """记录全链路总耗时"""
        if not self._check(): return
        self.total_latency.labels(tenant_id=tenant_id).observe(ms)

    def record_parse_time(self, ms: float, tenant_id: str = "global"):
        """记录文档解析耗时"""
        if not self._check(): return
        self.parse_latency.labels(tenant_id=tenant_id).observe(ms)

    def record_top5_hit(self, tenant_id: str):
        """记录一次 Top-5 检索命中"""
        if not self._check(): return
        self.top5_hit.labels(tenant_id=tenant_id).inc()

    def record_table_extraction(self, count: int, tenant_id: str = "global"):
        """记录表格提取成功（count 个表格）"""
        if not self._check(): return
        self.table_extracted.labels(tenant_id=tenant_id).inc(count)  # inc(n)：一次加 n

    def record_rate_limited(self, tenant_id: str):
        """记录一次被限流拒绝的请求"""
        if not self._check(): return
        self.rate_limited.labels(tenant_id=tenant_id).inc()

    def set_doc_count(self, tenant_id: str, count: int):
        """更新某租户的文档数量（Gauge 类型，可以设置任意值）"""
        if not self._check(): return
        self.doc_count.labels(tenant_id=tenant_id).set(count)  # set()：直接设置值

    def generate_metrics(self) -> tuple[bytes, str]:
        """
        生成 Prometheus 文本格式的指标数据
        供 /metrics 接口调用，Prometheus 定期来抓取这个接口
        返回：(字节内容, Content-Type 头)
        """
        if not self._check():
            return b"# Prometheus disabled", "text/plain"  # 监控未启用，返回提示文本
        return generate_latest(), CONTENT_TYPE_LATEST
        # generate_latest()：把所有已注册的指标序列化为 Prometheus 文本格式


# 全局监控服务实例（整个应用共享）
metrics = MetricsService()
