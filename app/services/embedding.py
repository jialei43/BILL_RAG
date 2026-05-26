# app/services/embedding.py
# 混合向量化服务
# 核心任务：把文字转换成数字向量，让计算机能"理解"和比较文本的语义相似度
#
# 三种向量化方式：
# 1. BGE-M3 稠密向量：每段文字→1024个浮点数，捕捉语义信息
# 2. BGE-M3 稀疏语义权重：词语→重要性权重，类似关键词索引
# 3. jieba + BM25：传统关键词检索算法，专门针对中文票据词汇优化

import json      # JSON 序列化/反序列化（这里备用）
import pickle    # Python 对象序列化，用于把 BM25 索引对象转换成字节流存到 Redis
from pathlib import Path     # 跨平台路径处理
from typing import Optional  # 可选类型
import numpy as np           # NumPy：科学计算库，向量运算的核心工具
from loguru import logger    # 日志

# tqdm >= 4.66 移除了 tqdm.tqdm._lock，但 FlagEmbedding 内部仍直接访问它
# 在导入 FlagEmbedding 前补丁，避免 "has no attribute '_lock'" 报错
import tqdm as _tqdm_pkg
if not hasattr(_tqdm_pkg.tqdm, '_lock'):
    import threading
    _tqdm_pkg.tqdm._lock = threading.RLock()

# 尝试导入 FlagEmbedding（BGE-M3 模型库），不存在时优雅降级
try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    # BGEM3FlagModel：BGE-M3 嵌入模型，把文字变成向量
    # FlagReranker：精排模型，对候选文档重新打分排序
    FLAGEMB_AVAILABLE = True   # 标记：模型库已安装
except ImportError:
    FLAGEMB_AVAILABLE = False  # 没装，后面用随机向量代替（仅用于测试）

# 尝试导入 jieba 中文分词库
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# 尝试导入 BM25 算法库
try:
    from rank_bm25 import BM25Okapi
    # BM25Okapi：BM25 算法的一种变体，是搜索引擎中最经典的关键词检索算法
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from config.settings import settings   # 导入配置


# ──────────────────────────────────────────────────────────────────────────────
# jieba 分词初始化
# ──────────────────────────────────────────────────────────────────────────────
def init_jieba():
    """
    加载票据行业专用词典
    普通 jieba 不认识"承兑汇票"、"贴现"、"背书"等专业词汇
    加载词典后，这些词会被当作完整词处理，而不是被拆开
    """
    if not JIEBA_AVAILABLE:
        logger.warning("[embedding] jieba 未安装，中文分词降级为逐字模式，BM25 召回质量会下降")  # 影响检索质量
        return   # 没安装 jieba，直接返回

    dict_path = Path(settings.JIEBA_USER_DICT)  # 词典文件路径
    if dict_path.exists():                       # 词典文件存在
        jieba.load_userdict(str(dict_path))      # 加载词典（每行一个词）
        logger.info(f"Loaded bill domain jieba dict: {dict_path}")
    else:
        logger.warning(f"Jieba user dict not found: {dict_path}")  # 词典不存在，发出警告


def tokenize_chinese(text: str) -> list[str]:
    """
    中文分词：把一段中文文本切成词语列表
    例如："银行承兑汇票贴现" → ["银行承兑汇票", "贴现"]（加载专业词典后）
    如果 jieba 不可用，退化为逐字处理
    """
    if not JIEBA_AVAILABLE:
        return list(text)              # 没有 jieba，每个字符作为一个"词"
    return list(jieba.cut(text))       # jieba 分词，返回词语列表


# ──────────────────────────────────────────────────────────────────────────────
# BGE-M3 模型（单例模式：整个程序只加载一次，避免重复占用大量内存）
# ──────────────────────────────────────────────────────────────────────────────
import threading as _threading

_bge_model: Optional["BGEM3FlagModel"] = None   # 全局变量，存储已加载的模型实例
_reranker: Optional["FlagReranker"] = None       # 全局变量，存储已加载的精排模型实例
_bge_model_lock = _threading.Lock()             # 防止多线程同时触发懒加载（2GB 模型只能串行初始化）
_reranker_lock  = _threading.Lock()
_encode_lock    = _threading.Lock()             # 串行化 encode()：CPU 推理多线程并发反而互相竞争慢，串行更高效


def get_bge_model() -> "BGEM3FlagModel":
    """
    懒加载 BGE-M3 模型（第一次调用时才加载，之后复用）
    使用双重检查锁（double-checked locking）：外层 if 避免每次都抢锁（已加载后无开销），
    内层 if 防止多个线程同时通过外层检查后重复初始化。
    """
    global _bge_model
    if _bge_model is None and FLAGEMB_AVAILABLE:
        with _bge_model_lock:
            if _bge_model is None:               # 二次确认：防止排队等锁的线程重复加载
                logger.info(f"Loading BGE-M3 from {settings.BGE_M3_MODEL_PATH}")
                try:
                    _bge_model = BGEM3FlagModel(
                        settings.BGE_M3_MODEL_PATH,
                        use_fp16=False,          # macOS MPS FP16 会触发 Metal 断言崩溃，强制 CPU FP32
                        device="cpu",            # 强制 CPU，避免 Apple MPS datatype mismatch
                    )
                    logger.info("[embedding] BGE-M3 模型加载成功")
                except Exception as e:
                    logger.error(f"[embedding] BGE-M3 模型加载失败: {e}", exc_info=True)
    return _bge_model


def get_reranker() -> "FlagReranker":
    """懒加载 BGE-Reranker 精排模型（同样使用双重检查锁）"""
    global _reranker
    if _reranker is None and FLAGEMB_AVAILABLE:
        with _reranker_lock:
            if _reranker is None:
                logger.info(f"Loading BGE-Reranker from {settings.BGE_RERANKER_MODEL_PATH}")
                try:
                    _reranker = FlagReranker(
                        settings.BGE_RERANKER_MODEL_PATH,
                        use_fp16=False,          # macOS MPS FP16 崩溃，强制 CPU FP32
                        device="cpu",
                    )
                    logger.info("[embedding] BGE-Reranker 模型加载成功")
                except Exception as e:
                    logger.error(f"[embedding] BGE-Reranker 模型加载失败: {e}", exc_info=True)
    return _reranker


# ──────────────────────────────────────────────────────────────────────────────
# BM25 租户级索引
# ──────────────────────────────────────────────────────────────────────────────
class TenantBM25Index:
    """
    每个租户独立的 BM25 关键词检索索引
    BM25 原理：给每个词赋予一个重要性分数（TF-IDF 的改进版），
    查询时计算问题词汇在文档中的综合得分，分越高越相关

    票据场景的优势：对"贴现"、"承兑"等低频但高重要性的专业词汇有更好的召回
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id         # 这个索引属于哪个租户
        self._corpus: list[list[str]] = [] # 语料库：所有文档片段的分词结果列表
        self._doc_ids: list[str] = []      # 与语料库一一对应的片段 ID 列表
        self._bm25: Optional["BM25Okapi"] = None  # BM25 模型实例（fit后才有）
        self._dirty = False                # 标记：有新数据加入但还没保存到 Redis

    def add_texts(self, texts: list[str], doc_ids: list[str]):
        """
        增量添加文本到 BM25 索引
        texts：文档片段的文本列表
        doc_ids：对应的片段 ID 列表（一一对应）
        """
        tokenized = [tokenize_chinese(t) for t in texts]  # 对每段文本进行分词
        self._corpus.extend(tokenized)     # 把分词结果追加到语料库
        self._doc_ids.extend(doc_ids)      # 追加对应的 ID
        self._fit()                        # 重新训练 BM25 模型（加入新数据后必须重新 fit）
        self._dirty = True                 # 标记有新数据，提醒保存到 Redis
        logger.debug(f"BM25 [{self.tenant_id}] fit with {len(self._corpus)} docs")

    def _fit(self):
        """用当前语料库训练（fit）BM25 模型"""
        if not BM25_AVAILABLE:
            logger.warning("[embedding] rank_bm25 未安装，BM25 检索路径不可用")  # 影响检索质量
            return
        if not self._corpus:
            logger.debug(f"[embedding] BM25[{self.tenant_id}] 语料库为空，跳过 fit")  # 空库无需训练
            return
        self._bm25 = BM25Okapi(self._corpus)        # 传入分词后的语料库，BM25 计算 IDF 权重

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        BM25 检索
        query：用户的查询问题
        top_k：返回前 top_k 个结果
        返回：[(片段ID, 相关性分数), ...] 按分数降序排列
        """
        if self._bm25 is None or not self._doc_ids:  # 索引为空，返回空结果
            logger.debug(f"[embedding] BM25[{self.tenant_id}] 索引为空，返回空结果")  # 提示还未入库文档
            return []

        tokens = tokenize_chinese(query)              # 对查询词分词
        scores = self._bm25.get_scores(tokens)        # BM25 计算每个文档的得分（NumPy 数组）

        # 归一化：把分数缩放到 [0, 1] 区间，便于和其他检索结果融合
        max_score = scores.max() if scores.max() > 0 else 1.0  # 防止除以零
        norm_scores = scores / max_score              # 所有分数除以最大分数

        # 取分数最高的 top_k 个
        top_indices = np.argsort(norm_scores)[::-1][:top_k]
        # argsort：返回从小到大排序的索引；[::-1]：反转为从大到小；[:top_k]：取前 k 个
        return [(self._doc_ids[i], float(norm_scores[i])) for i in top_indices]

    def serialize(self) -> bytes:
        """把 BM25 索引序列化为字节流（用于存到 Redis）"""
        return pickle.dumps({"corpus": self._corpus, "doc_ids": self._doc_ids})
        # pickle.dumps：把 Python 对象转成字节串

    @classmethod
    def deserialize(cls, tenant_id: str, data: bytes) -> "TenantBM25Index":
        """从字节流恢复 BM25 索引（从 Redis 加载时使用）"""
        obj = cls(tenant_id)              # 创建一个空的索引对象
        loaded = pickle.loads(data)       # 字节串转回 Python 对象
        obj._corpus = loaded["corpus"]    # 恢复语料库
        obj._doc_ids = loaded["doc_ids"]  # 恢复 ID 列表
        obj._fit()                        # 重新训练 BM25 模型
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# BM25 持久化存储（Redis 后端）
# ──────────────────────────────────────────────────────────────────────────────
class BM25Store:
    """
    管理所有租户的 BM25 索引
    - 内存缓存：避免每次都从 Redis 读取（Redis I/O 有延迟）
    - Redis 持久化：程序重启后索引不丢失
    """

    def __init__(self):
        self._cache: dict[str, TenantBM25Index] = {}  # 内存缓存：{租户ID: BM25索引}
        self._redis = None                             # Redis 连接（懒加载）

    def _get_redis(self):
        """懒加载 Redis 连接（第一次用时才连接）"""
        if self._redis is None:
            try:
                import redis
                self._redis = redis.from_url(settings.REDIS_URL)  # 从 URL 创建连接
            except Exception as e:
                logger.warning(f"Redis not available: {e}")       # Redis 不可用时继续运行
        return self._redis

    def get_index(self, tenant_id: str) -> TenantBM25Index:
        """
        获取指定租户的 BM25 索引
        查找顺序：内存缓存 → Redis → 新建空索引
        """
        if tenant_id in self._cache:    # 内存有缓存，直接返回（最快）
            return self._cache[tenant_id]

        # 尝试从 Redis 恢复（程序重启后缓存丢失，从 Redis 恢复）
        r = self._get_redis()
        if r:
            try:
                data = r.get(f"bm25:{tenant_id}")   # Redis key 格式：bm25:租户ID
                if data:                             # Redis 中有数据
                    idx = TenantBM25Index.deserialize(tenant_id, data)  # 反序列化
                    self._cache[tenant_id] = idx     # 放入内存缓存
                    logger.info(  # 从 Redis 恢复成功，记录语料库规模方便确认索引完整性
                        f"[embedding] BM25[{tenant_id}] 从 Redis 恢复成功 "
                        f"docs={len(idx._doc_ids)}"
                    )
                    return idx
            except Exception as e:
                logger.warning(f"Redis BM25 load error: {e}")

        # 都没有，创建新的空索引
        idx = TenantBM25Index(tenant_id)
        self._cache[tenant_id] = idx
        return idx

    def save_index(self, tenant_id: str):
        """把内存中的 BM25 索引保存到 Redis（新文档入库后调用）"""
        idx = self._cache.get(tenant_id)   # 从内存取
        if not idx:
            return                         # 内存没有，无需保存

        r = self._get_redis()
        if r:
            try:
                r.set(f"bm25:{tenant_id}", idx.serialize())  # 序列化后存入 Redis
            except Exception as e:
                logger.warning(f"Redis BM25 save error: {e}")


# 全局 BM25 存储实例（整个应用共享）
bm25_store = BM25Store()


# ──────────────────────────────────────────────────────────────────────────────
# 主向量化服务
# ──────────────────────────────────────────────────────────────────────────────
class EmbeddingService:
    """
    混合向量化服务，对外提供两个主要功能：
    1. encode_texts：把文本列表转换成稠密+稀疏向量（用于文档入库）
    2. rerank：用 BGE-Reranker 对候选文档重新精排（用于提升检索质量）
    """

    def encode_texts(
        self,
        texts: list[str],            # 要向量化的文本列表
        batch_size: int = None,      # 每批处理的文本数（None=使用配置中的默认值）
    ) -> dict:
        """
        把文本列表转换成向量
        返回格式：{
            "dense": numpy数组，形状为 (N, 1024)，每行是一段文字的1024维向量
            "sparse": 列表，每个元素是 {词ID: 权重} 字典（稀疏向量）
        }
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE  # 使用配置的批大小
        total_batches = (len(texts) + batch_size - 1) // batch_size  # 计算总批次数
        logger.info(  # 向量化入口：记录文本数和批次数，方便预估耗时
            f"[embedding] 开始向量化 texts={len(texts)} "
            f"batch_size={batch_size} batches={total_batches}"
        )

        model = get_bge_model()   # 获取模型（懒加载，第一次会下载模型）

        if model is None:
            # 模型未安装时的降级处理：用随机向量代替（只用于开发测试）
            logger.warning("BGE-M3 not available, using random vectors")
            return {
                "dense": np.random.randn(len(texts), settings.MILVUS_DENSE_DIM).astype(np.float32),
                # randn：生成标准正态分布的随机数，shape=(文本数, 1024)
                "sparse": [{} for _ in texts],   # 空的稀疏向量
            }

        all_dense = []    # 收集所有批次的稠密向量
        all_sparse = []   # 收集所有批次的稀疏向量

        # _encode_lock 保证同一时刻只有一个线程调用 model.encode()
        # CPU 推理本身已用满所有核心，多线程并发只会互相竞争 PyTorch 内部线程池导致死锁
        with _encode_lock:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_no = i // batch_size + 1
                logger.debug(
                    f"[embedding] 向量化 batch {batch_no}/{total_batches} size={len(batch)}"
                )
                output = model.encode(
                    batch,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                all_dense.append(output["dense_vecs"])
                all_sparse.extend(output["lexical_weights"])

        # 把所有批次的稠密向量拼接成一个大数组
        dense = np.vstack(all_dense).astype(np.float32)
        # vstack：垂直堆叠，如把 [(32,1024), (32,1024)] 合并成 (64,1024)

        # L2 归一化：把所有向量的长度变为 1，这样内积（IP）就等于余弦相似度
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        # norm：计算每个向量的 L2 范数（向量长度）；axis=1：对每行分别计算；keepdims：保持维度
        dense = dense / (norms + 1e-8)
        # 加 1e-8（极小值）防止除以零（当向量全为0时）

        logger.info(  # 向量化完成：记录最终输出形状，确认无批次丢失
            f"[embedding] 向量化完成 dense_shape={dense.shape} sparse_count={len(all_sparse)}"
        )
        return {"dense": dense, "sparse": all_sparse}

    def encode_query(self, query: str) -> dict:
        """
        对单个查询字符串进行向量化（直接复用 encode_texts）
        返回格式与 encode_texts 相同
        """
        return self.encode_texts([query])  # 包装成列表传入，返回的也是列表形式

    def rerank(self, query: str, candidates: list[str], top_n: int = None) -> list[tuple[int, float]]:
        """
        BGE-Reranker 精排：重新给候选文档打分，选出最相关的

        原理：把 (问题, 候选文档) 配对输入，模型输出相关性分数
        比纯向量检索更准确，但计算量更大（只对少量候选做精排）

        参数：
          query：用户的原始问题
          candidates：候选文档文本列表（向量检索初步筛选出的）
          top_n：最终保留多少个结果
        返回：[(原始序号, 相关性分数), ...] 按分数降序排列
        """
        top_n = top_n or settings.RERANK_TOP_N   # 默认保留 5 个
        reranker = get_reranker()                 # 获取精排模型
        logger.info(  # 精排入口：记录候选数量，用于判断精排是否正常缩减候选
            f"[embedding] 开始精排 candidates={len(candidates)} top_n={top_n}"
        )

        if reranker is None:
            # 精排模型不可用时的降级：按原来的顺序，分数递减
            logger.warning("Reranker not available, using identity ranking")
            return [(i, 1.0 / (i + 1)) for i in range(min(top_n, len(candidates)))]
            # i=0 得分 1.0，i=1 得分 0.5，i=2 得分 0.33...（保持原有排序）

        # 构建 (问题, 候选文档) 配对列表
        pairs = [(query, c) for c in candidates]  # 每个候选都和问题配成一对

        # 精排模型计算每对的相关性分数（同样串行化，避免并发调用 reranker）
        with _encode_lock:
            scores = reranker.compute_score(pairs, normalize=True)
        # normalize=True：把分数归一化到 [0, 1] 区间

        # 按分数从高到低排序，返回 (原始索引, 分数) 对
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        # enumerate：给每个分数标上原始序号；sorted：按分数降序排；reverse=True：从高到低
        result = [(idx, float(score)) for idx, score in indexed[:top_n]]
        logger.info(  # 精排完成：记录最高分和最低分，方便判断精排效果
            f"[embedding] 精排完成 returned={len(result)} "
            f"top_score={(result[0][1] if result else 0):.4f}"
        )
        return result  # 只取前 top_n 个，转换为 Python float 类型


# 全局向量化服务实例（整个应用共享一个，避免重复加载模型）
embedding_service = EmbeddingService()
