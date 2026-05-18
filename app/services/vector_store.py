# app/services/vector_store.py
# Milvus 向量存储服务
# 核心功能：存储和检索文档的向量表示
#
# 三路混合检索原理：
# - 路径1（稠密检索）：BGE-M3 稠密向量，捕捉语义相似（"贴现利率"能找到"折现率"的相关内容）
# - 路径2（稀疏检索）：BGE-M3 稀疏权重，类似关键词匹配（精确词语匹配）
# - 路径3（BM25检索）：经典关键词算法，对票据专业术语的低频词有更好的召回
# 三路结果用 RRF（倒数排名融合）算法合并，再用 BGE-Reranker 精排

import time      # 计时
import uuid      # 生成唯一 ID
from typing import Optional  # 可选类型
import numpy as np           # NumPy 数学运算
from loguru import logger    # 日志

# 尝试导入 Milvus 客户端库
try:
    from pymilvus import (
        connections,        # 管理 Milvus 连接（旧式 ORM API）
        Collection,         # 集合操作（查询/搜索/删除仍使用此 API）
        CollectionSchema,   # 集合的结构定义
        FieldSchema,        # 字段定义（类似列定义）
        DataType,           # 数据类型枚举（VARCHAR、FLOAT_VECTOR 等）
        utility,            # 工具函数（如检查集合是否存在）
        AnnSearchRequest,   # 近似最近邻搜索请求（用于 hybrid_search）
        RRFRanker,          # Milvus 内置的 RRF 融合排序器
        WeightedRanker,     # Milvus 内置的加权融合排序器
        MilvusClient,       # 新式客户端 API：pymilvus 2.4.3 Collection.insert() 对稀疏向量有 bug
                            # MilvusClient.insert() 走不同的序列化路径，稀疏向量可正常写入
    )
    MILVUS_AVAILABLE = True   # Milvus 库已安装
except ImportError:
    MILVUS_AVAILABLE = False  # 没装，向量检索功能不可用

# MilvusClient 实例（懒加载）：仅用于写入，查询/搜索仍走旧式 Collection API
_milvus_write_client: Optional["MilvusClient"] = None   # 全局写客户端，避免重复创建

from config.settings import settings                         # 配置
from app.services.embedding import embedding_service, bm25_store  # 向量化服务和 BM25 存储


# ──────────────────────────────────────────────────────────────────────────────
# Milvus 集合 Schema 定义（相当于建表语句）
# ──────────────────────────────────────────────────────────────────────────────
def _build_schema() -> "CollectionSchema":
    """
    定义 Milvus 集合的字段结构
    类比 SQL：CREATE TABLE bill_documents (id VARCHAR(100) PRIMARY KEY, ...)
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        # 主键：片段的唯一 ID（UUID 字符串，最长100字符）

        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=50,
                    is_partition_key=True),
        # 租户 ID：is_partition_key=True 表示这是分区键
        # Milvus 会按租户 ID 把数据分散到不同分区，实现物理隔离
        # 查询时自动过滤，A 租户完全看不到 B 租户的数据

        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
        # 所属文档 ID（用于按文档删除所有相关向量）

        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        # 在文档中的块序号（第几块）

        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        # 文本内容（Milvus VARCHAR 最大 65535 字符）

        FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=500),
        # 章节路径（如"第三条 贴现业务"）

        FieldSchema(name="page_num", dtype=DataType.INT64),
        # 页码

        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
        # 类型：text/table/image_ocr

        FieldSchema(name="md5_hash", dtype=DataType.VARCHAR, max_length=32),
        # 文件 MD5（用于幂等检查，防止重复入库）

        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR,
                    dim=settings.MILVUS_DENSE_DIM),
        # 稠密向量：1024 维浮点数组（BGE-M3 输出）

        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        # 稀疏向量：{词ID: 权重} 字典（只存非零值，节省空间）
    ]
    return CollectionSchema(
        fields=fields,
        description="票据业务智能顾问文档向量库",
        enable_dynamic_field=True,  # 允许存储 schema 之外的动态字段（灵活扩展）
    )


# ──────────────────────────────────────────────────────────────────────────────
# Milvus 连接管理
# ──────────────────────────────────────────────────────────────────────────────
def connect_milvus():
    """
    连接到 Milvus 服务器
    同时初始化旧式 ORM 连接（用于查询/搜索）和新式 MilvusClient（用于写入）
    返回 True=连接成功，False=连接失败
    """
    global _milvus_write_client    # 声明修改全局写客户端变量
    if not MILVUS_AVAILABLE:
        logger.warning("pymilvus not installed")
        return False
    try:
        connections.connect(
            alias="default",                   # 连接别名（整个程序用这个别名引用连接）
            host=settings.MILVUS_HOST,         # 服务器地址
            port=settings.MILVUS_PORT,         # 端口（默认 19530）
        )
        # 同时建立 MilvusClient 连接，用于写入（规避 Collection.insert() 稀疏向量 bug）
        _milvus_write_client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        )
        logger.info(f"Connected to Milvus {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        return True
    except Exception as e:
        logger.error(f"Milvus connection failed: {e}")
        return False


def get_or_create_collection() -> Optional["Collection"]:
    """
    获取已存在的集合，或者创建新集合
    同时创建索引（索引是加速检索的关键）
    """
    if not MILVUS_AVAILABLE:
        return None

    try:
        if utility.has_collection(settings.MILVUS_COLLECTION):
            # 集合已存在，直接加载到内存（Milvus 需要先 load 才能查询）
            col = Collection(settings.MILVUS_COLLECTION)
            col.load()            # 把索引数据加载到内存，查询才能生效

            # 打印 Milvus 里实际存储的字段列表，便于与代码 Schema 对比
            # 若字段不匹配（旧集合 vs 新代码），insert 会报 DataNotMatchException
            actual_fields = [f.name for f in col.schema.fields]   # 取当前集合的所有字段名
            expected_fields = [                                     # 代码期望的字段名列表
                "id", "tenant_id", "document_id", "chunk_index",
                "content", "section_path", "page_num", "chunk_type",
                "md5_hash", "dense_vector", "sparse_vector",
            ]
            missing = set(expected_fields) - set(actual_fields)    # 代码有但 Milvus 没有的字段
            extra   = set(actual_fields)   - set(expected_fields)  # Milvus 有但代码没有的字段
            if missing or extra:
                # 字段不一致时用 WARNING 级别提醒，并打印具体差异
                logger.warning(
                    f"Milvus schema mismatch! "
                    f"missing={missing}, extra={extra}. "
                    f"Actual fields: {actual_fields}. "
                    f"Run drop_and_reset_collection() to rebuild."
                )
            else:
                logger.info(f"Milvus collection schema OK: {actual_fields}")   # Schema 匹配正常

            return col

        # 集合不存在，创建新集合
        schema = _build_schema()
        col = Collection(
            name=settings.MILVUS_COLLECTION,
            schema=schema,
            num_partitions=64,    # 最多支持 64 个租户分区
        )

        # 为稠密向量创建 IVF_FLAT 索引
        # IVF（倒排文件索引）：把向量空间聚类，查询时只搜索最近的几个簇，大幅提速
        col.create_index(
            field_name="dense_vector",
            index_params={
                "metric_type": settings.MILVUS_METRIC_TYPE,  # IP=内积（余弦相似度）
                "index_type": settings.MILVUS_INDEX_TYPE,    # IVF_FLAT
                "params": {"nlist": settings.MILVUS_NLIST},  # 聚类数量（128个簇）
            },
        )

        # 为稀疏向量创建倒排索引（适合稀疏向量的高效索引结构）
        col.create_index(
            field_name="sparse_vector",
            index_params={
                "metric_type": "IP",                         # 内积
                "index_type": "SPARSE_INVERTED_INDEX",       # 稀疏倒排索引
                "params": {"drop_ratio_build": 0.2},         # 构建时丢弃权重最低的 20% 词，节省空间
            },
        )

        col.load()   # 创建完立即加载到内存
        logger.info(f"Created Milvus collection: {settings.MILVUS_COLLECTION}")
        return col

    except Exception as e:
        logger.error(f"Milvus collection error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# RRF（倒数排名融合）算法
# ──────────────────────────────────────────────────────────────────────────────
def rrf_fusion(
    result_lists: list[list[tuple[str, float]]],  # 多路检索结果（每路是按相关性排好序的列表）
    k: int = None,           # RRF 公式中的常数（默认60）
    weights: list[float] = None,  # 各路的权重（稠密0.6，稀疏0.4等）
) -> list[tuple[str, float]]:
    """
    倒数排名融合（Reciprocal Rank Fusion）
    原理：每个文档在每路检索中的排名越靠前，得分越高
    公式：得分 += 权重 / (k + 排名)
    优点：不受各路分数量纲不同的影响（只看排名，不看原始分数）
    """
    k = k or settings.RRF_K         # 默认 k=60（经验值，防止排名第1的文档得分过高）
    weights = weights or [1.0] * len(result_lists)  # 默认各路权重相等

    rrf_scores: dict[str, float] = {}   # 各文档的 RRF 累计得分

    for results, w in zip(result_lists, weights):  # 遍历每一路检索结果
        for rank, (doc_id, _) in enumerate(results):  # 遍历每个文档（rank=排名，从0开始）
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + w / (k + rank + 1)
            # 累加 RRF 得分：权重 / (常数 + 排名 + 1)
            # 排名 0（第1名）得 w/61，排名 1（第2名）得 w/62...排名越高得分越低

    # 按 RRF 总分从高到低排序
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results  # 返回 [(文档ID, RRF分数), ...] 降序排列


# ──────────────────────────────────────────────────────────────────────────────
# 向量存储服务（核心服务类）
# ──────────────────────────────────────────────────────────────────────────────
class VectorStoreService:
    """
    Milvus 向量存储服务（支持多租户）
    提供两个主要操作：upsert_chunks（写入）和 hybrid_search（检索）
    """

    def __init__(self):
        self._collection: Optional["Collection"] = None   # 集合对象（懒加载）

    def _get_collection(self) -> Optional["Collection"]:
        """懒加载集合对象"""
        if self._collection is None:
            self._collection = get_or_create_collection()
        return self._collection

    # ── 写入操作 ─────────────────────────────────────────────────────────────
    def upsert_chunks(
        self,
        tenant_id: str,
        document_id: str,
        chunks: list,         # list[TextChunk]（文档片段列表）
        md5_hash: str,        # 文件 MD5（用于幂等检查）
    ) -> int:
        """
        幂等写入文档向量（相同文件不重复写）
        返回实际写入的片段数（0 表示已存在，跳过）
        """
        col = self._get_collection()
        if col is None:
            logger.warning("Milvus not available, skipping vector upsert")
            return 0

        # ── 跨文档 MD5 去重：同一租户下相同内容的不同文档只入库一次 ──────────────
        # 关键：排除 document_id == 当前文档，否则断点续传时自己的部分写入会被误判为"已存在"
        dup_check = col.query(
            expr=f'md5_hash == "{md5_hash}" and tenant_id == "{tenant_id}" and document_id != "{document_id}"',
            output_fields=["id"],  # 只需要知道有没有，不关心内容
            limit=1,               # 找到一条即可，无需全量扫描
        )
        if dup_check:
            # 其他 document_id 已有相同 MD5，说明是重复上传，直接跳过
            logger.info(f"Doc {document_id} is a duplicate (md5={md5_hash}), skipping")
            return 0

        # ── Chunk 级幂等：查询该 document_id 已写入的 chunk，断点续传时跳过已有的 ──
        # Chunk ID 使用确定性格式 "{document_id}_{chunk_index}"，重试时 ID 完全相同
        # 好处：无论重试多少次，同一个 chunk 的 ID 永远不变，Milvus 不会产生重复数据
        all_ids = [f"{document_id}_{c.chunk_index}" for c in chunks]  # 全量确定性 ID 列表

        existing_res = col.query(
            expr=f'document_id == "{document_id}" and tenant_id == "{tenant_id}"',
            output_fields=["id"],   # 只取 id 字段，节省带宽
            limit=10000,            # 单文档 chunk 上限（512 token/chunk，10000 条约 500 万字）
        )
        existing_ids = {r["id"] for r in existing_res}  # 转为 set，O(1) 查找

        # 过滤出尚未写入 Milvus 的 chunk，保留 (id, chunk) 配对以便后续使用
        new_pairs = [(id_, c) for id_, c in zip(all_ids, chunks) if id_ not in existing_ids]
        if not new_pairs:
            # 所有 chunk 已全部写入，可能是重试时恰好全部完成，直接返回
            logger.info(f"Doc {document_id} already fully indexed ({len(all_ids)} chunks), skipping")
            return 0

        if len(new_pairs) < len(chunks):
            # 部分 chunk 已写入，说明是断点续传场景，打印进度信息
            logger.info(
                f"Doc {document_id} resuming: {len(existing_ids)} chunks already done, "
                f"{len(new_pairs)} remaining"
            )

        new_ids, new_chunks = zip(*new_pairs)   # zip(*...) 解压：[(id,chunk),...] → ([id,...], [chunk,...])
        new_ids = list(new_ids)                 # zip 返回元组，转为列表方便后续操作
        new_chunks = list(new_chunks)

        # ── 向量化 ───────────────────────────────────────────────────────────────
        texts = [c.content for c in new_chunks]              # 提取每个 chunk 的纯文本
        emb_result = embedding_service.encode_texts(texts)   # BGE-M3 批量编码，返回 dense + sparse
        dense_vecs = emb_result["dense"]                     # 稠密向量：numpy 数组，shape=(N, 1024)

        # ── 稀疏向量格式转换 ──────────────────────────────────────────────────
        # BGE-M3 lexical_weights 的输出格式因版本而异，可能是：
        #   ① {str: float}  — token piece 字符串键（如 "##承兑"），Milvus 不接受字符串键
        #   ② {np.int64: np.float16} — numpy 类型键值，Milvus 同样不接受
        # Milvus SPARSE_FLOAT_VECTOR 要求：{Python int: Python float}，键必须是非负整数
        raw_sparse = emb_result["sparse"]    # BGE-M3 原始稀疏输出，类型不确定
        sparse_vecs = []                     # 转换后的稀疏向量列表，最终传给 Milvus
        for s in raw_sparse:                 # 遍历每个 chunk 对应的稀疏向量字典
            converted = {}                   # 存放类型安全的 {int: float} 映射
            for k, v in s.items():           # 遍历原始稀疏向量的每个 token→权重对
                fv = float(v)                # 强制转为 Python float（处理 np.float16/float32）
                if fv <= 0:                  # 权重为零的项对检索无贡献，过滤掉节省存储
                    continue
                if isinstance(k, str):
                    # 字符串 token 键（版本①）：用 hash 映射到 20-bit 非负整数空间
                    # 20 bit = ~100 万个桶，实际词表约 10 万，碰撞率约 10%，可接受
                    ik = hash(k) & 0xFFFFF   # 取 hash 低 20 位，保证结果非负且范围合理
                else:
                    ik = int(k)              # 数字键（版本②）：直接转为 Python int
                converted[ik] = fv           # 写入转换后的字典
            # Milvus 不接受空稀疏向量，用低权重占位符保证至少有一项
            if not converted:
                converted[0] = 1e-9
            sparse_vecs.append(converted)    # 追加到结果列表

        # 记录第一个稀疏向量的条目数，便于观察稀疏向量质量
        if sparse_vecs:                              # 避免空列表越界
            logger.debug(f"sparse_vec[0] entries={len(sparse_vecs[0])}")  # 条目越多，关键词越丰富

        # ── 组装插入数据 ─────────────────────────────────────────────────────
        # 所有标量字段显式转为 Python 原生类型，防止 numpy 类型流入 Milvus 导致 schema 不匹配
        data = {
            "id":            [str(x) for x in new_ids],                         # 主键：确定性字符串 ID
            "tenant_id":     [str(tenant_id)] * len(new_chunks),                # 分区键：租户 ID
            "document_id":   [str(document_id)] * len(new_chunks),              # 所属文档 ID
            "chunk_index":   [int(c.chunk_index) for c in new_chunks],          # chunk 序号（Python int）
            "content":       [str(c.content)[:65530] for c in new_chunks],      # 文本内容，截断到 VARCHAR 上限
            "section_path":  [str(c.section_path or "")[:498] for c in new_chunks],  # 章节路径，None 安全处理
            "page_num":      [int(c.page_num) for c in new_chunks],             # 页码（Python int）
            "chunk_type":    [str(c.chunk_type) for c in new_chunks],           # 类型标签
            "md5_hash":      [str(md5_hash)] * len(new_chunks),                 # 文件 MD5
            "dense_vector":  [v.tolist() for v in dense_vecs],                  # numpy 行 → Python float 列表
            "sparse_vector": sparse_vecs,                                        # 转换后的稀疏向量
        }

        # ── MilvusClient 写入（规避 Collection.insert() 稀疏向量 bug）──────────────
        # pymilvus 2.4.3 的 Collection.insert() 在处理 SPARSE_FLOAT_VECTOR 时
        # 序列化路径有 bug，会导致 Milvus server 2.4.4 抛出 DataNotMatchException。
        # MilvusClient.insert() 使用不同的序列化实现，稀疏向量可正常写入。
        # 构造行格式（每行是一个完整的实体字典，MilvusClient 要求此格式）
        entities = []                                     # 收集所有行的实体字典列表
        for i in range(len(new_ids)):                     # 遍历每一个待写入的 chunk
            entities.append({
                "id":           data["id"][i],            # 主键：确定性字符串 ID
                "tenant_id":    data["tenant_id"][i],     # 分区键：租户 ID
                "document_id":  data["document_id"][i],   # 所属文档 ID
                "chunk_index":  data["chunk_index"][i],   # chunk 在文档中的序号
                "content":      data["content"][i],       # 文本内容
                "section_path": data["section_path"][i],  # 章节路径
                "page_num":     data["page_num"][i],      # 页码
                "chunk_type":   data["chunk_type"][i],    # 类型标签
                "md5_hash":     data["md5_hash"][i],      # 文件 MD5
                "dense_vector": data["dense_vector"][i],  # 1024 维稠密向量
                "sparse_vector":data["sparse_vector"][i], # {int: float} 稀疏向量
            })

        # 获取写客户端（全局单例，由 connect_milvus() 初始化）
        write_client = _milvus_write_client               # 写客户端：MilvusClient 实例
        if write_client is None:
            # 未初始化时兜底重建（通常不会走到这里）
            write_client = MilvusClient(
                uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            )

        write_client.insert(                              # 使用 MilvusClient 写入，正确处理稀疏向量
            collection_name=settings.MILVUS_COLLECTION,  # 目标集合名
            data=entities,                                # 行格式实体列表
        )
        col.flush()   # 使用 Collection 对象刷盘（MilvusClient 本身也可 flush，但 Collection 更稳定）

        # BM25 索引同步更新（只更新新增的 chunk，已有的不重复 fit）
        bm25_idx = bm25_store.get_index(tenant_id)   # 获取该租户的 BM25 索引实例
        bm25_idx.add_texts(texts, new_ids)            # 把新增文本加入 BM25 语料库，new_ids 用于精排时回溯
        bm25_store.save_index(tenant_id)              # 序列化到 Redis，确保重启后 BM25 索引不丢失

        logger.info(f"Inserted {len(new_chunks)} chunks for doc {document_id} tenant {tenant_id}")
        return len(new_chunks)

    def delete_document(self, tenant_id: str, document_id: str):
        """删除指定文档的所有向量（文档删除时调用）"""
        col = self._get_collection()
        if col is None:
            logger.warning(f"[vector_store] Milvus 不可用，跳过向量删除 doc={document_id}")  # 集合未就绪
            return
        logger.info(f"[vector_store] 删除文档向量 tenant={tenant_id} doc={document_id}")  # 删除操作入口
        try:
            col.delete(f'document_id == "{document_id}" and tenant_id == "{tenant_id}"')
            col.flush()  # 刷新确保删除生效
            logger.info(f"[vector_store] 向量删除完成 doc={document_id}")  # 确认删除成功
        except Exception as e:
            logger.error(f"[vector_store] 向量删除失败 doc={document_id} error={e}", exc_info=True)  # 含堆栈

    # ── 检索操作 ─────────────────────────────────────────────────────────────
    def hybrid_search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = None,
        rerank_top_n: int = None,
    ) -> list[dict]:
        """
        三路混合检索 + RRF 融合 + BGE-Reranker 精排
        这是整个检索系统的核心方法
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K        # 粗检索数量（默认20）
        rerank_top_n = rerank_top_n or settings.RERANK_TOP_N  # 精排数量（默认5）
        logger.info(  # 检索入口：记录关键参数，方便复现"为什么这次检索结果不对"
            f"[vector_store] 混合检索开始 tenant={tenant_id} "
            f"top_k={top_k} rerank_top_n={rerank_top_n} query={query[:50]!r}"
        )
        t0 = time.perf_counter()   # 记录检索开始时间

        # ── 向量化查询 ────────────────────────────────────────────────────────
        q_emb = embedding_service.encode_query(query)   # 把查询问题转成向量
        dense_vec = q_emb["dense"][0].tolist()          # 取第一个（也是唯一一个）稠密向量
        # 同样需要转换为 Python 原生类型，否则 Milvus 稀疏检索会报类型不匹配
        sparse_vec = {int(k): float(v) for k, v in q_emb["sparse"][0].items() if v > 0}

        col = self._get_collection()

        retrieval_results: list[list[tuple[str, float]]] = []  # 三路检索结果
        id_to_meta: dict[str, dict] = {}                       # 片段ID → 元数据映射（收集命中结果的内容）

        # ── 路径1：稠密向量检索（语义相似度） ─────────────────────────────────
        dense_results = self._dense_search(col, tenant_id, dense_vec, top_k, id_to_meta)
        retrieval_results.append(dense_results)  # 加入多路结果列表
        logger.debug(f"[vector_store] 稠密检索 hits={len(dense_results)}")  # 记录各路召回数量

        # ── 路径2：稀疏向量检索（词语权重匹配） ────────────────────────────────
        sparse_results = self._sparse_search(col, tenant_id, sparse_vec, top_k, id_to_meta)
        retrieval_results.append(sparse_results)
        logger.debug(f"[vector_store] 稀疏检索 hits={len(sparse_results)}")  # 同上

        # ── 路径3：BM25 关键词检索 ─────────────────────────────────────────────
        bm25_results = self._bm25_search(tenant_id, query, top_k)
        retrieval_results.append(bm25_results)
        logger.debug(f"[vector_store] BM25检索 hits={len(bm25_results)}")  # 同上

        t_retrieval = (time.perf_counter() - t0) * 1000   # 检索耗时

        # ── RRF 融合：合并三路结果 ──────────────────────────────────────────────
        weights = [
            settings.HYBRID_DENSE_WEIGHT,                   # 稠密检索权重（0.6）
            settings.HYBRID_SPARSE_WEIGHT,                  # 稀疏检索权重（0.4）
            settings.HYBRID_SPARSE_WEIGHT * 0.8,            # BM25 权重（稍低：0.32）
        ]
        fused = rrf_fusion(retrieval_results, weights=weights)  # 三路融合

        # 补全 BM25 命中但向量检索没命中的片段元数据（BM25 只有 ID，没有内容）
        missing_ids = [id_ for id_, _ in fused[:top_k] if id_ not in id_to_meta]
        if missing_ids and col is not None:
            logger.debug(f"[vector_store] BM25 命中需补查元数据 missing={len(missing_ids)}")  # 记录补查规模
            self._fetch_missing(col, tenant_id, missing_ids, id_to_meta)  # 从 Milvus 补查

        # 取融合结果中的 top_k 个候选
        candidates = []
        for id_, rrf_score in fused[:top_k]:
            meta = id_to_meta.get(id_)          # 获取该片段的内容和元数据
            if meta:
                meta["rrf_score"] = rrf_score   # 把 RRF 分数加入元数据
                candidates.append(meta)

        # ── BGE-Reranker 精排 ──────────────────────────────────────────────────
        # 用更强大的模型重新给候选文档打分，进一步提升准确率
        t1 = time.perf_counter()
        if candidates:
            candidate_texts = [c["content"] for c in candidates]   # 提取所有候选文本
            reranked = embedding_service.rerank(query, candidate_texts, top_n=rerank_top_n)
            # reranked：[(原始索引, 精排分数), ...] 降序排列
            final = [candidates[idx] | {"rerank_score": score} for idx, score in reranked]
            # candidates[idx]：取对应索引的候选；| 运算符：合并字典（添加精排分数）
        else:
            final = []   # 没有候选结果

        t_rerank = (time.perf_counter() - t1) * 1000   # 精排耗时

        logger.info(  # 升级为 INFO，检索全链路耗时是关键指标
            f"[vector_store] 混合检索完成 tenant={tenant_id} "
            f"retrieval={t_retrieval:.1f}ms rerank={t_rerank:.1f}ms "
            f"candidates={len(candidates)} final={len(final)}"
        )

        return final   # 返回精排后的最终结果列表

    def _dense_search(self, col, tenant_id: str, vec: list, top_k: int,
                      id_to_meta: dict) -> list[tuple[str, float]]:
        """
        稠密向量检索（语义相似度）
        col：Milvus 集合对象
        vec：查询的稠密向量（1024 维列表）
        id_to_meta：收集命中片段的内容，方便后续精排
        返回：[(片段ID, 相似度分数), ...] 降序
        """
        if col is None:
            return []
        try:
            results = col.search(
                data=[vec],                    # 查询向量（列表中的列表）
                anns_field="dense_vector",     # 在哪个字段上做近似最近邻搜索
                param={"metric_type": "IP", "params": {"nprobe": 16}},
                # metric_type=IP：用内积（内积越大越相似）
                # nprobe=16：搜索 16 个最近的聚类簇（越大越准确，但越慢）
                limit=top_k,                   # 最多返回 top_k 个结果
                expr=f'tenant_id == "{tenant_id}"',  # 只搜索该租户的数据
                output_fields=["content", "section_path", "chunk_type", "page_num",
                               "document_id", "chunk_index"],  # 要返回哪些字段
            )
            out = []
            for hits in results:               # results 是嵌套列表（每个查询向量一组结果）
                for hit in hits:               # 遍历命中结果
                    id_to_meta[hit.id] = {     # 存储元数据（供后续精排和展示用）
                        "id": hit.id,
                        "content": (hit.entity.get("content") or ""),
                        "section_path": (hit.entity.get("section_path") or ""),
                        "chunk_type": (hit.entity.get("chunk_type") or ""),
                        "page_num": (hit.entity.get("page_num") or 0),
                        "document_id": (hit.entity.get("document_id") or ""),
                    }
                    out.append((hit.id, float(hit.score)))  # 收集 (ID, 分数) 对
            return out
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return []

    def _sparse_search(self, col, tenant_id: str, sparse_vec: dict, top_k: int,
                       id_to_meta: dict) -> list[tuple[str, float]]:
        """
        稀疏向量检索（词语权重匹配，类似关键词检索但更智能）
        sparse_vec：{词ID: 权重} 格式的稀疏向量
        """
        if col is None:
            return []
        try:
            results = col.search(
                data=[sparse_vec],             # 查询稀疏向量
                anns_field="sparse_vector",    # 在稀疏向量字段上搜索
                param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
                # drop_ratio_search=0.2：搜索时忽略权重最低的 20% 的词，加速计算
                limit=top_k,
                expr=f'tenant_id == "{tenant_id}"',
                output_fields=["content", "section_path", "chunk_type", "page_num",
                               "document_id"],
            )
            out = []
            for hits in results:
                for hit in hits:
                    if hit.id not in id_to_meta:   # 避免重复存储（稠密检索已存的不覆盖）
                        id_to_meta[hit.id] = {
                            "id": hit.id,
                            "content": (hit.entity.get("content") or ""),
                            "section_path": (hit.entity.get("section_path") or ""),
                            "chunk_type": (hit.entity.get("chunk_type") or ""),
                            "page_num": (hit.entity.get("page_num") or 0),
                            "document_id": (hit.entity.get("document_id") or ""),
                        }
                    out.append((hit.id, float(hit.score)))
            return out
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return []

    def _bm25_search(self, tenant_id: str, query: str,
                     top_k: int) -> list[tuple[str, float]]:
        """
        BM25 关键词检索（从 Redis 加载该租户的 BM25 索引）
        注意：BM25 的 ID 是 Milvus 中的向量 ID，但元数据不在 BM25 中
        所以命中结果的内容需要后续从 Milvus 补查
        """
        try:
            bm25_idx = bm25_store.get_index(tenant_id)    # 获取该租户的 BM25 索引
            return bm25_idx.search(query, top_k=top_k)    # 执行关键词检索
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def _fetch_missing(self, col, tenant_id: str, ids: list[str], id_to_meta: dict):
        """
        从 Milvus 补查 BM25 命中但向量检索未命中的片段的内容
        因为 BM25 索引只存 ID，没有文本内容，需要回 Milvus 查
        """
        try:
            ids_str = '", "'.join(ids)   # 把 ID 列表拼成 Milvus 过滤表达式的格式
            results = col.query(
                expr=f'id in ["{ids_str}"] and tenant_id == "{tenant_id}"',
                # id in [...]：查询 ID 在列表中的所有记录
                output_fields=["id", "content", "section_path", "chunk_type",
                               "page_num", "document_id"],
            )
            for r in results:
                id_to_meta[r["id"]] = {        # 把补查结果加入元数据字典
                    "id": r["id"],
                    "content": r.get("content", ""),
                    "section_path": r.get("section_path", ""),
                    "chunk_type": r.get("chunk_type", ""),
                    "page_num": r.get("page_num", 0),
                    "document_id": r.get("document_id", ""),
                }
        except Exception as e:
            logger.error(f"Fetch missing chunks error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 集合重置工具（Schema 变更后执行一次，清空旧集合并按新 Schema 重建）
# 警告：会删除 Milvus 中所有向量数据，需要重新入库所有文档
# ──────────────────────────────────────────────────────────────────────────────
def drop_and_reset_collection() -> bool:
    """
    删除现有 Milvus 集合并按当前代码 Schema 重建
    适用场景：Schema 字段有增删改导致 DataNotMatchException 时
    注意：执行后所有向量数据丢失，需要重新触发所有文档入库
    返回 True=重建成功，False=失败
    """
    if not MILVUS_AVAILABLE:                              # pymilvus 未安装，无法操作
        logger.error("pymilvus not installed")
        return False
    try:
        connect_milvus()                                  # 确保连接已建立
        if utility.has_collection(settings.MILVUS_COLLECTION):   # 集合存在才需要删除
            utility.drop_collection(settings.MILVUS_COLLECTION)  # 删除旧集合（含索引和数据）
            logger.warning(f"Dropped Milvus collection: {settings.MILVUS_COLLECTION}")
        col = get_or_create_collection()                  # 按最新 Schema 重新建集合和索引
        if col is not None:
            logger.info(f"Recreated Milvus collection: {settings.MILVUS_COLLECTION}")
            return True                                   # 重建成功
        return False                                      # get_or_create_collection 内部失败
    except Exception as e:
        logger.error(f"drop_and_reset_collection failed: {e}")
        return False


# 全局向量存储服务实例（整个应用共享）
vector_store = VectorStoreService()
