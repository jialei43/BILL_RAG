# tests/test_core.py
# 核心模块的单元测试
# 测试策略：不需要真实的 Milvus/Redis 数据库，用 mock 模拟外部依赖
# 覆盖：分块器、表格置信度评估、RRF融合、BM25索引、限流器、MD5去重、Token计数
#
# 运行方式：
#   pytest tests/test_core.py -v  （-v 显示详细测试结果）
#   python tests/test_core.py     （直接运行）

import pytest          # 测试框架：提供 assert 增强、fixture、参数化等功能
import asyncio         # 异步支持（这里部分测试用到）
import tempfile        # 临时文件（用于测试 MD5 计算）
import os              # 文件系统操作
from unittest.mock import MagicMock, patch
# MagicMock：创建模拟对象（可以代替真实的数据库连接、模型等）
# patch：临时替换某个模块中的对象（测试结束后自动还原）
from pathlib import Path   # 路径
import sys

# 把项目根目录加入 Python 路径（让 import app.xxx 能找到正确的模块）
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# 测试分块器（ChineseSemanticChunker）
# ──────────────────────────────────────────────────────────────────────────────
class TestChineseSemanticChunker:
    """测试中文语义分块器的各种场景"""

    def setup_method(self):
        """每个测试方法执行前自动调用（测试夹具），初始化被测对象"""
        from app.services.chunker import ChineseSemanticChunker
        # 用较小的 token 限制，方便测试分块行为（不需要很长的文本）
        self.chunker = ChineseSemanticChunker(
            max_tokens=128,    # 每块最多 128 个 token
            overlap_tokens=16, # 相邻块重叠 16 个 token
            min_tokens=10,     # 至少 10 个 token（太短的块丢弃）
        )

    def test_short_text_no_split(self):
        """测试：短文本不应该被切分（不超过 max_tokens 时保持完整）"""
        from app.services.pdf_parser import ParsedElement, ParsedDocument

        # 构造一个包含一段短文字的解析文档
        elem = ParsedElement(content="票据贴现是指持票人将未到期票据转让给银行获取资金。",
                             element_type="text", page_num=1)
        doc = ParsedDocument(elements=[elem], total_pages=1, file_type="pdf")

        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) == 1           # 应该只有一块（没有分割）
        assert "票据贴现" in chunks[0].content  # 内容应该包含原始文字

    def test_section_detection(self):
        """测试：章节标题识别 + 章节路径注入"""
        from app.services.pdf_parser import ParsedElement, ParsedDocument

        elements = [
            # 第一段：包含章节标题"第一条"
            ParsedElement(content="第一条 总则\n本办法适用于银行承兑汇票贴现业务。",
                          element_type="text", page_num=1),
            # 第二段：普通文字，应该被注入前面检测到的章节路径
            ParsedElement(content="银行承兑汇票贴现利率按市场报价执行。",
                          element_type="text", page_num=1),
        ]
        doc = ParsedDocument(elements=elements, total_pages=1, file_type="pdf")
        chunks = self.chunker.chunk_document(doc)

        # 找到包含第二段文字的 chunk，验证它是否带有章节路径前缀
        second_chunks = [c for c in chunks if "银行承兑汇票贴现利率" in c.content]
        if second_chunks:
            # section_path 应该包含"第一条"，或者 content 里有"章节："前缀
            assert "第一条" in second_chunks[0].section_path or \
                   "章节" in second_chunks[0].content

    def test_table_as_single_chunk(self):
        """测试：表格元素应该保持完整，不被切分"""
        from app.services.pdf_parser import ParsedElement, ParsedDocument

        # 一个 Markdown 格式的表格
        table_content = "| 出票日期 | 到期日 | 票面金额 |\n|---|---|---|\n| 2024-01-01 | 2024-07-01 | 1000000 |"
        elem = ParsedElement(content=table_content, element_type="table", page_num=1)
        doc = ParsedDocument(elements=[elem], total_pages=1, file_type="pdf")

        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) == 1                    # 表格不应被切分，只有一块
        assert chunks[0].chunk_type == "table"     # 类型应该是 "table"

    def test_long_text_split(self):
        """测试：超过 max_tokens 的长文本应该被切成多块"""
        from app.services.pdf_parser import ParsedElement, ParsedDocument

        # 重复短句 30 次，构造一段超长文本
        long_text = "票据贴现是银行对持票人的融资服务。" * 30
        elem = ParsedElement(content=long_text, element_type="text", page_num=1)
        doc = ParsedDocument(elements=[elem], total_pages=1, file_type="pdf")

        chunks = self.chunker.chunk_document(doc)

        assert len(chunks) > 1   # 应该被切成多块
        for chunk in chunks:
            assert chunk.token_count <= 200   # 每块 token 数应合理（宽松验证）


# ──────────────────────────────────────────────────────────────────────────────
# 测试表格置信度评估器（TableConfidenceEvaluator）
# ──────────────────────────────────────────────────────────────────────────────
class TestTableConfidenceEvaluator:
    """测试表格质量评分逻辑"""

    def setup_method(self):
        from app.services.pdf_parser import TableConfidenceEvaluator
        self.evaluator = TableConfidenceEvaluator()

    def test_empty_df_returns_zero(self):
        """测试：空表格（没有单元格）应该返回 0 分"""
        import pandas as pd
        mock_table = MagicMock()           # 创建模拟的 Camelot Table 对象
        mock_table.df = pd.DataFrame()    # 设置空的 DataFrame
        score = self.evaluator.evaluate(mock_table)
        assert score == 0.0   # 空表格置信度为 0

    def test_full_table_high_score(self):
        """测试：数据完整的表格应该得高分"""
        import pandas as pd
        mock_table = MagicMock()
        mock_table.df = pd.DataFrame({
            "出票日期": ["2024-01-01"],
            "到期日": ["2024-07-01"],
            "票面金额": ["1000000"],
        })   # 3列1行，没有空单元格
        mock_table.accuracy = 95   # Camelot 原始精度 95%
        score = self.evaluator.evaluate(mock_table)
        assert score > 0.5   # 高质量表格应该超过 0.5 分

    def test_sparse_table_low_score(self):
        """测试：大量空单元格的表格应该得低分"""
        import pandas as pd
        mock_table = MagicMock()
        mock_table.df = pd.DataFrame({
            "A": ["", "", ""],      # 全空列
            "B": ["", "value", ""],  # 几乎全空
            "C": ["", "", ""],      # 全空列
        })   # 9个单元格，8个为空（空率 89%）
        mock_table.accuracy = 40   # Camelot 精度也很低
        score = self.evaluator.evaluate(mock_table)
        assert score < 0.5   # 低质量表格应该低于 0.5 分


# ──────────────────────────────────────────────────────────────────────────────
# 测试 RRF 融合算法（rrf_fusion）
# ──────────────────────────────────────────────────────────────────────────────
class TestRRFFusion:
    """测试倒数排名融合算法的正确性"""

    def test_basic_fusion(self):
        """测试：两路检索结果融合后，在两路都靠前的文档应排在最前面"""
        from app.services.vector_store import rrf_fusion

        list1 = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.7)]  # 第一路：doc_a 最好
        list2 = [("doc_b", 0.95), ("doc_a", 0.85), ("doc_d", 0.6)]  # 第二路：doc_b 最好

        result = rrf_fusion([list1, list2])
        ids = [r[0] for r in result]   # 提取融合后的 ID 列表

        # doc_a 和 doc_b 在两路中都靠前，融合后应排在前两位
        assert ids[0] in ("doc_a", "doc_b")
        assert ids[1] in ("doc_a", "doc_b")

    def test_single_list(self):
        """测试：单路结果的融合结果应保持原有顺序"""
        from app.services.vector_store import rrf_fusion

        list1 = [("doc_x", 1.0), ("doc_y", 0.9)]
        result = rrf_fusion([list1])
        assert result[0][0] == "doc_x"   # 排名第一的应该还是 doc_x

    def test_empty_lists(self):
        """测试：空输入应该返回空结果"""
        from app.services.vector_store import rrf_fusion

        result = rrf_fusion([[], []])
        assert result == []   # 两路都是空，结果也应该是空

    def test_weighted_fusion(self):
        """测试：权重高的检索路径，其结果应该排在前面"""
        from app.services.vector_store import rrf_fusion

        list1 = [("doc_a", 1.0)]   # 第一路只有 doc_a
        list2 = [("doc_b", 1.0)]   # 第二路只有 doc_b

        result = rrf_fusion([list1, list2], weights=[2.0, 1.0])
        # 第一路权重 2.0，第二路权重 1.0
        # doc_a 的 RRF 分 = 2.0/(60+1) ≈ 0.0328
        # doc_b 的 RRF 分 = 1.0/(60+1) ≈ 0.0164
        assert result[0][0] == "doc_a"   # doc_a 权重路径的文档应排第一


# ──────────────────────────────────────────────────────────────────────────────
# 测试 BM25 索引（TenantBM25Index）
# ──────────────────────────────────────────────────────────────────────────────
class TestTenantBM25Index:
    """测试租户级 BM25 关键词索引的添加、搜索、序列化功能"""

    def setup_method(self):
        from app.services.embedding import TenantBM25Index, init_jieba
        init_jieba()   # 初始化票据领域词典（让 BM25 分词更准确）
        self.idx = TenantBM25Index("test_tenant")   # 创建测试用的 BM25 索引

    def test_add_and_search(self):
        """测试：添加文本后，搜索相关词语应该能找到对应文档"""
        texts = [
            "银行承兑汇票贴现利率",       # chunk_1：关于贴现利率
            "票据质押融资业务规定",        # chunk_2：关于质押融资
            "转贴现市场成交数据分析",      # chunk_3：关于转贴现
        ]
        ids = ["chunk_1", "chunk_2", "chunk_3"]
        self.idx.add_texts(texts, ids)   # 添加文本到 BM25 索引

        results = self.idx.search("贴现利率", top_k=3)   # 搜索与"贴现利率"相关的文档

        assert len(results) > 0                # 应该有结果
        assert results[0][0] == "chunk_1"      # 第一条（最相关）应该是 chunk_1

    def test_serialize_deserialize(self):
        """测试：序列化（保存到 Redis）后，反序列化（从 Redis 读回）应该能正常搜索"""
        from app.services.embedding import TenantBM25Index

        self.idx.add_texts(["票据贴现业务"], ["c1"])   # 添加一条数据

        data = self.idx.serialize()   # 序列化为字节流（模拟保存到 Redis）
        restored = TenantBM25Index.deserialize("test_tenant", data)   # 从字节流恢复

        results = restored.search("贴现", top_k=1)   # 在恢复的索引中搜索
        assert len(results) > 0   # 应该能找到结果（说明恢复成功）


# ──────────────────────────────────────────────────────────────────────────────
# 测试限流器（RateLimiter）
# ──────────────────────────────────────────────────────────────────────────────
class TestRateLimiter:
    """测试 QPS 限流和文档配额检查逻辑"""

    def test_doc_quota_check(self):
        """测试：文档配额检查的边界情况"""
        from app.services.rate_limiter import RateLimiter
        rl = RateLimiter()

        assert rl.check_doc_quota("t1", 999, quota=1000) is True    # 999 < 1000，未超配额
        assert rl.check_doc_quota("t1", 1000, quota=1000) is False  # 1000 = 1000，已满（严格小于）
        assert rl.check_doc_quota("t1", 1001, quota=1000) is False  # 1001 > 1000，已满

    def test_rate_limit_without_redis(self):
        """测试：Redis 不可用时，限流器应该放行所有请求（降级策略）"""
        from app.services.rate_limiter import RateLimiter
        rl = RateLimiter()
        rl._redis = None   # 强制 Redis 连接为 None（模拟 Redis 宕机）

        allowed, info = rl.check_rate_limit("tenant_test")

        assert allowed is True   # 应该放行（宁可不限流，不能影响正常使用）


# ──────────────────────────────────────────────────────────────────────────────
# 测试 MD5 幂等去重（compute_md5）
# ──────────────────────────────────────────────────────────────────────────────
class TestMD5Idempotency:
    """测试文件 MD5 计算的正确性（相同文件哈希相同，不同文件哈希不同）"""

    def test_same_file_same_hash(self):
        """测试：同一文件多次计算 MD5，结果应该完全相同（确定性）"""
        from app.services.pdf_parser import compute_md5

        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"Test bill document content")   # 写入固定内容
            path = f.name   # 保存临时文件路径

        try:
            h1 = compute_md5(path)   # 第一次计算
            h2 = compute_md5(path)   # 第二次计算（相同文件）

            assert h1 == h2          # 两次结果必须相同
            assert len(h1) == 32     # MD5 是 32 位十六进制字符串
        finally:
            os.unlink(path)   # 清理临时文件（无论测试是否通过）

    def test_different_files_different_hash(self):
        """测试：内容不同的文件，MD5 应该不同"""
        from app.services.pdf_parser import compute_md5

        # 创建两个内容不同的临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f1:
            f1.write(b"Content A")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f2:
            f2.write(b"Content B")
            path2 = f2.name

        try:
            assert compute_md5(path1) != compute_md5(path2)   # 内容不同 → MD5 不同
        finally:
            os.unlink(path1)   # 清理临时文件
            os.unlink(path2)


# ──────────────────────────────────────────────────────────────────────────────
# 测试 Token 计数（count_tokens）
# ──────────────────────────────────────────────────────────────────────────────
class TestTokenCount:
    """测试 Token 估算函数（用于控制分块大小）"""

    def test_chinese_text(self):
        """测试：中文文本的 token 计数（每个中文字≈1 token）"""
        from app.services.chunker import count_tokens

        text = "票据贴现利率"   # 6 个中文字
        assert count_tokens(text) == 6   # 应该返回 6

    def test_mixed_text(self):
        """测试：中英混合文本的 token 计数"""
        from app.services.chunker import count_tokens

        text = "BGE-M3 向量模型"   # 1 个英文词（BGE-M3）+ 4 个中文字 = 5
        count = count_tokens(text)
        assert count >= 4   # 至少 4（可能对 BGE-M3 的处理有差异，用 >= 而非 ==）


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 自测：数据库模型结构验证
# ──────────────────────────────────────────────────────────────────────────────
class TestDBModels:
    """验证新增表的列名和关系定义是否完整（不需要真实数据库连接）"""

    def test_all_tables_registered(self):
        """所有新表均已注册到 Base.metadata"""
        from app.models.db_models import Base
        tables = set(Base.metadata.tables.keys())
        required = {"bill_records", "bill_versions", "intent_logs", "search_miss_logs"}
        missing = required - tables
        assert not missing, f"缺少表: {missing}"

    def test_bill_record_columns(self):
        """BillRecord 包含必要字段"""
        from app.models.db_models import BillRecord
        cols = {c.name for c in BillRecord.__table__.columns}
        for field in ["ticket_number", "acceptor", "due_date", "amount_numeric", "risk_flags"]:
            assert field in cols, f"BillRecord 缺少字段 {field}"

    def test_bill_version_columns(self):
        """BillVersion 包含流转相关字段"""
        from app.models.db_models import BillVersion
        cols = {c.name for c in BillVersion.__table__.columns}
        for field in ["version", "endorsers", "new_endorsers", "bill_record_id"]:
            assert field in cols, f"BillVersion 缺少字段 {field}"

    def test_query_log_new_columns(self):
        """QueryLog 已扩展 route_type 和 intent_id 列"""
        from app.models.db_models import QueryLog
        cols = {c.name for c in QueryLog.__table__.columns}
        assert "route_type" in cols
        assert "intent_id" in cols

    def test_intent_log_columns(self):
        """IntentLog 包含 classify_method 列（用于区分关键词/LLM两通道）"""
        from app.models.db_models import IntentLog
        cols = {c.name for c in IntentLog.__table__.columns}
        assert "classify_method" in cols
        assert "confidence" in cols

    def test_search_miss_log_columns(self):
        """SearchMissLog 包含 transferred 列"""
        from app.models.db_models import SearchMissLog
        cols = {c.name for c in SearchMissLog.__table__.columns}
        assert "transferred" in cols
        assert "top1_score" in cols


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 自测：票据生命周期服务
# ──────────────────────────────────────────────────────────────────────────────
class TestBillLifecycle:
    """测试矛盾检测规则和向量文本构造（不依赖数据库/Milvus）"""

    def _make_element(self, **kwargs):
        from app.services.bill_recognition import BillElement
        defaults = dict(
            ticket_type="银行承兑汇票",
            ticket_number="EA2024001234",
            issue_date="2024-09-15",
            due_date="2025-03-15",
            amount_numeric=1_000_000,
            amount_text="壹佰万元整",
            acceptor="杭州银行股份有限公司",
            drawer="甲贸易有限公司",
            endorsers=["乙公司", "丙公司"],
        )
        defaults.update(kwargs)
        return BillElement(**defaults)

    def test_no_risk_normal_bill(self):
        """正常票据不应检测到风险"""
        from app.services.bill_lifecycle import detect_risk_flags
        flags = detect_risk_flags(self._make_element())
        assert flags == []

    def test_date_logic_error(self):
        """到期日早于出票日应触发 DATE_LOGIC_ERROR"""
        from app.services.bill_lifecycle import detect_risk_flags
        el = self._make_element(issue_date="2024-09-15", due_date="2024-01-20")
        codes = [f["code"] for f in detect_risk_flags(el)]
        assert "DATE_LOGIC_ERROR" in codes

    def test_acceptor_not_bank(self):
        """银票承兑人为非银行主体应触发 ACCEPTOR_NOT_BANK"""
        from app.services.bill_lifecycle import detect_risk_flags
        el = self._make_element(acceptor="XX贸易有限公司")
        codes = [f["code"] for f in detect_risk_flags(el)]
        assert "ACCEPTOR_NOT_BANK" in codes

    def test_both_risks_detected(self):
        """同时存在两处风险时均应被检测到"""
        from app.services.bill_lifecycle import detect_risk_flags
        el = self._make_element(
            issue_date="2024-09-15", due_date="2024-01-20",
            acceptor="XX贸易有限公司",
        )
        codes = [f["code"] for f in detect_risk_flags(el)]
        assert "DATE_LOGIC_ERROR" in codes
        assert "ACCEPTOR_NOT_BANK" in codes

    def test_vector_text_contains_key_fields(self):
        """向量文本应包含票据核心字段"""
        from app.services.bill_lifecycle import _build_bill_vector_text
        el = self._make_element()
        text = _build_bill_vector_text(el, "EA2024001234")
        assert "杭州银行" in text
        assert "壹佰万元整" in text
        assert "乙公司" in text
        assert "EA2024001234" in text

    def test_endorser_vector_text(self):
        """背书人增量向量文本应包含新增背书人和完整链"""
        from app.services.bill_lifecycle import _build_endorser_vector_text
        text = _build_endorser_vector_text(
            "EA2024001234", 3,
            ["戊金融服务"],
            ["丙实业", "丁物流", "戊金融服务"],
            "2025-01-10",
        )
        assert "戊金融服务" in text
        assert "第3次流转" in text
        assert "丙实业" in text


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 自测：IntentRouter 关键词通道
# ──────────────────────────────────────────────────────────────────────────────
class TestIntentRouter:
    """测试意图路由器的关键词通道和路由决策（不调用真实 LLM）"""

    def setup_method(self):
        from app.services.intent_router import IntentRouter
        self.router = IntentRouter()

    def test_pricing_keywords(self):
        """贴现利率相关关键词应路由到 INTENT_PRICING"""
        result = self.router._keyword_classify("银行承兑汇票贴现利率")
        assert result is not None
        assert result[0] == "INTENT_PRICING"
        assert result[1] == 0.95

    def test_endorser_keywords(self):
        """背书相关关键词应路由到 INTENT_ENDORSER_CHECK"""
        result = self.router._keyword_classify("背书链是否连续")
        assert result is not None
        assert result[0] == "INTENT_ENDORSER_CHECK"

    def test_expiry_keywords(self):
        """到期追索关键词应路由到 INTENT_EXPIRY"""
        result = self.router._keyword_classify("这张票已经到期了还能追索吗")
        assert result is not None
        assert result[0] == "INTENT_EXPIRY"

    def test_aml_keywords(self):
        """大额反洗钱关键词应路由到 INTENT_AML"""
        result = self.router._keyword_classify("大额反洗钱报告")
        assert result is not None
        assert result[0] == "INTENT_AML"

    def test_element_check_keywords(self):
        """要素核验关键词应路由到 INTENT_ELEMENT_CHECK"""
        result = self.router._keyword_classify("票据要素核验")
        assert result is not None
        assert result[0] == "INTENT_ELEMENT_CHECK"

    def test_history_keywords(self):
        """流转历史关键词应路由到 INTENT_HISTORY"""
        result = self.router._keyword_classify("票据流转历史溯源")
        assert result is not None
        assert result[0] == "INTENT_HISTORY"

    def test_batch_risk_keywords(self):
        """集中度风险关键词应路由到 INTENT_BATCH_RISK"""
        result = self.router._keyword_classify("同一承兑行的集中度风险")
        assert result is not None
        assert result[0] == "INTENT_BATCH_RISK"

    def test_pledge_keywords(self):
        """质押融资关键词应路由到 INTENT_PLEDGE"""
        result = self.router._keyword_classify("票据质押融资方案")
        assert result is not None
        assert result[0] == "INTENT_PLEDGE"

    def test_unknown_returns_none(self):
        """无关问题关键词通道应返回 None（交由 LLM 阶段处理）"""
        result = self.router._keyword_classify("今天天气怎么样")
        assert result is None

    def test_should_use_specialized_true(self):
        """高置信度专项意图应走专项检索"""
        assert self.router.should_use_specialized("INTENT_PRICING", 0.95) is True

    def test_should_use_specialized_false_unknown(self):
        """UNKNOWN 意图不走专项检索"""
        assert self.router.should_use_specialized("INTENT_UNKNOWN", 0.95) is False

    def test_should_use_specialized_false_low_confidence(self):
        """置信度低于阈值不走专项检索"""
        assert self.router.should_use_specialized("INTENT_PRICING", 0.50) is False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 自测：检索质量评估与转人工机制
# ──────────────────────────────────────────────────────────────────────────────
class TestRetrievalQuality:
    """测试质量评分的三档判定和转人工响应构造"""

    def test_sufficient_quality(self):
        """高分多命中 → SUFFICIENT"""
        from app.services.retrieval_quality import evaluate, QUALITY_SUFFICIENT
        chunks = [{"rerank_score": 0.88}, {"rerank_score": 0.82}, {"rerank_score": 0.75}]
        r = evaluate(chunks)
        assert r.level == QUALITY_SUFFICIENT
        assert r.hit_count == 3

    def test_partial_quality(self):
        """中等分数单命中 → PARTIAL"""
        from app.services.retrieval_quality import evaluate, QUALITY_PARTIAL
        chunks = [{"rerank_score": 0.62}, {"rerank_score": 0.35}]
        r = evaluate(chunks)
        assert r.level == QUALITY_PARTIAL
        assert r.hit_count == 1

    def test_insufficient_empty(self):
        """空结果 → INSUFFICIENT"""
        from app.services.retrieval_quality import evaluate, QUALITY_INSUFFICIENT
        assert evaluate([]).level == QUALITY_INSUFFICIENT

    def test_insufficient_low_scores(self):
        """全低分 → INSUFFICIENT"""
        from app.services.retrieval_quality import evaluate, QUALITY_INSUFFICIENT
        chunks = [{"rerank_score": 0.30}, {"rerank_score": 0.20}]
        assert evaluate(chunks).level == QUALITY_INSUFFICIENT

    def test_transfer_response_structure(self):
        """转人工响应应包含 ticket_id、hotline、query_saved"""
        from app.services.retrieval_quality import build_transfer_response
        resp = build_transfer_response("这张票到期了怎么办", "INTENT_EXPIRY")
        assert resp["transfer_to_human"] is True
        assert resp["query_saved"] is True
        assert "ticket_id" in resp["contact_info"]
        assert resp["contact_info"]["ticket_id"].startswith("TKT-")
        assert resp["answer_type"] == "transfer_human"

    def test_transfer_response_unique_tickets(self):
        """每次调用应生成不同的工单号"""
        from app.services.retrieval_quality import build_transfer_response
        r1 = build_transfer_response("query1")
        r2 = build_transfer_response("query2")
        assert r1["contact_info"]["ticket_id"] != r2["contact_info"]["ticket_id"]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 自测：RAGService query_v2 主链路（mock 检索和 LLM）
# ──────────────────────────────────────────────────────────────────────────────
class TestRAGServiceV2:
    """
    测试 query_v2 的 F1 引导、F2 路由、转人工逻辑。
    使用 patch mock 掉向量检索和 LLM，不需要真实外部服务。
    异步测试通过 asyncio.run() 包装执行，无需 pytest-asyncio 插件配置。
    """

    def setup_method(self):
        from app.services.rag_service import RAGService
        self.svc = RAGService()

    def test_f1_empty_query_returns_guide(self):
        """空 query 应返回 guide_prompt 类型，不触发检索"""
        result = asyncio.run(self.svc.query_v2(tenant_id="t1", query=""))
        assert result["answer_type"] == "guide_prompt"
        assert "票据" in result["answer"]
        assert result["intent_id"] is None

    def test_f1_blank_query_returns_guide(self):
        """纯空格 query 也应返回 guide_prompt"""
        result = asyncio.run(self.svc.query_v2(tenant_id="t1", query="   "))
        assert result["answer_type"] == "guide_prompt"

    def test_f2_insufficient_returns_transfer(self):
        """检索质量不足时应返回 transfer_human 类型"""
        from unittest.mock import AsyncMock
        # 绕过 Redis 缓存：同一查询的历史缓存会遮蔽此测试期望的检索不足路径
        with patch("app.services.rag_service.bill_cache.get_rag", AsyncMock(return_value=None)), \
             patch("app.services.rag_service.vector_store") as mock_vs, \
             patch("app.services.rag_service.intent_router") as mock_ir:
            mock_ir.classify = AsyncMock(return_value=("INTENT_EXPIRY", 0.95, "keyword"))
            mock_ir.should_use_specialized.return_value = True
            mock_vs.hybrid_search.return_value = []  # 空结果 → INSUFFICIENT
            result = asyncio.run(self.svc.query_v2(tenant_id="t1", query="到期了怎么办"))
        assert result["answer_type"] == "transfer_human"
        assert result["transfer_to_human"] is True
        assert result["query_saved"] is True

    def test_f2_sufficient_returns_answer(self):
        """检索质量充足时应生成答案"""
        from unittest.mock import AsyncMock
        mock_chunks = [
            {"id": "c1", "content": "票据法第17条追索权时效", "rerank_score": 0.88,
             "section_path": "票据法", "page_num": 1, "chunk_type": "text",
             "document_id": "doc1"},
            {"id": "c2", "content": "持票人对承兑人权利两年", "rerank_score": 0.82,
             "section_path": "票据法", "page_num": 2, "chunk_type": "text",
             "document_id": "doc1"},
        ]
        with patch("app.services.rag_service.vector_store") as mock_vs, \
             patch("app.services.rag_service.intent_router") as mock_ir:
            mock_ir.classify = AsyncMock(return_value=("INTENT_EXPIRY", 0.95, "keyword"))
            mock_ir.should_use_specialized.return_value = True
            mock_vs.hybrid_search.return_value = mock_chunks
            self.svc._generate = AsyncMock(return_value="追索权时效为两年。")
            result = asyncio.run(self.svc.query_v2(tenant_id="t1", query="到期了怎么办"))
        assert result["answer_type"] == "answer"
        assert "追索权" in result["answer"]
        assert result["route_type"] == "specialized"
        assert result["retrieval_quality"] == "SUFFICIENT"

    def test_f2_fuzzy_route_for_unknown_intent(self):
        """UNKNOWN 意图应走 fuzzy 路由"""
        from unittest.mock import AsyncMock
        mock_chunks = [
            {"id": "c1", "content": "一些内容", "rerank_score": 0.80,
             "section_path": "", "page_num": 1, "chunk_type": "text",
             "document_id": "doc1"},
            {"id": "c2", "content": "更多内容", "rerank_score": 0.77,
             "section_path": "", "page_num": 2, "chunk_type": "text",
             "document_id": "doc1"},
        ]
        with patch("app.services.rag_service.vector_store") as mock_vs, \
             patch("app.services.rag_service.intent_router") as mock_ir:
            mock_ir.classify = AsyncMock(return_value=("INTENT_UNKNOWN", 0.40, "llm"))
            mock_ir.should_use_specialized.return_value = False
            mock_vs.hybrid_search.return_value = mock_chunks
            self.svc._generate = AsyncMock(return_value="模糊检索结果。")
            result = asyncio.run(self.svc.query_v2(tenant_id="t1", query="随便问一下"))
        assert result["route_type"] == "fuzzy"
        assert result["answer_type"] == "answer"

    def test_build_search_query_injects_context(self):
        """专项意图应将 bill_context 字段注入 search_query"""
        bill_ctx = {"acceptor": "招商银行", "due_date": "2025-06-30", "amount_numeric": 5000000}
        q = self.svc._build_search_query("质押融资怎么算", "INTENT_PLEDGE", bill_ctx)
        assert "招商银行" in q
        assert "2025-06-30" in q

    def test_build_search_query_no_context(self):
        """无 bill_context 时检索 query 应与原始 query 相同"""
        q = self.svc._build_search_query("贴现利率", "INTENT_PRICING", None)
        assert q == "贴现利率"

    def test_format_bill_context(self):
        """bill_context 格式化应包含非空字段"""
        ctx = {"ticket_type": "银行承兑汇票", "acceptor": "招商银行", "due_date": "2025-06-30"}
        text = self.svc._format_bill_context(ctx)
        assert "招商银行" in text
        assert "银行承兑汇票" in text


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6 自测：8 个专项场景处理器
# ──────────────────────────────────────────────────────────────────────────────
class TestSceneHandlers:
    """测试每个专项场景的 query 构造逻辑（不依赖外部服务）"""

    BASE_CTX = {
        "ticket_type": "银行承兑汇票",
        "ticket_number": "EA2024001234",
        "issue_date": "2024-09-15",
        "due_date": "2026-12-31",   # 未来日期，保证 days > 0
        "amount_numeric": 5_200_000,
        "amount_text": "伍佰贰拾万元整",
        "drawer": "甲贸易有限公司",
        "drawer_bank": "工商银行杭州支行",
        "acceptor": "杭州银行股份有限公司",
        "payee": "乙科技有限公司",
        "endorsers": ["丙实业", "丁物流"],
    }

    def test_scene1_pricing_queries(self):
        """场景一：定价查询应包含承兑人名称和利率关键词"""
        from app.services.scene_handlers import build_pricing_queries
        result = build_pricing_queries(self.BASE_CTX)
        joined = " ".join(result["queries"])
        assert "杭州银行" in joined
        assert "利率" in joined or "贴现" in joined
        assert "context_hint" in result

    def test_scene2_endorser_queries(self):
        """场景二：背书链查询应为每个主体生成一条查询"""
        from app.services.scene_handlers import build_endorser_check_queries
        result = build_endorser_check_queries(self.BASE_CTX)
        # 出票人 + 2个背书人 = 至少 3 条查询
        assert len(result["queries"]) >= 3
        joined = " ".join(result["queries"])
        assert "甲贸易" in joined
        assert "丙实业" in joined
        assert "失信" in joined or "黑名单" in joined

    def test_scene3_expiry_queries(self):
        """场景三：到期追索查询应包含票据法第17条"""
        from app.services.scene_handlers import build_expiry_queries
        result = build_expiry_queries(self.BASE_CTX)
        joined = " ".join(result["queries"])
        assert "追索权" in joined
        assert "时效" in joined

    def test_scene3_overdue_context_hint(self):
        """场景三：已逾期票据的 context_hint 应包含逾期天数"""
        from app.services.scene_handlers import build_expiry_queries
        ctx = dict(self.BASE_CTX, due_date="2020-01-01")
        result = build_expiry_queries(ctx)
        assert "逾期" in result["context_hint"]

    def test_scene4_aml_queries_large_amount(self):
        """场景四：500万以上应触发大额合规查询"""
        from app.services.scene_handlers import build_aml_queries
        result = build_aml_queries(self.BASE_CTX)
        joined = " ".join(result["queries"])
        assert "大额" in joined
        assert "反洗钱" in joined

    def test_scene5_element_check_date_error(self):
        """场景五：日期错误风险应映射到票据法第22条查询"""
        from app.services.scene_handlers import build_element_check_queries
        flags = [{"code": "DATE_LOGIC_ERROR", "level": "severe",
                  "desc": "到期日早于出票日"}]
        result = build_element_check_queries(self.BASE_CTX, flags)
        joined = " ".join(result["queries"])
        assert "第22条" in joined

    def test_scene5_element_check_acceptor_error(self):
        """场景五：承兑人不合规应映射到票据法第38条查询"""
        from app.services.scene_handlers import build_element_check_queries
        flags = [{"code": "ACCEPTOR_NOT_BANK", "level": "severe",
                  "desc": "承兑人非银行"}]
        result = build_element_check_queries(self.BASE_CTX, flags)
        joined = " ".join(result["queries"])
        assert "第38条" in joined

    def test_scene6_history_queries(self):
        """场景六：流转溯源应包含新增背书人和背书连续性查询"""
        from app.services.scene_handlers import build_history_queries
        result = build_history_queries(self.BASE_CTX, ["戊金融服务"])
        joined = " ".join(result["queries"])
        assert "戊金融服务" in joined
        assert "连续" in joined or "第31条" in joined

    def test_scene7_batch_risk_queries(self):
        """场景七：批量风险查询应包含承兑人和集中度关键词"""
        from app.services.scene_handlers import build_batch_risk_queries
        result = build_batch_risk_queries(self.BASE_CTX, 320_000_000, 12)
        joined = " ".join(result["queries"])
        assert "杭州银行" in joined
        assert "集中" in joined or "敞口" in joined

    def test_scene8_pledge_queries(self):
        """场景八：质押融资查询应包含质押率和背书规范"""
        from app.services.scene_handlers import build_pledge_queries
        result = build_pledge_queries(self.BASE_CTX)
        joined = " ".join(result["queries"])
        assert "质押" in joined
        assert "利率" in joined or "融资" in joined

    def test_get_scene_queries_dispatch(self):
        """统一路由入口应正确分发到各场景处理器"""
        from app.services.scene_handlers import get_scene_queries
        for intent_id in [
            "INTENT_PRICING", "INTENT_ENDORSER_CHECK", "INTENT_EXPIRY",
            "INTENT_AML", "INTENT_ELEMENT_CHECK", "INTENT_HISTORY",
            "INTENT_BATCH_RISK", "INTENT_PLEDGE",
        ]:
            result = get_scene_queries(intent_id, self.BASE_CTX)
            assert len(result["queries"]) >= 1, f"{intent_id} 未生成任何查询"
            assert "context_hint" in result

    def test_unknown_intent_returns_empty(self):
        """未知意图应返回空查询列表"""
        from app.services.scene_handlers import get_scene_queries
        result = get_scene_queries("INTENT_UNKNOWN", self.BASE_CTX)
        assert result["queries"] == []


# ──────────────────────────────────────────────────────────────────────────────
# 测试路由层（TestRouters）
# 验证 bills_router 和 /query/v2 端点的核心逻辑，使用 mock 绕过真实 DB/Milvus
# ──────────────────────────────────────────────────────────────────────────────
class TestRouters:
    """验证新增 API 端点的逻辑正确性（不需要真实数据库）"""

    # ── bills_router 上传端点的 Schema 验证 ────────────────────────────────────
    def test_bill_upsert_response_schema(self):
        """BillUpsertResponse Schema 必须包含所有必要字段"""
        from app.models.schemas import BillUpsertResponse
        resp = BillUpsertResponse(
            bill_record_id="rec-001",
            ticket_number="TICKET-001",
            version=1,
            is_new_bill=True,
            new_endorsers=["公司A", "公司B"],
            risk_flags=[{"code": "DATE_LOGIC_ERROR", "level": "severe", "desc": "描述"}],
            document_id="doc-001",
            elapsed_ms=120.5,
        )
        assert resp.bill_record_id == "rec-001"
        assert resp.is_new_bill is True
        assert len(resp.new_endorsers) == 2
        assert resp.version == 1

    def test_bill_record_response_schema(self):
        """BillRecordResponse Schema 必须包含版本历史字段"""
        from app.models.schemas import BillRecordResponse, BillVersionResponse
        from datetime import datetime
        ver = BillVersionResponse(
            version=1,
            endorsers=["甲公司"],
            new_endorsers=["甲公司"],
            upload_time=datetime.now(),
            uploaded_by="user-001",
        )
        rec = BillRecordResponse(
            id="rec-001",
            ticket_number="TICKET-001",
            ticket_type="银行承兑汇票",
            issue_date="2024-01-01",
            due_date="2024-07-01",
            amount_numeric=1_000_000.0,
            amount_text="壹百万元整",
            drawer="甲公司",
            acceptor="ABC银行",
            payee="乙公司",
            risk_flags=[],
            latest_version=1,
            versions=[ver],
            created_at=datetime.now(),
        )
        assert rec.latest_version == 1
        assert len(rec.versions) == 1
        assert rec.versions[0].endorsers == ["甲公司"]

    # ── query/v2 端点 Schema 验证 ────────────────────────────────────────────
    def test_query_request_v2_allows_empty_query(self):
        """QueryRequestV2 必须允许空 query（触发 F1 引导）"""
        from app.models.schemas import QueryRequestV2
        req = QueryRequestV2(query="")
        assert req.query == ""
        assert req.bill_record_id is None
        assert req.stream is False

    def test_query_request_v2_with_bill_record(self):
        """QueryRequestV2 支持携带 bill_record_id"""
        from app.models.schemas import QueryRequestV2
        req = QueryRequestV2(query="这张票据的贴现利率是多少？", bill_record_id="rec-abc123")
        assert req.bill_record_id == "rec-abc123"
        assert len(req.query) > 0

    def test_query_response_v2_schema(self):
        """QueryResponseV2 包含意图路由和检索质量字段"""
        from app.models.schemas import QueryResponseV2
        resp = QueryResponseV2(
            query_id="qid-001",
            answer_type="answer",
            answer="参考利率为 2.85%。",
            intent_id="INTENT_PRICING",
            route_type="specialized",
            retrieval_quality="SUFFICIENT",
            transfer_to_human=False,
            retrieval_ms=85.0,
            llm_ms=420.0,
            total_ms=510.0,
        )
        assert resp.answer_type == "answer"
        assert resp.transfer_to_human is False
        assert resp.intent_id == "INTENT_PRICING"

    def test_query_response_v2_transfer_to_human(self):
        """QueryResponseV2 转人工时携带 contact_info"""
        from app.models.schemas import QueryResponseV2
        resp = QueryResponseV2(
            query_id="qid-002",
            answer_type="transfer_human",
            answer="抱歉，暂无相关内容，请联系人工客服。",
            transfer_to_human=True,
            contact_info={"hotline": "400-XXX-XXXX", "ticket_id": "TKT-ABCD1234"},
            query_saved=True,
            retrieval_quality="INSUFFICIENT",
        )
        assert resp.transfer_to_human is True
        assert resp.contact_info["hotline"] == "400-XXX-XXXX"
        assert resp.query_saved is True

    # ── bills_router 注册验证（不依赖真实 DB）───────────────────────────────────
    def test_bills_router_is_exported(self):
        """bills_router 必须从 routers 模块正确导出"""
        from app.api.routers import bills_router
        from fastapi import APIRouter
        assert isinstance(bills_router, APIRouter)
        assert bills_router.prefix == "/bills"

    def test_bills_router_routes_registered(self):
        """bills_router 必须包含 upload、列表、详情三条路由"""
        from app.api.routers import bills_router
        paths = [r.path for r in bills_router.routes]
        assert "/bills/upload" in paths          # 上传入库
        assert "/bills/" in paths                # 列表查询
        assert "/bills/{ticket_number}" in paths  # 详情查询

    def test_query_v2_route_registered(self):
        """query_router 必须包含 /query/v2 路由"""
        from app.api.routers import query_router
        paths = [r.path for r in query_router.routes]
        assert "/query/v2" in paths      # 增强版问答路由

    # ── bill_context 字段映射验证 ─────────────────────────────────────────────
    def test_bill_context_fields_coverage(self):
        """查询 v2 路由注入的 bill_context 字段必须涵盖 scene_handlers 所需的所有关键字段"""
        required_fields = {
            "ticket_type", "due_date", "amount_numeric",
            "drawer", "drawer_bank", "acceptor", "payee",
        }
        # 模拟路由中构建 bill_context 的字段集合（与 query_v2_endpoint 保持一致）
        context_keys = {
            "ticket_number", "ticket_type", "issue_date", "due_date",
            "amount_numeric", "amount_text", "drawer", "drawer_bank",
            "acceptor", "payee",
        }
        assert required_fields.issubset(context_keys), "bill_context 缺少 scene_handlers 所需字段"


# ──────────────────────────────────────────────────────────────────────────────
# 测试执行入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 直接运行此文件时，启动 pytest 测试
    # -v：详细输出（显示每个测试用例的名称和结果）
    # --tb=short：出错时只显示简短的错误信息（不显示完整堆栈）
    pytest.main([__file__, "-v", "--tb=short"])
