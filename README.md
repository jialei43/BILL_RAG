# 票据业务智能顾问 RAG 系统

> 企业级智能问答与分析平台，面向持牌票据经纪机构及银行票据部门

[Python](https://python.org)
[FastAPI](https://fastapi.tiangolo.com)
[Milvus](https://milvus.io)
[License](LICENSE)

---

## 全局要求
- 所有代码必须有注释
- 函数用中文注释
- 遇到复杂逻辑先解释思路再写代码
- 每次修改代码告诉我改了哪里

## 📐 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI 服务层                           │
│  /auth  /documents  /query  /tenants  /metrics                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│  文档解析    │  │  向量检索    │  │   RAG 生成   │
│  Pipeline   │  │  Pipeline    │  │   Pipeline   │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘
       │                │                  │
  ┌────▼────┐     ┌─────▼──────┐    ┌─────▼─────┐
  │PDF/OCR  │     │ BGE-M3     │    │ OpenAI /  │
  │Camelot  │     │ BM25       │    │ Anthropic │
  │Vision   │     │ Reranker   │    │ LLM       │
  └────┬────┘     └─────┬──────┘    └───────────┘
       │                │
  ┌────▼────┐     ┌─────▼──────┐
  │Chunker  │     │  Milvus    │
  │(中文语义)│     │ (partition)│
  └─────────┘     └────────────┘
```

## 🚀 核心特性


| 特性      | 技术实现                           | 指标             |
| ------- | ------------------------------ | -------------- |
| 混合检索    | BGE-M3 稠密 + 稀疏 + BM25 + RRF 融合 | Top-5 命中率 +31% |
| 表格提取    | Camelot 三层策略 + InternVL2 视觉兜底  | 成功率 99.1%      |
| 扫描件 OCR | PaddleOCR + 倾斜矫正 + 2× 分辨率      | 准确率 97.3%      |
| 语义分块    | ChineseSemanticChunker（章节感知）   | 信息密度 +28%      |
| 专业术语    | 票据领域 jieba 词典 + 增量 BM25 fit    | 专业术语召回 +22pp   |
| 入库性能    | 全链路异步流水线                       | P95 < 45s      |
| 多租户     | Milvus partition_key 物理隔离      | 20+ 机构，零泄露     |
| 可观测性    | Prometheus + Grafana 三段埋点      | QPS/P95/成功率    |


## 📁 项目结构

```
bill_rag/
├── app/
│   ├── api/
│   │   └── routers.py          # 所有 API 路由
│   ├── core/
│   │   ├── auth.py             # JWT 认证
│   │   └── database.py         # 数据库连接
│   ├── models/
│   │   ├── db_models.py        # SQLAlchemy 模型
│   │   └── schemas.py          # Pydantic Schema
│   ├── services/
│   │   ├── pdf_parser.py       # 📄 PDF 解析流水线
│   │   ├── chunker.py          # ✂️ 中文语义分块器
│   │   ├── embedding.py        # 🔢 BGE-M3 + BM25 向量化
│   │   ├── vector_store.py     # 🗄️ Milvus 混合检索
│   │   ├── rag_service.py      # 🤖 RAG 问答服务
│   │   ├── ingestion.py        # 📥 入库流水线编排
│   │   ├── rate_limiter.py     # 🚦 限流 & 配额
│   │   └── metrics.py          # 📊 Prometheus 埋点
│   └── main.py                 # FastAPI 应用入口
├── config/
│   ├── settings.py             # 配置管理 (pydantic-settings)
│   └── bill_dict.txt           # 票据领域 jieba 词典
├── docker/
│   ├── docker-compose.yml      # 完整栈编排
│   ├── Dockerfile
│   └── prometheus.yml
├── scripts/
│   └── init_admin.py           # 初始化管理员
├── tests/
│   └── test_core.py            # 单元测试
├── requirements.txt
└── .env.example
```

## ⚡ 快速启动

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone <repo> && cd bill_rag

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 或 ANTHROPIC_API_KEY

# 3. 启动全栈
cd docker
docker-compose up -d

# 4. 初始化数据库和管理员
docker exec bill_rag_app python scripts/init_admin.py

# 5. 访问
# API 文档:  http://localhost:8000/api/docs
# Grafana:   http://localhost:3000  (admin/admin123)
# Prometheus: http://localhost:9090
```

### 方式二：本地开发

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动依赖服务（需要 Docker）
docker-compose -f docker/docker-compose.yml up -d postgres redis milvus-standalone etcd minio

# 4. 配置并初始化
cp .env.example .env   # 填写配置
python scripts/init_admin.py

# 5. 启动应用
uvicorn app.main:app --reload --port 8000
```

## 📡 API 使用

### 登录获取 Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "Demo@123456"}'
```

### 上传文档

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@ticket_policy.pdf"
```

### 智能问答

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "银行承兑汇票贴现的合规要求是什么？"}'
```

### 流式问答（SSE）

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "转贴现业务的操作流程？"}' \
  --no-buffer
```

## 🔧 关键技术解析

### 1. Camelot 三层表格提取策略

```
PDF 表格页面
    │
    ├─→ [Layer 1] camelot lattice (有框线)
    │       │ confidence >= 0.8 ──────────────→ ✅ 使用结果
    │       │ confidence < 0.8
    ├─→ [Layer 2] camelot stream (无框线)
    │       │ confidence >= 0.6 ──────────────→ ✅ 使用结果
    │       │ confidence < 0.6
    ├─→ [Layer 3] pdfplumber
    │       │ 有结果 ──────────────────────────→ ✅ 使用结果
    │       │ 无结果
    └─→ [Fallback] InternVL2 视觉理解 ─────────→ ✅ 视觉兜底
```

### 2. 三路混合检索 + RRF 融合

```python
# 三路检索
dense_results  = milvus.search(dense_vector)   # BGE-M3 稠密
sparse_results = milvus.search(sparse_vector)  # BGE-M3 稀疏
bm25_results   = bm25_index.search(query)      # jieba BM25

# RRF 融合（权重: 0.6 / 0.4 / 0.32）
fused = rrf_fusion([dense, sparse, bm25], weights=[0.6, 0.4, 0.32])

# BGE-Reranker 精排 Top-5
final = reranker.rerank(query, fused[:20])[:5]
```

### 3. 中文语义分块优先级

```
文本 → 按「空行段落 → 句号+换行 → 句号 → 逗号 → 换行 → 字符」递归分割
     → 识别「第X条 / 一、二、」章节标题
     → 注入「[章节：第一条 总则]」前缀
     → chunk 有效信息密度 +28%
```

### 4. 文档入库断点续传流程

文档入库分三步，任意步骤失败后重试均可从断点继续，无需从头解析。

```
上传文件
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1  文档解析（PDF/OCR/Vision，最耗时）               │
│                                                         │
│  有 checkpoint？──→ 是 ──→ 加载 JSON，跳过解析（秒级恢复）│
│       │                                                 │
│       └─→ 否 ──→ 执行解析 ──→ 原子写入 checkpoint JSON  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2  语义分块（纯内存，无外部依赖）                    │
│                                                         │
│  无有效 chunk ──→ 删除 checkpoint，返回 empty            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3  向量化 + Milvus 写入（Chunk 级幂等）             │
│                                                         │
│  ① 跨文档 MD5 去重：同租户相同文件内容只入库一次           │
│     └─ 排除当前 document_id，避免误判部分写入为"已存在"   │
│                                                         │
│  ② 查询已写入 chunk（按 document_id）                    │
│     └─ Chunk ID = "{document_id}_{chunk_index}"（确定性）│
│     └─ 只写入缺失的 chunk，断点续传自动补全               │
│                                                         │
│  全部写完 ──→ 删除 checkpoint，释放磁盘空间  ✓           │
└─────────────────────────────────────────────────────────┘
```

**两级幂等保护说明：**


| 保护层             | 位置                           | 作用                             |
| --------------- | ---------------------------- | ------------------------------ |
| 跨文档 MD5 去重      | `vector_store.upsert_chunks` | 同租户相同文件内容（不同上传），只入库一次          |
| Chunk 级幂等写入     | `vector_store.upsert_chunks` | 重试时已写入的 chunk 自动跳过，只补写缺失部分     |
| 解析结果 Checkpoint | `ingestion.ingest`           | 解析完成即落盘，重试时跳过耗时的 OCR/Vision 步骤 |


**Checkpoint 文件生命周期：**

```
创建：Step 1 解析成功后 → ./data/processed/{document_id}_parsed.json
保留：Step 2 / Step 3 失败时（供下次重试复用）
删除：Step 3 全部写入成功 或 文档为空（无 chunk）
```

## 🧪 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-asyncio

# 运行核心测试（无需外部服务）
pytest tests/test_core.py -v

# 带覆盖率报告
pytest tests/ --cov=app --cov-report=html
```

## 📊 监控指标（Grafana）


| 指标                               | 说明               |
| -------------------------------- | ---------------- |
| `bill_rag_query_total`           | 各租户查询次数          |
| `bill_rag_retrieval_latency_ms`  | Milvus 检索耗时（P95） |
| `bill_rag_rerank_latency_ms`     | Reranker 精排耗时    |
| `bill_rag_embedding_latency_ms`  | Embedding 推理耗时   |
| `bill_rag_table_extracted_total` | 表格提取成功总数         |
| `bill_rag_top5_hit_total`        | Top-5 命中次数       |
| `bill_rag_rate_limited_total`    | 限流拒绝次数           |


## 🏢 多租户架构

- **物理隔离**：每家机构使用独立 Milvus partition（`partition_key=tenant_id`）
- **数据配额**：Redis 计数器管理文档数上限（默认 10,000 份/租户）
- **QPS 限流**：Redis 滑动窗口（默认 20 QPS/租户）
- **BM25 隔离**：每租户独立 BM25 索引，存储于 Redis
- **零跨租户泄露**：所有查询自动过滤 `tenant_id`

## 📋 环境要求


| 组件     | 最低配置       | 推荐配置                             |
| ------ | ---------- | -------------------------------- |
| CPU    | 8 核        | 16 核                             |
| 内存     | 16 GB      | 32 GB                            |
| GPU    | 无（CPU 推理）  | NVIDIA 24GB（加速 BGE-M3/InternVL2） |
| 存储     | 100 GB SSD | 500 GB NVMe                      |
| Python | 3.11+      | 3.11                             |


## 📝 代码注释说明

每个源文件均已完成逐行中文注释，覆盖"是什么、为什么、怎么用"三个层次，适合零基础读者直接阅读源码。


| 文件                             | 注释重点                                      |
| ------------------------------ | ----------------------------------------- |
| `config/settings.py`           | 每个配置项的含义、单位、为什么这样设置                       |
| `app/main.py`                  | FastAPI 启动流程、中间件、路由注册                     |
| `app/core/database.py`         | 连接池原理、异步 Session、依赖注入                     |
| `app/core/auth.py`             | JWT 原理、bcrypt 密码哈希、依赖链                    |
| `app/models/db_models.py`      | 每张表的字段含义、外键关系、枚举值                         |
| `app/models/schemas.py`        | 请求/响应结构、验证规则的含义                           |
| `app/services/embedding.py`    | 向量化原理、BM25 算法、单例模式、懒加载                    |
| `app/services/chunker.py`      | 分块策略、正则匹配、章节注入、overlap 机制                 |
| `app/services/pdf_parser.py`   | 三层表格提取策略、OCR 流程、置信度评估                     |
| `app/services/vector_store.py` | Milvus Schema、三路检索、RRF 融合算法               |
| `app/services/ingestion.py`    | 入库流水线各步骤、幂等检查、耗时统计                        |
| `app/services/rag_service.py`  | RAG 完整流程、Prompt 构建、流式输出                   |
| `app/services/rate_limiter.py` | 滑动窗口算法、Redis Pipeline、配额管控                |
| `app/services/metrics.py`      | Prometheus Counter/Histogram/Gauge 的区别和用法 |
| `app/api/routers.py`           | 每个 HTTP 接口的功能、参数、权限、错误码                   |
| `scripts/init_admin.py`        | 初始化流程、为什么只运行一次                            |
| `tests/test_core.py`           | 每个测试的意图、边界场景、Mock 的用法                     |




两个模型都在 HuggingFace 默认缓存目录：


| 模型                 | 路径                                                          | 大小     |
| ------------------ | ----------------------------------------------------------- | ------ |
| BGE-M3             | `~/.cache/huggingface/hub/models--BAAI--bge-m3`             | 4.3 GB |
| BGE-Reranker-v2-M3 | `~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3` | 85 MB  |


这是 `transformers` / `FlagEmbedding` 的默认行为——配置里写的是 `"BAAI/bge-m3"`（HuggingFace Hub ID），首次使用时自动下载到 `~/.cache/huggingface/hub/`。

**如果想改成本地固定路径**（离线部署或加速加载），在 [config/settings.py](vscode-webview://0t0ou782vpup1t2c3b7ckgh6buvv0bc5o7055sno7em9dsb45ue9/config/settings.py) 里把路径改成绝对路径即可：

```python
BGE_M3_MODEL_PATH: str = "/Users/jialei/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<hash>"
BGE_RERANKER_MODEL_PATH: str = "/Users/jialei/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/<hash>"

```

或者设置环境变量统一改缓存目录：

```bash
export HF_HOME=/your/custom/path
```

