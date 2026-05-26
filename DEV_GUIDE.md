# 票据业务智能顾问 RAG 系统 — 开发与运维指南

> 面向开发人员、运维人员和新员工的完整技术文档。

---

## 目录

1. [项目概览](#1-项目概览)
2. [系统架构](#2-系统架构)
3. [本地开发环境搭建](#3-本地开发环境搭建)
4. [Docker 部署](#4-docker-部署)
5. [配置项说明](#5-配置项说明)
6. [数据库模型](#6-数据库模型)
7. [API 接口清单](#7-api-接口清单)
8. [核心服务层详解](#8-核心服务层详解)
9. [监控与运维](#9-监控与运维)
10. [测试](#10-测试)
11. [常见问题排查](#11-常见问题排查)

---

## 1. 项目概览

**票据业务智能顾问 RAG 系统**是面向持牌票据经纪机构及银行票据部门的企业级智能问答与分析平台，基于检索增强生成（RAG）技术构建。

### 核心能力

| 能力 | 技术方案 |
|------|----------|
| 混合检索 | BGE-M3 稠密向量 + 稀疏权重 + BM25，RRF 融合 + Reranker 精排 |
| 文档解析 | PDF/Word/Excel/图像，含扫描件 OCR（倾斜矫正） |
| 多租户隔离 | Milvus partition_key 物理隔离，零跨租户数据泄露 |
| 票据识别 | 视觉大模型（Qwen-VL）提取票据要素，支持生命周期管理 |
| 意图路由 | 关键词快通道 + LLM 语义识别，8 个专项检索场景 |
| 可观测性 | Prometheus + Grafana 全链路监控，结构化 JSON 日志 |

### 技术栈

```
后端框架：    FastAPI + Uvicorn（ASGI）
语言模型：    OpenAI API（兼容通义千问 qwen-max）/ Anthropic Claude
向量模型：    BGE-M3（1024维稠密+稀疏）+ BGE-Reranker-v2-m3
向量数据库：  Milvus 2.4（IVF_FLAT 索引，IP 相似度）
关系数据库：  PostgreSQL 16（asyncpg 异步驱动）
缓存/限流：   Redis 7（BM25 索引 + 滑动窗口限流）
PDF 解析：    PyMuPDF + Camelot + pdfplumber + PaddleOCR
日志：        loguru（三路输出：stderr / app.log / error.log）
监控：        prometheus-client + prometheus-fastapi-instrumentator
容器化：      Docker Compose
```

---

## 2. 系统架构

### 整体架构图

```
用户请求
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 服务层（:8000）                  │
│  /api/v1/auth  /api/v1/documents  /api/v1/query             │
│  /api/v1/tenants  /api/v1/bills  /metrics  /health          │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼────────────────┐
         ▼               ▼                ▼
  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
  │  文档入库    │  │  智能问答    │  │  票据识别    │
  │  Pipeline   │  │  Pipeline    │  │  Pipeline    │
  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
         │                │                  │
  ┌──────▼──────┐  ┌──────▼───────┐  ┌──────▼───────┐
  │ PDF解析      │  │ 意图路由     │  │ 视觉大模型   │
  │ Chunker     │  │ 混合检索     │  │ Qwen-VL      │
  │ Embedding   │  │ Reranker     │  │              │
  └──────┬──────┘  └──────┬───────┘  └──────────────┘
         │                │
         ▼                ▼
  ┌─────────────────────────────────┐
  │         数据存储层               │
  │  Milvus（向量） PostgreSQL（元数据）│
  │  Redis（缓存+限流）               │
  └─────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────┐
  │         可观测性层               │
  │  Prometheus（:9090）             │
  │  Grafana（:3000）                │
  │  loguru 结构化日志（./logs/）     │
  └─────────────────────────────────┘
```

### 目录结构

```
bill_rag/
├── app/
│   ├── main.py                  # FastAPI 应用入口（启动/关闭/路由/中间件）
│   ├── api/
│   │   └── routers.py           # 所有 HTTP 路由定义
│   ├── core/
│   │   ├── auth.py              # JWT 认证与权限管理
│   │   ├── database.py          # PostgreSQL 连接池（异步）
│   │   └── logging.py           # 三路日志配置（loguru）
│   ├── models/
│   │   ├── db_models.py         # SQLAlchemy ORM 模型（8 张表）
│   │   └── schemas.py           # Pydantic 请求/响应结构（20+ Schema）
│   └── services/
│       ├── pdf_parser.py        # 文档解析（四层兜底）
│       ├── chunker.py           # 中文语义分块器
│       ├── embedding.py         # BGE-M3 + BM25 混合向量化
│       ├── vector_store.py      # Milvus 三路混合检索 + RRF 融合
│       ├── rag_service.py       # RAG 完整流程 + LLM 生成
│       ├── ingestion.py         # 文档入库流水线（断点续传）
│       ├── bill_recognition.py  # 票据要素视觉识别（P1）
│       ├── bill_lifecycle.py    # 票据生命周期管理（P1）
│       ├── intent_router.py     # 意图识别路由（P2）
│       ├── scene_handlers.py    # 8 个专项检索场景处理器（P2）
│       ├── retrieval_quality.py # 检索质量评估与转人工（P3）
│       ├── rate_limiter.py      # 租户 QPS 限流与配额管理
│       └── metrics.py           # Prometheus 监控埋点
├── config/
│   ├── settings.py              # 全局配置（60+ 项，支持 .env 覆盖）
│   └── bill_dict.txt            # 票据行业 jieba 专用词典
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml       # 完整栈（app/postgres/redis/prometheus/grafana）
│   └── prometheus.yml           # Prometheus 抓取规则
├── scripts/
│   ├── init_admin.py            # 首次部署：创建管理员账号和演示租户
│   └── generate_test_pdf.py     # 生成复杂票据测试 PDF
├── tests/
│   └── test_core.py             # 单元测试（分块/RRF/BM25/限流）
├── data/
│   ├── uploads/                 # 用户上传的原始文件
│   └── processed/               # 解析处理后的中间文件（checkpoint JSON）
├── logs/                        # 运行时日志（app.log / error.log）
├── requirements.txt
└── .env                         # 本地环境变量（不提交 Git）
```

---

## 3. 本地开发环境搭建

### 前置依赖

- Python 3.11+
- PostgreSQL 16
- Milvus 2.4（或使用已有实例）
- Redis 7

### 步骤一：克隆项目，创建虚拟环境

```bash
git clone <repo_url>
cd bill_rag

python -m venv bill_rag_env
source bill_rag_env/bin/activate      # Windows: bill_rag_env\Scripts\activate

pip install -r requirements.txt
```

### 步骤二：配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，最少需要填写以下字段：

```bash
# 数据库（确保 PostgreSQL 已启动并建好 bill_rag 库）
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/bill_rag

# LLM API Key（二选一）
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxx...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # 使用通义千问时填写
OPENAI_MODEL=qwen-max

# 向量模型本地路径（从 HuggingFace 下载后填绝对路径）
BGE_M3_MODEL_PATH=/path/to/bge-m3
BGE_RERANKER_MODEL_PATH=/path/to/bge-reranker-v2-m3

# 视觉模型（可选，用于表格提取兜底）
VISION_ENABLED=true
VISION_MODEL=qwen-vl-max

# 开发模式
DEBUG=true
```

### 步骤三：初始化数据库

```bash
# 确保 PostgreSQL 已启动，先建数据库
psql -U postgres -c "CREATE DATABASE bill_rag;"

# 初始化表结构 + 创建管理员账号
python scripts/init_admin.py
```

初始化完成后会打印：
```
管理员账号: admin / Admin@123456
演示账号:   demo  / Demo@123456
```

### 步骤四：启动服务

**方式 A：命令行（推荐开发时使用）**

```bash
# 在项目根目录执行
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**方式 B：直接运行 main.py（Cursor/PyCharm 调试时使用）**

```bash
python app/main.py
```

> `app/main.py` 已在顶部自动将项目根目录加入 `sys.path`，IDE 直接运行无需额外配置。

### 步骤五：验证服务

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/api/docs` | Swagger 交互式 API 文档 |
| `http://localhost:8000/api/redoc` | ReDoc 格式 API 文档 |
| `http://localhost:8000/health` | 健康检查（返回 `{"status": "ok"}`） |
| `http://localhost:8000/metrics` | Prometheus 原始指标数据 |

---

## 4. Docker 部署

### 启动完整栈

```bash
# 项目根目录执行
cd docker

# 启动所有服务（首次会拉镜像，需等待）
docker compose up -d

# 查看服务状态
docker compose ps
```

**服务端口映射：**

| 服务 | 端口 | 说明 |
|------|------|------|
| app | 8000 | FastAPI 应用 |
| postgres | 5432 | PostgreSQL |
| redis | 6379 | Redis |
| prometheus | 9090 | Prometheus UI |
| grafana | 3000 | Grafana 面板（admin/admin123） |

> Milvus 使用外部已有实例（通过 `milvus_default` 网络连接），如需独立启动，取消注释 `docker-compose.yml` 中对应部分。

### 初始化数据库（首次部署）

```bash
docker exec bill_rag_app python scripts/init_admin.py
```

### 常用 Docker 运维命令

```bash
# 查看应用日志
docker logs -f bill_rag_app

# 重启应用（不重建镜像）
docker compose restart app

# 重新构建并启动（代码变更后）
docker compose up -d --build app

# 停止所有服务
docker compose down

# 停止并清除所有数据卷（危险：会删除数据库数据）
docker compose down -v
```

---

## 5. 配置项说明

所有配置通过 `config/settings.py` 管理，支持 `.env` 文件或环境变量覆盖，**环境变量优先级高于 `.env` 文件**。

### 应用基础

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `APP_NAME` | 票据业务智能顾问 RAG 系统 | 显示在 API 文档标题 |
| `APP_VERSION` | 1.0.0 | 版本号 |
| `DEBUG` | false | true=详细日志+热重载，生产必须为 false |
| `SECRET_KEY` | change-me-... | JWT 加密密钥，**生产环境必须替换** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 1440 | Token 有效期（分钟，默认 24 小时） |

### 数据库

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DATABASE_URL` | postgresql+asyncpg://postgres:postgres@localhost:5432/bill_rag | PostgreSQL 连接串 |
| `MILVUS_HOST` | localhost | Milvus 服务地址 |
| `MILVUS_PORT` | 19530 | Milvus 端口 |
| `MILVUS_COLLECTION` | bill_documents | 向量集合名 |
| `MILVUS_DENSE_DIM` | 1024 | BGE-M3 向量维度（不可随意修改） |
| `MILVUS_METRIC_TYPE` | IP | 相似度计算（IP=内积） |
| `REDIS_URL` | redis://localhost:6379/0 | Redis 连接地址 |

### 限流与配额

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TENANT_QPS_LIMIT` | 20 | 每租户每秒最大请求数 |
| `TENANT_QPS_WINDOW` | 60 | 限流滑动窗口大小（秒） |
| `TENANT_DOC_QUOTA` | 10000 | 每租户最大文档数 |

### 向量模型

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `BGE_M3_MODEL_PATH` | （本地路径） | BGE-M3 模型快照绝对路径 |
| `BGE_RERANKER_MODEL_PATH` | （本地路径） | Reranker 模型快照绝对路径 |
| `EMBEDDING_BATCH_SIZE` | 32 | 每批 Embedding 条数（避免显存溢出） |
| `RERANKER_TOP_N` | 5 | 精排保留 Top-N |

### 检索参数

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `RETRIEVAL_TOP_K` | 20 | 粗检索候选数量 |
| `RERANK_TOP_N` | 5 | 精排保留数量 |
| `HYBRID_DENSE_WEIGHT` | 0.6 | 稠密向量权重 |
| `HYBRID_SPARSE_WEIGHT` | 0.4 | 稀疏向量权重 |
| `RRF_K` | 60 | RRF 融合常数（标准值，通常无需修改） |

### 文档解析与分块

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CAMELOT_LATTICE_THRESHOLD` | 0.8 | 有框线表格置信度合格线 |
| `CAMELOT_STREAM_THRESHOLD` | 0.6 | 无框线表格置信度合格线 |
| `OCR_USE_ANGLE_CLS` | true | 自动纠正扫描件倾斜 |
| `OCR_DPI_SCALE` | 2 | 图像放大倍数（提高 OCR 准确率） |
| `CHUNK_MAX_TOKENS` | 512 | 每块最大 token 数（约 256-512 字） |
| `CHUNK_OVERLAP_TOKENS` | 64 | 相邻块重叠 token 数 |
| `CHUNK_MIN_TOKENS` | 50 | 最小块大小（低于此值丢弃） |
| `VISION_ENABLED` | true | 表格提取质量差时用视觉模型兜底 |
| `VISION_CONFIDENCE_THRESHOLD` | 0.8 | 触发视觉兜底的置信度阈值 |

### LLM

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | openai | openai / anthropic |
| `OPENAI_API_KEY` | （空） | OpenAI 或通义千问的 API Key |
| `OPENAI_BASE_URL` | （空） | 通义千问填 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `OPENAI_MODEL` | gpt-4o | 使用通义千问时填 `qwen-max` |
| `ANTHROPIC_API_KEY` | （空） | Claude API Key |
| `ANTHROPIC_MODEL` | claude-3-5-sonnet-20241022 | Claude 模型版本 |
| `LLM_MAX_TOKENS` | 2048 | 答案最大 token 数 |
| `LLM_TEMPERATURE` | 0.1 | 生成随机性（0=确定性，1=随机） |

### 日志与监控

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_DIR` | ./logs | 日志文件目录 |
| `LOG_LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR |
| `LOG_ROTATION` | 50 MB | 单文件超过此大小自动轮转 |
| `LOG_RETENTION` | 30 days | 日志保留天数 |
| `PROMETHEUS_ENABLED` | true | 是否采集 Prometheus 指标 |

---

## 6. 数据库模型

PostgreSQL 中共 8 张表，通过 SQLAlchemy ORM 管理，`scripts/init_admin.py` 自动建表。

### 表关系总览

```
Tenant ──┬── User (1:N)
         ├── Document (1:N) ── DocumentChunk (1:N)
         └── BillRecord (1:N) ── BillVersion (1:N)

QueryLog      （独立，关联 tenant_id / user_id）
IntentLog     （独立，关联 tenant_id / user_id）
SearchMissLog （独立，关联 tenant_id）
```

### Tenant（租户）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| name | String(200) | 机构名称 |
| code | String(50) | 机构代码（唯一） |
| license_no | String(100) | 牌照编号 |
| status | Enum | active / suspended / trial |
| doc_quota | Integer | 文档配额（默认 10000） |
| qps_limit | Integer | QPS 限制（默认 20） |
| milvus_partition | String(100) | Milvus 分区名（物理隔离） |

### User（用户）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| tenant_id | FK→Tenant | 所属机构 |
| username | String(100) | 登录名（全系统唯一） |
| email | String(200) | 邮箱（唯一，可为空） |
| hashed_password | String(200) | bcrypt 哈希密码（不存明文） |
| is_admin | Boolean | 是否管理员 |
| is_active | Boolean | 账号是否启用 |

### Document（文档）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| tenant_id | FK→Tenant | 所属机构（隔离） |
| filename | String(500) | 原始文件名 |
| file_type | String(20) | pdf / docx / xlsx / image |
| file_size | Integer | 字节数 |
| md5_hash | String(32) | 内容哈希，用于跨文档去重 |
| status | Enum | pending / processing / completed / failed |
| chunk_count | Integer | 切片数量 |
| error_msg | Text | 处理失败原因 |
| parse_meta | JSON | 解析统计信息 |

### DocumentChunk（文档切片）

| 字段 | 类型 | 说明 |
|------|------|------|
| document_id | FK→Document | 所属文档 |
| chunk_index | Integer | 切片序号（从0开始） |
| content | Text | 切片文本内容 |
| section_path | String(500) | 所属章节路径（如"第一条 总则"） |
| page_num | Integer | 原文档页码 |
| chunk_type | String(50) | text / table / image_ocr |
| milvus_id | String(100) | 对应 Milvus 中的向量 ID |

### QueryLog（查询日志）

记录每次用户问答的完整链路数据，用于性能分析和用户反馈收集。

关键字段：`retrieval_ms`、`rerank_ms`、`llm_ms`、`total_ms`、`feedback_score`（1-5星）、`top_k_hit`（Top-5 是否命中）。

### BillRecord / BillVersion（票据生命周期，P1）

- **BillRecord**：以票据号码（`ticket_number`）为业务唯一键，同一张票据无论上传多少次都对应同一条记录
- **BillVersion**：每次上传追加一条版本记录，`new_endorsers` 字段记录新增背书人，支持任意版本对比

### IntentLog / SearchMissLog（P2/P3）

- **IntentLog**：记录每次意图识别结果（关键词命中 vs LLM 推断），用于调优词表
- **SearchMissLog**：记录检索质量不足的问题，按 `intent_id` 聚合，指导知识库扩充优先级

---

## 7. API 接口清单

所有接口详情见 Swagger 文档：`http://localhost:8000/api/docs`

### 认证

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | /api/v1/auth/login | 登录，返回 JWT Token | 无 |
| POST | /api/v1/auth/register | 注册新用户 | 无 |

**登录示例：**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "Admin@123456"}'
```

后续所有请求携带 `Authorization: Bearer <token>` 请求头。

### 文档管理

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | /api/v1/documents/upload | 上传文件（支持批量） | 需要 |
| GET | /api/v1/documents | 文档列表（分页 + 状态过滤） | 需要 |
| DELETE | /api/v1/documents/{doc_id} | 删除文档（同步删除 Milvus 向量） | 需要 |

### 智能问答

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | /api/v1/query | 同步问答 | 需要 |
| POST | /api/v1/query/stream | 流式问答（SSE，逐字推送） | 需要 |
| POST | /api/v1/query/{query_id}/feedback | 提交 1-5 星反馈 | 需要 |

**问答请求体：**
```json
{
  "query": "银行承兑汇票的贴现利率如何计算？",
  "top_k": 5,
  "stream": false
}
```

### 票据管理（P1）

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | /api/v1/bill-recognition | 票据识别（仅识别，不入库） | 需要 |
| POST | /api/v1/bills | 上传票据（识别 + 入库 + 版本管理） | 需要 |
| GET | /api/v1/bills/{ticket_number} | 查询票据及流转历史 | 需要 |

### 租户管理（管理员专用）

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | /api/v1/tenants | 租户列表 | 管理员 |
| POST | /api/v1/tenants | 创建新租户 | 管理员 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查（k8s 就绪探针） |
| GET | /metrics | Prometheus 指标（纯文本） |

---

## 8. 核心服务层详解

### 8.1 文档入库流水线（ingestion.py）

**三步流水线，支持断点续传：**

```
Step1: 文档解析（最耗时）
  ├─ 检查 ./data/processed/{md5}.checkpoint.json 是否存在
  ├─ 有则直接加载（秒级恢复，跳过重复解析）
  └─ 无则执行解析 → 原子写入 checkpoint

Step2: 语义分块（纯内存，极快）
  └─ 如无有效 chunk，删除 checkpoint，返回 empty

Step3: 向量化 + Milvus 写入（chunk 级幂等）
  ├─ 跨文档 MD5 去重（同租户相同内容只入库一次）
  ├─ 查询已写入 chunk（chunk_id = {document_id}_{chunk_index}）
  ├─ 只写入缺失部分（真正的断点续传）
  └─ 全部完成后删除 checkpoint
```

**性能目标：P95 < 45 秒（标准 A4 票据 PDF，不含 OCR）**

### 8.2 文档解析（pdf_parser.py）

**四层兜底策略（按优先级）：**

```
PDF 表格提取
  └─ Layer 1：Camelot lattice（有框线，置信度≥0.8）
  └─ Layer 2：Camelot stream（无框线，置信度≥0.6）
  └─ Layer 3：pdfplumber（Camelot 完全失败时备用）
  └─ Layer 4：Qwen-VL 视觉大模型（前三层置信度均不足时）

扫描件 PDF
  └─ PaddleOCR（中文+英文，2× 分辨率，自动纠偏）
```

### 8.3 混合检索（vector_store.py）

**三路检索 + RRF 融合 + Reranker 精排：**

```
Query
  ├─ BGE-M3 稠密向量检索  → Top-20（权重 0.6）
  ├─ BGE-M3 稀疏权重检索  → Top-20（权重 0.4）
  └─ BM25 关键词检索      → Top-20（权重 0.32）
          │
          ▼
      RRF 融合排序
      score(d) = Σ 1/(k + rank_r(d))，k=60
          │
          ▼
      BGE-Reranker 精排 → Top-5
          │
          ▼
      LLM 生成答案
```

**Milvus Collection Schema：**

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(200) | 主键（{doc_id}_{chunk_index}） |
| content | VARCHAR(1000) | 切片文本 |
| dense_vector | FLOAT_VECTOR(1024) | BGE-M3 稠密向量 |
| sparse_vector | SPARSE_FLOAT_VECTOR | BGE-M3 稀疏权重 |
| tenant_id | VARCHAR(100) | 租户 ID（物理隔离） |
| document_id | VARCHAR(200) | 文档 ID |
| chunk_index | INT | 切片序号 |
| page_num | INT | 页码 |

### 8.4 意图路由（intent_router.py，P2）

**两阶段识别，兼顾速度和准确性：**

| 阶段 | 方式 | 耗时 | 触发条件 |
|------|------|------|----------|
| 1 | 关键词快通道 | <1ms | 命中词表 |
| 2 | LLM 语义识别 | 200-500ms | 阶段1未命中 |

**8 个意图场景：**

| 意图 ID | 场景 | 对应服务 |
|---------|------|----------|
| INTENT_PRICING | 贴现利率查询 | scene_handlers.py |
| INTENT_ENDORSER_CHECK | 背书合规性审查 | scene_handlers.py |
| INTENT_EXPIRY | 到期处理与追索权 | scene_handlers.py |
| INTENT_AML | 反洗钱与大额交易 | scene_handlers.py |
| INTENT_ELEMENT_CHECK | 票据要素合规审查 | scene_handlers.py |
| INTENT_HISTORY | 票据流转溯源 | scene_handlers.py |
| INTENT_BATCH_RISK | 多票据风险汇总 | scene_handlers.py |
| INTENT_PLEDGE | 质押融资方案 | scene_handlers.py |

### 8.5 检索质量评估（retrieval_quality.py，P3）

| 等级 | 判断条件 | 处理方式 |
|------|----------|----------|
| SUFFICIENT | top1 分数≥0.75 且命中数≥2 | 直接生成答案 |
| PARTIAL | top1 分数≥0.55 且命中数≥1 | 生成答案 + 免责提示 |
| INSUFFICIENT | 其他 | 转人工，返回工单号和客服热线 |

### 8.6 限流器（rate_limiter.py）

使用 **Redis Sorted Set 滑动窗口算法**：

```python
# key = "rl:{tenant_id}"
# 每次请求：
# 1. 删除窗口外旧记录（score < now - window_size）
# 2. 统计窗口内请求数
# 3. 超限则返回 429，否则插入当前时间戳
```

两层保护：**QPS 限制**（默认 20 req/s）+ **文档配额**（默认 10000 docs/tenant）

---

## 9. 监控与运维

### 9.1 Prometheus 指标

应用在 `http://localhost:8000/metrics` 暴露以下关键指标：

| 指标名 | 类型 | 标签 | 说明 |
|--------|------|------|------|
| `bill_rag_query_total` | Counter | tenant_id, status | 查询总数 |
| `bill_rag_retrieval_latency_ms` | Histogram | — | 检索耗时（P95 目标 <200ms） |
| `bill_rag_rerank_latency_ms` | Histogram | — | 精排耗时 |
| `bill_rag_embedding_latency_ms` | Histogram | — | Embedding 推理耗时 |
| `bill_rag_table_extracted_total` | Counter | tenant_id | 表格提取成功数 |
| `bill_rag_top5_hit_total` | Counter | intent_id | Top-5 命中数 |

**访问 Prometheus UI：**`http://localhost:9090`（需通过 Docker Compose 启动）

**访问 Grafana：**`http://localhost:3000`，账号 `admin / admin123`

### 9.2 日志

日志文件位于 `./logs/`：

| 文件 | 级别 | 格式 | 说明 |
|------|------|------|------|
| stderr | INFO+ | 彩色文本 | 开发调试、容器日志采集 |
| logs/app.log | INFO+ | JSON | 结构化日志，50MB 轮转，30天保留 |
| logs/error.log | ERROR+ | JSON | 错误专用，供告警系统消费 |

**查看实时日志：**
```bash
# 本地
tail -f logs/app.log | python -m json.tool   # 格式化 JSON

# Docker
docker logs -f bill_rag_app
```

**日志字段说明（JSON 格式）：**
- `time`：时间戳
- `level`：日志级别
- `message`：日志内容
- `request_id`：请求唯一 ID（全链路追踪）
- `name`、`function`、`line`：代码位置

### 9.3 健康检查

```bash
curl http://localhost:8000/health
# 返回：{"status": "ok", "timestamp": "..."}
```

可作为 Docker / k8s 的 readiness probe。

---

## 10. 测试

### 运行单元测试

```bash
# 在项目根目录
pytest tests/test_core.py -v
```

**覆盖范围：**
- `ChineseSemanticChunker`：分块逻辑与章节识别
- Camelot 置信度评估
- RRF 融合算法正确性
- BM25 索引构建与搜索
- 限流器（QPS / 配额）
- MD5 去重
- Token 计数

### 生成测试 PDF

```bash
python scripts/generate_test_pdf.py
```

生成包含有框线表格、无框线表格、图表、模拟扫描件的复杂测试 PDF，用于验证多层解析兜底逻辑。

### 手动 API 测试

推荐使用 `http://localhost:8000/api/docs` 的 Swagger UI 直接在浏览器中测试所有接口，无需 Postman。

---

## 11. 常见问题排查

### Q1：启动报 `ModuleNotFoundError: No module named 'config'`

**原因：** IDE 直接运行 `app/main.py` 时，`app/` 目录被加入 `sys.path`，但项目根目录未加入。

**解决：** 已在 `app/main.py` 顶部自动处理，确保文件头部有：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Q2：启动报 `[Errno 48] Address already in use`

**原因：** 8000 端口被之前的进程占用。

**解决：**
```bash
# 查找占用端口的进程
lsof -i :8000 -P -n

# 杀掉进程（替换 <PID>）
kill <PID>
```

### Q3：Milvus 连接失败，向量检索不可用

**现象：** 启动日志显示 `⚠️ Milvus 未连接，向量检索不可用`

**排查：**
```bash
# 检查 Milvus 是否在运行
curl http://localhost:9091/healthz

# 检查 .env 中的配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

系统在 Milvus 不可用时仍可启动，但 `/api/v1/query` 接口会返回错误。

### Q4：文档处理状态一直是 `processing`

**原因：** 处理过程中进程崩溃，状态未更新。

**解决：**
```bash
# 连接数据库，手动重置状态
psql -U postgres -d bill_rag -c \
  "UPDATE documents SET status='failed', error_msg='进程崩溃，手动重置' WHERE status='processing';"
```

然后重新上传文档，断点续传机制会自动跳过已处理的部分。

### Q5：BGE-M3 模型加载慢或失败

**原因：** 模型路径配置错误，或路径指向网络地址而非本地快照。

**解决：** 配置 `BGE_M3_MODEL_PATH` 为模型的本地绝对路径：
```bash
# 查找本地 HuggingFace 缓存
find ~/.cache/huggingface/hub -name "*.json" -path "*/bge-m3/*" | head -1
```

将 `snapshots/xxxxx` 目录的绝对路径填入 `.env`。

### Q6：Prometheus 抓取目标显示 DOWN

**原因：** Docker Compose 中 Prometheus 配置的目标是 `app:8000`（Docker 内部网络），本地直接运行时无法通过此地址访问。

**解决：**
- 若使用 Docker Compose 全栈部署：确保 `app` 容器正常运行
- 若仅本地开发：可直接访问 `http://localhost:8000/metrics` 查看原始指标，无需 Prometheus

### Q7：`WARNING: pkg_resources is deprecated` 警告

这是 jieba 库内部使用了旧版 `pkg_resources` API 的警告，**不影响功能**，等 jieba 官方更新修复。

### Q8：Pydantic `model_` 命名空间警告

```
Field "model_used" has conflict with protected namespace "model_"
```

在 `schemas.py` 中对应的 Schema 类添加以下配置即可消除：
```python
class Config:
    protected_namespaces = ()
```

---

## 附录：环境变量完整示例（.env.example）

```bash
# === 应用基础 ===
APP_NAME=票据业务智能顾问 RAG 系统
APP_VERSION=1.0.0
DEBUG=false
SECRET_KEY=your-super-secret-key-at-least-32-chars-long

# === 数据库 ===
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/bill_rag
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_URL=redis://localhost:6379/0

# === LLM（通义千问示例）===
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxx...
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
VISION_MODEL=qwen-vl-max

# === LLM（Anthropic Claude 示例）===
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-xxxx...
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# === 向量模型（填写本地绝对路径）===
BGE_M3_MODEL_PATH=/path/to/bge-m3/snapshots/xxxxx
BGE_RERANKER_MODEL_PATH=/path/to/bge-reranker-v2-m3/snapshots/xxxxx

# === 限流 ===
TENANT_QPS_LIMIT=20
TENANT_DOC_QUOTA=10000

# === 日志 ===
LOG_LEVEL=INFO

# === 监控 ===
PROMETHEUS_ENABLED=true
```
