# Agent 系统详细介绍

> 文档版本：v3.0.0 | 更新时间：2026-05-27  
> 本文档覆盖全部 12 个 Agent 的调用路径、功能实现、技术原理，供代码阅读和系统熟悉使用。

---

## 目录

1. [整体三层架构](#1-整体三层架构)
2. [LangGraph 执行引擎](#2-langgraph-执行引擎)
3. [OrchestratorAgent — 编排调度](#3-orchestratoragent--编排调度)
4. [DocumentParserAgent — 文档解析（M3）](#4-documentparseragent--文档解析m3)
5. [ElementExtractionAgent — 要素抽取（M3）](#5-elementextractionagent--要素抽取m3)
6. [ComplianceRetrievalAgent — 合规检索（M4）](#6-complianceretrievalagent--合规检索m4)
7. [EndorsementChainAgent — 背书链验证（M4）](#7-endorsementchainagent--背书链验证m4)
8. [ContractReviewAgent — 合同审核（M5）](#8-contractreviewagent--合同审核m5)
9. [RiskAssessmentAgent — 风险评分（M5）](#9-riskassessmentagent--风险评分m5)
10. [ReportGenerationAgent — 报告生成（M6）](#10-reportgenerationagent--报告生成m6)
11. [FlowTrackingAgent — 流转追踪（M7）](#11-flowtrackingagent--流转追踪m7)
12. [FraudDetectionAgent — 欺诈检测（M8）](#12-frauddetectionagent--欺诈检测m8)
13. [BatchSchedulingAgent — 批量调度（M9）](#13-batchschedulingagent--批量调度m9)
14. [BillIssuanceAgent — 出票预检（M10）](#14-billissuanceagent--出票预检m10)
15. [BaseAgent 框架层](#15-baseagent-框架层)
16. [shared_data 数据流全景](#16-shared_data-数据流全景)
17. [8 种业务图执行拓扑速查](#17-8-种业务图执行拓扑速查)

---

## 1. 整体三层架构

```
┌──────────────────────────────────────────────────────────────┐
│  层一：MCP 工具层（app/mcp/server.py + tools/）               │
│  FastMCP 实例注册 11 个 @mcp.tool()                           │
│  ─────────────────────────────────────────────────────────   │
│  外部：POST /mcp  ← Claude API / Claude Desktop 调用          │
│  内部：LangGraph 节点 import 直调（无 HTTP 开销）              │
├──────────────────────────────────────────────────────────────┤
│  层二：LangGraph 执行引擎（app/graph/）                        │
│  BillAuditState（TypedDict）+ AsyncPostgresSaver（外置状态）   │
│  build_audit_graph(task_type) → 按业务类型编译 StateGraph     │
│  graph.astream() 流式执行，每节点完成写 DB 进度               │
├──────────────────────────────────────────────────────────────┤
│  层三：专项 Agent 业务逻辑层（app/agents/）                    │
│  11 个 Agent.run()，完全不感知 LangGraph / MCP               │
│  只依赖 AgentContext / AsyncSession，业务逻辑独立封装          │
└──────────────────────────────────────────────────────────────┘
```

### 调用链（以 POST /api/v1/audit/tasks 为入口）

```
HTTP POST /api/v1/audit/tasks
    │
    ▼  app/api/audit_router.py → create_audit_task()
    │  创建 AuditTask 记录，status=PENDING，后台启动异步任务
    │
    ▼  OrchestratorAgent.execute(ctx, db)
    │     └─ OrchestratorAgent.run()
    │           ├─ build_audit_graph(task_type, checkpointer=get_checkpointer())
    │           ├─ graph.astream(initial_state, config={thread_id: task_id})
    │           │    每节点完成 → _update_progress_from_event()
    │           └─ graph.aget_state(config) → _sync_state_to_context(ctx)
    │
    ▼  LangGraph 节点函数（app/graph/nodes.py）
    │  每个节点 → from app.mcp.tools.xxx import tool_fn → tool_fn()
    │
    ▼  MCP 工具函数（app/mcp/tools/xxx_tools.py）
    │  构建 AgentContext → Agent.execute(ctx, db)
    │
    ▼  专项 Agent（app/agents/xxx_agent.py）
       Agent.run() → 业务逻辑 → 写 DB → 写 shared_data
```

---

## 2. LangGraph 执行引擎

### 关键文件

| 文件 | 职责 |
|------|------|
| [app/graph/state.py](app/graph/state.py) | `BillAuditState` TypedDict 定义，所有节点读写的共享状态 |
| [app/graph/builder.py](app/graph/builder.py) | `build_audit_graph(task_type)` — 按业务类型构建 StateGraph |
| [app/graph/nodes.py](app/graph/nodes.py) | 11 个节点函数，每个节点调用对应 MCP 工具 |
| [app/graph/edges.py](app/graph/edges.py) | 条件边函数，控制关键节点失败后的路由 |
| [app/graph/\_\_init\_\_.py](app/graph/__init__.py) | `setup_checkpointer()` / `get_checkpointer()` — PostgresSaver 单例 |

### BillAuditState 核心字段

```python
class BillAuditState(TypedDict, total=False):
    # 任务标识（只读）
    audit_task_id: str      # 与 audit_tasks.id 对应
    tenant_id: str
    task_type: str
    trace_id: str

    # 文件输入
    file_path: Optional[str]        # 票据文件路径
    contract_text: Optional[str]    # 合同文本

    # 各 Agent 输出（按执行顺序填充）
    parsed_doc: Optional[dict]          # DocumentParserAgent 输出
    bill_element: Optional[dict]        # ElementExtractionAgent 输出
    compliance_summary: Optional[dict]  # ComplianceRetrievalAgent 输出
    endorsement_result: Optional[dict]  # EndorsementChainAgent 输出
    fraud_result: Optional[dict]        # FraudDetectionAgent 输出
    contract_result: Optional[dict]     # ContractReviewAgent 输出
    risk_result: Optional[dict]         # RiskAssessmentAgent 输出
    report_result: Optional[dict]       # ReportGenerationAgent 输出
    issuance_result: Optional[dict]     # BillIssuanceAgent 输出

    # 执行控制（条件边路由依据）
    failed_nodes: list      # 失败节点名称列表
    skipped_nodes: list     # 被跳过的节点列表
    errors: dict            # {node_name: error_msg}
```

### 状态外置原理（PostgresSaver）

LangGraph 默认使用内存 Checkpointer，容器重启即丢失状态。本系统使用 `AsyncPostgresSaver` 将每步状态序列化为 JSON 写入 PostgreSQL 的检查点表，`thread_id = audit_task_id`。

- **写入时机**：每个 LangGraph 节点函数返回后，框架自动将增量合并到完整状态并调用 Saver 写库
- **恢复时机**：`graph.aget_state(config)` 读取最新检查点，任意新容器副本可无缝接管
- **关键优势**：K8s 滚动更新时，全流程 120s 的审核任务不会因容器重启而丢失

### 条件边路由逻辑

```python
# app/graph/edges.py 示意
def after_parse(state: BillAuditState) -> str:
    """parse 节点完成后的路由"""
    if "document_parser" in state.get("failed_nodes", []):
        return "end"   # 关键节点失败 → 终止整个图
    return "extract"   # 成功 → 继续执行要素抽取

def after_parallel_cef_to_contract(state: BillAuditState) -> str:
    """合规+背书+欺诈并行层完成后的路由"""
    failed = set(state.get("failed_nodes", []))
    # compliance 或 endorsement 失败 → 终止（欺诈失败可容忍）
    if "compliance_retrieval" in failed or "endorsement_chain" in failed:
        return "end"
    return "contract"
```

---

## 3. OrchestratorAgent — 编排调度

### 文件位置

[app/agents/orchestrator_agent.py](app/agents/orchestrator_agent.py)

### 职责

OrchestratorAgent 是整个多智能体系统的**指挥中枢**，本身不执行任何业务逻辑，专注于：
1. 根据 `task_type` 调用 `build_audit_graph()` 构建对应的 LangGraph 图
2. 创建 / 更新 `audit_tasks` 表记录（状态、进度百分比、当前 Agent）
3. 通过 `graph.astream()` 流式执行，每节点完成更新 DB 进度
4. 执行结束后将最终状态同步回 `ctx.shared_data`（向后兼容 API 层）

### 调用路径

```
POST /api/v1/audit/tasks
    ↓
audit_router.py → create_audit_task()
    ↓
asyncio.create_task(
    _run_audit_background(task_id, tenant_id, task_type, ...)
)
    ↓
OrchestratorAgent.execute(ctx, db)   ← BaseAgent.execute() 统一计时/日志
    ↓
OrchestratorAgent.run(ctx, db)
    ├── _create_audit_task_record(ctx, db, task_type)
    │       写入 audit_tasks 表，status=RUNNING
    ├── build_audit_graph(task_type, checkpointer=get_checkpointer())
    │       按业务类型构建并编译 StateGraph
    ├── initial_state = make_initial_state(...)
    │       构造 BillAuditState 初始值
    ├── config = {"configurable": {"thread_id": ctx.audit_task_id}}
    │
    ├── async for event in graph.astream(initial_state, config):
    │       # 每个节点完成时推送一个 event
    │       _update_progress_from_event(event, ctx.audit_task_id, db)
    │           # 计算进度百分比，更新 audit_tasks.progress_pct
    │
    ├── final_state = await graph.aget_state(config)
    │       # 从 PostgreSQL 读取最终状态
    └── _sync_state_to_context(final_state, ctx)
            # 将 LangGraph 状态字段写回 ctx.shared_data（供 API 层读取）
```

### 关键技术

**LangGraph `astream()` 事件格式**：
每次节点完成，LangGraph 推送一个 `{node_name: delta_dict}` 格式的事件。`_update_progress_from_event()` 从已知的节点顺序推算完成百分比，写入 `audit_tasks.progress_pct`，前端轮询 `GET /api/v1/audit/tasks/{id}` 可实时展示进度。

**`thread_id` 绑定**：
`config = {"configurable": {"thread_id": ctx.audit_task_id}}` 是 LangGraph 状态持久化的核心参数。`thread_id` 与 `audit_task_id` 一一绑定，同一 task_id 的多次 `astream()` 调用会从上次中断点续跑（断点续跑）。

### 数据库操作

| 表 | 操作 | 时机 |
|----|------|------|
| `audit_tasks` | INSERT (status=RUNNING) | 任务启动时 |
| `audit_tasks` | UPDATE (progress_pct, current_agent) | 每个节点完成时 |
| `audit_tasks` | UPDATE (status=COMPLETED/FAILED) | 整个图执行完成 |

---

## 4. DocumentParserAgent — 文档解析（M3）

### 文件位置

[app/agents/document_parser_agent.py](app/agents/document_parser_agent.py)

### 职责

将票据原始文件（PDF / 图片）解析为结构化文本，供后续 ElementExtractionAgent 进行要素识别。

### 调用路径

```
LangGraph 节点 "parse"
    ↓
app/graph/nodes.py → node_document_parser(state)
    ↓
from app.mcp.tools.document_tools import parse_bill_document
parse_bill_document(file_path, audit_task_id, tenant_id)
    ↓
构建 AgentContext(audit_task_id, tenant_id, document_id)
DocumentParserAgent.execute(ctx, db)   ← BaseAgent 统一计时
    ↓
DocumentParserAgent.run(ctx, db, file_path=...)
    ├── _resolve_file_path(ctx, db, file_path)
    │       优先使用传入 file_path
    │       否则 SELECT file_path FROM documents WHERE id=ctx.document_id
    ├── os.path.exists(resolved_path) → 文件存在校验
    ├── _parse_file(resolved_path)
    │       asyncio.to_thread(_sync_parse)   ← 线程池执行，不阻塞事件循环
    │           BillPDFParser().parse(file_path)
    │           ↓（五级降级策略）
    │           Camelot lattice → Camelot stream → pdfplumber → PaddleOCR → Qwen-VL
    ├── _calculate_confidence(parsed_doc)
    │       parse_stats["ocr_avg_confidence"] → 元素均值 → 兜底 1.0
    ├── confidence < 0.85 → logger.WARNING（不阻断流程）
    └── ctx.shared_data["parsed_doc"] = doc_dict
            写入 shared_data，后续 Agent 从此读取
```

### 五级降级解析策略详解

| 级别 | 工具 | 适用场景 | 置信度来源 |
|------|------|---------|-----------|
| 1 | Camelot lattice | 有明确边框线的规整表格 | 解析成功率 |
| 2 | Camelot stream | 无边框或虚线表格 | 文本密度估算 |
| 3 | pdfplumber | 复杂布局 PDF | 文本提取完整度 |
| 4 | PaddleOCR | 扫描件 / 图片型 PDF | OCR 逐字符置信度均值 |
| 5 | Qwen-VL（兜底） | 表格跨行合并、斜线表头等 | 模型自报置信度 |

**降级触发条件**：每级解析后计算空单元格比例和斜线表头检测。空单元格比例过高（>40%）或检测到斜线表头，自动降至下一级。

**`asyncio.to_thread` 的必要性**：Camelot / PaddleOCR 均是同步阻塞库，直接在 FastAPI 异步事件循环中调用会阻塞所有请求处理。`asyncio.to_thread` 将调用放入默认线程池（`ThreadPoolExecutor`），事件循环继续处理其他请求，解析完成后协程被唤醒。

### 输出格式（写入 shared_data["parsed_doc"]）

```python
{
    "elements": [
        {
            "text": "出票日期：2024年01月15日",
            "type": "table_cell",        # text / table / image_ocr
            "metadata": {"page": 1, "confidence": 0.97}
        },
        ...
    ],
    "parse_stats": {
        "page_count": 2,
        "ocr_page_count": 1,
        "table_count": 3,
        "ocr_avg_confidence": 0.94
    },
    "confidence": 0.94,       # 附加字段（非原始 ParsedDocument 字段）
    "file_path": "/data/uploads/xxx.pdf"
}
```

### 数据库操作

本 Agent **不写入数据库**（文档解析结果仅通过 shared_data 传递），所有数据库操作预留为扩展接口。

---

## 5. ElementExtractionAgent — 要素抽取（M3）

### 文件位置

[app/agents/element_extraction_agent.py](app/agents/element_extraction_agent.py)

### 职责

将解析后的文档文本转换为 18 个结构化票据要素字段（15 基础 + 3 扩展），写入 `bill_elements` 表，并在 `shared_data` 中为后续所有 Agent 提供要素访问入口。

### 调用路径

```
LangGraph 节点 "extract"
    ↓
app/graph/nodes.py → node_element_extraction(state)
    ↓
from app.mcp.tools.extraction_tools import extract_bill_elements
extract_bill_elements(file_path, audit_task_id, tenant_id, prefilled_element)
    ↓
ElementExtractionAgent.execute(ctx, db)
    ↓
ElementExtractionAgent.run(ctx, db, file_bytes, filename)
    │
    ├── 快速路径：ctx.shared_data.get("bill_element") 非空
    │       → _handle_prefilled_element(ctx, db)
    │           直接将预填充要素写入 bill_elements 表
    │           extraction_method = "prefilled"（跳过 Qwen-VL，节省 3~8s）
    │
    └── 正常路径：
        ├── _resolve_file_bytes(ctx, db, file_bytes, filename)
        │       优先使用传入 bytes
        │       否则从 shared_data["parsed_doc"]["file_path"] 读取文件
        │       否则 SELECT file_path FROM documents WHERE id=ctx.document_id
        ├── asyncio.to_thread(_sync_recognize, file_bytes, filename)
        │       BillRecognitionService().recognize_file(file_bytes, filename)
        │           ↓（视觉大模型 Qwen-VL）
        │           逐页渲染 PDF → Qwen-VL 提取结构化要素 → BillElementSchema
        │           金额大小写一致性校验
        ├── _calc_confidence(recognition_result)
        │       统计非空字段数 / 总字段数 → 综合置信度（0.0~1.0）
        ├── _calc_maturity_days(bill.due_date)
        │       datetime.strptime(due_date, "%Y-%m-%d") - datetime.now()
        ├── _extract_extended_fields(recognition_result.raw_texts)
        │       关键词匹配提取：
        │       trade_purpose（"贸易背景："后的文本）
        │       acceptance_clause（"承兑条款："后的文本）
        │       special_remarks（"其他记载事项："后的文本）
        ├── db.add(BillElement(id, audit_task_id, 18字段, confidence_score, ...))
        └── ctx.shared_data["bill_element"] = {18字段 dict}
```

### 18 个要素字段说明

| 分类 | 字段 | 来源 |
|------|------|------|
| 基础 15 个 | ticket_number / ticket_type / issue_date / due_date / amount_numeric / amount_text / currency / drawer / drawer_account / drawer_bank / acceptor / payee / drawee_bank / endorsers / maturity_days | Qwen-VL 直接识别 |
| 扩展 3 个 | trade_purpose / acceptance_clause / special_remarks | OCR 文本关键词匹配 |

### 预填充快速路径

聊天咨询场景（`POST /api/v1/query/consult`）会在调用 Agent 链之前先通过 BillRecognitionService 识别要素，并将结果注入 `ctx.shared_data["bill_element"]`。ElementExtractionAgent 检测到预填充后直接走快速路径，**跳过视觉模型调用**，节省 3~8s 和 1000~2000 token。

### 数据库操作

| 表 | 操作 | 字段 |
|----|------|------|
| `bill_elements` | INSERT | id / audit_task_id / 18字段 / confidence_score / field_confidences / raw_ocr_result / extraction_method |

---

## 6. ComplianceRetrievalAgent — 合规检索（M4）

### 文件位置

[app/agents/compliance_retrieval_agent.py](app/agents/compliance_retrieval_agent.py)

### 职责

为票据 18 个要素字段**并行发起 18 个 RAG 查询**，每个字段对应一条法规条文引用，判断字段值是否合规，是系统中 Token 消耗最多的单次操作（约 36,000~54,000 tokens）。

### 调用路径

```
LangGraph 节点 "parallel"（asyncio.gather 并行层之一）
    ↓
app/graph/nodes.py → node_parallel_compliance_endorse_fraud(state)
    asyncio.gather(
        _run_compliance(state),   # 调用 check_compliance MCP 工具
        _run_endorsement(state),  # 调用 analyze_endorsement_chain MCP 工具
        _run_fraud(state),        # 调用 detect_fraud MCP 工具
    )
    ↓
from app.mcp.tools.compliance_tools import check_compliance
check_compliance(bill_element_dict, audit_task_id, tenant_id)
    ↓
ComplianceRetrievalAgent.execute(ctx, db)
    ↓
ComplianceRetrievalAgent.run(ctx, db)
    │
    ├── bill_element = ctx.shared_data["bill_element"]
    │
    ├── bill_fp = compute_bill_fingerprint(bill_element)
    │       SHA-256(18字段 JSON)[:24] → 96 bit 指纹
    │
    ├── cached = await bill_cache.get_compliance(tenant_id, bill_fp)
    │       Redis Key: compliance:{tenant_id}:{bill_fp}  TTL=6h
    │       命中 → 直接写 DB，跳过全部 RAG 查询
    │
    ├── tasks = [
    │       _check_field(field_name, field_value, query_tpl, regulation)
    │       for field_name, (query_tpl, regulation) in FIELD_QUERY_TEMPLATES.items()
    │   ]
    │   # 18 个并发任务，每个任务：
    │   #   构建查询句 = "xxx要求是什么？（当前值：{field_value}）"
    │   #   调用 RAGService.query(text, tenant_id)
    │   #   根据 RAG 返回分数（≥0.70 为合规）判断该字段合规性
    │
    ├── check_results = await asyncio.gather(*tasks)
    │       # 等待全部 18 个 RAG 查询完成
    │
    ├── for check in check_results:
    │       db.add(ComplianceCheck(
    │           element_field, element_value, regulation_ref,
    │           rag_query, rag_answer, rag_score,
    │           is_compliant, violation_level, violation_desc
    │       ))
    │
    ├── ctx.shared_data["compliance_summary"] = {
    │       total_fields, violation_count, severe_count,
    │       compliance_rate, is_overall_compliant
    │   }
    │
    └── await bill_cache.set_compliance(tenant_id, bill_fp, cache_payload)
            # 写入 Redis 缓存，TTL=6h
```

### 合规判断规则

```python
# FIELD_QUERY_TEMPLATES 定义了每个字段的查询模板和对应法规
FIELD_QUERY_TEMPLATES = {
    "ticket_number": ("票据号码格式和唯一性要求是什么？", "票据法第22条"),
    "due_date":      ("票据到期日的规定，最长期限是多少？", "商业汇票承兑贴现办法第8条"),
    # ... 共 18 个字段
}

# 合规判断逻辑
def _determine_compliance(field_name, field_value, rag_score):
    if not field_value and field_name in REQUIRED_FIELDS:
        return is_compliant=False, violation_level=SEVERE
    if field_name == "trade_purpose" and not field_value:
        return is_compliant=False, violation_level=WARNING
    if rag_score >= 0.70 and field_value:
        return is_compliant=True
    return is_compliant=False, violation_level=WARNING
```

### 缓存机制

合规检索是系统中**最贵的单次操作**（18 次 LLM 调用，约 30~40s，36000+ tokens）。五层缓存中第 2 层专门用于此场景：
- **Key**：`compliance:{tenant_id}:{bill_fingerprint}`
- **TTL**：6h（法规一个工作日内通常不修改）
- **命中收益**：节省 36000 tokens 和 30~40s

### 数据库操作

| 表 | 操作 | 说明 |
|----|------|------|
| `compliance_checks` | INSERT × 18 | 每个字段一条记录，含 RAG 查询/答案/得分 |

---

## 7. EndorsementChainAgent — 背书链验证（M4）

### 文件位置

[app/agents/endorsement_chain_agent.py](app/agents/endorsement_chain_agent.py)

### 职责

从票据要素中提取背书人列表，重建背书**有向图**，运行 9 类违规检测算法，输出背书链完整性报告。与 ComplianceRetrievalAgent 和 FraudDetectionAgent 并行执行。

### 调用路径

```
LangGraph 并行节点 "parallel"（asyncio.gather 第 2 个任务）
    ↓
from app.mcp.tools.endorsement_tools import analyze_endorsement_chain
    ↓
EndorsementChainAgent.execute(ctx, db)
    ↓
EndorsementChainAgent.run(ctx, db)
    │
    ├── bill_element = ctx.shared_data["bill_element"]
    │   drawer = bill_element["drawer"]       # 出票人 → 图的起点
    │   payee  = bill_element["payee"]        # 初始收款人
    │   endorsers = bill_element["endorsers"] # 背书人顺序列表 ["A公司", "B银行", ...]
    │
    ├── _build_graph(drawer, payee, endorsers)
    │       构建有向图：
    │       nodes: [{id: "n0", name: "出票人"}, {id: "n1", name: "收款人"}, ...]
    │       edges: [{from: "n0", to: "n1"}, {from: "n1", to: "n2"}, ...]
    │       full_chain = [drawer, payee] + endorsers → 逐对建边
    │
    ├── _detect_violations(drawer, payee, endorsers, graph_edges)
    │       运行 9 类检测（见下表）
    │       返回 {violation_code: description} dict
    │
    ├── is_continuous = "EN01" not in violations
    │   has_cycle     = "EN03" in violations
    │   blank_count   = endorsers.count(None) + endorsers.count("")
    │
    ├── db.add(EndorsementChain(
    │       chain_graph={nodes, edges},
    │       endorser_count, is_continuous,
    │       violation_codes, violation_details,
    │       max_chain_depth, has_cycle, blank_endorsement_count
    │   ))
    │
    └── ctx.shared_data["endorsement_result"] = {
            chain_id, endorser_count, is_continuous,
            has_cycle, violation_count, violation_codes
        }
```

### 9 类违规检测算法

| 代码 | 违规类型 | 检测算法 |
|------|---------|---------|
| EN01 | 背书链断裂 | 逐对检查：`payee[i]` 是否等于 `endorsers[i]`（前手收款人 = 背书人） |
| EN02 | 重复背书 | `len(set(endorsers)) < len(endorsers)` → 集合大小比较 |
| EN03 | 背书闭环 | DFS 三色着色（WHITE/GRAY/BLACK），GRAY 节点的后继节点为 GRAY → 有环 |
| EN04 | 空白背书 | `None` 或 `""` 出现在 endorsers 列表中 |
| EN05 | 背书日期异常 | 需要背书日期列表（目前占位实现，生产需传入 endorsement_dates） |
| EN06 | 超出最大层数 | `len(endorsers) > MAX_ENDORSEMENT_DEPTH(10)` |
| EN07 | 背书人=出票人 | `drawer in endorsers` |
| EN08 | 背书撤销后流转 | 需要外部撤销状态数据（目前 Mock 实现） |
| EN09 | 自我背书 | `endorsers[i] == endorsers[i+1]`（相邻背书人相同） |

### DFS 三色着色环路检测原理

```
WHITE = 未访问，GRAY = 正在递归（在当前 DFS 路径上），BLACK = 已完成

从每个 WHITE 节点出发 DFS：
  访问节点 v → 标记为 GRAY
  遍历 v 的所有出边 → 邻居 u
    if u == GRAY → 发现环（EN03）
    if u == WHITE → 递归 DFS(u)
  v 的所有邻居处理完 → 标记为 BLACK

优势：时间复杂度 O(V+E)，能准确区分 "正在处理中" 和 "已完成" 的节点。
```

### 数据库操作

| 表 | 操作 | 关键字段 |
|----|------|---------|
| `endorsement_chains` | INSERT | chain_graph(JSON) / is_continuous / has_cycle / violation_codes(JSON 数组) |

---

## 8. ContractReviewAgent — 合同审核（M5）

### 文件位置

[app/agents/contract_review_agent.py](app/agents/contract_review_agent.py)

### 职责

将票据 6 个核心要素与合同文本进行比对，评估贸易背景真实性。无合同时**降级处理**（给予默认 50 分），不阻断审核流程。

### 调用路径

```
LangGraph 节点 "contract"（并行层之后）
    ↓
from app.mcp.tools.contract_tools import review_contract
    ↓
ContractReviewAgent.execute(ctx, db)
    ↓
ContractReviewAgent.run(ctx, db, contract_text=..., contract_document_id=...)
    │
    ├── bill_element = ctx.shared_data["bill_element"]
    │   bill_amount  = bill_element["amount_numeric"]
    │   bill_drawer  = bill_element["drawer"]
    │   bill_payee   = bill_element["payee"]
    │   bill_date    = bill_element["issue_date"]
    │   bill_purpose = bill_element["trade_purpose"]
    │
    ├── 有 contract_text → _parse_contract_elements(contract_text)
    │       正则提取：金额（r"金额[：:]\s*(\d+\.?\d*)"）
    │       关键词定位：甲方（"甲方：" 后面的文本）
    │       关键词定位：用途（"合同用途：" 后面的文本）
    │
    ├── 无 contract_text → contract_elements = {}（降级：所有匹配项默认 True）
    │
    ├── 6 项比对：
    │   amount_match  = abs(bill_amount - contract_amount) / contract_amount ≤ 0.05
    │   party_match   = bill_drawer in contract_parties OR bill_payee in contract_parties
    │   date_match    = contract_start ≤ bill_date ≤ contract_end
    │   purpose_match = 票据贸易背景关键词与合同用途的词集交集非空
    │   （另外 2 项为内部扩展字段：currency_match / term_match）
    │
    ├── mismatch_details = [] → 收集不匹配的字段详情
    │
    ├── match_score = _calc_match_score(...)
    │       有合同：4 项比对加权 → 0~100
    │       无合同：DEFAULT_SCORE_NO_CONTRACT = 50.0（降级分）
    │
    ├── trade_background_score = _calc_trade_background_score(...)
    │       基于 trade_purpose 关键词丰富度 + purpose_match 布尔值
    │
    ├── db.add(ContractReview(...))
    │
    └── ctx.shared_data["contract_result"] = {
            match_score, trade_background_score,
            amount_match, party_match, date_match, purpose_match,
            mismatch_details, has_contract=bool(contract_text)
        }
```

### 金额匹配容错逻辑

```python
AMOUNT_TOLERANCE_RATE = 0.05  # ±5%
amount_match = (
    abs(bill_amount - contract_amount) / contract_amount
) <= AMOUNT_TOLERANCE_RATE
```

实际业务中票据金额与合同金额允许有小幅差异（如汇率波动、部分付款），5% 容差是行业惯例。

### 无合同降级原理

ContractReviewAgent 在 FULL_AUDIT 中是**非关键节点**（`after_contract_optional` 条件边），失败也只跳到 `risk`，不终止整个图。RiskAssessmentAgent 检测到 `contract_result` 缺失时，使用默认分 50 分（`DEFAULT_SCORE_MISSING_DIMENSION["contract"] = 50.0`），标记为缺失维度。

### 数据库操作

| 表 | 操作 | 关键字段 |
|----|------|---------|
| `contract_reviews` | INSERT | match_score / trade_background_score / amount_match / party_match / mismatch_details(JSON) |

---

## 9. RiskAssessmentAgent — 风险评分（M5）

### 文件位置

[app/agents/risk_assessment_agent.py](app/agents/risk_assessment_agent.py)

### 职责

聚合合规/背书/合同/欺诈四个维度的中间结果，通过**四维加权公式**计算综合评分，按五档阈值判定风险等级，是整个 Agent 链路的核心决策节点。

### 调用路径

```
LangGraph 节点 "risk"
    ↓
from app.mcp.tools.risk_tools import assess_risk
    ↓
RiskAssessmentAgent.execute(ctx, db)
    ↓
RiskAssessmentAgent.run(ctx, db)
    │
    ├── 读取 4 个维度数据：
    │   compliance_summary  = ctx.shared_data.get("compliance_summary")
    │   endorsement_result  = ctx.shared_data.get("endorsement_result")
    │   contract_result     = ctx.shared_data.get("contract_result")
    │   fraud_result        = ctx.shared_data.get("fraud_result")
    │
    ├── 各维度得分计算（缺失则降级）：
    │   compliance_score  = _calc_compliance_score(compliance_summary)
    │       合规率 × 100 - 严重违规数 × 15（下限 0）
    │   endorsement_score = _calc_endorsement_score(endorsement_result)
    │       无违规=100，每个违规-10，闭环/断链额外-20
    │   contract_score    = _calc_contract_score(contract_result)
    │       match_score（0~100）
    │   fraud_score       = _calc_fraud_score(fraud_result)
    │       (1 - overall_fraud_score) × 100（欺诈分越低反欺诈得分越高）
    │
    ├── composite_score = (
    │       compliance_score  × 0.30 +
    │       endorsement_score × 0.30 +
    │       contract_score    × 0.20 +
    │       fraud_score       × 0.20
    │   )
    │
    ├── risk_level = _determine_risk_level(composite_score)
    │       ≥ 85 → LOW        （自动通过）
    │       70~84 → MEDIUM_LOW （快速审核）
    │       50~69 → MEDIUM_HIGH（全面复核）
    │       30~49 → HIGH       （建议拒绝）
    │       < 30  → CRITICAL   （强制拒绝）
    │
    ├── db.add(RiskAssessment(
    │       compliance_score, endorsement_score, contract_score, fraud_score,
    │       composite_score, risk_level, dimension_details, missing_dimensions
    │   ))
    │
    └── ctx.shared_data["risk_result"] = {composite_score, risk_level, ...}
```

### 四维加权公式

```
综合评分 = 合规维度 × 0.30
         + 背书维度 × 0.30
         + 合同维度 × 0.20
         + 欺诈维度 × 0.20
```

**权重设计依据**：
- 合规（30%）和背书（30%）是票据法律合规的核心，权重相等且最高
- 合同（20%）体现贸易背景真实性，重要但可降级
- 欺诈（20%）兜底，主要由 FraudDetectionAgent 的硬性规则（重复票据强制 1.0）覆盖极端情形

### 缺失维度降级分

| 维度 | 降级分 | 设计原因 |
|------|-------|---------|
| 合规 | 60.0 | 中间偏低，提示风险存在 |
| 背书 | 70.0 | 较宽松，无背书的简单票据不应被误判 |
| 合同 | 50.0 | 强制进入人工复核区间 |
| 欺诈 | 80.0 | 无欺诈检测结果默认清洁 |

### 数据库操作

| 表 | 操作 | 关键字段 |
|----|------|---------|
| `risk_assessments` | INSERT | composite_score / risk_level / dimension_details(JSON) / missing_dimensions(JSON 数组) |

---

## 10. ReportGenerationAgent — 报告生成（M6）

### 文件位置

[app/agents/report_generation_agent.py](app/agents/report_generation_agent.py)  
[app/agents/utils/report_renderer.py](app/agents/utils/report_renderer.py)

### 职责

聚合所有 Agent 的 `shared_data` 输出，构建 **9 节结构化 JSON 报告**，并通过 `reportlab` 渲染为 A4 格式 PDF，是整个 Agent 链路的最终产出节点。

### 调用路径

```
LangGraph 节点 "report"（风险评估之后）
    ↓
from app.mcp.tools.report_tools import generate_report
    ↓
ReportGenerationAgent.execute(ctx, db)
    ↓
ReportGenerationAgent.run(ctx, db, training_mode=False, is_final=False)
    │
    ├── report_json = _build_report_json(ctx, training_mode)
    │       聚合 9 节数据：
    │       "summary"    → 任务基本信息 + 综合结论
    │       "elements"   → 18 个要素字段
    │       "compliance" → 18 条合规检查结果
    │       "endorsement"→ 背书链图 + 违规列表
    │       "contract"   → 合同匹配分 + 不匹配详情
    │       "fraud"      → 五维欺诈评分
    │       "risk"       → 四维加权分 + 风险等级
    │       "flow"       → 七步流转状态（若执行了 FlowTracking）
    │       "conclusion" → 审核决策（APPROVE/REJECT/...）+ LLM 摘要
    │
    ├── 生成审核决策（基于 risk_level）：
    │   LOW       → APPROVE
    │   MEDIUM_LOW → APPROVE_WITH_REVIEW
    │   MEDIUM_HIGH → MANUAL_REVIEW
    │   HIGH       → REJECT
    │   CRITICAL   → MANDATORY_REJECT
    │
    ├── renderer = ReportRenderer(output_dir=self._output_dir)
    │   pdf_path = await asyncio.to_thread(
    │       renderer.render, report_json, ctx.audit_task_id, training_mode
    │   )
    │   # asyncio.to_thread：reportlab 是同步库，线程池执行避免阻塞事件循环
    │   #
    │   # ReportRenderer.render() 内部步骤：
    │   #   1. 创建 A4 PDF（SimpleDocTemplate）
    │   #   2. 封面：任务 ID + 风险等级 + 时间戳
    │   #   3. 7 个内容节（对应 report_json 7 节）
    │   #   4. 风险色彩高亮：LOW=绿#27AE60 / MEDIUM=橙 / HIGH=红 / CRITICAL=紫#8E44AD
    │   #   5. 可选培训注解页（training_mode=True 时追加）
    │   # PDF 渲染失败不阻断流程（JSON 仍保存）
    │
    ├── pdf_size = os.path.getsize(pdf_path)
    │
    ├── db.add(AuditReport(
    │       report_json, pdf_path, pdf_size_bytes,
    │       is_final, training_mode, version=1
    │   ))
    │
    └── ctx.shared_data["report_result"] = {
            report_id, pdf_path, pdf_size_bytes, has_pdf
        }
```

### 9 节 JSON 报告结构

```json
{
  "summary": { "task_id": "...", "created_at": "...", "decision": "APPROVE" },
  "elements": { "ticket_number": "...", "amount_numeric": 500000, ... },
  "compliance": [
    { "field": "ticket_number", "is_compliant": true, "rag_score": 0.92 },
    ...
  ],
  "endorsement": { "is_continuous": true, "has_cycle": false, "violations": [] },
  "contract":    { "match_score": 85.0, "amount_match": true },
  "fraud":       { "overall_fraud_score": 0.12, "fraud_level": "clean" },
  "risk":        { "composite_score": 88.5, "risk_level": "LOW" },
  "flow":        { "completed_steps": 7, "timeout_level": "NORMAL" },
  "conclusion":  { "decision": "APPROVE", "summary": "各维度均符合规范..." }
}
```

### 培训模式（training_mode）

当 `training_mode=True` 时，PDF 在正文后追加**评分原理注解页**，详细说明：
- 每个维度得分的计算过程
- 每条违规规则的法规依据原文
- 风险阈值和业务建议的行业背景

用于机构内部培训合规人员，帮助理解系统评分逻辑。

### 数据库操作

| 表 | 操作 | 关键字段 |
|----|------|---------|
| `audit_reports` | INSERT | report_json(JSON-9节) / pdf_path / pdf_size_bytes / is_final / training_mode |

---

## 11. FlowTrackingAgent — 流转追踪（M7）

### 文件位置

[app/agents/flow_tracking_agent.py](app/agents/flow_tracking_agent.py)  
[app/agents/utils/flow_channel_mock.py](app/agents/utils/flow_channel_mock.py)

### 职责

驱动**七步报文流转状态机**，模拟票据在发起行 → 前置机 → 票交所 → 对手行的完整报文流转过程，记录每步耗时并判断超时预警级别。通过适配器模式隔离外部接口，可无缝切换 Mock 和真实接口。

### 调用路径

```
POST /api/v1/tracking/tasks（独立端点，不走 OrchestratorAgent）
    ↓
tracking_router.py → create_tracking_task()
    ↓
FlowTrackingAgent.execute(ctx, db)
    ↓
FlowTrackingAgent.run(ctx, db, business_type, ticket_number)
    │
    ├── flow_task_id = uuid.uuid4()
    │   db.add(FlowTrackingTask(total_steps=7, status=INITIATED))
    │
    ├── for step in range(1, TOTAL_STEPS + 1):  # 步骤 1~7，串行执行（有顺序依赖）
    │       step_config = FLOW_STEP_NODES[step]
    │       # step_config = {"send": "发起行-报文组装", "receive": "发起行-前置机发送"}
    │       │
    │       ├── send_msg = await self._channel.send_message(step, business_type, metadata)
    │       │       FlowChannelMock.send_message()：
    │       │           asyncio.sleep(模拟延迟 10~50ms)
    │       │           返回 {status: "SENT", processing_ms: 32, msg_type: "..."}
    │       │
    │       ├── db.add(FlowMessage(step_index=step, node_name=send_node, status=SENT))
    │       │
    │       ├── recv_msg = await self._channel.receive_ack(step, business_type)
    │       │       模拟应答（可配置失败率）
    │       │
    │       ├── db.add(FlowMessage(step_index=step, node_name=recv_node, status=ACK))
    │       │
    │       ├── total_elapsed_ms += step_elapsed_ms
    │       ├── timeout_level = _check_timeout(total_elapsed_ms)
    │       │
    │       └── 步骤失败 → final_status = FAILED，break
    │
    ├── UPDATE FlowTrackingTask SET
    │       completed_steps, status, timeout_level, summary_text
    │
    └── ctx.shared_data["flow_result"] = {
            flow_task_id, completed_steps, status, timeout_level
        }
```

### 七步流转协议

| 步骤 | 发送节点 | 接收节点 |
|------|---------|---------|
| 1 | 发起行-报文组装 | 发起行-前置机发送 |
| 2 | 前置机-票交所发送 | 票交所-接收确认 |
| 3 | 票交所-报文处理 | 票交所-转发对手行 |
| 4 | 对手行-接收 | 对手行-处理中 |
| 5 | 对手行-应答生成 | 对手行-票交所应答 |
| 6 | 票交所-应答接收 | 票交所-发起行转发 |
| 7 | 发起行前置机-应答接收 | 发起行-业务完成 |

### 超时预警五级

| 级别 | 累计耗时 | 处理建议 |
|------|---------|---------|
| NORMAL | < 30s | 正常，无需干预 |
| WATCH | 30~60s | 开始关注 |
| WARNING | 60~120s | 告警通知 |
| URGENT | 120~300s | 紧急处理 |
| OVERDUE | > 300s | 人工介入，触发超期处理流程 |

### 适配器模式设计

```python
class FlowChannelMock:
    """Mock 适配器：模拟前置机/票交所/对手行三个外部接口"""
    async def send_message(self, step, business_type, metadata) -> dict: ...
    async def receive_ack(self, step, business_type) -> dict: ...

# 生产环境替换：
class FlowChannelReal:
    """真实适配器：调用真实票交所 HTTP API"""
    async def send_message(self, step, business_type, metadata) -> dict:
        response = await http_client.post(ECDS_API_URL, ...)
        return parse_ecds_response(response)
```

生产环境只需实现 `FlowChannelReal`，注入 `FlowTrackingAgent(channel=FlowChannelReal())`，Agent 业务逻辑零修改。

### 数据库操作

| 表 | 操作 | 说明 |
|----|------|------|
| `flow_tracking_tasks` | INSERT + UPDATE | 主记录，含 7 步进度和超时级别 |
| `flow_messages` | INSERT × 14 | 每步 2 条（发送+接收），共 14 条 |

---

## 12. FraudDetectionAgent — 欺诈检测（M8）

### 文件位置

[app/agents/fraud_detection_agent.py](app/agents/fraud_detection_agent.py)

### 职责

通过**五维并行检测**（每维 2 秒超时）综合评估票据欺诈风险。重复票据是**硬性规则**：一旦在历史库中找到完全相同的票据，综合评分强制置 1.0（最高欺诈风险）。与 ComplianceRetrievalAgent 和 EndorsementChainAgent 并行执行。

### 调用路径

```
LangGraph 并行节点 "parallel"（asyncio.gather 第 3 个任务）
    ↓
from app.mcp.tools.fraud_tools import detect_fraud
    ↓
FraudDetectionAgent.execute(ctx, db)
    ↓
FraudDetectionAgent.run(ctx, db)
    │
    ├── bill_element       = ctx.shared_data["bill_element"]
    │   endorsement_result = ctx.shared_data.get("endorsement_result", {})
    │
    ├── bill_fp = compute_bill_fingerprint(bill_element)
    │   cached  = await bill_cache.get_fraud(tenant_id, bill_fp)  TTL=2h
    │   命中 → 直接写 DB，返回
    │
    ├── tasks = [
    │       _run_with_timeout("seal",      _check_seal(bill_element), 2s),
    │       _run_with_timeout("duplicate", _check_duplicate(bill_element), 2s),
    │       _run_with_timeout("tamper",    _check_tamper(bill_element), 2s),
    │       _run_with_timeout("network",   _check_network(endorsement_result), 2s),
    │       _run_with_timeout("blacklist", _check_blacklist(bill_element), 2s),
    │   ]
    │   results = await asyncio.gather(*tasks)  # 五维完全并行
    │
    ├── _check_seal(bill_element)
    │       OCR 置信度分析（parse_stats.ocr_avg_confidence）
    │       + blacklist_entities 向量比对（印章图像特征）
    │       → seal_score（0.0=正常，1.0=高风险）
    │
    ├── _check_duplicate(bill_element)
    │       SELECT COUNT(*) FROM bill_elements
    │       WHERE ticket_number = bill_element["ticket_number"]
    │       AND audit_task_id != ctx.audit_task_id
    │       # 精确查询历史票据，相同票号即命中
    │       → duplicate_score = 1.0（命中）/ 0.0（未命中）
    │
    ├── _check_tamper(bill_element)
    │       OCR 置信度分布分析（标准差过大 → 可能被局部篡改）
    │       → tamper_score
    │
    ├── _check_network(endorsement_result)
    │       直接复用 EndorsementChainAgent 的结果：
    │       has_cycle=True → network_score += 0.8
    │       → network_score
    │
    ├── _check_blacklist(bill_element)
    │       SELECT * FROM blacklist_entities
    │       WHERE entity_name ILIKE %drawer% OR %acceptor% OR %payee%
    │       → blacklist_score
    │
    ├── 重复票据硬性规则：
    │   if duplicate_score >= 1.0:
    │       overall_fraud_score = 1.0  # 强制最高分
    │   else:
    │       overall_fraud_score = sum(score × weight for each dimension)
    │
    ├── fraud_level = _determine_fraud_level(overall_fraud_score)
    │       ≥ 0.8 → "fraud"
    │       0.6~0.8 → "high_risk"
    │       0.3~0.6 → "suspicious"
    │       < 0.3  → "clean"
    │
    ├── db.add(FraudDetection(seal/duplicate/tamper/network/blacklist_score, ...))
    │
    ├── await bill_cache.set_fraud(tenant_id, bill_fp, fraud_result)  TTL=2h
    │
    └── ctx.shared_data["fraud_result"] = {overall_fraud_score, fraud_level, ...}
```

### 五维权重配置

| 维度 | 权重 | 检测方法 | 特殊规则 |
|------|------|---------|---------|
| 印章真伪（seal） | 30% | OCR 置信度 + 黑名单向量比对 | — |
| 重复票据（duplicate） | 25% | PostgreSQL 精确历史查询 | **命中强制总分=1.0** |
| 图像篡改（tamper） | 25% | OCR 置信度分布分析（标准差） | — |
| 关联网络（network） | 15% | 复用背书链的闭环检测结果 | — |
| 历史黑名单（blacklist） | 5% | blacklist_entities 模糊匹配 | — |

### 超时保护机制

```python
async def _run_with_timeout(self, name, coro, timeout=2.0):
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        # 超时：该维度得分归零，不阻断其他维度和主流程
        logger.warning(f"[{self.agent_name}] 维度 {name} 超时，得分归零")
        return (0.0, {}, None)
```

每维最多等待 2 秒，超时返回 0 分（而非失败），确保整体检测时间可控（最差情况 ≤ 2s + gather 开销）。

### 数据库操作

| 表 | 操作 | 关键字段 |
|----|------|---------|
| `fraud_detections` | INSERT | seal/duplicate/tamper/network/blacklist_score / overall_fraud_score / fraud_level |

---

## 13. BatchSchedulingAgent — 批量调度（M9）

### 文件位置

[app/agents/batch_scheduling_agent.py](app/agents/batch_scheduling_agent.py)

### 职责

接收批量审核任务（多个文档 ID），通过 `asyncio.Semaphore` 控制并发上限，为每个子任务创建独立的 `AgentContext` 并调用完整的 OrchestratorAgent 链路。子任务失败**完全隔离**，不影响其他子任务。

### 调用路径

```
POST /api/v1/batch/tasks
    ↓
batch_router.py → create_batch_task()
    ↓
BatchSchedulingAgent.execute(ctx, db)
    ↓
BatchSchedulingAgent.run(ctx, db, items=[...], concurrency=10, task_type="FULL_AUDIT")
    │
    ├── concurrency = min(concurrency, MAX_CONCURRENCY=100)
    │
    ├── batch_task_id = uuid.uuid4()
    │   db.add(BatchTask(total_count=len(items), status=RUNNING))
    │
    ├── item_records = [BatchTaskItem(...) for item in items]
    │   db.add_all(item_records)  # 批量写入子任务记录
    │
    ├── semaphore = asyncio.Semaphore(concurrency)
    │       # 并发槽位：最多 concurrency 个子任务同时运行
    │
    ├── async def _run_item(item_record):
    │       async with semaphore:  # 获取槽位
    │           try:
    │               sub_ctx = AgentContext(
    │                   audit_task_id=str(uuid.uuid4()),  # 独立子任务 ID
    │                   tenant_id=ctx.tenant_id,
    │                   document_id=item_record.document_id,
    │               )
    │               result = await self._subtask_runner(sub_ctx, db, task_type)
    │               # 默认 _default_subtask_runner → OrchestratorAgent().execute()
    │               item_record.status = "COMPLETED"
    │           except Exception as e:
    │               item_record.status = "FAILED"
    │               item_record.error_msg = str(e)
    │               # 子任务失败只更新该条记录，不影响其他子任务
    │
    ├── await asyncio.gather(
    │       *[_run_item(r) for r in item_records]
    │   )
    │   # 所有子任务并发启动，Semaphore 控制实际并发数
    │
    ├── 统计结果：completed_count / failed_count / failure_rate
    │
    ├── 整体状态判定：
    │   failure_rate >= BATCH_FAILURE_THRESHOLD(0.5) → FAILED
    │   否则 → COMPLETED
    │
    └── UPDATE BatchTask SET completed/failed_count, status, result_summary
```

### `asyncio.Semaphore` 并发控制原理

```python
semaphore = asyncio.Semaphore(10)  # 允许 10 个协程同时持有许可

async def _run_item(item):
    async with semaphore:    # 等待许可（阻塞直到有空闲槽位）
        await do_work(item)  # 执行实际工作
    # 退出 async with 时自动释放许可，让下一个等待的协程继续
```

`asyncio.gather` 并发启动所有任务，但 Semaphore 确保任意时刻最多 N 个任务真正执行，其余等待。这样既保持高吞吐，又不会因过度并发撑爆数据库连接池和 LLM API 速率限制。

### 测试友好设计

`subtask_runner` 通过构造函数注入，测试时可注入 Mock 函数，避免真实调用 OrchestratorAgent：
```python
async def mock_runner(ctx, db, task_type):
    return AgentResult(agent_name="mock", success=True)

agent = BatchSchedulingAgent(subtask_runner=mock_runner)
```

### 数据库操作

| 表 | 操作 | 说明 |
|----|------|------|
| `batch_tasks` | INSERT + UPDATE | 批量任务主记录 |
| `batch_task_items` | INSERT × N + UPDATE × N | 每个子任务的状态和错误信息 |

---

## 14. BillIssuanceAgent — 出票预检（M10）

### 文件位置

[app/agents/bill_issuance_agent.py](app/agents/bill_issuance_agent.py)

### 职责

在票据正式提交出票前，**并行执行三项快速预检**（要素完整性 + 黑名单 + 授信额度），秒级响应，用于出票表单填写阶段的即时校验，节约后续全流程审核的计算资源。

### 调用路径（两种使用场景）

**场景 1：实时预检（不创建 AuditTask）**
```
POST /api/v1/audit/issuance/check
    ↓
audit_router.py → quick_issuance_check()
    ↓
BillIssuanceAgent.execute(ctx, db)  # ctx.audit_task_id 为临时 ID，不写 audit_tasks 表
```

**场景 2：出票正式审核（走完整 OrchestratorAgent 链路）**
```
POST /api/v1/audit/tasks {task_type: "issuance_check"}
    ↓
OrchestratorAgent → build_audit_graph("ISSUANCE_CHECK")
    ↓
LangGraph: parse → extract → parallel(compliance + issuance) → risk → report
    ↓
node_parallel_compliance_issuance → asyncio.gather(
    check_compliance MCP,       # ComplianceRetrievalAgent
    check_bill_issuance MCP,    # BillIssuanceAgent ← 在此处调用
)
```

**BillIssuanceAgent 内部执行**
```
BillIssuanceAgent.run(ctx, db)
    │
    ├── bill_element = ctx.shared_data["bill_element"]
    │
    ├── elements_result, blacklist_result, credit_result =
    │       await asyncio.gather(
    │           _check_required_elements(bill_element),   # 要素完整性
    │           _check_blacklist(bill_element, db),       # 黑名单查询
    │           _check_credit_system(bill_element),       # 授信系统
    │       )
    │
    ├── _check_required_elements(bill_element)
    │       检查 8 个必填字段（ISSUANCE_REQUIRED_FIELDS）是否非空
    │       + 金额范围：10000 ≤ amount_numeric ≤ 500000000
    │       + 期限：maturity_days ≤ 365
    │
    ├── _check_blacklist(bill_element, db)
    │       SELECT * FROM blacklist_entities
    │       WHERE entity_name ILIKE %drawer%
    │          OR entity_name ILIKE %acceptor%
    │          OR entity_name ILIKE %payee%
    │          AND is_active = true
    │          AND (expired_at IS NULL OR expired_at > NOW())
    │
    ├── _check_credit_system(bill_element)
    │       if CREDIT_SYSTEM_MOCK:
    │           模拟授信额度查询（随机返回 Mock 结果）
    │       else:
    │           await http_client.post(CREDIT_SYSTEM_URL, ...)  # 真实接口
    │       # 出票金额 > 授信额度 → 预检失败
    │
    ├── failed_checks = elements_failures + blacklist_failures + credit_failures
    │   is_eligible  = len(failed_checks) == 0
    │
    └── ctx.shared_data["issuance_result"] = {
            is_eligible, failed_checks, failed_count,
            elements_passed, blacklist_passed, credit_passed,
            recommendation
        }
```

### 三项预检规则

| 预检项 | 规则 | 失败处理 |
|--------|------|---------|
| 要素完整性 | 8 个必填字段非空；金额 1万~5亿；期限 ≤ 365天 | 列出所有缺失/异常字段 |
| 黑名单 | 出票人/承兑人/收款人均不在黑名单中 | 标注命中的实体和黑名单类型 |
| 授信系统 | 出票金额 ≤ 出票人授信额度 | 标注授信缺口金额 |

### 数据库操作

| 表 | 操作 | 说明 |
|----|------|------|
| `blacklist_entities` | SELECT | 黑名单查询（只读） |
| `shared_data` | 写入 issuance_result | 不单独写 DB 表（实时预检场景） |

---

## 15. BaseAgent 框架层

### 文件位置

[app/agents/base_agent.py](app/agents/base_agent.py)

### 核心数据类

```python
@dataclass
class AgentContext:
    """Agent 执行上下文：贯穿整个调用链"""
    audit_task_id: str          # 主任务 UUID（与 audit_tasks.id 对应）
    tenant_id: str              # 租户 ID（数据隔离）
    bill_record_id: Optional[str] = None
    document_id: Optional[str] = None
    shared_data: dict = field(default_factory=dict)  # Agent 间共享中间结果

@dataclass
class AgentResult:
    """标准化 Agent 输出"""
    agent_name: str
    success: bool
    data: Any = None
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    elapsed_ms: float = 0.0          # 由 execute() 自动填入
```

### BaseAgent.execute() 执行流程

```python
async def execute(self, ctx, db, **kwargs) -> AgentResult:
    start_time = time.monotonic()

    # 1. 记录启动日志
    logger.info(f"[{self.agent_name}] start | task={ctx.audit_task_id}")

    # 2. 更新 audit_tasks.current_agent（让前端知道当前在执行哪个 Agent）
    await self._update_current_agent(ctx.audit_task_id, db)

    try:
        # 3. 调用子类实现的核心逻辑（子类无需 try-except）
        result = await self.run(ctx, db, **kwargs)

    except Exception as e:
        # 4. 统一异常捕获（子类任何未处理异常都在此兜底）
        return AgentResult(
            agent_name=self.agent_name,
            success=False,
            error_code="SYS_AGENT_EXCEPTION",
            error_msg=f"{type(e).__name__}: {str(e)}",
            elapsed_ms=(time.monotonic() - start_time) * 1000,
        )

    # 5. 自动填入耗时
    result.elapsed_ms = (time.monotonic() - start_time) * 1000
    logger.info(f"[{self.agent_name}] done | elapsed={result.elapsed_ms:.1f}ms")
    return result
```

### 子类开发约定

1. **继承 BaseAgent，覆盖 `agent_name`**（用于日志标识和 current_agent 更新）
2. **实现 `run()` 方法**，不需要 try-except（`execute()` 统一兜底）
3. **读取 `ctx.shared_data`** 获取前序 Agent 的输出
4. **写入 `ctx.shared_data`** 供后续 Agent 读取
5. **调用 `db.add()` 写数据库**，不需要 `commit()`（由外层调用者统一提交）

---

## 16. shared_data 数据流全景

`shared_data` 是 Agent 间数据传递的核心载体（通过 LangGraph 的 `BillAuditState` 持久化到 PostgreSQL）。

```
DocumentParserAgent
    └─ shared_data["parsed_doc"] = {elements, parse_stats, confidence, file_path}
                                       ↓
ElementExtractionAgent（读取 parsed_doc，或从文件重新识别）
    └─ shared_data["bill_element"] = {18字段, confidence_score}
                                       ↓
         ┌─────────────────────────────┼────────────────────────────┐
         ↓                             ↓                            ↓
ComplianceRetrievalAgent    EndorsementChainAgent         FraudDetectionAgent
（读 bill_element）          （读 bill_element）           （读 bill_element + endorsement_result）
└─ shared_data["compliance_summary"]  └─ shared_data["endorsement_result"]  └─ shared_data["fraud_result"]
                ↓                             ↓                            ↓
                └─────────────────────────────┼────────────────────────────┘
                                              ↓
                                    ContractReviewAgent（可选）
                                    （读 bill_element + contract_text）
                                    └─ shared_data["contract_result"]
                                              ↓
                                    RiskAssessmentAgent
                                    （读 compliance + endorsement + contract + fraud）
                                    └─ shared_data["risk_result"]
                                              ↓
                                    ReportGenerationAgent
                                    （读全部 shared_data）
                                    └─ shared_data["report_result"]
```

---

## 17. 8 种业务图执行拓扑速查

| 业务类型 | task_type | 拓扑 | contract 是否关键 |
|---------|-----------|------|-------------------|
| 全流程审核 | FULL_AUDIT | parse→extract→**并行(合规+背书+欺诈)**→contract→risk→report | 否（失败继续） |
| 出票预检 | ISSUANCE_CHECK | parse→extract→**并行(合规+出票预检)**→risk→report | — |
| 贴现申请 | DISCOUNT_APPLY | parse→extract→**并行(合规+背书+欺诈)**→contract→risk→report | **是**（失败终止） |
| 提示承兑 | ACCEPTANCE_PROMPT | parse→extract→**并行(合规+背书)**→risk→report | — |
| 背书转让 | ENDORSEMENT | parse→extract→**并行(背书+欺诈)**→risk→report | — |
| 提示付款 | PAYMENT_PROMPT | 同 FULL_AUDIT | 否 |
| 质押融资 | PLEDGE | 同 FULL_AUDIT | 否 |
| 托收代收 | COLLECTION | 同 FULL_AUDIT | 否 |

> **贴现申请（DISCOUNT_APPLY）中 contract 为关键节点**：贴现业务必须有合同支撑贸易背景，无合同或合同严重不匹配时直接拒绝，不进入风险评分。

### 条件边失败路由规则

| 节点 | 条件 | 路由 |
|------|------|------|
| parse | 失败 | → END（无法解析则无法继续） |
| extract | 失败 | → END（无要素则无从检索） |
| parallel_cef | compliance 或 endorsement 失败 | → END |
| parallel_cef | 仅 fraud 失败 | → contract（欺诈检测失败可容忍） |
| contract (关键) | 失败 | → END |
| contract (非关键) | 失败 | → risk（使用降级默认分） |
| risk | 失败 | → END |
| report | 始终 | → END（最终节点） |

---

*文档由代码自动提炼，如代码有更新请同步修改此文档。*
