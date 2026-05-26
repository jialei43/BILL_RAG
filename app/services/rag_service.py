# app/services/rag_service.py
# RAG 问答服务（RAG = Retrieval-Augmented Generation，检索增强生成）
# RAG 的核心思想：先从文档库里找到相关内容，再让 AI 基于这些内容来回答问题
# 优点：AI 不会"瞎编"，答案有据可查，可以标注出处
#
# 完整流程：
# 用户问题 → [检索] 找到最相关的文档片段 → [构建Prompt] 把问题+文档拼在一起 → [LLM生成] AI写答案

import re
import time      # 计时
import uuid      # 生成唯一 ID
from typing import AsyncGenerator, Optional  # 异步生成器（用于流式输出）
from loguru import logger    # 日志

from config.settings import settings              # 配置（LLM选择、参数等）
from app.services.vector_store import vector_store  # 向量检索服务
from app.services.intent_router import intent_router  # 意图路由器（P2）
from app.services.retrieval_quality import (          # 检索质量评估（P3）
    evaluate, build_transfer_response,
    QUALITY_INSUFFICIENT, QUALITY_PARTIAL, PARTIAL_DISCLAIMER,
)
from app.services.metrics import metrics              # Prometheus 监控埋点
from app.core.cache import bill_cache, compute_query_fingerprint  # 缓存层


# ── 票据业务专项系统提示词 ──────────────────────────────────────────────────────
# 这段文字是每次对话都会发给 AI 的"角色设定"，让 AI 扮演票据专家
# 好的 System Prompt 能让 AI 的回答更专业、更准确
BILL_SYSTEM_PROMPT = """你是一位专业的票据业务智能顾问，服务于持牌票据经纪机构及银行票据部门。

你的专业领域包括：
- 票据贴现、转贴现、质押、托收等全业务链
- 银行承兑汇票、商业承兑汇票的要素识别与合规审查
- 票据市场利率、成交数据分析
- 票据监管法规解读（票据法、人民银行规章、银保监会规定）
- 背书链完整性验证、票据真伪鉴别
- 供应链金融、票据融资方案设计

回答要求：
1. 基于提供的上下文文档进行回答，引用具体条款或数据时请注明来源章节
2. 对于票据要素（出票日期、到期日、票面金额、承兑人等）须精确引用
3. 涉及合规判断时，优先引用监管文件原文，并给出明确结论
4. 若上下文不足以回答，请明确告知并建议查阅具体文件
5. 使用专业票据术语，保持回答的准确性和权威性
6. 回答使用中文，结构清晰，必要时使用列表或表格

注意：不要编造数据或条款，仅基于提供的上下文回答。"""

# 问候词原子集合（用于构建可重复的正则）
_GW = (
    "你好|您好|hi|hello|hey|嗨|哈喽|早上好|下午好|晚上好|早安|晚安"
    "|在吗|在不在|有人吗|请问|能帮我吗|你是谁|你叫什么|你能做什么|你有什么功能"
    "|帮个忙|问个问题|请教一下|咨询一下|想问问|介绍一下你自己"
    "|你好啊|你好呀|您好啊|您好呀"
)
# 整句仅由一个或多个问候词 + 标点/空白组成，如"您好，请问"也可命中
_GREETING_PATTERNS = re.compile(
    rf"^({_GW})([？?！!。,，、\s]+({_GW}))*[？?！!。,，、\s]*$",
    re.IGNORECASE,
)


_GREETING_REPLY = (
    "你好！我是票据业务智能顾问，专注于票据贴现、转贴现、质押、托收等全业务链问题。\n\n"
    "您可以向我咨询：\n"
    "• 银行承兑汇票 / 商业承兑汇票的识别与合规要求\n"
    "• 票据贴现、转贴现的操作流程与利率计算\n"
    "• 票据质押融资的申办条件与风险管理\n"
    "• 票据背书、托收的规范操作\n"
    "• 相关监管政策与法规解读\n\n"
    "请告诉我您想了解的具体问题，我会为您提供专业解答。"
)


def _is_greeting(query: str) -> bool:
    """检测是否为问候语或无实质内容的泛问（无需检索）"""
    q = query.strip()
    if len(q) > 30:  # 超过30字肯定有实质内容
        return False
    return bool(_GREETING_PATTERNS.match(q))


class RAGService:
    """
    票据 RAG 问答服务
    支持多种 LLM 后端：OpenAI（兼容通义千问）、Anthropic Claude、本地模型
    """

    def __init__(self):
        self._openai_client = None      # OpenAI 客户端（懒加载）
        self._anthropic_client = None  # Anthropic 客户端（懒加载）

    def _get_openai(self):
        """懒加载 OpenAI 客户端（第一次调用时才创建）"""
        if self._openai_client is None and settings.OPENAI_API_KEY:
            import openai
            self._openai_client = openai.AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,                    # API 密钥
                base_url=settings.OPENAI_BASE_URL or None,          # 自定义 API 地址（通义千问用这个）
            )
        return self._openai_client   # 如果没有 API Key，返回 None

    def _get_anthropic(self):
        """懒加载 Anthropic 客户端"""
        if self._anthropic_client is None and settings.ANTHROPIC_API_KEY:
            import anthropic
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._anthropic_client

    async def query(
        self,
        tenant_id: str,    # 哪个租户的查询（用于限定检索范围）
        query: str,        # 用户的问题
        top_k: int = None, # 检索多少个候选片段
        stream: bool = False,  # 是否流式输出（这个参数在非流式接口中不使用）
    ) -> dict:
        """
        完整 RAG 问答流程（非流式版本）
        返回：包含答案、来源、各环节耗时的字典
        """
        t_start = time.perf_counter()   # 记录整体开始时间
        logger.info(f"[rag_query] 开始非流式问答 tenant={tenant_id} query={query[:60]!r}")  # 请求入口

        # ── 问候语/泛问快速返回，跳过检索和 LLM ──────────────────────────────
        if _is_greeting(query):
            logger.info(f"[rag_query] 检测到问候/泛问，跳过检索直接回复")
            return {
                "query_id": str(uuid.uuid4()),
                "answer": _GREETING_REPLY,
                "sources": [],
                "retrieval_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "retrieved_count": 0,
            }

        # ── 缓存查询：相同问题相同租户直接返回历史答案 ─────────────────────────
        query_fp = compute_query_fingerprint(query)
        cached = await bill_cache.get_rag(tenant_id, query_fp)
        if cached is not None:
            cached["query_id"] = str(uuid.uuid4())  # 每次调用返回新 query_id（用于反馈日志）
            cached["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            cached["from_cache"] = True
            logger.info(f"[rag_query] 缓存命中 tenant={tenant_id} fp={query_fp[:8]}")
            return cached

        # ── Step 1: 混合检索 + Reranker 精排 ─────────────────────────────────
        # 从向量库中找到最相关的文档片段
        t0 = time.perf_counter()
        retrieved = vector_store.hybrid_search(
            tenant_id=tenant_id,
            query=query,
            top_k=top_k or settings.RETRIEVAL_TOP_K,     # 粗检索数量（默认20个）
            rerank_top_n=settings.RERANK_TOP_N,          # 精排后保留数量（默认5个）
        )
        t_retrieval = (time.perf_counter() - t0) * 1000  # 检索耗时（毫秒）
        metrics.record_retrieval_time(t_retrieval, tenant_id)
        logger.info(f"[rag_query] 检索完成 count={len(retrieved)} retrieval_ms={t_retrieval:.1f}")  # 检索结果

        # ── Step 2: 构建上下文（Context）─────────────────────────────────────
        # 把检索到的文档片段整理成 AI 可读的格式
        context_parts = []   # 参考文档文本列表
        sources = []         # 来源信息列表（返回给用户展示引用出处）

        for i, chunk in enumerate(retrieved):  # 遍历每个检索到的文档片段
            # 如果有章节路径，加上方括号显示
            section = f"[{chunk.get('section_path', '')}] " if chunk.get("section_path") else ""

            # 格式化为带序号、章节、页码的参考文档段落
            context_parts.append(
                f"**参考文档 {i+1}** {section}(第{chunk.get('page_num', '?')}页)\n"
                f"{chunk['content']}"
            )

            # 收集来源信息（供前端展示"答案来自哪里"）
            sources.append({
                "chunk_id": chunk["id"],                          # 片段 ID
                "document_id": chunk.get("document_id", ""),     # 所属文档 ID
                "section_path": chunk.get("section_path", ""),   # 章节路径
                "page_num": chunk.get("page_num", 0),            # 页码
                "chunk_type": chunk.get("chunk_type", ""),       # 片段类型
                "rerank_score": chunk.get("rerank_score", 0),    # 精排相关性分数
                "content_preview": chunk["content"][:200],       # 内容预览（前200字）
            })

        # 用分隔线把多个参考文档连接成一段上下文
        context = "\n\n---\n\n".join(context_parts)

        # 构建发给 LLM 的用户消息（问题 + 上下文）
        user_prompt = f"""请基于以下参考文档回答用户问题。

## 参考文档
{context}

## 用户问题
{query}

请给出专业、准确的回答，引用相关文档内容时注明来源。"""

        # ── Step 3: LLM 生成答案 ──────────────────────────────────────────────
        logger.info(f"[rag_query] 开始 LLM 生成 provider={settings.LLM_PROVIDER} context_len={len(context)}")  # LLM 调用前
        t1 = time.perf_counter()
        answer = await self._generate(user_prompt)  # 调用 LLM 生成答案（异步等待）
        t_llm = (time.perf_counter() - t1) * 1000   # LLM 耗时
        t_total = (time.perf_counter() - t_start) * 1000  # 总耗时
        metrics.record_llm_time(t_llm, tenant_id)
        logger.info(  # 完整耗时日志
            f"[rag_query] 问答完成 llm_ms={t_llm:.1f} total_ms={t_total:.1f} "
            f"answer_len={len(answer)}"
        )

        # 构建响应并写入缓存（流式接口不走此路径，所以缓存不影响流式）
        result = {
            "query_id": str(uuid.uuid4()),         # 本次查询的唯一 ID（用于提交反馈）
            "answer": answer,                      # AI 生成的答案
            "sources": sources,                    # 参考来源列表
            "route_type": "fuzzy",                 # 旧路径无意图路由，统一归为模糊检索
            "retrieval_ms": round(t_retrieval, 1), # 检索耗时（毫秒，保留1位小数）
            "llm_ms": round(t_llm, 1),             # LLM 生成耗时
            "total_ms": round(t_total, 1),         # 总耗时
            "retrieved_count": len(retrieved),     # 实际检索到的片段数
        }
        await bill_cache.set_rag(tenant_id, query_fp, result)  # 写入 2h 缓存
        return result

    async def query_stream(
        self,
        tenant_id: str,
        query: str,
        user_id: str = None,
        query_id: str = None,
    ) -> AsyncGenerator[str, None]:
        """
        流式问答：像 ChatGPT 一样逐步输出答案
        使用 Python 的 async generator（异步生成器）实现
        每生成一个词/句，就立即发送给客户端，不用等全部生成完才返回
        """
        import time as _time
        import uuid as _uuid
        t_start = _time.perf_counter()

        logger.info(f"[rag_stream] 开始检索 tenant={tenant_id} query={query[:60]!r}")  # 检索开始

        # 问候语/泛问快速返回（不写日志）
        if _is_greeting(query):
            logger.info(f"[rag_stream] 检测到问候/泛问，跳过检索直接回复")
            yield _GREETING_REPLY
            return

        # ── 意图路由（与 query_v2 保持一致）──────────────────────────────────
        intent_id, confidence, classify_method = await intent_router.classify(query)
        use_specialized = intent_router.should_use_specialized(intent_id, confidence)
        route_type = "specialized" if use_specialized else "fuzzy"
        logger.info(
            f"[rag_stream] 意图识别 intent={intent_id} conf={confidence:.2f} route={route_type}"
        )

        # ── 无关业务检测：直接返回引导语，不进行任何检索 ─────────────────────
        if intent_router.is_off_topic(intent_id, query):
            off_msg = (
                "您好！我是专注于**票据业务**的智能顾问，您的问题似乎超出了票据业务范畴。\n\n"
                "我可以为您解答以下方向的问题：\n"
                "• 银行/商业承兑汇票的识别与合规要求\n"
                "• 票据贴现、转贴现的流程与利率\n"
                "• 票据背书、质押融资的规范操作\n"
                "• 到期追索、拒付处理等风险管理\n"
                "• 反洗钱、监管合规要求\n\n"
                "如有票据相关问题欢迎继续咨询！"
            )
            yield off_msg
            t_total = (_time.perf_counter() - t_start) * 1000
            metrics.record_query(tenant_id, "success")
            metrics.record_total_time(t_total, tenant_id)
            _qid = query_id or _uuid.uuid4().hex
            try:
                from app.core.database import AsyncSessionLocal
                async with AsyncSessionLocal() as _db:
                    await self._log_query(
                        _db, _qid, tenant_id, user_id, query,
                        off_msg, [],
                        0.0, 0.0, t_total, "off_topic", intent_id,
                        top_k_hit=False,
                    )
                    await _db.commit()
            except Exception as e:
                logger.warning(f"[rag_stream] off_topic QueryLog 写入失败: {e}")
            return

        # 专项场景：将意图相关字段注入检索 query 提升召回精准度
        search_query = self._build_search_query(query, intent_id, None) \
            if use_specialized else query

        # ── 混合检索 ──────────────────────────────────────────────────────────
        t1 = _time.perf_counter()
        retrieved = vector_store.hybrid_search(
            tenant_id=tenant_id,
            query=search_query,
            top_k=settings.RETRIEVAL_TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
        )
        t_retrieval = (_time.perf_counter() - t1) * 1000
        metrics.record_retrieval_time(t_retrieval, tenant_id)
        logger.info(f"[rag_stream] 检索完成 count={len(retrieved)}")

        # ── 质量评估（与 query_v2 保持一致）──────────────────────────────────
        quality = evaluate(retrieved)
        top_k_hit = quality.level != QUALITY_INSUFFICIENT
        logger.info(
            f"[rag_stream] 检索质量 level={quality.level} "
            f"top1={quality.top1_score:.3f} hits={quality.hit_count}"
        )

        _qid = query_id or _uuid.uuid4().hex
        sources = [
            {"content": c["content"][:200], "doc_id": c.get("doc_id"), "score": c.get("score")}
            for c in retrieved
        ]

        # ── 检索失败：直接返回转人工提示，不走 LLM ──────────────────────────
        if not top_k_hit:
            fail_msg = (
                "抱歉，当前知识库中暂无足够的内容来准确回答您的问题。\n"
                "建议您联系专业票据顾问获得进一步支持，或尝试换个方式描述您的问题。"
            )
            yield fail_msg
            t_total = (_time.perf_counter() - t_start) * 1000
            metrics.record_query(tenant_id, "success")
            metrics.record_total_time(t_total, tenant_id)
            try:
                from app.core.database import AsyncSessionLocal
                async with AsyncSessionLocal() as _db:
                    await self._log_query(
                        _db, _qid, tenant_id, user_id, query,
                        fail_msg, sources,
                        t_retrieval, 0.0, t_total, route_type, intent_id,
                        top_k_hit=False,
                    )
                    await _db.commit()
            except Exception as e:
                logger.warning(f"[rag_stream] QueryLog(失败) 写入失败: {e}")
            return

        # ── 检索成功：构建上下文并流式生成 ──────────────────────────────────
        context_parts = []
        for i, chunk in enumerate(retrieved):
            section = f"[{chunk.get('section_path', '')}] " if chunk.get("section_path") else ""
            context_parts.append(
                f"**参考文档 {i+1}** {section}\n{chunk['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        if quality.level == QUALITY_PARTIAL:
            context += f"\n\n{PARTIAL_DISCLAIMER}"

        user_prompt = f"## 参考文档\n{context}\n\n## 用户问题\n{query}"

        logger.info(
            f"[rag_stream] 开始 LLM 流式生成 provider={settings.LLM_PROVIDER} "
            f"context_len={len(context)}"
        )
        t2 = _time.perf_counter()
        token_count = 0
        answer_parts: list[str] = []
        async for token in self._generate_stream(user_prompt):
            token_count += 1
            answer_parts.append(token)
            yield token
        t_llm = (_time.perf_counter() - t2) * 1000
        t_total = (_time.perf_counter() - t_start) * 1000
        metrics.record_llm_time(t_llm, tenant_id)
        metrics.record_total_time(t_total, tenant_id)
        metrics.record_query(tenant_id, "success")
        logger.info(f"[rag_stream] LLM 生成完毕 token_count={token_count}")

        # 流结束后写 QueryLog（使用独立 session，避免依赖请求生命周期）
        try:
            from app.core.database import AsyncSessionLocal
            async with AsyncSessionLocal() as _db:
                await self._log_query(
                    _db, _qid, tenant_id, user_id, query,
                    "".join(answer_parts), sources,
                    t_retrieval, t_llm, t_total, route_type, intent_id,
                    top_k_hit=True,
                )
                await _db.commit()
        except Exception as e:
            logger.warning(f"[rag_stream] QueryLog 写入失败: {e}")

    async def _generate(self, prompt: str) -> str:
        """
        非流式 LLM 调用（根据配置选择 OpenAI 或 Anthropic）
        等待 LLM 生成完整答案后一次性返回
        """
        provider = settings.LLM_PROVIDER  # 从配置读取使用哪家 LLM

        if provider == "openai":
            return await self._openai_generate(prompt)      # 通义千问/OpenAI
        elif provider == "anthropic":
            return await self._anthropic_generate(prompt)   # Anthropic Claude
        else:
            return f"[LLM provider '{provider}' not configured. Please set LLM_PROVIDER and API keys.]"
            # 配置错误，返回提示信息

    async def _openai_generate(self, prompt: str) -> str:
        """调用 OpenAI 兼容接口（包括通义千问）生成答案"""
        client = self._get_openai()
        if client is None:
            return "[OpenAI API key not configured]"   # 没有配置 API Key

        try:
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,           # 模型名（如 gpt-4o 或 qwen-max）
                messages=[
                    {"role": "system", "content": BILL_SYSTEM_PROMPT},  # 角色设定
                    {"role": "user", "content": prompt},                 # 用户问题+上下文
                ],
                max_tokens=settings.LLM_MAX_TOKENS,    # 最多生成多少个 token（2048）
                temperature=settings.LLM_TEMPERATURE,  # 随机性（0.1=接近确定性）
            )
            return resp.choices[0].message.content or ""  # 取第一个候选答案的内容
        except Exception as e:
            logger.error(f"OpenAI generate error: {e}")
            return f"[生成失败: {str(e)}]"    # 出错时返回错误信息（不抛出异常，保证接口可用）

    async def _anthropic_generate(self, prompt: str) -> str:
        """调用 Anthropic Claude API 生成答案"""
        client = self._get_anthropic()
        if client is None:
            return "[Anthropic API key not configured]"

        try:
            resp = await client.messages.create(
                model=settings.ANTHROPIC_MODEL,        # Claude 模型版本
                system=BILL_SYSTEM_PROMPT,             # 系统提示（Anthropic API 单独传）
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            return resp.content[0].text   # 取第一个内容块的文本
        except Exception as e:
            logger.error(f"Anthropic generate error: {e}")
            return f"[生成失败: {str(e)}]"

    async def _generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        流式 LLM 调用（逐步返回生成的文字）
        使用 Python 的 async generator：每生成一小段就 yield 出去
        """
        provider = settings.LLM_PROVIDER
        logger.info(f"[_generate_stream] provider={provider} model={settings.OPENAI_MODEL if provider == 'openai' else settings.ANTHROPIC_MODEL}")  # 确认实际使用的模型

        if provider == "openai":
            client = self._get_openai()
            if client is None:  # API Key 未配置，提前返回错误
                logger.error("[_generate_stream] OpenAI client 未初始化，请检查 OPENAI_API_KEY 配置")  # 配置缺失告警
                yield "[OpenAI not configured]"
                return

            try:
                logger.debug(f"[_generate_stream] 发起 OpenAI stream 请求 model={settings.OPENAI_MODEL}")  # 请求发起
                # 创建流式请求（stream=True）
                stream = await client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": BILL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE,
                    stream=True,    # 开启流式模式
                )
                # 逐个读取流式响应的 chunk（每个 chunk 包含一小段生成的文字）
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content  # 取本次增量内容
                    if delta:                               # 非空才 yield
                        yield delta

            except Exception as e:
                logger.error(f"[_generate_stream] OpenAI 流式请求失败: {e}", exc_info=True)  # 含堆栈的错误日志
                yield f"[流式生成失败: {e}]"

        elif provider == "anthropic":
            client = self._get_anthropic()
            if client is None:  # API Key 未配置
                logger.error("[_generate_stream] Anthropic client 未初始化，请检查 ANTHROPIC_API_KEY 配置")  # 配置缺失告警
                yield "[Anthropic not configured]"
                return

            try:
                logger.debug(f"[_generate_stream] 发起 Anthropic stream 请求 model={settings.ANTHROPIC_MODEL}")  # 请求发起
                # Anthropic 的流式 API 使用上下文管理器
                async with client.messages.stream(
                    model=settings.ANTHROPIC_MODEL,
                    system=BILL_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings.LLM_MAX_TOKENS,
                ) as stream:
                    async for text in stream.text_stream:  # 逐个 token 读取
                        yield text

            except Exception as e:
                logger.error(f"[_generate_stream] Anthropic 流式请求失败: {e}", exc_info=True)  # 含堆栈的错误日志
                yield f"[流式生成失败: {e}]"

        else:
            logger.error(f"[_generate_stream] 未知 LLM_PROVIDER={provider!r}，请检查配置")  # 未知 provider 告警
            yield "[LLM provider not configured]"   # 未知的 LLM 提供商


    async def query_v2(
        self,
        tenant_id: str,
        query: str,
        top_k: int = None,
        user_id: str = None,
        bill_context: dict = None,   # 可选：已识别票据的结构化字段（用于专项检索）
        db=None,                     # 可选：AsyncSession，用于写日志（P4）
    ) -> dict:
        """
        增强版问答入口（F1 + F2）：
          F1：query 为空 → 返回友情引导，不写任何日志
          F2：query 非空 → 意图路由 → 专项/模糊检索 → 质量评估 → 答复/转人工
        返回与 QueryResponseV2 兼容的字典。
        """
        t_start = time.perf_counter()
        query_id = str(uuid.uuid4())

        # ── F1 友情引导（空 query 直接返回，不进入检索流程）────────────────────
        if not query.strip():
            return {
                "query_id": query_id,
                "answer_type": "guide_prompt",
                "answer": GUIDE_PROMPT,
                "sources": [],
                "intent_id": None,
                "route_type": None,
                "retrieval_quality": None,
                "transfer_to_human": False,
                "contact_info": None,
                "query_saved": False,
                "retrieval_ms": 0.0,
                "llm_ms": 0.0,
                "total_ms": 0.0,
            }

        logger.info(f"[rag_v2] 开始 tenant={tenant_id} query={query[:60]!r}")

        # ── 缓存查询（query_v2 支持 bill_context，将其纳入指纹避免不同票据命中同一缓存）
        query_fp = compute_query_fingerprint(query, bill_context)
        cached_v2 = await bill_cache.get_rag(tenant_id, query_fp)
        if cached_v2 is not None:
            cached_v2["query_id"] = query_id   # 替换为本次 query_id
            cached_v2["query_saved"] = False    # 缓存命中不写日志（避免重复记录）
            cached_v2["from_cache"] = True
            cached_v2["total_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
            logger.info(f"[rag_v2] 缓存命中 tenant={tenant_id} fp={query_fp[:8]}")
            return cached_v2

        # ── F2 Step1：意图识别（两阶段：关键词 → qwen-max）────────────────────
        intent_id, confidence, classify_method = await intent_router.classify(query)
        use_specialized = intent_router.should_use_specialized(intent_id, confidence)
        route_type = "specialized" if use_specialized else "fuzzy"

        logger.info(
            f"[rag_v2] 意图识别 intent={intent_id} conf={confidence:.2f} "
            f"method={classify_method} route={route_type}"
        )

        # ── F2 Step2：构造检索 query（专项场景注入 bill_context 字段）──────────
        search_query = self._build_search_query(query, intent_id, bill_context) \
            if use_specialized else query

        # ── F2 Step3：混合检索 ───────────────────────────────────────────────
        t0 = time.perf_counter()
        retrieved = vector_store.hybrid_search(
            tenant_id=tenant_id,
            query=search_query,
            top_k=top_k or settings.RETRIEVAL_TOP_K,
            rerank_top_n=settings.RERANK_TOP_N,
        )
        t_retrieval = (time.perf_counter() - t0) * 1000
        metrics.record_retrieval_time(t_retrieval, tenant_id)

        # ── F2 Step4：检索质量评估 ───────────────────────────────────────────
        quality = evaluate(retrieved)
        logger.info(
            f"[rag_v2] 检索质量 level={quality.level} top1={quality.top1_score:.3f} "
            f"hits={quality.hit_count} retrieval_ms={t_retrieval:.1f}"
        )

        # ── F2 Step5a：质量不足 → 转人工 + 写未命中日志 ─────────────────────
        if quality.level == QUALITY_INSUFFICIENT:
            await self._log_search_miss(
                db, tenant_id, query, intent_id, route_type,
                quality.top1_score, t_retrieval,
            )
            await self._log_intent(
                db, tenant_id, user_id, query, intent_id,
                confidence, classify_method, "no_result",
            )
            resp = build_transfer_response(query, intent_id)
            t_total_v2 = round((time.perf_counter() - t_start) * 1000, 1)
            metrics.record_total_time(t_total_v2, tenant_id)
            resp.update({
                "query_id": query_id,
                "intent_id": intent_id,
                "route_type": route_type,
                "retrieval_ms": round(t_retrieval, 1),
                "llm_ms": 0.0,
                "total_ms": t_total_v2,
            })
            return resp

        # ── F2 Step5b：质量充足/部分 → 生成答案 ────────────────────────────
        context_parts, sources = self._build_context(retrieved)
        context = "\n\n---\n\n".join(context_parts)

        # 专项场景：将 bill_context 的结构化字段注入 prompt，提升答案精准度
        bill_info_str = self._format_bill_context(bill_context) if bill_context else ""
        bill_section = ("## 当前票据信息\n" + bill_info_str + "\n") if bill_info_str else ""
        user_prompt = (
            bill_section
            + "## 参考文档\n" + context + "\n\n"
            + "## 用户问题\n" + query + "\n\n"
            + "请给出专业、准确的回答，引用相关文档内容时注明来源。"
        )

        t1 = time.perf_counter()
        answer = await self._generate(user_prompt)
        t_llm = (time.perf_counter() - t1) * 1000
        t_total = (time.perf_counter() - t_start) * 1000
        metrics.record_llm_time(t_llm, tenant_id)
        metrics.record_total_time(t_total, tenant_id)

        # PARTIAL 级别附加免责提示
        if quality.level == QUALITY_PARTIAL:
            answer += PARTIAL_DISCLAIMER

        # ── F2 Step6：写命中日志（QueryLog + IntentLog）──────────────────────
        await self._log_query(
            db, query_id, tenant_id, user_id, query, answer,
            sources, t_retrieval, t_llm, t_total, route_type, intent_id,
        )
        await self._log_intent(
            db, tenant_id, user_id, query, intent_id,
            confidence, classify_method, route_type,
        )

        logger.info(
            f"[rag_v2] 完成 route={route_type} llm_ms={t_llm:.1f} total_ms={t_total:.1f}"
        )
        result_v2 = {
            "query_id": query_id,
            "answer_type": "answer",
            "answer": answer,
            "sources": sources,
            "intent_id": intent_id,
            "route_type": route_type,
            "retrieval_quality": quality.level,
            "transfer_to_human": False,
            "contact_info": None,
            "query_saved": False,
            "retrieval_ms": round(t_retrieval, 1),
            "llm_ms": round(t_llm, 1),
            "total_ms": round(t_total, 1),
        }
        await bill_cache.set_rag(tenant_id, query_fp, result_v2)  # 写入 2h 缓存
        return result_v2

    # ── 内部辅助方法 ──────────────────────────────────────────────────────────

    def _build_search_query(
        self, raw_query: str, intent_id: str, bill_context: Optional[dict]
    ) -> str:
        """
        专项场景：将 bill_context 中与本意图相关的字段拼入检索 query，
        提升向量检索的召回精准度。bill_context 为 None 时退化为原始 query。
        """
        if not bill_context:
            return raw_query
        # 各意图关注的核心字段
        INTENT_FIELDS = {
            "INTENT_PRICING":        ["acceptor", "ticket_type", "due_date", "amount_numeric"],
            "INTENT_ENDORSER_CHECK": ["endorsers", "drawer", "payee"],
            "INTENT_EXPIRY":         ["due_date", "ticket_type", "issue_date"],
            "INTENT_AML":            ["amount_numeric", "drawer", "drawer_bank", "ticket_type"],
            "INTENT_ELEMENT_CHECK":  ["ticket_type", "acceptor", "issue_date", "due_date",
                                      "amount_numeric", "amount_text"],
            "INTENT_HISTORY":        ["ticket_number"],
            "INTENT_BATCH_RISK":     ["acceptor", "drawer", "amount_numeric"],
            "INTENT_PLEDGE":         ["ticket_type", "acceptor", "due_date",
                                      "amount_numeric", "endorsers"],
        }
        fields = INTENT_FIELDS.get(intent_id, [])
        extras = []
        for f in fields:
            val = bill_context.get(f)
            if val and val != "null":
                extras.append(f"{f}:{val}")
        if extras:
            return raw_query + " " + " ".join(extras)
        return raw_query

    def _build_context(self, retrieved: list) -> tuple[list[str], list[dict]]:
        """将检索结果格式化为 context 文本列表和 sources 列表。"""
        parts, sources = [], []
        for i, chunk in enumerate(retrieved):
            section = f"[{chunk.get('section_path', '')}] " if chunk.get("section_path") else ""
            parts.append(
                f"**参考文档 {i+1}** {section}(第{chunk.get('page_num', '?')}页)\n"
                f"{chunk['content']}"
            )
            sources.append({
                "chunk_id": chunk["id"],
                "document_id": chunk.get("document_id", ""),
                "section_path": chunk.get("section_path", ""),
                "page_num": chunk.get("page_num", 0),
                "chunk_type": chunk.get("chunk_type", ""),
                "rerank_score": chunk.get("rerank_score", 0),
                "content_preview": chunk["content"][:200],
            })
        return parts, sources

    def _format_bill_context(self, bill_context: dict) -> str:
        """将票据结构化字段格式化为 LLM 可读的文本块，注入专项 prompt。"""
        lines = []
        field_labels = {
            "ticket_type": "票据类型", "ticket_number": "票据号码",
            "issue_date": "出票日期", "due_date": "到期日",
            "amount_numeric": "金额（数字）", "amount_text": "大写金额",
            "drawer": "出票人", "drawer_bank": "出票人开户行",
            "acceptor": "承兑人", "payee": "收款人",
            "endorsers": "背书人列表", "drawee_bank": "付款行",
        }
        for key, label in field_labels.items():
            val = bill_context.get(key)
            if val:
                lines.append(f"- {label}：{val}")
        return "\n".join(lines)

    async def _log_intent(self, db, tenant_id, user_id, query, intent_id,
                          confidence, classify_method, route_type) -> None:
        """写 IntentLog（P4）。db 为 None 时静默跳过（如测试环境）。"""
        if db is None:
            return
        try:
            from app.models.db_models import IntentLog
            log = IntentLog(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                query=query,
                intent_id=intent_id,
                confidence=confidence,
                classify_method=classify_method,
                route_type=route_type,
            )
            db.add(log)
            await db.flush()
        except Exception as e:
            logger.warning(f"[rag_v2] IntentLog 写入失败: {e}")

    async def _log_search_miss(self, db, tenant_id, query, intent_id,
                               route_type, top1_score, retrieval_ms) -> None:
        """写 SearchMissLog（P4）。db 为 None 时静默跳过。"""
        if db is None:
            return
        try:
            from app.models.db_models import SearchMissLog
            log = SearchMissLog(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                query=query,
                intent_id=intent_id,
                route_type=route_type,
                top1_score=top1_score,
                retrieval_ms=retrieval_ms,
                transferred=True,
            )
            db.add(log)
            await db.flush()
        except Exception as e:
            logger.warning(f"[rag_v2] SearchMissLog 写入失败: {e}")

    async def _log_query(self, db, query_id, tenant_id, user_id, query, answer,
                         sources, retrieval_ms, llm_ms, total_ms,
                         route_type, intent_id, top_k_hit: bool = None) -> None:
        """写 QueryLog（P4）。db 为 None 时静默跳过。"""
        if db is None:
            return
        try:
            from app.models.db_models import QueryLog
            log = QueryLog(
                id=query_id,
                tenant_id=tenant_id,
                user_id=user_id,
                query=query,
                answer=answer,
                retrieved_chunks=sources,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
                total_ms=total_ms,
                route_type=route_type,
                intent_id=intent_id,
                top_k_hit=top_k_hit,
            )
            db.add(log)
            await db.flush()
        except Exception as e:
            logger.warning(f"[rag_v2] QueryLog 写入失败: {e}")


# ── 友情引导文案（F1）────────────────────────────────────────────────────────
GUIDE_PROMPT = """您好！我是票据业务智能顾问，可以帮您处理以下方面的问题：

📋 票据要素查询    — 票据号码、金额、出票人、到期日等信息核查
📊 贴现定价参考    — 根据承兑人和期限查询当前市场贴现利率区间
🔗 背书链合规审查  — 背书人连续性、空白背书、签章规范核验
⏰ 到期日追索咨询  — 追索权时效计算、拒付处理流程
🏦 大额交易合规    — 反洗钱报告义务、客户尽职调查要求
📜 监管法规解读    — 票据法、人行规章、银保监规定条款查询
🔍 票据风险评估    — 承兑人风险、背书人黑名单、要素矛盾检测

请问您需要了解哪方面的内容？
（也可以直接上传票据图片，我会自动识别票面信息并为您提供分析）"""


# 全局 RAG 服务实例（整个应用共享）
rag_service = RAGService()
