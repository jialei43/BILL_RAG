# app/services/bill_recognition.py
# 票据要素结构化识别服务
# 与通用文档解析不同，此模块专注于从票据图片中提取标准化要素
#
# 核心能力：
# - 接受图片（PNG/JPG/TIFF）或 PDF 文件
# - 调用视觉大模型，输出 JSON 格式的结构化票据字段
# - 一次可识别多张背书页（PDF 多页）
# - 字段包括：票据类型、票号、出票日期、到期日、金额、出票人、承兑人、背书人等

import io           # 字节流处理，把字节数据包装成"像文件一样"可读的对象
import json         # 解析视觉模型返回的 JSON 字符串
import time         # 计时（记录识别耗时）
import base64       # Base64 编码，把二进制图片转为 API 可接受的文本格式
from dataclasses import dataclass, field   # 数据类，自动生成 __init__ 等方法
from pathlib import Path                   # 跨平台路径处理
from typing import Optional                # 可选类型标注

from loguru import logger   # 日志

from config.settings import settings   # 系统配置（视觉模型地址、密钥等）
from app.core.cache import compute_file_fingerprint  # 文件指纹计算（无 async 依赖）


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构：票据要素
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BillElement:
    """
    一张票据的所有结构化要素
    dataclass：自动生成 __init__，无需手写构造函数
    所有字段均为可选，识别失败时保持 None
    """
    is_bill_found: bool = True             # 是否在图片中找到票据
    ticket_type: Optional[str] = None      # 票据类型：银行承兑汇票 / 商业承兑汇票 / 本票等
    ticket_number: Optional[str] = None    # 票据号码（通常印在票面右上角或下方）
    issue_date: Optional[str] = None       # 出票日期（格式：YYYY-MM-DD 或原始格式）
    due_date: Optional[str] = None         # 到期日（付款期限）
    amount_numeric: Optional[float] = None # 票面金额（数字形式，便于计算）
    amount_text: Optional[str] = None      # 大写金额（如：壹佰万元整）
    currency: Optional[str] = None         # 币种（人民币 / USD 等，默认人民币）
    drawer: Optional[str] = None           # 出票人名称（开票方）
    drawer_account: Optional[str] = None   # 出票人银行账号
    drawer_bank: Optional[str] = None      # 出票人开户行
    acceptor: Optional[str] = None         # 承兑人（银行承兑汇票为银行，商业承兑为出票人自己）
    payee: Optional[str] = None            # 收款人（第一手持票人）
    endorsers: list = field(default_factory=list)  # 背书人列表（按背书顺序排列）
    drawee_bank: Optional[str] = None      # 付款行（承兑行）


@dataclass
class RecognitionResult:
    """
    票据识别的完整结果
    一个 PDF 可能包含多页（正面 + 背书页），每页可能有一张票据
    """
    bills: list = field(default_factory=list)   # 识别出的所有票据要素列表（BillElement）
    raw_texts: list = field(default_factory=list)  # 每页的原始视觉识别文本（调试用）
    elapsed_ms: float = 0.0                     # 总识别耗时（毫秒）
    model_used: str = ""                        # 实际使用的模型名称
    page_count: int = 0                         # 处理的页数/图片数量


# ──────────────────────────────────────────────────────────────────────────────
# 票据识别服务
# ──────────────────────────────────────────────────────────────────────────────
class BillRecognitionService:
    """
    票据要素识别服务
    核心方法：recognize_file()
    调用流程：文件读取 → 渲染为图片 → 视觉 API 提取 → JSON 解析 → 结构化返回
    """

    # 要求视觉模型严格按此 JSON schema 返回，避免自由格式难以解析
    _BILL_EXTRACT_PROMPT = """请仔细识别图片中的票据，提取所有要素并以 **严格 JSON 格式** 返回，不要有任何额外说明文字。

返回格式：
{
  "is_bill_found": true,
  "ticket_type": "银行承兑汇票",
  "ticket_number": "票据号码",
  "issue_date": "出票日期（YYYY-MM-DD 或原始格式）",
  "due_date": "到期日（YYYY-MM-DD 或原始格式）",
  "amount_numeric": 1000000.00,
  "amount_text": "大写金额",
  "currency": "人民币",
  "drawer": "出票人名称",
  "drawer_account": "出票人账号",
  "drawer_bank": "出票人开户行",
  "acceptor": "承兑人名称",
  "payee": "收款人名称",
  "endorsers": ["背书人1", "背书人2"],
  "drawee_bank": "付款行名称"
}

如果图片中没有票据，返回：{"is_bill_found": false}
所有字段如果在票据上找不到，设置为 null。"""

    def _get_sync_redis(self):
        """
        获取同步 Redis 客户端（recognize_file 是同步方法，不能 await 异步客户端）
        与 rate_limiter 的 Redis 实例独立，避免跨模块共享状态引发竞态。
        """
        if not settings.CACHE_ENABLED:
            return None
        try:
            import redis as _redis
            r = _redis.from_url(settings.REDIS_URL, decode_responses=True,
                                socket_connect_timeout=1)
            r.ping()   # 快速验证连接可用（1 秒超时自动失败）
            return r
        except Exception:
            return None   # 连接失败时返回 None，后续缓存操作静默跳过

    def recognize_file(self, file_bytes: bytes, filename: str) -> RecognitionResult:
        """
        从上传的文件字节中识别票据要素
        file_bytes：文件的原始字节内容
        filename：原始文件名（用于判断文件类型）
        返回：RecognitionResult 包含所有识别到的票据
        """
        t_start = time.perf_counter()   # 记录开始时间，用于计算总耗时

        # ── 缓存查询：同一文件内容只调用一次视觉模型 ─────────────────────────────
        # 使用文件内容 MD5 作为 Key：相同物理文件 MD5 完全一致
        file_md5 = compute_file_fingerprint(file_bytes)
        r = self._get_sync_redis()
        if r is not None:
            try:
                raw = r.get(f"recognition:{file_md5}")
                if raw:
                    cached = json.loads(raw)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    logger.info(
                        f"BillRecognition: 缓存命中 md5={file_md5[:8]} "
                        f"bills={len(cached.get('bills', []))} elapsed_ms={elapsed_ms:.1f}ms(cached)"
                    )
                    bills = [BillElement(**b) for b in cached.get("bills", [])]
                    return RecognitionResult(
                        bills=bills,
                        raw_texts=cached.get("raw_texts", []),
                        elapsed_ms=elapsed_ms,
                        model_used=cached.get("model_used", settings.VISION_MODEL) + "(cached)",
                        page_count=cached.get("page_count", 0),
                    )
            except Exception as e:
                logger.debug(f"BillRecognition: 缓存读取跳过: {e}")

        suffix = Path(filename).suffix.lower()   # 提取文件后缀（小写），用于类型判断

        if suffix == ".pdf":
            # PDF 文件：逐页渲染为图片再识别
            image_list = self._pdf_to_images(file_bytes)   # PDF 各页转为 PNG 字节列表
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
            # 图片文件：直接包装成单元素列表
            image_list = [file_bytes]   # 只有一张图
        else:
            # 不支持的文件类型，返回空结果
            logger.warning(f"BillRecognition: unsupported file type: {suffix}")
            return RecognitionResult()   # 所有字段默认值，bills 为空

        bills = []        # 收集每页识别出的票据要素
        raw_texts = []    # 收集每页的原始文本（调试用）

        for i, img_bytes in enumerate(image_list):   # 逐页处理
            logger.info(f"BillRecognition: processing page {i + 1}/{len(image_list)}")
            raw_text = self._call_vision(img_bytes)   # 调用视觉模型，返回原始 JSON 字符串
            raw_texts.append(raw_text)                # 保存原始文本

            bill = self._parse_bill_json(raw_text)    # 尝试解析 JSON 为 BillElement
            if bill is not None:                      # 解析成功
                bills.append(bill)                    # 加入结果列表

        elapsed_ms = (time.perf_counter() - t_start) * 1000   # 计算总耗时（毫秒）
        logger.info(
            f"BillRecognition: done, {len(bills)} bills found "
            f"in {len(image_list)} pages, {elapsed_ms:.0f}ms"
        )

        result = RecognitionResult(
            bills=bills,               # 所有识别出的票据
            raw_texts=raw_texts,       # 各页原始文本
            elapsed_ms=round(elapsed_ms, 1),   # 保留 1 位小数
            model_used=settings.VISION_MODEL,  # 记录使用的模型
            page_count=len(image_list),        # 处理的总页数
        )

        # ── 写入识别缓存（识别成功时才缓存，避免缓存空结果）───────────────────────
        if bills and r is not None:
            try:
                import dataclasses
                cache_data = {
                    "bills": [
                        dataclasses.asdict(b) for b in bills   # dataclass 转 dict
                    ],
                    "raw_texts": raw_texts,
                    "model_used": settings.VISION_MODEL,
                    "page_count": len(image_list),
                }
                r.set(
                    f"recognition:{file_md5}",
                    json.dumps(cache_data, ensure_ascii=False, default=str),
                    ex=settings.CACHE_RECOGNITION_TTL,   # TTL=24h
                )
                logger.debug(f"BillRecognition: 缓存已写入 md5={file_md5[:8]}")
            except Exception as e:
                logger.debug(f"BillRecognition: 缓存写入跳过: {e}")

        return result

    def _pdf_to_images(self, pdf_bytes: bytes) -> list[bytes]:
        """
        把 PDF 的每一页渲染成 PNG 图片字节列表
        2× 分辨率放大，提高视觉模型识别准确率（低分辨率容易漏识）
        """
        try:
            import fitz   # PyMuPDF：PDF 渲染库
        except ImportError:
            logger.error("PyMuPDF not installed, cannot convert PDF to images")
            return []   # 没有 PyMuPDF 返回空列表

        images = []                                  # 收集每页的 PNG 字节
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # 从字节流打开 PDF（不需要磁盘文件）
        for page_idx in range(len(doc)):             # 遍历每一页
            page = doc[page_idx]
            mat = fitz.Matrix(2.0, 2.0)              # 2× 放大矩阵（144DPI，比默认 72DPI 清晰）
            pix = page.get_pixmap(matrix=mat, alpha=False)   # 渲染为像素图（不含透明通道）
            images.append(pix.tobytes("png"))        # 转为 PNG 字节并追加
        doc.close()   # 关闭文档，释放内存
        return images   # 返回所有页的图片字节列表

    def _call_vision(self, image_bytes: bytes) -> str:
        """
        调用视觉大模型识别图片中的票据要素
        优先使用配置的 API（Qwen-VL/GPT-4o），失败时返回空字符串
        """
        if not settings.OPENAI_API_KEY:
            # API 密钥未配置，无法调用视觉模型
            logger.warning("BillRecognition: OPENAI_API_KEY not set, cannot call vision model")
            return ""   # 返回空字符串，上层会解析为识别失败

        try:
            import openai   # OpenAI 兼容客户端（也支持通义千问等兼容接口）

            client = openai.OpenAI(
                api_key=settings.OPENAI_API_KEY,              # API 密钥
                base_url=settings.OPENAI_BASE_URL or None,    # 自定义接口地址（通义千问等）
            )

            b64 = base64.b64encode(image_bytes).decode()   # 图片字节 → Base64 字符串

            response = client.chat.completions.create(
                model=settings.VISION_MODEL,   # 视觉模型（如 qwen-vl-max / gpt-4o）
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                            # 以 data URI 格式把图片内联到消息中
                        },
                        {
                            "type": "text",
                            "text": self._BILL_EXTRACT_PROMPT,   # 要求返回 JSON 格式的票据要素
                        },
                    ]
                }],
                max_tokens=2048,       # 票据字段较多，给够 token 空间
                temperature=0.0,       # 温度为 0：最确定性的输出，避免模型自由发挥
            )

            result = response.choices[0].message.content or ""   # 取模型返回的文字
            logger.debug(f"BillRecognition vision raw: {result[:200]}")   # 打印前 200 字调试
            return result   # 返回原始文本（可能含 JSON，也可能有额外说明）

        except Exception as e:
            logger.error(f"BillRecognition vision API error: {e}")
            return ""   # API 调用失败，返回空字符串

    def _parse_bill_json(self, raw_text: str) -> Optional[BillElement]:
        """
        从视觉模型的原始输出中解析 JSON 票据要素
        模型输出可能包含 JSON 之外的说明文字，需要先提取 JSON 部分
        """
        if not raw_text.strip():
            # 空字符串直接返回 None（表示识别失败）
            return None

        # 尝试从文本中提取 JSON 块（模型可能在 JSON 前后加说明文字）
        json_str = self._extract_json_block(raw_text)   # 从文本中提取 { ... } 块

        if not json_str:
            # 找不到 JSON，说明模型返回格式不对，记录警告
            logger.warning(f"BillRecognition: no JSON found in model output: {raw_text[:100]}")
            return BillElement(is_bill_found=False)   # 返回"未找到票据"的对象

        try:
            data = json.loads(json_str)   # 解析 JSON 字符串为 Python 字典
        except json.JSONDecodeError as e:
            logger.warning(f"BillRecognition: JSON parse error: {e}, raw: {json_str[:100]}")
            return BillElement(is_bill_found=False)   # JSON 格式错误，返回失败对象

        if not data.get("is_bill_found", True):
            # 模型明确说图片中没有票据
            return BillElement(is_bill_found=False)

        # 从字典中安全提取每个字段（字段缺失时保持 None）
        return BillElement(
            is_bill_found=True,
            ticket_type=data.get("ticket_type"),         # 票据类型
            ticket_number=data.get("ticket_number"),     # 票号
            issue_date=data.get("issue_date"),           # 出票日期
            due_date=data.get("due_date"),               # 到期日
            amount_numeric=self._parse_float(data.get("amount_numeric")),  # 金额（数字）
            amount_text=data.get("amount_text"),         # 大写金额
            currency=data.get("currency", "人民币"),      # 币种，默认人民币
            drawer=data.get("drawer"),                   # 出票人
            drawer_account=data.get("drawer_account"),  # 出票人账号
            drawer_bank=data.get("drawer_bank"),         # 出票人开户行
            acceptor=data.get("acceptor"),               # 承兑人
            payee=data.get("payee"),                     # 收款人
            endorsers=data.get("endorsers") or [],       # 背书人列表，None 时转为空列表
            drawee_bank=data.get("drawee_bank"),         # 付款行
        )

    def _extract_json_block(self, text: str) -> str:
        """
        从可能包含说明文字的字符串中提取第一个完整的 JSON 对象
        策略：找第一个 '{' 和最后一个 '}' 之间的内容（包括嵌套花括号）
        """
        start = text.find("{")   # 找第一个左花括号
        end = text.rfind("}")    # 找最后一个右花括号（rfind = 从右向左找）
        if start == -1 or end == -1 or start >= end:
            return ""   # 没有找到完整的 {} 对，返回空字符串
        return text[start: end + 1]   # 切出 JSON 部分（含首尾括号）

    def _parse_float(self, value) -> Optional[float]:
        """
        安全地把金额值转为 float
        视觉模型可能返回字符串形式（"1,000,000.00"）或数字
        """
        if value is None:
            return None   # None 直接透传
        try:
            if isinstance(value, str):
                clean = value.replace(",", "").replace("，", "").strip()
                # 去掉千位分隔符（英文逗号和中文逗号）再转浮点数
                return float(clean)
            return float(value)   # 数字类型直接转 float
        except (ValueError, TypeError):
            logger.warning(f"BillRecognition: cannot parse amount: {value!r}")
            return None   # 转换失败返回 None，不抛异常


# 全局单例（整个应用共享，避免重复初始化）
bill_recognition_service = BillRecognitionService()
