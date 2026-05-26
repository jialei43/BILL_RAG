# app/services/pdf_parser.py
# 票据文档解析流水线
# 把各种格式的文件（PDF/Word/Excel/图片）转换成结构化的文本元素
#
# 解析策略（三层兜底，保证表格提取质量）：
# 1. Camelot lattice：提取有框线表格（汇票要素表等有边框的表格）
# 2. Camelot stream：提取无框线表格（数据对齐但无边框的表格）
# 3. pdfplumber：最终备用（Camelot 失败时）
# 如果置信度仍低于阈值：触发 Qwen-VL 视觉模型，让 AI "看图识字"
# 最终兜底：PaddleOCR（纯文字识别，丢失表格结构）

import io            # 字节流处理（把图片字节转为可操作的文件对象）
import hashlib       # MD5 哈希计算（用于文件去重）
import json          # checkpoint 序列化
import os            # 文件删除（清理临时文件）
import platform      # 判断操作系统（macOS/Linux），选择不同的 .doc 转换工具
import subprocess    # 调用系统命令（textutil / LibreOffice）
import tempfile      # 临时文件（处理过程中的中间文件）
from dataclasses import dataclass, field, asdict   # 数据类装饰器
from pathlib import Path                   # 路径处理
from typing import Any, Optional           # 类型标注
from loguru import logger                  # 日志

# 各依赖库的可选导入（没装时优雅降级，不影响其他功能）
try:
    import fitz   # PyMuPDF：读取 PDF 文件、提取文字、渲染页面为图片
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    import camelot  # Camelot：专门提取 PDF 中的表格（比 pdfplumber 更强）
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber  # pdfplumber：另一个 PDF 解析库（Camelot 备用）
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from paddleocr import PaddleOCR  # PaddleOCR：百度开源 OCR 库，支持倾斜矫正
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

from config.settings import settings   # 导入配置


@dataclass
class ParsedElement:
    """
    一个解析出的文档元素（一段文字、一个表格、一段OCR识别结果）
    dataclass 自动生成 __init__ 方法，不用手写
    """
    content: str               # 元素的文本内容
    element_type: str          # 类型："text"=正文 / "table"=表格 / "image_ocr"=扫描识别 / "title"=标题
    page_num: int              # 在文档中的页码（从1开始）
    section_path: str = ""     # 所属章节路径（由分块器填写）
    confidence: float = 1.0   # 置信度（0-1，OCR和表格提取结果小于1）
    metadata: dict = field(default_factory=dict)   # 额外信息（如提取方法、原始置信度）
    # field(default_factory=dict)：每个实例都得到独立的空字典（不共享同一个字典对象）


@dataclass
class ParsedDocument:
    """整个文档的解析结果"""
    elements: list[ParsedElement]   # 所有解析出的元素（按文档顺序排列）
    total_pages: int                # 总页数
    file_type: str                  # 文件类型（pdf/docx/xlsx/image）
    parse_stats: dict = field(default_factory=dict)  # 解析统计（表格数量、OCR页数等）

    def save_checkpoint(self, path: str) -> None:
        """将解析结果序列化为 JSON checkpoint 文件，使用临时文件原子写入防止半写。"""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在，parents=True 递归创建
        tmp = dest.with_suffix(".tmp")                   # 先写到同目录的 .tmp 临时文件
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False)
                # asdict：把 dataclass 递归转为普通字典（包含嵌套的 ParsedElement 列表）
                # ensure_ascii=False：允许直接写中文，不转义为 \uXXXX
            tmp.replace(dest)   # 原子替换：rename 在同一文件系统内是原子操作，
                                # 进程崩溃时要么旧文件完整，要么新文件完整，不会出现半写状态
        except Exception:
            tmp.unlink(missing_ok=True)  # 写入失败时清理临时文件，missing_ok=True 防止二次报错
            raise                        # 继续向上抛出，让调用方感知到失败

    @classmethod
    def load_checkpoint(cls, path: str) -> "ParsedDocument":
        """从 JSON checkpoint 文件反序列化，加载失败时抛出异常由调用方决定是否重新解析。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)                          # 读取整个 JSON 文件到内存字典
        elements = [ParsedElement(**e) for e in data["elements"]]
        # 把每个元素字典解包为 ParsedElement 对象，**e 等价于逐一传入所有字段
        return cls(
            elements=elements,                           # 重建元素列表
            total_pages=data["total_pages"],             # 原始总页数
            file_type=data["file_type"],                 # 原始文件类型
            parse_stats=data.get("parse_stats", {}),    # 解析统计，旧版 checkpoint 可能没有此字段
        )


# ──────────────────────────────────────────────────────────────────────────────
# ── OCR 实例池 ────────────────────────────────────────────────────────────────
# PaddleOCR 单实例非线程安全（Tensor 状态共享），多线程并发调用会触发
# "Tensor holds no memory" PreconditionNotMet 错误。
# 解决方案：预创建 N 个独立实例放入 Queue，并发调用时每线程从池中取一个，
# 用完归还。允许最多 N 路并发 OCR，且每个实例内部状态完全隔离。
import threading
import queue as _queue

_ocr_pool: "_queue.Queue[PaddleOCR]" = _queue.Queue()
_ocr_pool_lock = threading.Lock()   # 防止多线程同时初始化池
_ocr_pool_ready = False             # 标记池是否已初始化


def _ensure_ocr_pool():
    """懒初始化 OCR 实例池（首次调用时创建 OCR_POOL_SIZE 个实例）"""
    global _ocr_pool_ready
    if _ocr_pool_ready or not PADDLE_AVAILABLE:
        return
    with _ocr_pool_lock:
        if _ocr_pool_ready:   # double-check
            return
        pool_size = settings.OCR_POOL_SIZE
        logger.info(f"[OCR] 初始化实例池 size={pool_size}，每实例约 600MB 内存")
        for i in range(pool_size):
            engine = PaddleOCR(
                use_angle_cls=settings.OCR_USE_ANGLE_CLS,
                lang=settings.OCR_LANG,
                use_gpu=settings.OCR_USE_GPU,
                show_log=False,
            )
            _ocr_pool.put(engine)
            logger.info(f"[OCR] 实例 {i+1}/{pool_size} 初始化完成")
        _ocr_pool_ready = True
        logger.info(f"[OCR] 实例池就绪，支持 {pool_size} 路并发 OCR")


class _OcrInstance:
    """上下文管理器：从池中借出一个 OCR 实例，with 块结束后自动归还"""
    def __enter__(self) -> "PaddleOCR":
        _ensure_ocr_pool()
        if not PADDLE_AVAILABLE or _ocr_pool.empty() and not _ocr_pool_ready:
            return None
        self._engine = _ocr_pool.get()   # 阻塞直到有可用实例
        return self._engine

    def __exit__(self, *_):
        if hasattr(self, "_engine") and self._engine is not None:
            _ocr_pool.put(self._engine)  # 归还实例到池中


# ──────────────────────────────────────────────────────────────────────────────
# 表格置信度评估器
# ──────────────────────────────────────────────────────────────────────────────
class TableConfidenceEvaluator:
    """
    评估 Camelot 提取的表格质量
    如果质量太低（如大量空单元格、识别混乱），就触发视觉模型兜底
    """

    def evaluate(self, table) -> float:
        """
        计算表格置信度得分（0.0 - 1.0）
        得分越高表示提取质量越好
        """
        try:
            df = table.df          # 表格的 DataFrame 表示（pandas DataFrame）
            total_cells = df.size  # 总单元格数 = 行数 × 列数

            if total_cells == 0:
                return 0.0         # 空表格，置信度为 0

            # 惩罚项1：空单元格比例（空格越多，说明提取越不准确）
            empty_cells = (df == "").sum().sum() + df.isna().sum().sum()
            # (df == "").sum().sum()：统计空字符串的单元格数
            # df.isna().sum().sum()：统计 NaN（空值）的单元格数
            empty_ratio = empty_cells / total_cells   # 空单元格占比

            # 惩罚项2：Camelot 自带的精度评分（0-100，除以100转为0-1）
            camelot_acc = float(table.accuracy) / 100.0 if hasattr(table, "accuracy") else 0.5
            # hasattr：检查对象是否有 accuracy 属性（防止 AttributeError）

            # 惩罚项3：斜线表头检测（票据表格常见，但 Camelot 处理不好）
            has_diagonal = self._detect_diagonal_header(df)
            diagonal_penalty = 0.15 if has_diagonal else 0.0  # 有斜线表头扣 0.15 分

            # 综合得分：Camelot精度 × (1 - 空格惩罚) - 斜线惩罚
            score = camelot_acc * (1 - empty_ratio * 0.5) - diagonal_penalty
            return max(0.0, min(1.0, score))   # 钳制到 [0, 1] 范围内

        except Exception:
            return 0.0   # 评估出错，保守返回 0

    def _detect_diagonal_header(self, df) -> bool:
        """
        检测表格是否有斜线表头
        斜线表头特征：第一行包含 "/" 字符（如"产品/规格"）
        """
        if df.empty:
            return False
        first_row = df.iloc[0].astype(str)       # 取第一行，转为字符串类型
        slash_count = first_row.str.contains("/").sum()  # 统计包含 "/" 的单元格数
        return slash_count > 0   # 有任何 "/" 就认为可能有斜线表头


# ──────────────────────────────────────────────────────────────────────────────
# Camelot 三层表格提取策略
# ──────────────────────────────────────────────────────────────────────────────
class CamelotTableExtractor:
    """
    三层策略提取 PDF 表格：
    1. lattice（格子模式）：适合有明显边框线的表格
    2. stream（流模式）：适合依靠空白对齐的无边框表格
    3. pdfplumber：最终备选方案
    """

    def __init__(self):
        self.evaluator = TableConfidenceEvaluator()   # 表格质量评估器

    def extract(self, pdf_path: str, page_num: int) -> tuple[str, float, str]:
        """
        尝试三层策略提取指定页的表格
        返回：(Markdown格式的表格文本, 置信度0-1, 使用的方法名)
        如果所有方法都失败，返回 ("", 0.0, "none")
        """
        page_str = str(page_num)   # Camelot 需要字符串格式的页码

        # 第一层：lattice 模式（有框线表格，质量要求 0.8）
        result, conf = self._try_camelot(pdf_path, page_str, flavor="lattice")
        if conf >= settings.CAMELOT_LATTICE_THRESHOLD:   # 质量合格
            return result, conf, "camelot_lattice"

        # 第二层：stream 模式（无框线表格，质量要求 0.6，略低）
        result, conf = self._try_camelot(pdf_path, page_str, flavor="stream")
        if conf >= settings.CAMELOT_STREAM_THRESHOLD:
            return result, conf, "camelot_stream"

        # 第三层：pdfplumber 备选
        result, conf = self._try_pdfplumber(pdf_path, page_num)
        if result:
            return result, conf, "pdfplumber"

        return "", 0.0, "none"   # 全部失败

    def _try_camelot(self, pdf_path: str, page_str: str, flavor: str) -> tuple[str, float]:
        """尝试用 Camelot 提取表格，返回 (Markdown文本, 置信度)"""
        if not CAMELOT_AVAILABLE:
            return "", 0.0   # Camelot 未安装

        try:
            tables = camelot.read_pdf(pdf_path, pages=page_str, flavor=flavor)
            # flavor="lattice"：用边框线识别表格
            # flavor="stream"：用空白间距识别表格

            if not tables:
                return "", 0.0   # 没有找到表格

            # 找到置信度最高的表格（一页可能有多个表格）
            best_table = None
            best_conf = 0.0
            for t in tables:
                conf = self.evaluator.evaluate(t)  # 评估当前表格质量
                if conf > best_conf:
                    best_conf = conf
                    best_table = t

            if best_table is None:
                return "", 0.0

            return self._df_to_markdown(best_table.df), best_conf   # 转为 Markdown 格式

        except Exception as e:
            logger.debug(f"Camelot {flavor} failed: {e}")
            return "", 0.0

    def _try_pdfplumber(self, pdf_path: str, page_num: int) -> tuple[str, float]:
        """用 pdfplumber 提取表格（Camelot 的备选方案）"""
        if not PDFPLUMBER_AVAILABLE:
            return "", 0.0

        try:
            with pdfplumber.open(pdf_path) as pdf:   # 打开 PDF
                if page_num - 1 >= len(pdf.pages):   # 页码超出范围
                    return "", 0.0

                page = pdf.pages[page_num - 1]        # 取对应页（0-indexed）
                tables = page.extract_tables()        # 提取该页所有表格

                if not tables:
                    return "", 0.0

                # 选择最大的表格（行数×列数最多的）
                biggest = max(tables, key=lambda t: len(t) * len(t[0]) if t else 0)
                md = self._list_to_markdown(biggest)

                return md, 0.65   # pdfplumber 固定置信度 0.65（中等质量）

        except Exception as e:
            logger.debug(f"pdfplumber failed p{page_num}: {e}")
            return "", 0.0

    @staticmethod
    def _df_to_markdown(df) -> str:
        """把 pandas DataFrame 转换为 Markdown 表格格式"""
        try:
            import pandas as pd
            rows = []
            header = df.iloc[0].tolist()   # 第一行作为表头
            rows.append("| " + " | ".join(str(h) for h in header) + " |")
            rows.append("| " + " | ".join(["---"] * len(header)) + " |")
            # "---" 是 Markdown 表格的分隔行，表示这行上面是表头

            for _, row in df.iloc[1:].iterrows():   # 从第2行开始遍历数据行
                rows.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")

            return "\n".join(rows)   # 用换行连接所有行

        except Exception:
            return str(df.values.tolist())   # 转换失败，退化为列表字符串

    @staticmethod
    def _list_to_markdown(table: list) -> str:
        """把嵌套列表形式的表格转换为 Markdown 格式"""
        rows = []
        if not table:
            return ""

        rows.append("| " + " | ".join(str(c or "") for c in table[0]) + " |")  # 表头行
        rows.append("| " + " | ".join(["---"] * len(table[0])) + " |")          # 分隔行

        for row in table[1:]:  # 数据行
            rows.append("| " + " | ".join(str(c or "") for c in row) + " |")

        return "\n".join(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 视觉模型表格提取（置信度不足时的兜底）
# ──────────────────────────────────────────────────────────────────────────────
class VisionTableExtractor:
    """
    当 Camelot 置信度低于阈值时，用视觉大模型"看图识字"
    支持两种方式：
    1. Qwen-VL API（推荐）：能理解表格结构，输出 Markdown 格式
    2. PaddleOCR（兜底）：只识别文字，丢失表格结构
    """

    def __init__(self):
        self._model = None        # 本地模型实例（备用）
        self._tokenizer = None    # 本地分词器（备用）
        self._use_api = settings.LLM_PROVIDER == "openai"   # True=用 API，False=用本地模型

    def extract_from_image(self, image_bytes: bytes, prompt: str = None) -> str:
        """
        从图片中提取表格内容
        image_bytes：图片的字节数据
        prompt：告诉视觉模型要提取什么信息（默认提取票据要素）
        """
        if prompt is None:
            # 通用提取 prompt：适用于任何文档类型（合规文档、票据、合同等）
            # 不再要求"必须有票据要素"，避免合规文档因找不到票据字段而返回空结果
            prompt = (
                "请识别图片中的所有文字和表格内容，"
                "保持原始结构，表格用 Markdown 表格格式输出，"
                "普通文字按段落输出，不要遗漏任何内容。"
            )

        if self._use_api:
            return self._call_vision_api(image_bytes, prompt)    # 优先用 Qwen-VL API
        return self._call_local_model(image_bytes, prompt)       # 没有 API 则用本地

    def _call_vision_api(self, image_bytes: bytes, prompt: str) -> str:
        """
        调用 Qwen-VL API（通义千问视觉语言模型）
        如果 API 调用失败，自动降级到 PaddleOCR
        """
        import base64   # Base64 编码：把二进制图片转为可以放在 JSON 里的文本
        import openai

        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY 未配置，视觉兜底降级到 PaddleOCR")
            return self._call_local_model(image_bytes, prompt)

        client = openai.OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
        )
        b64 = base64.b64encode(image_bytes).decode()   # 图片字节 → Base64 字符串

        try:
            response = client.chat.completions.create(
                model=settings.VISION_MODEL,  # 视觉模型（qwen-vl-max）
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        # 把图片以 Base64 格式嵌入消息（data:image/png;base64,...）
                        {"type": "text", "text": prompt},   # 文字提示
                    ]
                }],
                max_tokens=1024,   # 最多输出 1024 个 token
            )
            return response.choices[0].message.content or ""   # 取 AI 的回答

        except Exception as e:
            logger.error(f"Qwen-VL API error: {e}，降级到 PaddleOCR")
            return self._call_local_model(image_bytes, prompt)  # API 失败，降级到 OCR

    def _call_local_model(self, image_bytes: bytes, prompt: str) -> str:
        """
        PaddleOCR 兜底：当 Qwen-VL API 不可用时使用
        注意：只能识别文字，无法理解表格结构，金融数字务必人工复核
        """
        try:
            import numpy as np
            from PIL import Image   # PIL：Python 图像处理库

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # BytesIO：把字节流转为"像文件一样"的对象；convert("RGB")：确保是 RGB 格式（非 RGBA）
            with _OcrInstance() as ocr:   # 从实例池借出，支持并发
                if ocr is None:
                    return ""
                result = ocr.ocr(np.array(img), cls=settings.OCR_USE_ANGLE_CLS)
            # np.array(img)：PIL 图片转 NumPy 数组（OCR 需要这种格式）

            if not result or not result[0]:
                return ""

            # 过滤低置信度的识别结果（低于 0.8 的可能是误识别）
            lines = [
                line[1][0]           # line[1][0]：识别出的文字
                for line in result[0]
                if line and len(line) >= 2 and line[1][1] >= 0.8  # line[1][1]：置信度
                # 只保留置信度 >= 80% 的识别结果
            ]
            return "\n".join(lines)   # 把所有行用换行连接

        except Exception as e:
            logger.error(f"PaddleOCR vision fallback error: {e}")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# 票据 PDF 主解析器（编排整个解析流程）
# ──────────────────────────────────────────────────────────────────────────────
class BillPDFParser:
    """
    支持多种文件格式的票据文档解析器
    对外提供统一的 parse(file_path) 接口，内部自动根据文件类型选择解析策略
    """

    def __init__(self):
        self.table_extractor = CamelotTableExtractor()  # 三层表格提取器
        self.vision_extractor = VisionTableExtractor()  # 视觉模型兜底

    def parse(self, file_path: str) -> ParsedDocument:
        """
        解析文件，根据扩展名选择对应的解析方法
        """
        suffix = Path(file_path).suffix.lower()  # 获取文件扩展名（小写）

        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self._parse_image(file_path)  # 图片文件直接 OCR
        elif suffix == ".docx":
            return self._parse_docx(file_path)
        elif suffix == ".doc":
            return self._parse_doc(file_path)   # 旧版 Word，先转 .docx 再解析
        elif suffix in (".xlsx", ".xls"):
            return self._parse_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")  # 不支持的格式

    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """
        PDF 文件解析主流程
        每页处理：判断是否扫描件 → 提取文字 → 提取表格 → 必要时触发视觉兜底
        """
        if not FITZ_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed")

        elements: list[ParsedElement] = []   # 收集所有解析出的元素

        # 解析统计字典（记录各方法的使用次数）
        stats = {
            "tables_lattice": 0,      # lattice 方法提取的表格数
            "tables_stream": 0,       # stream 方法提取的表格数
            "tables_pdfplumber": 0,   # pdfplumber 提取的表格数
            "tables_vision": 0,       # 视觉模型提取的表格数
            "text_pages": 0,          # 普通文字页数
            "ocr_pages": 0,           # OCR 识别的扫描页数
        }

        doc = fitz.open(file_path)   # 用 PyMuPDF 打开 PDF
        total_pages = len(doc)       # 总页数

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1   # 页码从 1 开始（方便用户理解）

            # ── 判断是否是扫描件 ──────────────────────────────────────────────
            text = page.get_text("text").strip()  # 提取页面所有文字（PyMuPDF 原生文字层）
            is_scanned = len(text) < 50           # 文字极少（<50字符）视为扫描件/纯图片页

            if is_scanned:
                # ── 扫描件路径：PaddleOCR 识别 ───────────────────────────────
                ocr_text = self._ocr_page(doc, page_idx)   # OCR 识别页面文字（含倾斜矫正）
                if ocr_text:
                    # OCR 识别成功，直接入库
                    elements.append(ParsedElement(
                        content=ocr_text,
                        element_type="image_ocr",           # 标记为 OCR 识别内容
                        page_num=page_num,
                        confidence=0.97,                    # PaddleOCR 整体置信度约 97%
                        metadata={"method": "paddleocr_with_angle_cls"},
                    ))
                    stats["ocr_pages"] += 1   # 统计 OCR 处理页数
                else:
                    # OCR 识别失败（低质量扫描、模糊、倾斜过大等）→ 视觉大模型兜底
                    # 确保扫描页内容不会因 OCR 失败而静默丢失
                    logger.warning(
                        f"[p{page_num}] PaddleOCR returned empty, "
                        f"falling back to vision model"
                    )
                    img_bytes = self._render_page_image(doc, page_idx)   # 渲染为高分辨率图片
                    vision_result = self.vision_extractor.extract_from_image(img_bytes)  # 视觉识别
                    if vision_result:
                        # 视觉模型成功识别出内容
                        elements.append(ParsedElement(
                            content=vision_result,
                            element_type="image_ocr",       # 与 OCR 同类型，便于后续统一处理
                            page_num=page_num,
                            confidence=0.85,                # 视觉模型置信度略低于 PaddleOCR
                            metadata={"method": "vision_ocr_fallback"},  # 标记为视觉兜底
                        ))
                        stats["ocr_pages"] += 1             # 计入 OCR 页数（最终都是文字识别）
                        stats["tables_vision"] += 1         # 同时计入视觉模型使用次数
                    else:
                        # OCR 和视觉模型都失败，页面可能是空白页或严重损坏
                        logger.warning(
                            f"[p{page_num}] Both OCR and vision failed, "
                            f"page content cannot be extracted"
                        )

            else:
                # ── 普通文字页路径 ────────────────────────────────────────────
                blocks = page.get_text("dict")["blocks"]   # 获取页面所有内容块（含位置信息）

                # 检测页面是否含有嵌入图片块（type==1）
                # 嵌入图片可能是截图表格、流程图等 PyMuPDF 无法从中提取文字的内容
                has_image_blocks = any(b.get("type") == 1 for b in blocks)

                # 提取所有文字块（type==0）
                for block in blocks:
                    if block.get("type") == 0:   # type=0 = 文字块；type=1 = 图片块
                        # 拼接该文字块内所有行、所有字体区间（span）的文字
                        block_text = " ".join(
                            span["text"]                        # 每个 span 的文字内容
                            for line in block.get("lines", [])  # 遍历行
                            for span in line.get("spans", [])   # 遍历 span（字体/颜色区间）
                        ).strip()

                        if block_text:   # 过滤空字符串
                            elements.append(ParsedElement(
                                content=block_text,
                                element_type="text",   # 普通文字
                                page_num=page_num,
                            ))
                stats["text_pages"] += 1   # 统计普通文字页数

                # ── 表格提取（三层策略）+ 嵌入图片视觉识别 ─────────────────────
                table_content, conf, method = self.table_extractor.extract(file_path, page_num)
                # table_extractor 依次尝试 camelot_lattice → camelot_stream → pdfplumber
                # 返回：(提取内容, 置信度, 使用的方法名)；无表格时返回 ("", 0.0, "none")

                if method != "none" and conf >= settings.VISION_CONFIDENCE_THRESHOLD:
                    # 置信度达标（≥ VISION_CONFIDENCE_THRESHOLD），直接采用提取结果
                    elements.append(ParsedElement(
                        content=table_content,
                        element_type="table",   # 结构化表格
                        page_num=page_num,
                        confidence=conf,
                        metadata={"method": method},   # 记录使用了哪种提取方法
                    ))
                    key = f"tables_{method.split('_')[-1]}"   # "camelot_lattice" → key "lattice"
                    stats[key] = stats.get(key, 0) + 1        # 更新对应方法的计数

                elif table_content or (method == "none" and has_image_blocks):
                    # 两种情况均需视觉兜底：
                    # ① table_content 非空但置信度不足（Camelot 抽到了但质量差）
                    # ② 未检测到表格，但页面有嵌入图片块（截图表格/流程图/图片中的文字）
                    #    → 纯文字页（无 image_blocks）不触发，避免对已完整提取的页做无效识别
                    img_bytes = self._render_page_image(doc, page_idx)   # 渲染整页为高清图片
                    vision_result = self.vision_extractor.extract_from_image(img_bytes)   # 视觉识别

                    if vision_result:
                        # 判断是 API（Qwen-VL/GPT-4o）还是 PaddleOCR 兜底
                        is_api_result = self.vision_extractor._use_api and settings.OPENAI_API_KEY
                        elements.append(ParsedElement(
                            content=vision_result,
                            # API 能理解表格结构，标记 table；OCR 只有文字，标记 image_ocr
                            element_type="table" if is_api_result else "image_ocr",
                            page_num=page_num,
                            confidence=0.90 if is_api_result else 0.70,
                            metadata={
                                "method": "qwen_vl" if is_api_result else "paddleocr_fallback",
                                "camelot_conf": conf,          # 保留 Camelot 原始置信度供分析
                                "had_image_blocks": has_image_blocks,  # 是否因图片块触发
                            },
                        ))
                        stats["tables_vision"] += 1   # 统计视觉模型使用次数

        doc.close()   # 关闭 PDF（释放文件句柄）

        return ParsedDocument(
            elements=elements,
            total_pages=total_pages,
            file_type="pdf",
            parse_stats=stats,
        )

    def _ocr_page(self, doc, page_idx: int) -> str:
        """
        用 PaddleOCR 识别扫描页（含 2× 分辨率放大 + 方向矫正）
        2× 放大：把 72DPI 的 PDF 页面放大到 144DPI，提高 OCR 准确率
        """
        try:
            page = doc[page_idx]
            mat = fitz.Matrix(settings.OCR_DPI_SCALE, settings.OCR_DPI_SCALE)
            # Matrix：变换矩阵，(2, 2) 表示宽高都放大 2 倍
            pix = page.get_pixmap(matrix=mat)   # 渲染页面为像素图
            img_bytes = pix.tobytes("png")      # 转为 PNG 格式的字节数据

            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(img_bytes))  # 字节 → PIL 图片
            img_arr = np.array(img)                  # PIL 图片 → NumPy 数组

            with _OcrInstance() as ocr:   # 从实例池借出，支持并发
                if ocr is None:
                    return ""
                result = ocr.ocr(img_arr, cls=settings.OCR_USE_ANGLE_CLS)
            # cls=True：开启文字方向分类（能处理旋转90°、180°的文字）

            if not result or not result[0]:
                return ""

            # 提取所有识别出的文字行
            lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]   # line 的格式：[坐标框, [文字, 置信度]]
                    lines.append(text)

            return "\n".join(lines)   # 按行连接

        except Exception as e:
            logger.error(f"OCR error page {page_idx}: {e}")
            return ""

    def _render_page_image(self, doc, page_idx: int) -> bytes:
        """
        把 PDF 页面渲染为高分辨率图片（供视觉模型处理）
        """
        page = doc[page_idx]
        mat = fitz.Matrix(settings.OCR_DPI_SCALE, settings.OCR_DPI_SCALE)  # 2× 放大
        pix = page.get_pixmap(matrix=mat)   # 渲染
        return pix.tobytes("png")           # 返回 PNG 字节

    def _parse_image(self, file_path: str) -> ParsedDocument:
        """
        直接解析图片文件（如扫描版汇票图片）
        完整图片 → OCR → 文字结果
        """
        try:
            with open(file_path, "rb") as f:   # rb=二进制读取
                img_bytes = f.read()            # 读取整个图片文件

            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 打开图片并转为 RGB
            img_arr = np.array(img)

            with _OcrInstance() as ocr:   # 从实例池借出，支持并发
                if ocr is None:
                    return ParsedDocument(elements=[], total_pages=1, file_type="image")
                result = ocr.ocr(img_arr, cls=True)   # 执行 OCR（cls=True 开启方向矫正）
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_lines.append(line[1][0])   # 提取文字

            # 把所有文字合并为一个元素
            elements = [ParsedElement(
                content="\n".join(text_lines),
                element_type="image_ocr",
                page_num=1,
                confidence=0.97,
                metadata={"method": "paddleocr"},
            )] if text_lines else []   # 没有识别到文字时返回空列表

            return ParsedDocument(elements=elements, total_pages=1, file_type="image")

        except Exception as e:
            logger.error(f"Image parse error: {e}")
            return ParsedDocument(elements=[], total_pages=1, file_type="image")

    def _parse_doc(self, file_path: str) -> ParsedDocument:
        """
        解析旧版 Word 文档（.doc 格式）
        策略：先用系统工具把 .doc 转换为 .docx，再复用 _parse_docx 解析。
          - macOS：textutil（系统内置，无需安装）
          - Linux：LibreOffice headless（需提前安装 libreoffice）
        转换生成的临时 .docx 文件在解析完毕后自动删除。
        """
        tmp_docx = None  # 记录临时文件路径，确保 finally 能清理
        try:
            # 创建临时 .docx 文件路径（与原文件同目录，避免跨设备移动问题）
            tmp_docx = str(Path(file_path).with_suffix(".tmp_converted.docx"))

            if platform.system() == "Darwin":
                # macOS 内置 textutil，直接转换，无需任何依赖安装
                result = subprocess.run(
                    ["textutil", "-convert", "docx", "-output", tmp_docx, file_path],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"textutil 转换失败: {result.stderr.decode(errors='replace')}"
                    )
            else:
                # Linux/其他系统：使用 LibreOffice headless 转换
                # 安装：apt install libreoffice 或 yum install libreoffice
                out_dir = str(Path(tmp_docx).parent)
                result = subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "docx",
                     "--outdir", out_dir, file_path],
                    capture_output=True, timeout=60,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"LibreOffice 转换失败: {result.stderr.decode(errors='replace')}"
                    )
                # LibreOffice 输出文件名 = 原文件名.docx（忽略 --output 参数）
                lo_output = Path(out_dir) / (Path(file_path).stem + ".docx")
                if lo_output.exists():
                    lo_output.rename(tmp_docx)  # 统一重命名到 tmp_docx 路径
                else:
                    raise RuntimeError(f"LibreOffice 未生成预期输出文件: {lo_output}")

            logger.info(f"[_parse_doc] .doc → .docx 转换成功: {Path(file_path).name}")
            parsed = self._parse_docx(tmp_docx)   # 复用已有的 docx 解析逻辑
            parsed.file_type = "doc"               # 标记原始文件类型为 doc
            return parsed

        except Exception as e:
            logger.error(f"DOC parse error: {e}")
            return ParsedDocument(elements=[], total_pages=1, file_type="doc")
        finally:
            if tmp_docx and os.path.exists(tmp_docx):
                os.remove(tmp_docx)  # 无论成败都清理临时文件，避免磁盘泄漏

    def _parse_docx(self, file_path: str) -> ParsedDocument:
        """
        解析 Word 文档（.docx 格式）
        提取段落文字和表格
        """
        try:
            from docx import Document as DocxDocument   # python-docx 库
            import pandas as pd

            doc = DocxDocument(file_path)   # 打开 Word 文档
            elements = []

            # 提取所有段落
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():   # 跳过空段落
                    # para.style 在部分 DOCX 文件中可能为 None（段落未设置样式）
                    style_name = para.style.name if para.style else "Normal"
                    elements.append(ParsedElement(
                        content=para.text.strip(),
                        element_type="text",
                        page_num=1,   # Word 文档没有固定页码概念，统一用 1
                        metadata={"style": style_name},   # 段落样式（标题1、正文等）
                    ))

            # 提取所有表格
            for table in doc.tables:
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                # 双重列表推导：表格行 → 每行的单元格文本
                if rows:
                    md = CamelotTableExtractor._list_to_markdown(rows)  # 转为 Markdown
                    elements.append(ParsedElement(
                        content=md,
                        element_type="table",
                        page_num=1,
                    ))

            return ParsedDocument(elements=elements, total_pages=1, file_type="docx")

        except Exception as e:
            logger.error(f"DOCX parse error: {e}")
            return ParsedDocument(elements=[], total_pages=1, file_type="docx")

    def _parse_excel(self, file_path: str) -> ParsedDocument:
        """
        解析 Excel 文件（.xlsx/.xls 格式）
        每个工作表作为一个表格元素
        """
        try:
            import pandas as pd

            # .xlsx 用 openpyxl 引擎，.xls 用 xlrd 引擎（两者不可互换）
            ext = Path(file_path).suffix.lower()
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"

            try:
                xf = pd.ExcelFile(file_path, engine=engine)  # 明确指定引擎，避免自动检测出错
            except ImportError:
                # xlrd 未安装时给出明确提示，而不是让 pandas 的错误信息透传
                logger.error(
                    f"Excel parse error: 解析 {ext} 格式需要安装 xlrd，"
                    f"请执行: pip install 'xlrd>=2.0.1'"
                )
                return ParsedDocument(elements=[], total_pages=1, file_type="excel")

            elements = []
            for sheet_name in xf.sheet_names:   # 遍历所有工作表
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
                md = df.to_markdown(index=False)   # 转为 Markdown 格式（index=False 不显示行号）
                elements.append(ParsedElement(
                    content=f"## 工作表: {sheet_name}\n\n{md}",  # 加上工作表名作为标题
                    element_type="table",
                    page_num=1,
                    metadata={"sheet": sheet_name},  # 记录工作表名
                ))

            return ParsedDocument(elements=elements, total_pages=1, file_type="excel")

        except Exception as e:
            logger.error(f"Excel parse error: {e}")
            return ParsedDocument(elements=[], total_pages=1, file_type="excel")


def compute_md5(file_path: str) -> str:
    """
    计算文件的 MD5 哈希值（32位十六进制字符串）
    用于文档去重：相同内容的文件 MD5 相同，避免重复入库浪费存储
    分块读取（每次8KB）避免把大文件整个加载进内存
    """
    h = hashlib.md5()                           # 创建 MD5 哈希计算器
    with open(file_path, "rb") as f:            # 以二进制模式读取文件
        for chunk in iter(lambda: f.read(8192), b""):
            # iter(callable, sentinel)：不断调用 callable，直到返回值等于 sentinel（b"" 表示文件结尾）
            # f.read(8192)：每次读取 8KB（8192字节）
            h.update(chunk)                     # 把这一块数据喂给哈希计算器
    return h.hexdigest()                        # 返回最终的 MD5 字符串（32位十六进制）
