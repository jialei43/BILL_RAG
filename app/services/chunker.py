# app/services/chunker.py
# 中文语义分块器
# 核心问题：一篇几十页的 PDF 不能整体输入向量模型（有 token 长度限制）
# 解决方案：把文档切成小块（chunk），每块单独向量化，再存入向量库
#
# 特色：
# - 按语义边界切割（优先在句号、段落处切，而不是机械按字数切）
# - 自动识别章节标题，给每块加上"章节路径"前缀（方便检索时了解上下文）

import re            # 正则表达式，用于检测章节标题
from dataclasses import dataclass  # dataclass：自动生成 __init__、__repr__ 等方法的装饰器
from typing import Optional        # 可选类型
from loguru import logger          # 日志

from config.settings import settings                             # 配置
from app.services.pdf_parser import ParsedElement, ParsedDocument  # 解析结果数据结构


@dataclass
class TextChunk:
    """一个文档片段（chunk）的数据结构"""
    content: str          # 片段文本内容（含章节路径前缀）
    section_path: str     # 所属章节路径（如"第一条 总则"）
    chunk_index: int      # 在文档中的序号（第几块，从0开始）
    page_num: int         # 所在页码
    chunk_type: str       # 类型："text"=正文 / "table"=表格 / "image_ocr"=扫描识别文字
    token_count: int      # 估算的 token 数量（用于控制块大小）
    metadata: dict        # 其他元数据（如提取方法、置信度）


# ── 章节标题检测正则表达式 ────────────────────────────────────────────────────
# 这些正则用于识别票据法规文件中的各级标题
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千\d]+条\s*"),
    # 匹配"第X条"格式：如"第一条"、"第23条"
    re.compile(r"^[一二三四五六七八九十]+、"),
    # 匹配"一、二、三、"格式的一级列表
    re.compile(r"^（[一二三四五六七八九十]+）"),
    # 匹配"（一）（二）"格式的括号序号
    re.compile(r"^\d+\.\s+"),
    # 匹配"1. 2. 3."格式的数字列表
    re.compile(r"^第[一二三四五六七八九十百千\d]+章\s*"),
    # 匹配"第X章"格式
    re.compile(r"^(总则|分则|附则|附录|附件)\s*"),
    # 匹配固定关键词章节名
]

# ── 中文分割符，按优先级从高到低排列 ──────────────────────────────────────────
# 优先在语义完整的地方切割，最不得已才按单个字符切
SPLIT_PRIORITIES = [
    "\n\n",           # 最高优先级：空行（段落分隔，语义最完整）
    "。\n",           # 句号+换行（一个完整句子结束）
    "；\n",           # 分号+换行
    "。",             # 句号（一个完整句子）
    "；",             # 分号（一个语义单元）
    "，",             # 逗号（较小的语义单元）
    "\n",             # 换行（最后的语义边界）
]


def count_tokens(text: str) -> int:
    """
    估算文本的 token 数量
    简化规则：一个中文字符 ≈ 1 token，一个英文单词 ≈ 1 token
    （真实 tokenizer 更复杂，但这个估算足够用于分块控制）
    """
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    # [一-鿿]：中文字符的 Unicode 范围，findall 找出所有中文字符
    english_words = len(re.findall(r"[a-zA-Z0-9]+", text))
    # [a-zA-Z0-9]+：连续的英文字母或数字（算作一个词）
    return chinese_chars + english_words


def detect_section(text: str) -> Optional[str]:
    """
    检测一段文本是否是章节标题
    如果是，返回标题的前 30 个字符（用作 section_path）
    如果不是，返回 None
    """
    text = text.strip()               # 去掉首尾空白
    for pat in CHAPTER_PATTERNS:      # 逐个正则尝试匹配
        m = pat.match(text)           # match：从文本开头开始匹配
        if m:
            return text[:min(30, len(text))].strip()  # 截取前 30 字符作为标题
    return None                       # 没有匹配，不是标题


class ChineseSemanticChunker:
    """
    中文语义分块器
    核心功能：把解析出的文档元素智能切成适合向量化的小块
    """

    def __init__(
        self,
        max_tokens: int = None,      # 每块最大 token 数（None 用配置默认值）
        overlap_tokens: int = None,  # 相邻块重叠的 token 数（防止语义截断）
        min_tokens: int = None,      # 每块最小 token 数（太短的块直接丢弃）
    ):
        self.max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS      # 默认 512
        self.overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS  # 默认 64
        self.min_tokens = min_tokens or settings.CHUNK_MIN_TOKENS      # 默认 50
        logger.debug(  # 初始化参数日志，排查"块太大/太小"时先看这里
            f"[chunker] 初始化 max_tokens={self.max_tokens} "
            f"overlap={self.overlap_tokens} min={self.min_tokens}"
        )

    def chunk_document(self, parsed_doc: ParsedDocument) -> list[TextChunk]:
        """
        对整个文档进行分块
        parsed_doc：PDF/Word/Excel 解析后的结构化数据
        返回：所有文档片段的列表
        """
        logger.info(  # 分块入口：记录元素总数，用于与分块结果数对比
            f"[chunker] 开始分块 elements={len(parsed_doc.elements)} "
            f"file_type={parsed_doc.file_type}"
        )
        chunks: list[TextChunk] = []  # 收集所有块
        current_section = ""          # 当前所在章节（随着遍历文档元素动态更新）
        chunk_idx = 0                 # 块序号计数器

        for element in parsed_doc.elements:  # 遍历文档的每个元素（文字段、表格、OCR文字）
            # 表格和 OCR 扫描文字：直接作为完整的一块，不再切分
            if element.element_type in ("table", "image_ocr"):
                section_prefixed = self._inject_prefix(element.content, current_section)
                # 给内容加上章节前缀（如"[章节：第三章 贴现业务] | 出票日期 | 到期日 |"）
                chunk = TextChunk(
                    content=section_prefixed,
                    section_path=current_section,
                    chunk_index=chunk_idx,
                    page_num=element.page_num,
                    chunk_type=element.element_type,
                    token_count=count_tokens(section_prefixed),
                    metadata=element.metadata,
                )
                chunks.append(chunk)
                chunk_idx += 1
                continue               # 跳到下一个元素，不执行后面的文字处理逻辑

            # 文字元素：先检测是否是章节标题
            detected = detect_section(element.content)
            if detected:               # 发现章节标题，更新当前章节路径
                current_section = detected

            # 对文字内容进行语义分块
            text_chunks = self._split_text(element.content)
            for tc in text_chunks:
                if count_tokens(tc) < self.min_tokens:  # 太短的块直接丢弃
                    continue

                section_prefixed = self._inject_prefix(tc, current_section)
                chunk = TextChunk(
                    content=section_prefixed,
                    section_path=current_section,
                    chunk_index=chunk_idx,
                    page_num=element.page_num,
                    chunk_type="text",
                    token_count=count_tokens(section_prefixed),
                    metadata=element.metadata,
                )
                chunks.append(chunk)
                chunk_idx += 1

        # 按类型统计，方便排查"为什么 chunk 数和预期不符"
        type_counts = {}
        for c in chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
        logger.info(  # 分块结果：总数 + 类型分布
            f"[chunker] 分块完成 total={len(chunks)} distribution={type_counts}"
        )
        return chunks

    def _inject_prefix(self, text: str, section_path: str) -> str:
        """
        给文本注入章节路径前缀
        目的：让每个块都携带"它属于哪个章节"的信息，方便用户定位原文
        例如：section_path="第三条 贴现利率"，text="贴现利率按市场报价执行"
        结果："[章节：第三条 贴现利率] 贴现利率按市场报价执行"
        """
        if section_path and not text.startswith(section_path):
            # 只在有章节路径且文本不是以章节路径开头时才加前缀（避免重复）
            return f"[章节：{section_path}] {text}"
        return text   # 不需要加前缀，原样返回

    def _split_text(self, text: str) -> list[str]:
        """
        递归语义分块：按优先级找分隔符，把长文本切成合适大小的块
        """
        text = text.strip()            # 去掉首尾空白

        if not text:
            return []                  # 空文本，返回空列表

        if count_tokens(text) <= self.max_tokens:
            return [text]              # 文本够短，不需要切割，直接返回

        # 按优先级尝试各个分隔符
        for sep in SPLIT_PRIORITIES:
            parts = text.split(sep)    # 用分隔符切分
            if len(parts) > 1:         # 成功切出多块
                return self._merge_parts(parts, sep)  # 合并成不超过 max_tokens 的块

        # 所有分隔符都没找到（非常罕见），最后手段：强制按字符数截断
        logger.warning(  # 字符级截断是最差情况，可能在词语中间断开，需关注
            f"[chunker] 无语义分隔符，降级为字符截断 text_len={len(text)}"
        )
        return self._char_split(text)

    def _merge_parts(self, parts: list[str], sep: str) -> list[str]:
        """
        把切分出的部分重新合并成合适大小的块（含重叠处理）
        例如：
          parts = ["A", "B", "C", "D"]，max_tokens=10
          如果 A+B<=10，A+B+C>10：则第一块=A+B，第二块从重叠部分+C开始
        """
        chunks = []          # 最终结果
        current = ""         # 当前正在累积的块
        overlap_buffer = ""  # 重叠缓冲区（上一块末尾的内容，用于下一块的开头）

        for part in parts:
            # 尝试把当前 part 加入正在累积的块
            candidate = current + sep + part if current else part
            if count_tokens(candidate) <= self.max_tokens:
                current = candidate        # 没超限，继续累积
            else:
                if current:
                    chunks.append(current)  # 已有内容，先保存当前块
                    # 计算重叠内容：取当前块末尾的 overlap_tokens 个 token
                    overlap_buffer = self._tail_tokens(current, self.overlap_tokens)
                # 新块从重叠内容开始（保证语义连续性）
                current = overlap_buffer + sep + part if overlap_buffer else part
                overlap_buffer = ""        # 清空重叠缓冲区

        if current:
            chunks.append(current)         # 把最后一块也加入

        # 递归处理：如果某块仍然超长（可能是 sep 分割后单个 part 就很长），继续细分
        result = []
        for chunk in chunks:
            if count_tokens(chunk) > self.max_tokens:
                result.extend(self._split_text(chunk))  # 递归切分
            else:
                result.append(chunk)
        return result

    def _char_split(self, text: str) -> list[str]:
        """
        字符级截断（最后手段，当所有语义分割都失败时使用）
        按字符数量强制切分，可能会在词语中间截断
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.max_tokens * 2  # 字符数 ≈ token 数 × 2（因为一个中文字算1 token，但占2字节）
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap_tokens * 2  # 下一块从重叠部分开始（回退 overlap 个字符）
        return chunks

    @staticmethod
    def _tail_tokens(text: str, n_tokens: int) -> str:
        """
        取文本末尾约 n_tokens 个 token 对应的字符
        用于生成重叠内容（overlap）
        """
        chars = n_tokens * 2          # 估算字符数（中文字符 token≈字符数，但保守估计用×2）
        return text[-chars:] if len(text) > chars else text
        # 如果文本够长，取末尾 chars 个字符；否则取整个文本
