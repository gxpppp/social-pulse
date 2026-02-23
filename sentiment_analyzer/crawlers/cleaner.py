"""
数据清洗管道模块

提供文本清洗、内容标准化、语言检测和数据去重功能。
"""

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse


@dataclass
class CleanedContent:
    """
    清洗后的内容

    Attributes:
        text: 清洗后的文本
        original_length: 原始文本长度
        cleaned_length: 清洗后文本长度
        language: 检测到的语言
        urls: 提取的URL列表
        hashtags: 提取的话题标签列表
        mentions: 提取的用户提及列表
        removed_elements: 被移除的元素统计
    """

    text: str
    original_length: int = 0
    cleaned_length: int = 0
    language: Optional[str] = None
    urls: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    removed_elements: dict[str, int] = field(default_factory=dict)


class DataCleaner:
    """
    数据清洗管道

    提供文本清洗、内容标准化、语言检测和数据提取功能。

    Attributes:
        remove_html: 是否移除HTML标签
        remove_urls: 是否移除URL
        remove_mentions: 是否移除用户提及
        remove_hashtags: 是否移除话题标签
        normalize_whitespace: 是否标准化空白字符
        lowercase: 是否转换为小写
        remove_emoji: 是否移除表情符号
        remove_special_chars: 是否移除特殊字符

    Example:
        >>> cleaner = DataCleaner(remove_html=True, normalize_whitespace=True)
        >>> result = cleaner.clean_text("<p>Hello World!</p>")
        >>> print(result)
        Hello World!
    """

    URL_PATTERN = re.compile(
        r"(?:https?://|www\.)[^\s<>'\"{}|\\^`[\]]+",
        re.IGNORECASE,
    )
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    MENTION_PATTERN = re.compile(r"@(\w+)")
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    WHITESPACE_PATTERN = re.compile(r"\s+")
    SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s\u4e00-\u9fff.,!?;:'\"-]")
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = False,
        remove_mentions: bool = False,
        remove_hashtags: bool = False,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
        remove_emoji: bool = False,
        remove_special_chars: bool = False,
        min_length: int = 1,
        max_length: Optional[int] = None,
    ) -> None:
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.remove_emoji = remove_emoji
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """
        清洗文本

        根据配置执行一系列清洗操作。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        result = text

        result = self.CONTROL_CHAR_PATTERN.sub(" ", result)

        result = unicodedata.normalize("NFKC", result)

        if self.remove_html:
            result = self._remove_html(result)

        if self.remove_emoji:
            result = self.EMOJI_PATTERN.sub("", result)

        if self.remove_urls:
            result = self.URL_PATTERN.sub("", result)

        if self.remove_mentions:
            result = self.MENTION_PATTERN.sub("", result)

        if self.remove_hashtags:
            result = self.HASHTAG_PATTERN.sub("", result)

        if self.remove_special_chars:
            result = self.SPECIAL_CHAR_PATTERN.sub("", result)

        if self.normalize_whitespace:
            result = self.WHITESPACE_PATTERN.sub(" ", result).strip()

        if self.lowercase:
            result = result.lower()

        if self.max_length and len(result) > self.max_length:
            result = result[: self.max_length]

        return result

    def _remove_html(self, text: str) -> str:
        """移除HTML标签和实体"""
        result = self.HTML_TAG_PATTERN.sub(" ", text)

        result = html.unescape(result)

        result = self.HTML_ENTITY_PATTERN.sub(" ", result)

        return result

    def normalize_content(self, content: str | dict[str, Any]) -> CleanedContent:
        """
        内容标准化

        执行完整的清洗流程并返回详细结果。

        Args:
            content: 原始内容（字符串或字典）

        Returns:
            清洗后的内容对象
        """
        if isinstance(content, dict):
            text = content.get("text", "") or content.get("content", "") or ""
        else:
            text = content

        original_length = len(text)

        urls = self.extract_urls(text)
        hashtags = self.extract_hashtags(text)
        mentions = self.extract_mentions(text)

        removed_elements: dict[str, int] = {}

        cleaned = self.clean_text(text)

        removed_elements["html_tags"] = len(self.HTML_TAG_PATTERN.findall(text))
        removed_elements["control_chars"] = len(self.CONTROL_CHAR_PATTERN.findall(text))

        if len(cleaned) < original_length:
            removed_elements["total_chars"] = original_length - len(cleaned)

        language = self.detect_language(cleaned)

        return CleanedContent(
            text=cleaned,
            original_length=original_length,
            cleaned_length=len(cleaned),
            language=language,
            urls=urls,
            hashtags=hashtags,
            mentions=mentions,
            removed_elements=removed_elements,
        )

    def detect_language(self, text: str) -> Optional[str]:
        """
        语言检测

        基于字符分布进行简单的语言检测。

        Args:
            text: 要检测的文本

        Returns:
            语言代码（zh/en/other）或None
        """
        if not text:
            return None

        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        ascii_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return None

        chinese_ratio = chinese_chars / total_chars
        ascii_ratio = ascii_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh"
        elif ascii_ratio > 0.5:
            return "en"
        else:
            return "other"

    def extract_urls(self, text: str) -> list[str]:
        """
        提取URL

        Args:
            text: 包含URL的文本

        Returns:
            URL列表
        """
        urls = self.URL_PATTERN.findall(text)
        return [self.normalize_url(url) for url in urls]

    def normalize_url(self, url: str) -> str:
        """
        URL标准化

        Args:
            url: 原始URL

        Returns:
            标准化后的URL
        """
        url = url.strip()

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            parsed = urlparse(url)

            netloc = parsed.netloc.lower()

            path = parsed.path.rstrip("/") or "/"

            return urlunparse(
                (
                    parsed.scheme,
                    netloc,
                    path,
                    parsed.params,
                    parsed.query,
                    "",
                )
            )
        except Exception:
            return url

    def extract_hashtags(self, text: str) -> list[str]:
        """
        提取话题标签

        Args:
            text: 包含话题标签的文本

        Returns:
            话题标签列表（不含#符号）
        """
        return self.HASHTAG_PATTERN.findall(text)

    def extract_mentions(self, text: str) -> list[str]:
        """
        提取用户提及

        Args:
            text: 包含用户提及的文本

        Returns:
            用户名列表（不含@符号）
        """
        return self.MENTION_PATTERN.findall(text)

    def remove_duplicates(
        self, items: list[Any], key: Optional[str] = None
    ) -> list[Any]:
        """
        去重

        Args:
            items: 要去重的列表
            key: 用于比较的键（如果元素是字典）

        Returns:
            去重后的列表
        """
        if not items:
            return []

        seen: set[Any] = set()
        result: list[Any] = []

        for item in items:
            if key and isinstance(item, dict):
                compare_value = item.get(key)
            else:
                compare_value = item

            if isinstance(compare_value, (dict, list)):
                compare_value = str(compare_value)

            if compare_value not in seen:
                seen.add(compare_value)
                result.append(item)

        return result

    def clean_batch(
        self, texts: list[str], parallel: bool = False
    ) -> list[str]:
        """
        批量清洗文本

        Args:
            texts: 文本列表
            parallel: 是否并行处理（当前版本暂不支持）

        Returns:
            清洗后的文本列表
        """
        return [self.clean_text(text) for text in texts]

    def normalize_batch(
        self, contents: list[str | dict[str, Any]]
    ) -> list[CleanedContent]:
        """
        批量标准化内容

        Args:
            contents: 内容列表

        Returns:
            清洗后的内容对象列表
        """
        return [self.normalize_content(content) for content in contents]

    def get_stats(self, results: list[CleanedContent]) -> dict[str, Any]:
        """
        获取清洗统计信息

        Args:
            results: 清洗结果列表

        Returns:
            统计信息字典
        """
        if not results:
            return {
                "total": 0,
                "avg_original_length": 0,
                "avg_cleaned_length": 0,
                "avg_reduction": 0,
                "languages": {},
                "total_urls": 0,
                "total_hashtags": 0,
                "total_mentions": 0,
            }

        total_original = sum(r.original_length for r in results)
        total_cleaned = sum(r.cleaned_length for r in results)

        languages: dict[str, int] = {}
        for r in results:
            if r.language:
                languages[r.language] = languages.get(r.language, 0) + 1

        return {
            "total": len(results),
            "avg_original_length": total_original / len(results),
            "avg_cleaned_length": total_cleaned / len(results),
            "avg_reduction": (total_original - total_cleaned) / len(results),
            "languages": languages,
            "total_urls": sum(len(r.urls) for r in results),
            "total_hashtags": sum(len(r.hashtags) for r in results),
            "total_mentions": sum(len(r.mentions) for r in results),
        }


def create_default_cleaner() -> DataCleaner:
    """创建默认配置的数据清洗器"""
    return DataCleaner(
        remove_html=True,
        remove_urls=False,
        remove_mentions=False,
        remove_hashtags=False,
        normalize_whitespace=True,
        lowercase=False,
        remove_emoji=False,
        remove_special_chars=False,
    )


def create_strict_cleaner() -> DataCleaner:
    """创建严格模式的数据清洗器（移除所有非文本内容）"""
    return DataCleaner(
        remove_html=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=True,
        normalize_whitespace=True,
        lowercase=True,
        remove_emoji=True,
        remove_special_chars=True,
    )
