"""
爬虫模块

提供平台适配器和爬虫基础类。
"""

from .base import (
    AuthenticationException,
    BaseCrawler,
    CrawlerException,
    CrawlerStatus,
    NetworkException,
    ParseException,
    ProxyInfo,
    ProxyPool,
    RateLimitConfig,
    RateLimitException,
    RateLimiter,
)
from .bloom_filter import BloomFilter, ScalableBloomFilter
from .cleaner import CleanedContent, DataCleaner, create_default_cleaner, create_strict_cleaner
from .proxy_pool import ProxyInfo as DetailedProxyInfo
from .proxy_pool import ProxyPool as DetailedProxyPool
from .proxy_pool import ProxyStrategy
from .rate_limiter import MultiRateLimiter, RateLimit, TokenBucket
from .reddit import (
    RedditComment,
    RedditConfig,
    RedditCrawler,
    RedditParser,
)
from .telegram import (
    TelegramChannel,
    TelegramChat,
    TelegramConfig,
    TelegramCrawler,
    TelegramParser,
)

__all__ = [
    "BaseCrawler",
    "CrawlerStatus",
    "RateLimitConfig",
    "ProxyInfo",
    "ProxyPool",
    "RateLimiter",
    "CrawlerException",
    "RateLimitException",
    "ParseException",
    "NetworkException",
    "AuthenticationException",
    "TokenBucket",
    "RateLimit",
    "MultiRateLimiter",
    "DetailedProxyInfo",
    "DetailedProxyPool",
    "ProxyStrategy",
    "BloomFilter",
    "ScalableBloomFilter",
    "DataCleaner",
    "CleanedContent",
    "create_default_cleaner",
    "create_strict_cleaner",
    "RedditConfig",
    "RedditCrawler",
    "RedditParser",
    "RedditComment",
    "TelegramConfig",
    "TelegramCrawler",
    "TelegramParser",
    "TelegramChannel",
    "TelegramChat",
]
