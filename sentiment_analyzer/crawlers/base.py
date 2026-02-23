"""
平台适配器接口定义

定义爬虫抽象基类和通用接口，供各平台具体实现继承。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional, Protocol

from storage.models import Platform, Post, User


class CrawlerStatus(str, Enum):
    """爬虫状态枚举"""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class RateLimitConfig:
    """
    限流配置

    Attributes:
        requests_per_second: 每秒请求数限制
        requests_per_minute: 每分钟请求数限制
        requests_per_hour: 每小时请求数限制
        burst_size: 突发请求大小
        backoff_factor: 退避因子
        max_backoff: 最大退避时间（秒）
    """

    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 5
    backoff_factor: float = 2.0
    max_backoff: float = 300.0


@dataclass
class ProxyInfo:
    """
    代理信息

    Attributes:
        url: 代理URL
        protocol: 代理协议
        country: 代理所在国家
        last_used: 最后使用时间
        success_count: 成功次数
        failure_count: 失败次数
        avg_response_time: 平均响应时间
    """

    url: str
    protocol: str = "http"
    country: Optional[str] = None
    last_used: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total


class RateLimiter:
    """
    限流器

    实现令牌桶算法进行请求限流，支持多级限流配置。
    """

    def __init__(self, config: RateLimitConfig):
        """
        初始化限流器

        Args:
            config: 限流配置
        """
        self.config = config
        self._tokens = float(config.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self._request_times: list[float] = []
        self._consecutive_failures = 0

    async def acquire(self) -> None:
        """
        获取请求许可

        如果超过限流阈值，将阻塞等待直到可以执行请求。
        """
        async with self._lock:
            now = time.monotonic()

            self._request_times = [t for t in self._request_times if now - t < 3600]

            if len(self._request_times) >= self.config.requests_per_hour:
                sleep_time = 3600 - (now - self._request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.monotonic()

            minute_ago = now - 60
            minute_requests = sum(1 for t in self._request_times if t > minute_ago)
            if minute_requests >= self.config.requests_per_minute:
                sleep_time = 60 - (now - min(t for t in self._request_times if t > minute_ago))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                now = time.monotonic()

            elapsed = now - self._last_update
            self._tokens = min(
                float(self.config.burst_size),
                self._tokens + elapsed * self.config.requests_per_second,
            )

            if self._tokens < 1:
                sleep_time = (1 - self._tokens) / self.config.requests_per_second
                await asyncio.sleep(sleep_time)
                self._tokens = 0
            else:
                self._tokens -= 1

            self._last_update = time.monotonic()
            self._request_times.append(self._last_update)

    def record_success(self) -> None:
        """记录请求成功，重置连续失败计数"""
        self._consecutive_failures = 0

    def record_failure(self) -> float:
        """
        记录请求失败，计算退避时间

        Returns:
            建议的退避等待时间
        """
        self._consecutive_failures += 1
        backoff = min(
            self.config.backoff_factor**self._consecutive_failures,
            self.config.max_backoff,
        )
        return backoff


class ProxyPool(Protocol):
    """
    代理池接口协议

    定义代理池管理的基本接口。
    """

    async def get_proxy(self) -> Optional[ProxyInfo]:
        """
        获取一个可用代理

        Returns:
            代理信息，如果没有可用代理则返回None
        """
        ...

    async def release_proxy(self, proxy: ProxyInfo, success: bool) -> None:
        """
        释放代理并更新状态

        Args:
            proxy: 要释放的代理
            success: 请求是否成功
        """
        ...

    async def add_proxy(self, proxy: ProxyInfo) -> None:
        """
        添加新代理到池中

        Args:
            proxy: 要添加的代理
        """
        ...

    async def remove_proxy(self, proxy_url: str) -> None:
        """
        从池中移除代理

        Args:
            proxy_url: 要移除的代理URL
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """
        获取代理池统计信息

        Returns:
            包含统计信息的字典
        """
        ...


class BaseCrawler(ABC):
    """
    爬虫抽象基类

    定义所有平台爬虫必须实现的接口。各平台爬虫需要继承此类并实现所有抽象方法。

    Attributes:
        platform: 爬虫对应的平台
        rate_limiter: 限流器实例
        proxy_pool: 代理池实例（可选）
        status: 当前爬虫状态
    """

    platform: Platform

    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        proxy_pool: Optional[ProxyPool] = None,
    ):
        """
        初始化爬虫

        Args:
            rate_limit_config: 限流配置，如果为None则使用默认配置
            proxy_pool: 代理池实例
        """
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.proxy_pool = proxy_pool
        self._status = CrawlerStatus.IDLE
        self._current_proxy: Optional[ProxyInfo] = None
        self._stats: dict[str, Any] = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "users_collected": 0,
            "posts_collected": 0,
            "start_time": None,
            "last_request_time": None,
        }

    @property
    def status(self) -> CrawlerStatus:
        """获取当前爬虫状态"""
        return self._status

    @status.setter
    def status(self, value: CrawlerStatus) -> None:
        """设置爬虫状态"""
        self._status = value

    @abstractmethod
    async def crawl(
        self,
        query: Optional[str] = None,
        user_ids: Optional[list[str]] = None,
        post_ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[User | Post]:
        """
        执行爬取任务

        Args:
            query: 搜索查询字符串
            user_ids: 要爬取的用户ID列表
            post_ids: 要爬取的帖子ID列表
            **kwargs: 其他平台特定参数

        Yields:
            爬取到的用户或帖子对象

        Raises:
            CrawlerException: 爬取过程中发生错误
        """
        ...

    @abstractmethod
    async def parse_user(self, data: dict[str, Any]) -> User:
        """
        解析用户数据

        将平台原始数据转换为统一的User模型。

        Args:
            data: 平台返回的原始用户数据

        Returns:
            解析后的User对象

        Raises:
            ParseException: 解析过程中发生错误
        """
        ...

    @abstractmethod
    async def parse_post(self, data: dict[str, Any]) -> Post:
        """
        解析帖子数据

        将平台原始数据转换为统一的Post模型。

        Args:
            data: 平台返回的原始帖子数据

        Returns:
            解析后的Post对象

        Raises:
            ParseException: 解析过程中发生错误
        """
        ...

    async def before_crawl(self) -> None:
        """
        爬取前的准备工作

        子类可以覆盖此方法实现平台特定的初始化逻辑。
        """
        self._stats["start_time"] = datetime.utcnow()
        self.status = CrawlerStatus.RUNNING

    async def after_crawl(self) -> None:
        """
        爬取后的清理工作

        子类可以覆盖此方法实现平台特定的清理逻辑。
        """
        self.status = CrawlerStatus.IDLE

    async def on_error(self, error: Exception) -> None:
        """
        错误处理回调

        Args:
            error: 发生的异常
        """
        self._stats["requests_failed"] += 1
        if self._current_proxy and self.proxy_pool:
            await self.proxy_pool.release_proxy(self._current_proxy, success=False)
            self._current_proxy = None

    async def get_request_context(self) -> dict[str, Any]:
        """
        获取请求上下文

        包含代理信息和限流控制。

        Returns:
            请求上下文字典
        """
        await self.rate_limiter.acquire()

        context: dict[str, Any] = {}

        if self.proxy_pool:
            self._current_proxy = await self.proxy_pool.get_proxy()
            if self._current_proxy:
                context["proxy"] = self._current_proxy.url

        self._stats["requests_total"] += 1
        self._stats["last_request_time"] = datetime.utcnow()

        return context

    async def release_request_context(self, success: bool = True) -> None:
        """
        释放请求上下文

        Args:
            success: 请求是否成功
        """
        if success:
            self.rate_limiter.record_success()
            self._stats["requests_success"] += 1
        else:
            backoff = self.rate_limiter.record_failure()
            await asyncio.sleep(backoff)

        if self._current_proxy and self.proxy_pool:
            await self.proxy_pool.release_proxy(self._current_proxy, success=success)
            self._current_proxy = None

    def get_stats(self) -> dict[str, Any]:
        """
        获取爬虫统计信息

        Returns:
            包含统计信息的字典
        """
        stats = self._stats.copy()
        stats["status"] = self.status.value
        stats["platform"] = self.platform.value
        if stats["start_time"]:
            stats["uptime_seconds"] = (datetime.utcnow() - stats["start_time"]).total_seconds()
        return stats

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "users_collected": 0,
            "posts_collected": 0,
            "start_time": None,
            "last_request_time": None,
        }


class CrawlerException(Exception):
    """爬虫异常基类"""

    def __init__(self, message: str, platform: Optional[Platform] = None, **context: Any):
        super().__init__(message)
        self.platform = platform
        self.context = context


class RateLimitException(CrawlerException):
    """限流异常"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.retry_after = retry_after


class ParseException(CrawlerException):
    """解析异常"""

    def __init__(
        self,
        message: str,
        raw_data: Optional[dict[str, Any]] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.raw_data = raw_data


class NetworkException(CrawlerException):
    """网络异常"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.status_code = status_code


class AuthenticationException(CrawlerException):
    """认证异常"""

    def __init__(
        self,
        message: str = "Authentication failed",
        **context: Any,
    ):
        super().__init__(message, **context)
