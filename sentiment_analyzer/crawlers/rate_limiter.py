"""
限流器模块

提供令牌桶算法和多级限流器实现，用于控制请求频率。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucket:
    """
    令牌桶算法实现

    令牌桶算法是一种流量整形算法，以固定速率向桶中添加令牌，
    请求需要获取令牌才能执行，从而控制请求速率。

    Attributes:
        rate: 令牌添加速率（令牌/秒）
        capacity: 桶容量（最大令牌数）
        tokens: 当前令牌数
        last_update: 上次更新时间

    Example:
        >>> bucket = TokenBucket(rate=10, capacity=100)
        >>> await bucket.acquire(5)  # 获取5个令牌
        >>> bucket.try_acquire(3)    # 非阻塞获取3个令牌
    """

    rate: float
    capacity: float
    tokens: float = field(default=0.0)
    last_update: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.tokens == 0.0:
            self.tokens = self.capacity
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def _refill(self) -> None:
        """根据时间流逝补充令牌"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        获取令牌（阻塞式）

        如果当前令牌不足，将阻塞等待直到有足够的令牌。

        Args:
            tokens: 需要获取的令牌数量

        Returns:
            总是返回 True

        Raises:
            ValueError: 请求数量超过桶容量
        """
        if tokens > self.capacity:
            raise ValueError(
                f"Requested tokens ({tokens}) exceed capacity ({self.capacity})"
            )

        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                wait_time = self.wait_time(tokens)
                await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        尝试获取令牌（非阻塞）

        如果当前令牌不足，立即返回 False。

        Args:
            tokens: 需要获取的令牌数量

        Returns:
            是否成功获取令牌
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: float = 1.0) -> float:
        """
        计算获取指定数量令牌需要等待的时间

        Args:
            tokens: 需要获取的令牌数量

        Returns:
            需要等待的秒数
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.rate

    @property
    def available(self) -> float:
        """当前可用令牌数"""
        self._refill()
        return self.tokens

    def reset(self) -> None:
        """重置桶为满状态"""
        self.tokens = self.capacity
        self.last_update = time.monotonic()


@dataclass
class RateLimit:
    """
    单级限流配置

    Attributes:
        period: 时间周期（秒）
        limit: 周期内允许的请求数
        bucket: 对应的令牌桶
    """

    period: float
    limit: int
    bucket: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        rate = self.limit / self.period
        self.bucket = TokenBucket(rate=rate, capacity=float(self.limit))

    async def acquire(self, tokens: float = 1.0) -> bool:
        """获取令牌"""
        return await self.bucket.acquire(tokens)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """尝试获取令牌"""
        return self.bucket.try_acquire(tokens)


class MultiRateLimiter:
    """
    多级限流器

    支持同时配置多个时间维度的限流策略（如每秒、每分钟、每小时）。
    请求需要通过所有级别的限流检查才能执行。

    Attributes:
        limits: 各级限流配置列表

    Example:
        >>> limiter = MultiRateLimiter()
        >>> limiter.add_limit(period=1, limit=10)      # 每秒10次
        >>> limiter.add_limit(period=60, limit=100)    # 每分钟100次
        >>> limiter.add_limit(period=3600, limit=1000) # 每小时1000次
        >>> await limiter.acquire()
    """

    def __init__(self) -> None:
        self._limits: list[RateLimit] = []
        self._lock = asyncio.Lock()

    def add_limit(self, period: float, limit: int) -> "MultiRateLimiter":
        """
        添加限流级别

        Args:
            period: 时间周期（秒）
            limit: 周期内允许的请求数

        Returns:
            self，支持链式调用
        """
        rate_limit = RateLimit(period=period, limit=limit)
        self._limits.append(rate_limit)
        return self

    def set_per_second(self, limit: int) -> "MultiRateLimiter":
        """设置每秒限制"""
        return self.add_limit(period=1.0, limit=limit)

    def set_per_minute(self, limit: int) -> "MultiRateLimiter":
        """设置每分钟限制"""
        return self.add_limit(period=60.0, limit=limit)

    def set_per_hour(self, limit: int) -> "MultiRateLimiter":
        """设置每小时限制"""
        return self.add_limit(period=3600.0, limit=limit)

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        获取请求许可

        需要通过所有级别的限流检查。如果任一级别需要等待，
        将等待最长时间的那个级别。

        Args:
            tokens: 需要获取的令牌数量
        """
        async with self._lock:
            max_wait = 0.0
            for rate_limit in self._limits:
                wait_time = rate_limit.bucket.wait_time(tokens)
                max_wait = max(max_wait, wait_time)

            if max_wait > 0:
                await asyncio.sleep(max_wait)

            for rate_limit in self._limits:
                await rate_limit.acquire(tokens)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        尝试获取请求许可（非阻塞）

        Args:
            tokens: 需要获取的令牌数量

        Returns:
            是否成功获取所有级别的许可
        """
        for rate_limit in self._limits:
            if not rate_limit.try_acquire(tokens):
                return False
        return True

    def wait_time(self, tokens: float = 1.0) -> float:
        """
        计算需要等待的时间

        Args:
            tokens: 需要获取的令牌数量

        Returns:
            最长的等待时间
        """
        max_wait = 0.0
        for rate_limit in self._limits:
            wait_time = rate_limit.bucket.wait_time(tokens)
            max_wait = max(max_wait, wait_time)
        return max_wait

    def update_rate(self, period: float, new_limit: int) -> bool:
        """
        动态更新指定周期的限流速率

        Args:
            period: 时间周期（秒）
            new_limit: 新的请求数限制

        Returns:
            是否成功更新
        """
        for rate_limit in self._limits:
            if rate_limit.period == period:
                new_rate = RateLimit(period=period, limit=new_limit)
                rate_limit.bucket = new_rate.bucket
                return True
        return False

    def get_status(self) -> dict:
        """
        获取限流器状态

        Returns:
            包含各级别状态的字典
        """
        status = {}
        for i, rate_limit in enumerate(self._limits):
            status[f"level_{i}"] = {
                "period": rate_limit.period,
                "limit": rate_limit.limit,
                "available": rate_limit.bucket.available,
                "wait_time": rate_limit.bucket.wait_time(),
            }
        return status

    @classmethod
    def from_config(
        cls,
        requests_per_second: Optional[float] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
    ) -> "MultiRateLimiter":
        """
        从配置创建多级限流器

        Args:
            requests_per_second: 每秒请求数限制
            requests_per_minute: 每分钟请求数限制
            requests_per_hour: 每小时请求数限制

        Returns:
            配置好的多级限流器
        """
        limiter = cls()
        if requests_per_second is not None:
            limiter.set_per_second(int(requests_per_second))
        if requests_per_minute is not None:
            limiter.set_per_minute(requests_per_minute)
        if requests_per_hour is not None:
            limiter.set_per_hour(requests_per_hour)
        return limiter
