"""
代理池模块

提供代理管理和选择策略实现，支持多种代理获取策略和健康检查。
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional


class ProxyStrategy(str, Enum):
    """代理选择策略"""

    RANDOM = "random"
    WEIGHTED = "weighted"
    GEO = "geo"
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"


@dataclass
class ProxyInfo:
    """
    代理信息数据类

    存储代理的详细信息和统计数据。

    Attributes:
        url: 代理URL（包含认证信息）
        protocol: 代理协议（http/https/socks5）
        country: 代理所在国家代码
        quality_score: 代理质量分数（0-100）
        last_used: 最后使用时间
        success_rate: 成功率（0.0-1.0）
        success_count: 成功请求次数
        failure_count: 失败请求次数
        avg_response_time: 平均响应时间（秒）
        created_at: 创建时间
        tags: 标签列表
    """

    url: str
    protocol: str = "http"
    country: Optional[str] = None
    quality_score: float = 50.0
    last_used: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    @property
    def total_requests(self) -> int:
        """总请求数"""
        return self.success_count + self.failure_count

    @property
    def is_healthy(self) -> bool:
        """判断代理是否健康"""
        if self.total_requests < 5:
            return True
        return self.success_rate >= 0.5 and self.quality_score >= 30

    @property
    def weight(self) -> float:
        """计算代理权重（用于加权选择）"""
        base_weight = self.quality_score / 100.0
        success_weight = self.success_rate
        response_weight = 1.0 / (1.0 + self.avg_response_time)
        return base_weight * 0.4 + success_weight * 0.4 + response_weight * 0.2

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "url": self.url,
            "protocol": self.protocol,
            "country": self.country,
            "quality_score": self.quality_score,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "success_rate": self.success_rate,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_response_time": self.avg_response_time,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }


class ProxyPool:
    """
    代理池管理类

    管理代理的添加、获取、健康检查和状态更新。

    Attributes:
        min_quality: 最低质量分数阈值
        max_failures: 最大连续失败次数
        stale_timeout: 代理过期时间（秒）
        health_check_interval: 健康检查间隔（秒）

    Example:
        >>> pool = ProxyPool()
        >>> pool.add_proxy(ProxyInfo(url="http://proxy:8080"))
        >>> proxy = await pool.get_proxy(strategy='weighted')
        >>> pool.report_success(proxy)
    """

    def __init__(
        self,
        min_quality: float = 30.0,
        max_failures: int = 10,
        stale_timeout: float = 3600.0,
        health_check_interval: float = 300.0,
    ) -> None:
        self._proxies: dict[str, ProxyInfo] = {}
        self._lock = asyncio.Lock()
        self._min_quality = min_quality
        self._max_failures = max_failures
        self._stale_timeout = stale_timeout
        self._health_check_interval = health_check_interval
        self._round_robin_index = 0
        self._consecutive_failures: dict[str, int] = {}

    def add_proxy(self, proxy: ProxyInfo) -> bool:
        """
        添加代理到池中

        Args:
            proxy: 代理信息

        Returns:
            是否成功添加
        """
        if not proxy.url:
            return False

        self._proxies[proxy.url] = proxy
        self._consecutive_failures[proxy.url] = 0
        return True

    def add_proxies(self, proxies: list[ProxyInfo]) -> int:
        """
        批量添加代理

        Args:
            proxies: 代理列表

        Returns:
            成功添加的数量
        """
        count = 0
        for proxy in proxies:
            if self.add_proxy(proxy):
                count += 1
        return count

    def remove_proxy(self, proxy_url: str) -> bool:
        """
        从池中移除代理

        Args:
            proxy_url: 代理URL

        Returns:
            是否成功移除
        """
        if proxy_url in self._proxies:
            del self._proxies[proxy_url]
            self._consecutive_failures.pop(proxy_url, None)
            return True
        return False

    async def get_proxy(
        self,
        strategy: ProxyStrategy | str = ProxyStrategy.RANDOM,
        country: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[ProxyInfo]:
        """
        获取代理

        Args:
            strategy: 选择策略（random/weighted/geo/round_robin/least_used）
            country: 指定国家代码（用于geo策略）
            tags: 指定标签过滤

        Returns:
            代理信息，如果没有可用代理则返回None
        """
        async with self._lock:
            candidates = self._get_candidates(country, tags)

            if not candidates:
                return None

            strategy_enum = (
                ProxyStrategy(strategy) if isinstance(strategy, str) else strategy
            )

            if strategy_enum == ProxyStrategy.RANDOM:
                proxy = random.choice(candidates)
            elif strategy_enum == ProxyStrategy.WEIGHTED:
                proxy = self._weighted_select(candidates)
            elif strategy_enum == ProxyStrategy.GEO:
                proxy = self._geo_select(candidates, country)
            elif strategy_enum == ProxyStrategy.ROUND_ROBIN:
                proxy = self._round_robin_select(candidates)
            elif strategy_enum == ProxyStrategy.LEAST_USED:
                proxy = self._least_used_select(candidates)
            else:
                proxy = random.choice(candidates)

            proxy.last_used = datetime.utcnow()
            return proxy

    def _get_candidates(
        self, country: Optional[str] = None, tags: Optional[list[str]] = None
    ) -> list[ProxyInfo]:
        """获取符合条件的候选代理"""
        candidates = []

        for proxy in self._proxies.values():
            if not proxy.is_healthy:
                continue

            if proxy.quality_score < self._min_quality:
                continue

            if self._consecutive_failures.get(proxy.url, 0) >= self._max_failures:
                continue

            if country and proxy.country != country:
                continue

            if tags and not any(tag in proxy.tags for tag in tags):
                continue

            candidates.append(proxy)

        return candidates

    def _weighted_select(self, candidates: list[ProxyInfo]) -> ProxyInfo:
        """加权随机选择"""
        weights = [p.weight for p in candidates]
        total = sum(weights)
        if total == 0:
            return random.choice(candidates)

        r = random.uniform(0, total)
        cumulative = 0.0
        for proxy, weight in zip(candidates, weights):
            cumulative += weight
            if r <= cumulative:
                return proxy

        return candidates[-1]

    def _geo_select(
        self, candidates: list[ProxyInfo], country: Optional[str]
    ) -> ProxyInfo:
        """地理位置优先选择"""
        if country:
            geo_candidates = [p for p in candidates if p.country == country]
            if geo_candidates:
                return self._weighted_select(geo_candidates)

        return self._weighted_select(candidates)

    def _round_robin_select(self, candidates: list[ProxyInfo]) -> ProxyInfo:
        """轮询选择"""
        proxy = candidates[self._round_robin_index % len(candidates)]
        self._round_robin_index += 1
        return proxy

    def _least_used_select(self, candidates: list[ProxyInfo]) -> ProxyInfo:
        """最少使用选择"""
        return min(candidates, key=lambda p: p.total_requests)

    def report_success(
        self, proxy: ProxyInfo, response_time: Optional[float] = None
    ) -> None:
        """
        报告代理使用成功

        Args:
            proxy: 代理信息
            response_time: 响应时间（秒）
        """
        if proxy.url in self._proxies:
            stored_proxy = self._proxies[proxy.url]
            stored_proxy.success_count += 1
            self._consecutive_failures[proxy.url] = 0

            if response_time is not None:
                if stored_proxy.avg_response_time == 0:
                    stored_proxy.avg_response_time = response_time
                else:
                    stored_proxy.avg_response_time = (
                        stored_proxy.avg_response_time * 0.8 + response_time * 0.2
                    )

            stored_proxy.quality_score = min(
                100.0, stored_proxy.quality_score + 1.0
            )

    def report_failure(self, proxy: ProxyInfo) -> None:
        """
        报告代理使用失败

        Args:
            proxy: 代理信息
        """
        if proxy.url in self._proxies:
            stored_proxy = self._proxies[proxy.url]
            stored_proxy.failure_count += 1
            self._consecutive_failures[proxy.url] = (
                self._consecutive_failures.get(proxy.url, 0) + 1
            )

            stored_proxy.quality_score = max(
                0.0, stored_proxy.quality_score - 5.0
            )

    async def health_check(self) -> dict[str, Any]:
        """
        执行健康检查

        检查所有代理的健康状态，移除不健康的代理。

        Returns:
            健康检查结果统计
        """
        async with self._lock:
            results = {
                "total": len(self._proxies),
                "healthy": 0,
                "unhealthy": 0,
                "removed": 0,
            }

            to_remove = []

            for proxy in self._proxies.values():
                if proxy.is_healthy:
                    results["healthy"] += 1
                else:
                    results["unhealthy"] += 1
                    if (
                        self._consecutive_failures.get(proxy.url, 0)
                        >= self._max_failures
                    ):
                        to_remove.append(proxy.url)

            for url in to_remove:
                self.remove_proxy(url)
                results["removed"] += 1

            return results

    def remove_stale(self) -> int:
        """
        移除过期代理

        移除超过指定时间未使用的代理。

        Returns:
            移除的代理数量
        """
        now = datetime.utcnow()
        stale_threshold = timedelta(seconds=self._stale_timeout)
        to_remove = []

        for url, proxy in self._proxies.items():
            if proxy.last_used and (now - proxy.last_used) > stale_threshold:
                to_remove.append(url)
            elif proxy.total_requests == 0 and (now - proxy.created_at) > stale_threshold:
                to_remove.append(url)

        for url in to_remove:
            self.remove_proxy(url)

        return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """
        获取代理池统计信息

        Returns:
            统计信息字典
        """
        if not self._proxies:
            return {
                "total": 0,
                "healthy": 0,
                "unhealthy": 0,
                "avg_quality": 0.0,
                "avg_success_rate": 0.0,
                "countries": [],
            }

        proxies = list(self._proxies.values())
        healthy_proxies = [p for p in proxies if p.is_healthy]

        return {
            "total": len(proxies),
            "healthy": len(healthy_proxies),
            "unhealthy": len(proxies) - len(healthy_proxies),
            "avg_quality": sum(p.quality_score for p in proxies) / len(proxies),
            "avg_success_rate": sum(p.success_rate for p in proxies) / len(proxies),
            "countries": list(set(p.country for p in proxies if p.country)),
        }

    def get_all_proxies(self) -> list[ProxyInfo]:
        """获取所有代理列表"""
        return list(self._proxies.values())

    def clear(self) -> None:
        """清空代理池"""
        self._proxies.clear()
        self._consecutive_failures.clear()
        self._round_robin_index = 0

    def __len__(self) -> int:
        return len(self._proxies)

    def __contains__(self, proxy_url: str) -> bool:
        return proxy_url in self._proxies
