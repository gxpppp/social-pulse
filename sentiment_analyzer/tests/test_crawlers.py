"""
爬虫模块测试

测试限流器、代理池、布隆过滤器和数据清洗功能。
"""

import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentiment_analyzer.crawlers.rate_limiter import TokenBucket, RateLimit, MultiRateLimiter
from sentiment_analyzer.crawlers.proxy_pool import ProxyPool, ProxyInfo, ProxyStrategy
from sentiment_analyzer.crawlers.bloom_filter import BloomFilter, ScalableBloomFilter
from sentiment_analyzer.crawlers.cleaner import DataCleaner, CleanedContent, create_default_cleaner, create_strict_cleaner


class TestTokenBucket:
    """令牌桶测试"""

    def test_token_bucket_initialization(self):
        """测试令牌桶初始化"""
        bucket = TokenBucket(rate=10, capacity=100)
        assert bucket.rate == 10
        assert bucket.capacity == 100
        assert bucket.tokens == 100

    def test_token_bucket_invalid_rate(self):
        """测试无效速率"""
        with pytest.raises(ValueError):
            TokenBucket(rate=0, capacity=100)
        with pytest.raises(ValueError):
            TokenBucket(rate=-1, capacity=100)

    def test_token_bucket_invalid_capacity(self):
        """测试无效容量"""
        with pytest.raises(ValueError):
            TokenBucket(rate=10, capacity=0)
        with pytest.raises(ValueError):
            TokenBucket(rate=10, capacity=-1)

    def test_try_acquire_success(self):
        """测试成功获取令牌"""
        bucket = TokenBucket(rate=10, capacity=100)
        result = bucket.try_acquire(5)
        assert result is True
        assert bucket.available == 95

    def test_try_acquire_insufficient_tokens(self):
        """测试令牌不足"""
        bucket = TokenBucket(rate=10, capacity=100)
        bucket.tokens = 3
        result = bucket.try_acquire(5)
        assert result is False

    def test_try_acquire_exceed_capacity(self):
        """测试请求数超过容量"""
        bucket = TokenBucket(rate=10, capacity=100)
        with pytest.raises(ValueError):
            bucket.try_acquire(150)

    @pytest.mark.asyncio
    async def test_acquire_blocking(self):
        """测试阻塞式获取令牌"""
        bucket = TokenBucket(rate=100, capacity=10)
        bucket.tokens = 0
        
        start_time = asyncio.get_event_loop().time()
        result = await bucket.acquire(1)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert result is True
        assert elapsed >= 0.01

    def test_wait_time_calculation(self):
        """测试等待时间计算"""
        bucket = TokenBucket(rate=10, capacity=100)
        bucket.tokens = 5
        wait_time = bucket.wait_time(15)
        assert wait_time == 1.0

    def test_wait_time_sufficient_tokens(self):
        """测试令牌充足时的等待时间"""
        bucket = TokenBucket(rate=10, capacity=100)
        bucket.tokens = 50
        wait_time = bucket.wait_time(10)
        assert wait_time == 0.0

    def test_reset(self):
        """测试重置令牌桶"""
        bucket = TokenBucket(rate=10, capacity=100)
        bucket.tokens = 10
        bucket.reset()
        assert bucket.tokens == 100

    def test_available_property(self):
        """测试可用令牌属性"""
        bucket = TokenBucket(rate=10, capacity=100)
        bucket.tokens = 50
        assert bucket.available == 50


class TestRateLimit:
    """单级限流配置测试"""

    def test_rate_limit_initialization(self):
        """测试限流配置初始化"""
        rate_limit = RateLimit(period=60, limit=100)
        assert rate_limit.period == 60
        assert rate_limit.limit == 100
        assert rate_limit.bucket is not None

    @pytest.mark.asyncio
    async def test_rate_limit_acquire(self):
        """测试限流配置获取令牌"""
        rate_limit = RateLimit(period=1, limit=10)
        result = await rate_limit.acquire(1)
        assert result is True

    def test_rate_limit_try_acquire(self):
        """测试限流配置非阻塞获取"""
        rate_limit = RateLimit(period=1, limit=10)
        result = rate_limit.try_acquire(1)
        assert result is True


class TestMultiRateLimiter:
    """多级限流器测试"""

    def test_add_limit(self):
        """测试添加限流级别"""
        limiter = MultiRateLimiter()
        limiter.add_limit(period=1, limit=10)
        limiter.add_limit(period=60, limit=100)
        assert len(limiter._limits) == 2

    def test_set_per_second(self):
        """测试设置每秒限制"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10)
        assert len(limiter._limits) == 1
        assert limiter._limits[0].period == 1.0

    def test_set_per_minute(self):
        """测试设置每分钟限制"""
        limiter = MultiRateLimiter()
        limiter.set_per_minute(100)
        assert len(limiter._limits) == 1
        assert limiter._limits[0].period == 60.0

    def test_set_per_hour(self):
        """测试设置每小时限制"""
        limiter = MultiRateLimiter()
        limiter.set_per_hour(1000)
        assert len(limiter._limits) == 1
        assert limiter._limits[0].period == 3600.0

    def test_chain_calls(self):
        """测试链式调用"""
        limiter = MultiRateLimiter()
        result = limiter.set_per_second(10).set_per_minute(100).set_per_hour(1000)
        assert result is limiter
        assert len(limiter._limits) == 3

    @pytest.mark.asyncio
    async def test_acquire_all_levels(self):
        """测试通过所有级别限流"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(100).set_per_minute(1000)
        await limiter.acquire(1)
        assert limiter._limits[0].bucket.available == 99
        assert limiter._limits[1].bucket.available == 999

    def test_try_acquire_success(self):
        """测试非阻塞获取成功"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10).set_per_minute(100)
        result = limiter.try_acquire(1)
        assert result is True

    def test_try_acquire_one_level_fails(self):
        """测试某一级限流失败"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10)
        limiter._limits[0].bucket.tokens = 0
        result = limiter.try_acquire(1)
        assert result is False

    def test_wait_time(self):
        """测试等待时间计算"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10)
        limiter._limits[0].bucket.tokens = 0
        wait_time = limiter.wait_time(1)
        assert wait_time > 0

    def test_update_rate(self):
        """测试更新限流速率"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10)
        result = limiter.update_rate(1.0, 20)
        assert result is True
        assert limiter._limits[0].limit == 20

    def test_update_rate_not_found(self):
        """测试更新不存在的限流级别"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10)
        result = limiter.update_rate(999, 20)
        assert result is False

    def test_get_status(self):
        """测试获取限流器状态"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(10).set_per_minute(100)
        status = limiter.get_status()
        assert "level_0" in status
        assert "level_1" in status
        assert status["level_0"]["period"] == 1.0
        assert status["level_1"]["period"] == 60.0

    def test_from_config(self):
        """测试从配置创建限流器"""
        limiter = MultiRateLimiter.from_config(
            requests_per_second=10,
            requests_per_minute=100,
            requests_per_hour=1000
        )
        assert len(limiter._limits) == 3


class TestProxyInfo:
    """代理信息测试"""

    def test_proxy_info_initialization(self):
        """测试代理信息初始化"""
        proxy = ProxyInfo(url="http://proxy:8080")
        assert proxy.url == "http://proxy:8080"
        assert proxy.protocol == "http"
        assert proxy.quality_score == 50.0

    def test_success_rate_no_requests(self):
        """测试无请求时的成功率"""
        proxy = ProxyInfo(url="http://proxy:8080")
        assert proxy.success_rate == 1.0

    def test_success_rate_with_requests(self):
        """测试有请求时的成功率"""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=8, failure_count=2)
        assert proxy.success_rate == 0.8

    def test_total_requests(self):
        """测试总请求数"""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=8, failure_count=2)
        assert proxy.total_requests == 10

    def test_is_healthy_few_requests(self):
        """测试请求数少时判断健康"""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=2, failure_count=1)
        assert proxy.is_healthy is True

    def test_is_healthy_good_proxy(self):
        """测试健康代理"""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=8, failure_count=2, quality_score=50)
        assert proxy.is_healthy is True

    def test_is_healthy_unhealthy_proxy(self):
        """测试不健康代理"""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=2, failure_count=8, quality_score=50)
        assert proxy.is_healthy is False

    def test_weight_calculation(self):
        """测试权重计算"""
        proxy = ProxyInfo(
            url="http://proxy:8080",
            quality_score=80,
            success_count=9,
            failure_count=1,
            avg_response_time=0.5
        )
        weight = proxy.weight
        assert 0 < weight <= 1

    def test_to_dict(self):
        """测试转换为字典"""
        proxy = ProxyInfo(url="http://proxy:8080", country="US")
        data = proxy.to_dict()
        assert data["url"] == "http://proxy:8080"
        assert data["country"] == "US"


class TestProxyPool:
    """代理池测试"""

    def test_add_proxy(self):
        """测试添加代理"""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy:8080")
        result = pool.add_proxy(proxy)
        assert result is True
        assert len(pool) == 1

    def test_add_proxy_empty_url(self):
        """测试添加空URL代理"""
        pool = ProxyPool()
        proxy = ProxyInfo(url="")
        result = pool.add_proxy(proxy)
        assert result is False

    def test_add_proxies_batch(self):
        """测试批量添加代理"""
        pool = ProxyPool()
        proxies = [
            ProxyInfo(url="http://proxy1:8080"),
            ProxyInfo(url="http://proxy2:8080"),
            ProxyInfo(url="http://proxy3:8080"),
        ]
        count = pool.add_proxies(proxies)
        assert count == 3
        assert len(pool) == 3

    def test_remove_proxy(self):
        """测试移除代理"""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy:8080")
        pool.add_proxy(proxy)
        result = pool.remove_proxy("http://proxy:8080")
        assert result is True
        assert len(pool) == 0

    def test_remove_proxy_not_found(self):
        """测试移除不存在的代理"""
        pool = ProxyPool()
        result = pool.remove_proxy("http://nonexistent:8080")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_proxy_random(self):
        """测试随机获取代理"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080"),
            ProxyInfo(url="http://proxy2:8080"),
        ])
        proxy = await pool.get_proxy(strategy=ProxyStrategy.RANDOM)
        assert proxy is not None
        assert proxy.url in ["http://proxy1:8080", "http://proxy2:8080"]

    @pytest.mark.asyncio
    async def test_get_proxy_weighted(self):
        """测试加权获取代理"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080", quality_score=90),
            ProxyInfo(url="http://proxy2:8080", quality_score=50),
        ])
        proxy = await pool.get_proxy(strategy=ProxyStrategy.WEIGHTED)
        assert proxy is not None

    @pytest.mark.asyncio
    async def test_get_proxy_round_robin(self):
        """测试轮询获取代理"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080"),
            ProxyInfo(url="http://proxy2:8080"),
        ])
        proxy1 = await pool.get_proxy(strategy=ProxyStrategy.ROUND_ROBIN)
        proxy2 = await pool.get_proxy(strategy=ProxyStrategy.ROUND_ROBIN)
        assert proxy1.url != proxy2.url

    @pytest.mark.asyncio
    async def test_get_proxy_least_used(self):
        """测试最少使用获取代理"""
        pool = ProxyPool()
        proxy1 = ProxyInfo(url="http://proxy1:8080", success_count=10)
        proxy2 = ProxyInfo(url="http://proxy2:8080", success_count=2)
        pool.add_proxies([proxy1, proxy2])
        proxy = await pool.get_proxy(strategy=ProxyStrategy.LEAST_USED)
        assert proxy.url == "http://proxy2:8080"

    @pytest.mark.asyncio
    async def test_get_proxy_by_country(self):
        """测试按国家获取代理"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080", country="US"),
            ProxyInfo(url="http://proxy2:8080", country="CN"),
        ])
        proxy = await pool.get_proxy(strategy=ProxyStrategy.GEO, country="US")
        assert proxy is not None
        assert proxy.country == "US"

    @pytest.mark.asyncio
    async def test_get_proxy_empty_pool(self):
        """测试空代理池获取"""
        pool = ProxyPool()
        proxy = await pool.get_proxy()
        assert proxy is None

    def test_report_success(self):
        """测试报告成功"""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy:8080")
        pool.add_proxy(proxy)
        pool.report_success(proxy, response_time=0.5)
        stored = pool._proxies["http://proxy:8080"]
        assert stored.success_count == 1
        assert stored.avg_response_time == 0.5

    def test_report_failure(self):
        """测试报告失败"""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy:8080")
        pool.add_proxy(proxy)
        pool.report_failure(proxy)
        stored = pool._proxies["http://proxy:8080"]
        assert stored.failure_count == 1
        assert stored.quality_score < 50

    @pytest.mark.asyncio
    async def test_health_check(self):
        """测试健康检查"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080", success_count=8, failure_count=2),
            ProxyInfo(url="http://proxy2:8080", success_count=2, failure_count=8),
        ])
        result = await pool.health_check()
        assert result["total"] == 2
        assert result["healthy"] == 1
        assert result["unhealthy"] == 1

    def test_get_stats(self):
        """测试获取统计信息"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080", quality_score=80, country="US"),
            ProxyInfo(url="http://proxy2:8080", quality_score=60, country="CN"),
        ])
        stats = pool.get_stats()
        assert stats["total"] == 2
        assert stats["avg_quality"] == 70.0
        assert set(stats["countries"]) == {"US", "CN"}

    def test_clear(self):
        """测试清空代理池"""
        pool = ProxyPool()
        pool.add_proxies([
            ProxyInfo(url="http://proxy1:8080"),
            ProxyInfo(url="http://proxy2:8080"),
        ])
        pool.clear()
        assert len(pool) == 0

    def test_contains(self):
        """测试代理是否存在"""
        pool = ProxyPool()
        pool.add_proxy(ProxyInfo(url="http://proxy:8080"))
        assert "http://proxy:8080" in pool
        assert "http://nonexistent:8080" not in pool


class TestBloomFilter:
    """布隆过滤器测试"""

    def test_initialization(self):
        """测试初始化"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        assert bf.capacity == 1000
        assert bf.error_rate == 0.01
        assert bf.element_count == 0

    def test_invalid_error_rate(self):
        """测试无效错误率"""
        with pytest.raises(ValueError):
            BloomFilter(capacity=1000, error_rate=0)
        with pytest.raises(ValueError):
            BloomFilter(capacity=1000, error_rate=1)

    def test_invalid_capacity(self):
        """测试无效容量"""
        with pytest.raises(ValueError):
            BloomFilter(capacity=0, error_rate=0.01)

    def test_add_and_contains(self):
        """测试添加和包含检查"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        bf.add("test_item")
        assert "test_item" in bf
        assert "nonexistent" not in bf

    def test_add_multiple_items(self):
        """测试添加多个元素"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bf.add(item)
        
        for item in items:
            assert item in bf
        
        assert len(bf) == 100

    def test_union(self):
        """测试并集操作"""
        bf1 = BloomFilter(capacity=1000, error_rate=0.01)
        bf2 = BloomFilter(capacity=1000, error_rate=0.01)
        
        bf1.add("item1")
        bf2.add("item2")
        
        union = bf1.union(bf2)
        assert "item1" in union
        assert "item2" in union

    def test_union_different_sizes(self):
        """测试不同大小的并集"""
        bf1 = BloomFilter(capacity=1000, error_rate=0.01)
        bf2 = BloomFilter(capacity=2000, error_rate=0.01)
        
        with pytest.raises(ValueError):
            bf1.union(bf2)

    def test_inplace_union(self):
        """测试原地并集操作"""
        bf1 = BloomFilter(capacity=1000, error_rate=0.01)
        bf2 = BloomFilter(capacity=1000, error_rate=0.01)
        
        bf1.add("item1")
        bf2.add("item2")
        
        bf1 |= bf2
        assert "item1" in bf1
        assert "item2" in bf1

    def test_save_and_load(self):
        """测试保存和加载"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        for i in range(100):
            bf.add(f"item_{i}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bloom") as f:
            temp_path = Path(f.name)
        
        try:
            bf.save(temp_path)
            loaded = BloomFilter.load(temp_path)
            
            assert loaded.capacity == bf.capacity
            assert loaded.element_count == bf.element_count
            
            for i in range(100):
                assert f"item_{i}" in loaded
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_clear(self):
        """测试清空过滤器"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        bf.add("item1")
        bf.add("item2")
        bf.clear()
        assert bf.element_count == 0
        assert "item1" not in bf

    def test_load_factor(self):
        """测试负载因子"""
        bf = BloomFilter(capacity=100, error_rate=0.01)
        assert bf.load_factor == 0.0
        
        for i in range(50):
            bf.add(f"item_{i}")
        assert bf.load_factor == 0.5

    def test_estimated_false_positive_rate(self):
        """测试估计的假阳性率"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        assert bf.estimated_false_positive_rate == 0.0
        
        for i in range(500):
            bf.add(f"item_{i}")
        assert bf.estimated_false_positive_rate > 0

    def test_get_info(self):
        """测试获取过滤器信息"""
        bf = BloomFilter(capacity=1000, error_rate=0.01)
        info = bf.get_info()
        assert "capacity" in info
        assert "error_rate" in info
        assert "bit_size" in info
        assert "hash_count" in info


class TestScalableBloomFilter:
    """可扩展布隆过滤器测试"""

    def test_initialization(self):
        """测试初始化"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        assert sbf._initial_capacity == 100
        assert len(sbf._filters) == 1

    def test_add_and_contains(self):
        """测试添加和包含检查"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        sbf.add("test_item")
        assert "test_item" in sbf
        assert "nonexistent" not in sbf

    def test_auto_expand(self):
        """测试自动扩展"""
        sbf = ScalableBloomFilter(initial_capacity=10, error_rate=0.01)
        
        for i in range(50):
            sbf.add(f"item_{i}")
        
        assert len(sbf._filters) > 1
        assert len(sbf) == 50

    def test_add_many(self):
        """测试批量添加"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        items = [f"item_{i}" for i in range(100)]
        sbf.add_many(items)
        assert len(sbf) == 100

    def test_add_duplicate(self):
        """测试添加重复元素"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        sbf.add("item1")
        sbf.add("item1")
        assert len(sbf) == 1

    def test_save_and_load(self):
        """测试保存和加载"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        for i in range(200):
            sbf.add(f"item_{i}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sbf") as f:
            temp_path = Path(f.name)
        
        try:
            sbf.save(temp_path)
            loaded = ScalableBloomFilter.load(temp_path)
            
            assert len(loaded) == len(sbf)
            for i in range(200):
                assert f"item_{i}" in loaded
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_clear(self):
        """测试清空过滤器"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        sbf.add("item1")
        sbf.add("item2")
        sbf.clear()
        assert len(sbf) == 0
        assert "item1" not in sbf

    def test_get_info(self):
        """测试获取过滤器信息"""
        sbf = ScalableBloomFilter(initial_capacity=100, error_rate=0.01)
        info = sbf.get_info()
        assert "initial_capacity" in info
        assert "error_rate" in info
        assert "element_count" in info
        assert "filter_count" in info


class TestDataCleaner:
    """数据清洗测试"""

    def test_initialization(self):
        """测试初始化"""
        cleaner = DataCleaner()
        assert cleaner.remove_html is True
        assert cleaner.remove_urls is False

    def test_clean_text_basic(self):
        """测试基本文本清洗"""
        cleaner = DataCleaner()
        text = "  Hello   World  "
        result = cleaner.clean_text(text)
        assert result == "Hello World"

    def test_clean_text_html(self):
        """测试HTML清洗"""
        cleaner = DataCleaner(remove_html=True)
        text = "<p>Hello <b>World</b></p>"
        result = cleaner.clean_text(text)
        assert "<p>" not in result
        assert "<b>" not in result

    def test_clean_text_urls(self):
        """测试URL清洗"""
        cleaner = DataCleaner(remove_urls=True)
        text = "Visit https://example.com for more info"
        result = cleaner.clean_text(text)
        assert "https://example.com" not in result

    def test_clean_text_mentions(self):
        """测试提及清洗"""
        cleaner = DataCleaner(remove_mentions=True)
        text = "Hello @user1 and @user2"
        result = cleaner.clean_text(text)
        assert "@user1" not in result
        assert "@user2" not in result

    def test_clean_text_hashtags(self):
        """测试话题标签清洗"""
        cleaner = DataCleaner(remove_hashtags=True)
        text = "This is #test #example"
        result = cleaner.clean_text(text)
        assert "#test" not in result
        assert "#example" not in result

    def test_clean_text_lowercase(self):
        """测试小写转换"""
        cleaner = DataCleaner(lowercase=True)
        text = "Hello World"
        result = cleaner.clean_text(text)
        assert result == "hello world"

    def test_clean_text_emoji(self):
        """测试表情符号清洗"""
        cleaner = DataCleaner(remove_emoji=True)
        text = "Hello 😀 World 🎉"
        result = cleaner.clean_text(text)
        assert "😀" not in result
        assert "🎉" not in result

    def test_clean_text_max_length(self):
        """测试最大长度限制"""
        cleaner = DataCleaner(max_length=10)
        text = "This is a very long text"
        result = cleaner.clean_text(text)
        assert len(result) == 10

    def test_clean_text_empty(self):
        """测试空文本"""
        cleaner = DataCleaner()
        result = cleaner.clean_text("")
        assert result == ""

    def test_normalize_content(self):
        """测试内容标准化"""
        cleaner = DataCleaner()
        content = "Check https://example.com #test @user"
        result = cleaner.normalize_content(content)
        
        assert isinstance(result, CleanedContent)
        assert result.original_length == len(content)
        assert len(result.urls) == 1
        assert len(result.hashtags) == 1
        assert len(result.mentions) == 1

    def test_normalize_content_dict(self):
        """测试字典内容标准化"""
        cleaner = DataCleaner()
        content = {"text": "Hello World", "extra": "data"}
        result = cleaner.normalize_content(content)
        assert result.text == "Hello World"

    def test_detect_language_chinese(self):
        """测试中文检测"""
        cleaner = DataCleaner()
        text = "这是一段中文文本"
        lang = cleaner.detect_language(text)
        assert lang == "zh"

    def test_detect_language_english(self):
        """测试英文检测"""
        cleaner = DataCleaner()
        text = "This is an English text"
        lang = cleaner.detect_language(text)
        assert lang == "en"

    def test_detect_language_empty(self):
        """测试空文本语言检测"""
        cleaner = DataCleaner()
        lang = cleaner.detect_language("")
        assert lang is None

    def test_extract_urls(self):
        """测试URL提取"""
        cleaner = DataCleaner()
        text = "Visit https://example.com and http://test.org"
        urls = cleaner.extract_urls(text)
        assert len(urls) == 2

    def test_extract_hashtags(self):
        """测试话题标签提取"""
        cleaner = DataCleaner()
        text = "This is #test and #example"
        hashtags = cleaner.extract_hashtags(text)
        assert "test" in hashtags
        assert "example" in hashtags

    def test_extract_mentions(self):
        """测试提及提取"""
        cleaner = DataCleaner()
        text = "Hello @user1 and @user2"
        mentions = cleaner.extract_mentions(text)
        assert "user1" in mentions
        assert "user2" in mentions

    def test_normalize_url(self):
        """测试URL标准化"""
        cleaner = DataCleaner()
        url = "HTTPS://Example.COM/Path/"
        result = cleaner.normalize_url(url)
        assert "https://" in result.lower()
        assert "example.com" in result.lower()

    def test_remove_duplicates(self):
        """测试去重"""
        cleaner = DataCleaner()
        items = ["a", "b", "a", "c", "b"]
        result = cleaner.remove_duplicates(items)
        assert len(result) == 3
        assert set(result) == {"a", "b", "c"}

    def test_remove_duplicates_dict(self):
        """测试字典去重"""
        cleaner = DataCleaner()
        items = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
            {"id": 1, "name": "c"},
        ]
        result = cleaner.remove_duplicates(items, key="id")
        assert len(result) == 2

    def test_clean_batch(self):
        """测试批量清洗"""
        cleaner = DataCleaner()
        texts = ["  Hello  ", "  World  ", "  Test  "]
        results = cleaner.clean_batch(texts)
        assert results == ["Hello", "World", "Test"]

    def test_normalize_batch(self):
        """测试批量标准化"""
        cleaner = DataCleaner()
        contents = ["Hello #test", "World #example"]
        results = cleaner.normalize_batch(contents)
        assert len(results) == 2
        assert all(isinstance(r, CleanedContent) for r in results)

    def test_get_stats(self):
        """测试获取统计信息"""
        cleaner = DataCleaner()
        contents = ["Hello World", "Test Content"]
        results = cleaner.normalize_batch(contents)
        stats = cleaner.get_stats(results)
        
        assert stats["total"] == 2
        assert stats["avg_original_length"] > 0

    def test_create_default_cleaner(self):
        """测试创建默认清洗器"""
        cleaner = create_default_cleaner()
        assert cleaner.remove_html is True
        assert cleaner.remove_urls is False

    def test_create_strict_cleaner(self):
        """测试创建严格清洗器"""
        cleaner = create_strict_cleaner()
        assert cleaner.remove_html is True
        assert cleaner.remove_urls is True
        assert cleaner.remove_mentions is True
        assert cleaner.lowercase is True
