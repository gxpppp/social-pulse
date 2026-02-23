"""
Twitter/X 爬虫模块

提供Twitter/X平台的数据采集功能，支持API模式和浏览器模式。
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from loguru import logger

from .base import (
    AuthenticationException,
    BaseCrawler,
    CrawlerException,
    NetworkException,
    ParseException,
    Platform,
    ProxyPool,
    RateLimitConfig,
    RateLimitException,
)
from storage.models import Post, User


class TwitterRateLimitException(RateLimitException):
    """Twitter速率限制异常"""

    def __init__(
        self,
        message: str = "Twitter rate limit exceeded",
        retry_after: Optional[float] = None,
        endpoint: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, retry_after=retry_after, **context)
        self.endpoint = endpoint


class TwitterAuthException(AuthenticationException):
    """Twitter认证异常"""

    def __init__(
        self,
        message: str = "Twitter authentication failed",
        auth_type: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, **context)
        self.auth_type = auth_type


class TwitterNotFoundException(CrawlerException):
    """Twitter资源不存在异常"""

    def __init__(
        self,
        message: str = "Twitter resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **context: Any,
    ):
        super().__init__(message, platform=Platform.TWITTER, **context)
        self.resource_type = resource_type
        self.resource_id = resource_id


@dataclass
class TwitterConfig:
    """
    Twitter爬虫配置类

    Attributes:
        bearer_token: API认证令牌（用于API v2）
        api_key: OAuth 1.0a API密钥
        api_secret: OAuth 1.0a API密钥密文
        access_token: OAuth 1.0a 访问令牌
        access_token_secret: OAuth 1.0a 访问令牌密文
        use_browser: 是否使用浏览器模式
        headless: 浏览器无头模式
        browser_timeout: 浏览器超时时间（秒）
        scroll_limit: 滚动加载次数限制
        cookies_path: Cookie存储路径
        user_agent: 用户代理字符串
    """

    bearer_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_token_secret: Optional[str] = None
    use_browser: bool = False
    headless: bool = True
    browser_timeout: int = 30
    scroll_limit: int = 10
    cookies_path: Optional[str] = None
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def validate_api_mode(self) -> bool:
        """验证API模式配置是否完整"""
        return self.bearer_token is not None

    def validate_oauth_mode(self) -> bool:
        """验证OAuth模式配置是否完整"""
        return all([
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_token_secret,
        ])


@dataclass
class TwitterMetrics:
    """推文互动指标"""

    likes: int = 0
    retweets: int = 0
    replies: int = 0
    quotes: int = 0
    views: int = 0
    bookmarks: int = 0


@dataclass
class TwitterTweet:
    """推文数据结构"""

    tweet_id: str
    author_id: str
    author_username: Optional[str] = None
    content: str = ""
    created_at: Optional[datetime] = None
    language: Optional[str] = None
    metrics: TwitterMetrics = field(default_factory=TwitterMetrics)
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    media_urls: list[str] = field(default_factory=list)
    is_retweet: bool = False
    is_quote: bool = False
    is_reply: bool = False
    parent_id: Optional[str] = None
    quoted_id: Optional[str] = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TwitterUser:
    """用户数据结构"""

    user_id: str
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    avatar_url: Optional[str] = None
    banner_url: Optional[str] = None
    followers_count: int = 0
    following_count: int = 0
    tweets_count: int = 0
    likes_count: int = 0
    verified: bool = False
    verified_type: Optional[str] = None
    created_at: Optional[datetime] = None
    raw_data: dict[str, Any] = field(default_factory=dict)


class TwitterParser:
    """
    Twitter数据解析器

    负责解析API响应和网页元素，转换为统一的数据模型。
    """

    @staticmethod
    def parse_tweet(data: dict[str, Any], includes: Optional[dict[str, Any]] = None) -> TwitterTweet:
        """
        解析单条推文

        Args:
            data: 推文原始数据
            includes: 包含的关联数据（用户等）

        Returns:
            TwitterTweet对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            includes = includes or {}
            entities = data.get("entities", {})
            public_metrics = data.get("public_metrics", {})

            hashtags = [h.get("tag", "") for h in entities.get("hashtags", [])]
            mentions = [m.get("username", "") for m in entities.get("mentions", [])]
            urls = [u.get("expanded_url", u.get("url", "")) for u in entities.get("urls", [])]

            media_urls = []
            if "attachments" in data:
                media_keys = data["attachments"].get("media_keys", [])
                for media in includes.get("media", []):
                    if media.get("media_key") in media_keys:
                        media_urls.append(media.get("url", ""))

            author_id = data.get("author_id", "")
            author_username = author_id
            for user in includes.get("users", []):
                if user.get("id") == author_id:
                    author_username = user.get("username", author_id)
                    break

            referenced_tweets = data.get("referenced_tweets", [])
            is_retweet = any(r.get("type") == "retweeted" for r in referenced_tweets)
            is_quote = any(r.get("type") == "quoted" for r in referenced_tweets)
            is_reply = any(r.get("type") == "replied_to" for r in referenced_tweets)

            parent_id = None
            quoted_id = None
            for ref in referenced_tweets:
                if ref.get("type") == "replied_to":
                    parent_id = ref.get("id")
                elif ref.get("type") == "quoted":
                    quoted_id = ref.get("id")

            metrics = TwitterMetrics(
                likes=public_metrics.get("like_count", 0),
                retweets=public_metrics.get("retweet_count", 0),
                replies=public_metrics.get("reply_count", 0),
                quotes=public_metrics.get("quote_count", 0),
                views=public_metrics.get("impression_count", 0),
                bookmarks=public_metrics.get("bookmark_count", 0),
            )

            created_at = None
            if data.get("created_at"):
                created_at = TwitterParser._parse_datetime(data["created_at"])

            return TwitterTweet(
                tweet_id=data.get("id", ""),
                author_id=author_id,
                author_username=author_username,
                content=data.get("text", ""),
                created_at=created_at,
                language=data.get("lang"),
                metrics=metrics,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                media_urls=media_urls,
                is_retweet=is_retweet,
                is_quote=is_quote,
                is_reply=is_reply,
                parent_id=parent_id,
                quoted_id=quoted_id,
                raw_data=data,
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse tweet: {e}",
                raw_data=data,
                platform=Platform.TWITTER,
            )

    @staticmethod
    def parse_user(data: dict[str, Any]) -> TwitterUser:
        """
        解析用户信息

        Args:
            data: 用户原始数据

        Returns:
            TwitterUser对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            public_metrics = data.get("public_metrics", {})

            created_at = None
            if data.get("created_at"):
                created_at = TwitterParser._parse_datetime(data["created_at"])

            return TwitterUser(
                user_id=data.get("id", ""),
                username=data.get("username", ""),
                display_name=data.get("name"),
                bio=data.get("description"),
                location=data.get("location"),
                website=data.get("url"),
                avatar_url=data.get("profile_image_url"),
                banner_url=data.get("profile_banner_url"),
                followers_count=public_metrics.get("followers_count", 0),
                following_count=public_metrics.get("following_count", 0),
                tweets_count=public_metrics.get("tweet_count", 0),
                likes_count=public_metrics.get("like_count", 0),
                verified=data.get("verified", False),
                verified_type=data.get("verified_type"),
                created_at=created_at,
                raw_data=data,
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse user: {e}",
                raw_data=data,
                platform=Platform.TWITTER,
            )

    @staticmethod
    def parse_metrics(element: Any) -> TwitterMetrics:
        """
        解析互动数据（浏览器模式）

        Args:
            element: 网页元素

        Returns:
            TwitterMetrics对象
        """
        metrics = TwitterMetrics()

        try:
            if hasattr(element, "query_selector_all"):
                pass
        except Exception:
            pass

        return metrics

    @staticmethod
    def parse_tweet_from_html(element: Any) -> Optional[TwitterTweet]:
        """
        从HTML元素解析推文（浏览器模式）

        Args:
            element: Playwright元素

        Returns:
            TwitterTweet对象或None
        """
        try:
            tweet_id = None
            tweet_link = element.query_selector('a[href*="/status/"]')
            if tweet_link:
                href = tweet_link.get_attribute("href") or ""
                match = re.search(r"/status/(\d+)", href)
                if match:
                    tweet_id = match.group(1)

            if not tweet_id:
                return None

            content_el = element.query_selector('[data-testid="tweetText"]')
            content = content_el.inner_text() if content_el else ""

            author_el = element.query_selector('[data-testid="User-Name"]')
            author_username = ""
            if author_el:
                username_el = author_el.query_selector("a")
                if username_el:
                    href = username_el.get_attribute("href") or ""
                    author_username = href.strip("/").split("/")[-1]

            author_id = author_username

            metrics = TwitterMetrics()

            metric_items = element.query_selector_all('[role="group"] [data-testid]')
            for item in metric_items:
                testid = item.get_attribute("data-testid") or ""
                aria_label = item.get_attribute("aria-label") or ""
                count = TwitterParser._extract_count(aria_label)

                if "reply" in testid:
                    metrics.replies = count
                elif "retweet" in testid:
                    metrics.retweets = count
                elif "like" in testid:
                    metrics.likes = count
                elif "bookmark" in testid:
                    metrics.bookmarks = count

            hashtags = re.findall(r"#(\w+)", content)
            mentions = re.findall(r"@(\w+)", content)
            urls = re.findall(r"https?://[^\s]+", content)

            return TwitterTweet(
                tweet_id=tweet_id,
                author_id=author_id,
                author_username=author_username,
                content=content,
                metrics=metrics,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
            )
        except Exception as e:
            logger.debug(f"Failed to parse tweet from HTML: {e}")
            return None

    @staticmethod
    def _extract_count(text: str) -> int:
        """从文本中提取数字"""
        match = re.search(r"[\d,]+", text)
        if match:
            num_str = match.group().replace(",", "")
            try:
                return int(num_str)
            except ValueError:
                pass

        if "K" in text or "k" in text:
            match = re.search(r"(\d+\.?\d*)[Kk]", text)
            if match:
                return int(float(match.group(1)) * 1000)
        if "M" in text or "m" in text:
            match = re.search(r"(\d+\.?\d*)[Mm]", text)
            if match:
                return int(float(match.group(1)) * 1000000)

        return 0

    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """解析ISO格式时间字符串"""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def to_post(tweet: TwitterTweet) -> Post:
        """将TwitterTweet转换为Post模型"""
        return Post(
            post_id=tweet.tweet_id,
            user_id=tweet.author_id,
            platform=Platform.TWITTER,
            content=tweet.content,
            language=tweet.language,
            posted_at=tweet.created_at,
            likes_count=tweet.metrics.likes,
            shares_count=tweet.metrics.retweets,
            comments_count=tweet.metrics.replies,
            hashtags=tweet.hashtags,
            mentions=tweet.mentions,
            urls=tweet.urls,
            media=[{"url": url} for url in tweet.media_urls],
        )

    @staticmethod
    def to_user(user: TwitterUser) -> User:
        """将TwitterUser转换为User模型"""
        return User(
            user_id=user.user_id,
            platform=Platform.TWITTER,
            username=user.username,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            created_at=user.created_at,
            followers_count=user.followers_count,
            friends_count=user.following_count,
            posts_count=user.tweets_count,
            verified=user.verified,
        )


class TwitterCrawler(BaseCrawler):
    """
    Twitter/X 爬虫

    支持API模式和浏览器模式采集Twitter数据。

    Attributes:
        platform: 平台标识（TWITTER）
        config: Twitter配置
        parser: 数据解析器
    """

    platform = Platform.TWITTER

    def __init__(
        self,
        config: TwitterConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
        proxy_pool: Optional[ProxyPool] = None,
    ) -> None:
        """
        初始化Twitter爬虫

        Args:
            config: Twitter配置
            rate_limit_config: 限流配置
            proxy_pool: 代理池
        """
        super().__init__(rate_limit_config, proxy_pool)
        self.config = config
        self.parser = TwitterParser()
        self._api_base = "https://api.twitter.com/2"
        self._session: Optional[Any] = None
        self._browser: Optional[Any] = None
        self._page: Optional[Any] = None
        self._is_running = False
        self._stream_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """初始化爬虫"""
        if self.config.use_browser:
            await self._init_browser()
        else:
            await self._init_api()

        self._is_running = True
        logger.info(f"Twitter crawler initialized (browser_mode={self.config.use_browser})")

    async def _init_api(self) -> None:
        """初始化API会话"""
        import aiohttp

        headers = {
            "User-Agent": self.config.user_agent,
            "Content-Type": "application/json",
        }

        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"

        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def _init_browser(self) -> None:
        """初始化浏览器"""
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
            )

            context_options = {
                "user_agent": self.config.user_agent,
                "viewport": {"width": 1280, "height": 720},
            }

            if self.config.cookies_path:
                import os
                if os.path.exists(self.config.cookies_path):
                    try:
                        with open(self.config.cookies_path, "r") as f:
                            cookies = json.load(f)
                            context_options["storage_state"] = {"cookies": cookies}
                    except Exception as e:
                        logger.warning(f"Failed to load cookies: {e}")

            self._context = await self._browser.new_context(**context_options)
            self._page = await self._context.new_page()
            self._page.set_default_timeout(self.config.browser_timeout * 1000)

            logger.info("Browser mode initialized")

        except ImportError:
            raise CrawlerException(
                "Playwright not installed. Run: pip install playwright && playwright install chromium",
                platform=Platform.TWITTER,
            )

    async def close(self) -> None:
        """关闭爬虫"""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        if self._browser:
            if self.config.cookies_path and self._page:
                try:
                    import os
                    os.makedirs(os.path.dirname(self.config.cookies_path) or ".", exist_ok=True)
                    storage = await self._context.storage_state()
                    with open(self.config.cookies_path, "w") as f:
                        json.dump(storage.get("cookies", []), f)
                except Exception as e:
                    logger.warning(f"Failed to save cookies: {e}")

            await self._browser.close()
            await self._playwright.stop()
            self._browser = None
            self._page = None

        self._is_running = False
        logger.info("Twitter crawler closed")

    async def _make_api_request(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        method: str = "GET",
        body: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        发送API请求

        Args:
            endpoint: API端点
            params: 查询参数
            method: HTTP方法
            body: 请求体

        Returns:
            响应数据

        Raises:
            TwitterRateLimitException: 速率限制
            TwitterAuthException: 认证失败
            TwitterNotFoundException: 资源不存在
        """
        if not self._session:
            raise RuntimeError("Crawler not initialized")

        url = f"{self._api_base}/{endpoint}"
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                await self.rate_limiter.acquire()

                if method == "GET":
                    async with self._session.get(url, params=params) as response:
                        return await self._handle_api_response(response, endpoint)
                else:
                    async with self._session.post(url, params=params, json=body) as response:
                        return await self._handle_api_response(response, endpoint)

            except TwitterRateLimitException as e:
                retry_after = e.retry_after or 60
                logger.warning(f"Rate limited on {endpoint}, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                retry_count += 1

            except TwitterAuthException:
                raise

            except Exception as e:
                logger.error(f"API request error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)

        return None

    async def _handle_api_response(self, response: Any, endpoint: str) -> Optional[dict[str, Any]]:
        """处理API响应"""
        status = response.status

        if status == 200:
            self.rate_limiter.record_success()
            return await response.json()

        elif status == 429:
            retry_after = int(response.headers.get("x-rate-limit-reset", 60))
            raise TwitterRateLimitException(
                f"Rate limit exceeded for {endpoint}",
                retry_after=retry_after,
                endpoint=endpoint,
            )

        elif status == 401:
            raise TwitterAuthException(
                "Authentication failed. Check your bearer token.",
                auth_type="bearer",
            )

        elif status == 403:
            raise TwitterAuthException(
                "Access forbidden. Check your API permissions.",
                auth_type="permissions",
            )

        elif status == 404:
            raise TwitterNotFoundException(
                f"Resource not found: {endpoint}",
                endpoint=endpoint,
            )

        elif status >= 500:
            raise NetworkException(
                f"Twitter API server error: {status}",
                status_code=status,
            )

        else:
            error_data = await response.json() if response.content_length else {}
            error_msg = error_data.get("detail", f"Request failed with status {status}")
            logger.error(f"API error: {error_msg}")
            return None

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
            user_ids: 用户ID列表
            post_ids: 帖子ID列表

        Yields:
            User或Post对象
        """
        await self.before_crawl()

        try:
            if query:
                limit = kwargs.get("limit", 100)
                async for post in self.crawl_keywords([query], limit):
                    yield post

            if user_ids:
                for user_id in user_ids:
                    user = await self.crawl_user_timeline(user_id, limit=1)
                    if user:
                        yield user

            if post_ids:
                for post_id in post_ids:
                    post = await self.crawl_post(post_id)
                    if post:
                        yield post

        finally:
            await self.after_crawl()

    async def crawl_keywords(
        self,
        keywords: list[str],
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        关键词搜索采集

        Args:
            keywords: 关键词列表
            limit: 采集数量限制

        Yields:
            Post对象
        """
        if self.config.use_browser:
            async for post in self._crawl_keywords_browser(keywords, limit):
                yield post
        else:
            async for post in self._crawl_keywords_api(keywords, limit):
                yield post

    async def _crawl_keywords_api(
        self,
        keywords: list[str],
        limit: int,
    ) -> AsyncIterator[Post]:
        """API模式关键词搜索"""
        query = " OR ".join(keywords)
        query += " -is:retweet lang:en"

        params = {
            "query": query,
            "max_results": min(100, limit),
            "tweet.fields": "created_at,public_metrics,entities,lang,referenced_tweets,attachments",
            "expansions": "author_id,attachments.media_keys",
            "user.fields": "username,name",
            "media.fields": "url,type",
        }

        count = 0
        next_token = None

        while count < limit:
            if next_token:
                params["next_token"] = next_token

            data = await self._make_api_request("tweets/search/recent", params)

            if not data or "data" not in data:
                break

            includes = data.get("includes", {})
            for tweet_data in data["data"]:
                if count >= limit:
                    break
                try:
                    tweet = self.parser.parse_tweet(tweet_data, includes)
                    yield self.parser.to_post(tweet)
                    count += 1
                except ParseException as e:
                    logger.warning(f"Failed to parse tweet: {e}")

            next_token = data.get("meta", {}).get("next_token")
            if not next_token:
                break

    async def _crawl_keywords_browser(
        self,
        keywords: list[str],
        limit: int,
    ) -> AsyncIterator[Post]:
        """浏览器模式关键词搜索"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        query = " OR ".join(keywords)
        search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"

        await self._page.goto(search_url, wait_until="networkidle")
        await asyncio.sleep(2)

        count = 0
        scroll_count = 0

        while count < limit and scroll_count < self.config.scroll_limit:
            tweets = await self._page.query_selector_all('[data-testid="tweet"]')

            for tweet_el in tweets:
                if count >= limit:
                    break

                tweet = self.parser.parse_tweet_from_html(tweet_el)
                if tweet:
                    yield self.parser.to_post(tweet)
                    count += 1

            await self._scroll_page()
            scroll_count += 1
            await asyncio.sleep(1)

    async def crawl_user_timeline(
        self,
        user_id: str,
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        用户时间线采集

        Args:
            user_id: 用户ID或用户名
            limit: 采集数量限制

        Yields:
            Post对象
        """
        if self.config.use_browser:
            async for post in self._crawl_user_timeline_browser(user_id, limit):
                yield post
        else:
            async for post in self._crawl_user_timeline_api(user_id, limit):
                yield post

    async def _crawl_user_timeline_api(
        self,
        user_id: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """API模式用户时间线"""
        params = {
            "max_results": min(100, limit),
            "tweet.fields": "created_at,public_metrics,entities,lang,referenced_tweets,attachments",
            "expansions": "attachments.media_keys",
            "media.fields": "url,type",
        }

        count = 0
        next_token = None

        while count < limit:
            if next_token:
                params["pagination_token"] = next_token

            data = await self._make_api_request(f"users/{user_id}/tweets", params)

            if not data or "data" not in data:
                break

            includes = data.get("includes", {})
            for tweet_data in data["data"]:
                if count >= limit:
                    break
                try:
                    tweet = self.parser.parse_tweet(tweet_data, includes)
                    yield self.parser.to_post(tweet)
                    count += 1
                except ParseException as e:
                    logger.warning(f"Failed to parse tweet: {e}")

            next_token = data.get("meta", {}).get("next_token")
            if not next_token:
                break

    async def _crawl_user_timeline_browser(
        self,
        user_id: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """浏览器模式用户时间线"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        profile_url = f"https://twitter.com/{user_id}"
        await self._page.goto(profile_url, wait_until="networkidle")
        await asyncio.sleep(2)

        count = 0
        scroll_count = 0

        while count < limit and scroll_count < self.config.scroll_limit:
            tweets = await self._page.query_selector_all('[data-testid="tweet"]')

            for tweet_el in tweets:
                if count >= limit:
                    break

                tweet = self.parser.parse_tweet_from_html(tweet_el)
                if tweet:
                    yield self.parser.to_post(tweet)
                    count += 1

            await self._scroll_page()
            scroll_count += 1
            await asyncio.sleep(1)

    async def crawl_hashtag(
        self,
        hashtag: str,
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        话题标签采集

        Args:
            hashtag: 话题标签（不含#）
            limit: 采集数量限制

        Yields:
            Post对象
        """
        if self.config.use_browser:
            async for post in self._crawl_hashtag_browser(hashtag, limit):
                yield post
        else:
            async for post in self._crawl_hashtag_api(hashtag, limit):
                yield post

    async def _crawl_hashtag_api(
        self,
        hashtag: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """API模式话题标签采集"""
        query = f"#{hashtag} -is:retweet"

        params = {
            "query": query,
            "max_results": min(100, limit),
            "tweet.fields": "created_at,public_metrics,entities,lang,referenced_tweets",
            "expansions": "author_id",
            "user.fields": "username,name",
        }

        count = 0
        next_token = None

        while count < limit:
            if next_token:
                params["next_token"] = next_token

            data = await self._make_api_request("tweets/search/recent", params)

            if not data or "data" not in data:
                break

            includes = data.get("includes", {})
            for tweet_data in data["data"]:
                if count >= limit:
                    break
                try:
                    tweet = self.parser.parse_tweet(tweet_data, includes)
                    yield self.parser.to_post(tweet)
                    count += 1
                except ParseException as e:
                    logger.warning(f"Failed to parse tweet: {e}")

            next_token = data.get("meta", {}).get("next_token")
            if not next_token:
                break

    async def _crawl_hashtag_browser(
        self,
        hashtag: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """浏览器模式话题标签采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        hashtag_url = f"https://twitter.com/hashtag/{hashtag}?f=live"
        await self._page.goto(hashtag_url, wait_until="networkidle")
        await asyncio.sleep(2)

        count = 0
        scroll_count = 0

        while count < limit and scroll_count < self.config.scroll_limit:
            tweets = await self._page.query_selector_all('[data-testid="tweet"]')

            for tweet_el in tweets:
                if count >= limit:
                    break

                tweet = self.parser.parse_tweet_from_html(tweet_el)
                if tweet:
                    yield self.parser.to_post(tweet)
                    count += 1

            await self._scroll_page()
            scroll_count += 1
            await asyncio.sleep(1)

    async def crawl_trending(self) -> list[dict[str, Any]]:
        """
        热门话题采集

        Returns:
            热门话题列表
        """
        if self.config.use_browser:
            return await self._crawl_trending_browser()
        else:
            return await self._crawl_trending_api()

    async def _crawl_trending_api(self) -> list[dict[str, Any]]:
        """API模式热门话题采集"""
        data = await self._make_api_request("trends/recent", {
            "trend.fields": "tweet_volume"
        })

        if not data or "data" not in data:
            return []

        trends = []
        for trend in data["data"]:
            trends.append({
                "name": trend.get("name", ""),
                "tweet_count": trend.get("tweet_volume", 0),
            })

        return trends

    async def _crawl_trending_browser(self) -> list[dict[str, Any]]:
        """浏览器模式热门话题采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.goto("https://twitter.com/explore", wait_until="networkidle")
        await asyncio.sleep(2)

        trends = []
        trend_items = await self._page.query_selector_all('[data-testid="trend"]')

        for item in trend_items[:10]:
            try:
                text_el = await item.query_selector("span")
                if text_el:
                    trend_name = await text_el.inner_text()
                    trends.append({"name": trend_name, "tweet_count": 0})
            except Exception:
                continue

        return trends

    async def stream_tweets(
        self,
        keywords: list[str],
    ) -> AsyncIterator[Post]:
        """
        流式API监听（异步生成器）

        实时监听包含指定关键词的推文。

        Args:
            keywords: 关键词列表

        Yields:
            Post对象
        """
        if self.config.use_browser:
            raise CrawlerException(
                "Streaming not supported in browser mode",
                platform=Platform.TWITTER,
            )

        stream_url = "https://api.twitter.com/2/tweets/search/stream"

        params = {
            "tweet.fields": "created_at,public_metrics,entities,lang,referenced_tweets",
            "expansions": "author_id",
            "user.fields": "username,name",
        }

        rules = [{"value": " OR ".join(keywords), "tag": "stream"}]

        add_rules_url = "https://api.twitter.com/2/tweets/search/stream/rules"
        await self._make_api_request(
            add_rules_url,
            method="POST",
            body={"add": rules},
        )

        logger.info(f"Started streaming for keywords: {keywords}")

        import aiohttp

        headers = {"Authorization": f"Bearer {self.config.bearer_token}"}

        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(stream_url, params=params) as response:
                async for line in response.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if "data" in data:
                            tweet = self.parser.parse_tweet(
                                data["data"],
                                data.get("includes", {}),
                            )
                            yield self.parser.to_post(tweet)
                    except json.JSONDecodeError:
                        continue
                    except ParseException as e:
                        logger.warning(f"Failed to parse streamed tweet: {e}")
                        continue

    async def crawl_user(self, user_id: str) -> Optional[User]:
        """爬取用户信息"""
        params = {
            "user.fields": "created_at,description,location,public_metrics,verified,verified_type,profile_image_url,url",
        }

        data = await self._make_api_request(f"users/{user_id}", params)

        if not data or "data" not in data:
            return None

        try:
            user = self.parser.parse_user(data["data"])
            return self.parser.to_user(user)
        except ParseException as e:
            logger.warning(f"Failed to parse user: {e}")
            return None

    async def crawl_post(self, post_id: str) -> Optional[Post]:
        """爬取单条帖子"""
        params = {
            "tweet.fields": "created_at,public_metrics,entities,lang,referenced_tweets,attachments",
            "expansions": "author_id,attachments.media_keys",
            "user.fields": "username,name",
            "media.fields": "url,type",
        }

        data = await self._make_api_request(f"tweets/{post_id}", params)

        if not data or "data" not in data:
            return None

        try:
            tweet = self.parser.parse_tweet(data["data"], data.get("includes", {}))
            return self.parser.to_post(tweet)
        except ParseException as e:
            logger.warning(f"Failed to parse post: {e}")
            return None

    async def parse_user(self, data: dict[str, Any]) -> User:
        """解析用户数据"""
        user = self.parser.parse_user(data)
        return self.parser.to_user(user)

    async def parse_post(self, data: dict[str, Any]) -> Post:
        """解析帖子数据"""
        tweet = self.parser.parse_tweet(data)
        return self.parser.to_post(tweet)

    async def _scroll_page(self) -> None:
        """滚动页面加载更多内容"""
        if self._page:
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)

    async def login_browser(self, username: str, password: str) -> bool:
        """
        浏览器模式登录

        Args:
            username: 用户名
            password: 密码

        Returns:
            登录是否成功
        """
        if not self._page:
            await self.initialize()

        try:
            await self._page.goto("https://twitter.com/i/flow/login", wait_until="networkidle")
            await asyncio.sleep(2)

            username_input = await self._page.query_selector('input[autocomplete="username"]')
            if username_input:
                await username_input.fill(username)
                await self._page.keyboard.press("Enter")
                await asyncio.sleep(2)

            password_input = await self._page.query_selector('input[name="password"]')
            if password_input:
                await password_input.fill(password)
                await self._page.keyboard.press("Enter")
                await asyncio.sleep(3)

            current_url = self._page.url
            if "home" in current_url:
                logger.info("Login successful")
                return True
            else:
                logger.warning("Login may have failed")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def __aenter__(self) -> "TwitterCrawler":
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.close()
