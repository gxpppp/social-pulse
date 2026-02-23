"""
微博平台适配器

实现微博爬虫、解析器和配置类。
"""

import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote, urlencode

from loguru import logger

from .base import (
    AuthenticationException,
    BaseCrawler,
    CrawlerException,
    NetworkException,
    ParseException,
    ProxyInfo,
    ProxyPool,
    RateLimitConfig,
    RateLimitException,
)
from ..storage.models import Interaction, InteractionType, Platform, Post, User


class WeiboAuthException(AuthenticationException):
    """微博认证异常"""

    def __init__(
        self,
        message: str = "Weibo authentication failed",
        **context: Any,
    ):
        super().__init__(message, platform=Platform.WEIBO, **context)


class WeiboRateLimitException(RateLimitException):
    """微博速率限制异常"""

    def __init__(
        self,
        message: str = "Weibo rate limit exceeded",
        retry_after: Optional[float] = None,
        **context: Any,
    ):
        super().__init__(message, retry_after=retry_after, platform=Platform.WEIBO, **context)


class WeiboBlockedException(CrawlerException):
    """微博账号被封异常"""

    def __init__(
        self,
        message: str = "Weibo account is blocked",
        **context: Any,
    ):
        super().__init__(message, platform=Platform.WEIBO, **context)


@dataclass
class WeiboConfig:
    """
    微博爬虫配置

    Attributes:
        cookies: 登录Cookie字符串
        use_mobile_api: 是否使用移动端API
        user_agent: 用户代理字符串
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        request_delay: 请求延迟范围（秒）
    """

    cookies: Optional[str] = None
    use_mobile_api: bool = True
    user_agent: str = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
    timeout: int = 30
    max_retries: int = 3
    request_delay: tuple[float, float] = (1.0, 3.0)

    def get_cookie_dict(self) -> dict[str, str]:
        """将Cookie字符串转换为字典"""
        if not self.cookies:
            return {}
        cookies: dict[str, str] = {}
        for item in self.cookies.split(";"):
            item = item.strip()
            if "=" in item:
                key, value = item.split("=", 1)
                cookies[key.strip()] = value.strip()
        return cookies


class WeiboParser:
    """
    微博数据解析器

    负责将微博API返回的原始数据转换为标准模型。
    """

    @staticmethod
    def parse_weibo(data: dict[str, Any]) -> Post:
        """
        解析微博内容

        Args:
            data: 微博原始数据

        Returns:
            Post对象

        Raises:
            ParseException: 解析失败
        """
        try:
            text = data.get("text", "")
            clean_text = WeiboParser._clean_html(text)

            user_data = data.get("user", {})
            user_id = str(user_data.get("id", data.get("user_id", "")))
            if not user_id:
                user_id = str(data.get("uid", ""))

            post_id = str(data.get("id", data.get("mid", "")))
            if not post_id:
                raise ParseException("Missing post ID", raw_data=data)

            created_at = WeiboParser._parse_weibo_time(data.get("created_at"))

            media: list[dict[str, Any]] = []
            pics = data.get("pics", [])
            if pics:
                for pic in pics:
                    if pic:
                        media.append({
                            "type": "image",
                            "url": pic.get("large", {}).get("url", pic.get("url", "")),
                            "thumbnail": pic.get("url", ""),
                        })

            page_info = data.get("page_info", {})
            if page_info:
                media_type = page_info.get("type", "")
                if media_type == "video":
                    media.append({
                        "type": "video",
                        "url": page_info.get("media_info", {}).get("stream_url", ""),
                        "thumbnail": page_info.get("page_pic", {}).get("url", ""),
                    })

            retweeted_status = data.get("retweeted_status")
            is_retweet = retweeted_status is not None

            return Post(
                post_id=post_id,
                user_id=user_id,
                platform=Platform.WEIBO,
                content=clean_text,
                language="zh",
                posted_at=created_at,
                likes_count=int(data.get("attitudes_count", data.get("like_count", 0))),
                shares_count=int(data.get("reposts_count", data.get("share_count", 0))),
                comments_count=int(data.get("comments_count", 0)),
                hashtags=WeiboParser._extract_hashtags(text),
                mentions=WeiboParser._extract_mentions(text),
                urls=WeiboParser._extract_urls(text),
                media=media,
            )
        except Exception as e:
            if isinstance(e, ParseException):
                raise
            raise ParseException(f"Failed to parse weibo: {e}", raw_data=data) from e

    @staticmethod
    def parse_user(data: dict[str, Any]) -> User:
        """
        解析用户信息

        Args:
            data: 用户原始数据

        Returns:
            User对象

        Raises:
            ParseException: 解析失败
        """
        try:
            user_id = str(data.get("id", ""))
            if not user_id:
                raise ParseException("Missing user ID", raw_data=data)

            return User(
                user_id=user_id,
                platform=Platform.WEIBO,
                username=data.get("screen_name", data.get("name", "")),
                display_name=data.get("screen_name", data.get("name", "")),
                bio=data.get("description", data.get("bio", "")),
                avatar_url=data.get("avatar_hd", data.get("avatar_large", data.get("profile_image_url", ""))),
                created_at=WeiboParser._parse_weibo_time(data.get("created_at")),
                followers_count=int(data.get("followers_count", data.get("followers", 0))),
                friends_count=int(data.get("follow_count", data.get("friends", 0))),
                posts_count=int(data.get("statuses_count", data.get("statuses", 0))),
                verified=data.get("verified", False),
            )
        except Exception as e:
            if isinstance(e, ParseException):
                raise
            raise ParseException(f"Failed to parse user: {e}", raw_data=data) from e

    @staticmethod
    def parse_comments(data: dict[str, Any]) -> list[Interaction]:
        """
        解析评论数据

        Args:
            data: 评论原始数据（通常是评论列表）

        Returns:
            Interaction对象列表

        Raises:
            ParseException: 解析失败
        """
        try:
            interactions: list[Interaction] = []
            comments = data if isinstance(data, list) else data.get("data", data.get("comments", []))

            for comment in comments:
                if not isinstance(comment, dict):
                    continue

                comment_id = str(comment.get("id", ""))
                if not comment_id:
                    continue

                user_data = comment.get("user", {})
                source_user_id = str(user_data.get("id", ""))
                target_post_id = str(comment.get("status", {}).get("id", "")) if comment.get("status") else ""

                interactions.append(Interaction(
                    interaction_id=f"weibo_comment_{comment_id}",
                    interaction_type=InteractionType.COMMENT,
                    source_user_id=source_user_id,
                    target_post_id=target_post_id,
                    timestamp=WeiboParser._parse_weibo_time(comment.get("created_at")),
                    content=WeiboParser._clean_html(comment.get("text", "")),
                ))

            return interactions
        except Exception as e:
            raise ParseException(f"Failed to parse comments: {e}", raw_data=data) from e

    @staticmethod
    def _clean_html(text: str) -> str:
        """清理HTML标签"""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    @staticmethod
    def _extract_hashtags(text: str) -> list[str]:
        """提取话题标签"""
        if not text:
            return []
        hashtags = re.findall(r"#([^#]+)#", text)
        return [h.strip() for h in hashtags if h.strip()]

    @staticmethod
    def _extract_mentions(text: str) -> list[str]:
        """提取提及用户"""
        if not text:
            return []
        mentions = re.findall(r"@([\w\u4e00-\u9fff\-_]+)", text)
        return [m.strip() for m in mentions if m.strip()]

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        """提取URL"""
        if not text:
            return []
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
        return [u.strip() for u in urls if u.strip()]

    @staticmethod
    def _parse_weibo_time(time_str: Optional[str]) -> Optional[datetime]:
        """解析微博时间格式"""
        if not time_str:
            return None
        formats = [
            "%a %b %d %H:%M:%S %z %Y",
            "%a %b %d %H:%M:%S %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        return None


class WeiboCrawler(BaseCrawler):
    """
    微博爬虫

    继承BaseCrawler，实现微博平台的数据采集功能。
    支持移动端API和网页模式两种采集方式。

    Attributes:
        platform: 平台标识（WEIBO）
        config: 微博配置
        parser: 数据解析器
    """

    platform = Platform.WEIBO

    def __init__(
        self,
        config: WeiboConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
        proxy_pool: Optional[ProxyPool] = None,
    ) -> None:
        """
        初始化微博爬虫

        Args:
            config: 微博配置
            rate_limit_config: 限流配置
            proxy_pool: 代理池
        """
        super().__init__(rate_limit_config, proxy_pool)
        self.config = config
        self.parser = WeiboParser()
        self._session: Optional[Any] = None
        self._browser: Optional[Any] = None
        self._context: Optional[Any] = None
        self._page: Optional[Any] = None

        if config.use_mobile_api:
            self._api_base = "https://m.weibo.cn/api"
        else:
            self._api_base = "https://weibo.com/ajax"

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
            query: 搜索关键词
            user_ids: 用户ID列表
            post_ids: 帖子ID列表
            **kwargs: 其他参数（如limit）

        Yields:
            用户或帖子对象
        """
        await self.before_crawl()
        limit = kwargs.get("limit", 100)

        try:
            if query:
                async for post in self.crawl_search(query, limit):
                    yield post
            elif user_ids:
                for user_id in user_ids:
                    user = await self.crawl_user(user_id)
                    if user:
                        yield user
                    async for post in self.crawl_user_posts(user_id, limit):
                        yield post
            elif post_ids:
                for post_id in post_ids:
                    post = await self._crawl_single_post(post_id)
                    if post:
                        yield post
        finally:
            await self.after_crawl()

    async def parse_user(self, data: dict[str, Any]) -> User:
        """解析用户数据"""
        return self.parser.parse_user(data)

    async def parse_post(self, data: dict[str, Any]) -> Post:
        """解析帖子数据"""
        return self.parser.parse_weibo(data)

    async def initialize(self) -> None:
        """初始化爬虫"""
        if self.config.use_mobile_api:
            await self._init_mobile_api()
        else:
            await self._init_browser()

    async def _init_mobile_api(self) -> None:
        """初始化移动端API模式"""
        import aiohttp

        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://m.weibo.cn/",
            "X-Requested-With": "XMLHttpRequest",
            "MWeibo-Pwa": "1",
        }

        if self.config.cookies:
            headers["Cookie"] = self.config.cookies

        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            cookie_jar=aiohttp.CookieJar(),
        )

        if self.config.cookies:
            cookie_dict = self.config.get_cookie_dict()
            for key, value in cookie_dict.items():
                self._session.cookie_jar.update_cookies({key: value})

        logger.info("Weibo mobile API crawler initialized")

    async def _init_browser(self) -> None:
        """初始化浏览器模式"""
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            self._context = await self._browser.new_context(
                user_agent=self.config.user_agent,
                viewport={"width": 375, "height": 812},
                device_scale_factor=3,
            )

            if self.config.cookies:
                cookies = []
                for item in self.config.cookies.split(";"):
                    item = item.strip()
                    if "=" in item:
                        key, value = item.split("=", 1)
                        cookies.append({
                            "name": key.strip(),
                            "value": value.strip(),
                            "domain": ".weibo.com",
                            "path": "/",
                        })
                if cookies:
                    await self._context.add_cookies(cookies)

            self._page = await self._context.new_page()
            logger.info("Weibo browser crawler initialized")
        except ImportError:
            logger.warning("Playwright not installed, falling back to mobile API")
            self.config.use_mobile_api = True
            await self._init_mobile_api()

    async def close(self) -> None:
        """关闭爬虫"""
        if self._session:
            await self._session.close()
            self._session = None

        if self._page:
            await self._page.close()
            self._page = None
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if hasattr(self, "_playwright") and self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Weibo crawler closed")

    async def crawl_search(
        self,
        keyword: str,
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        搜索采集

        Args:
            keyword: 搜索关键词
            limit: 最大采集数量

        Yields:
            Post对象
        """
        if self.config.use_mobile_api:
            async for post in self._search_mobile_api(keyword, limit):
                yield post
        else:
            async for post in self._search_browser(keyword, limit):
                yield post

    async def _search_mobile_api(
        self,
        keyword: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """移动端API搜索"""
        encoded_keyword = quote(keyword)
        containerid = f"100103type=1&q={encoded_keyword}"

        page = 1
        count = 0

        while count < limit:
            params = {
                "containerid": containerid,
                "page_type": "searchall",
                "page": page,
            }

            data = await self._make_request(
                f"{self._api_base}/container/getIndex",
                params,
            )

            if not data or data.get("ok") != 1:
                break

            cards = data.get("data", {}).get("cards", [])
            if not cards:
                break

            for card in cards:
                if count >= limit:
                    break

                card_type = card.get("card_type", 0)
                if card_type == 9:
                    mblog = card.get("mblog")
                    if mblog:
                        try:
                            post = self.parser.parse_weibo(mblog)
                            yield post
                            count += 1
                            self._stats["posts_collected"] += 1
                        except ParseException as e:
                            logger.warning(f"Failed to parse post: {e}")

                card_group = card.get("card_group", [])
                for item in card_group:
                    if count >= limit:
                        break
                    mblog = item.get("mblog")
                    if mblog:
                        try:
                            post = self.parser.parse_weibo(mblog)
                            yield post
                            count += 1
                            self._stats["posts_collected"] += 1
                        except ParseException as e:
                            logger.warning(f"Failed to parse post: {e}")

            page += 1
            await asyncio.sleep(self._calculate_delay())

    async def _search_browser(
        self,
        keyword: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """浏览器模式搜索"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        search_url = f"https://s.weibo.com/weibo?q={quote(keyword)}"
        await self._page.goto(search_url, wait_until="networkidle")

        count = 0
        last_height = 0

        while count < limit:
            cards = await self._page.query_selector_all(".card-wrap[action-type=feed_list_item]")

            for card in cards[count:]:
                if count >= limit:
                    break
                try:
                    post = await self._parse_weibo_card(card)
                    if post:
                        yield post
                        count += 1
                        self._stats["posts_collected"] += 1
                except Exception as e:
                    logger.warning(f"Failed to parse card: {e}")

            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(self._calculate_delay())

            new_height = await self._page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    async def crawl_user_posts(
        self,
        user_id: str,
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        用户主页采集

        Args:
            user_id: 用户ID
            limit: 最大采集数量

        Yields:
            Post对象
        """
        if self.config.use_mobile_api:
            async for post in self._user_posts_mobile_api(user_id, limit):
                yield post
        else:
            async for post in self._user_posts_browser(user_id, limit):
                yield post

    async def _user_posts_mobile_api(
        self,
        user_id: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """移动端API用户帖子采集"""
        containerid = f"107603{user_id}"

        page = 1
        count = 0

        while count < limit:
            params = {
                "containerid": containerid,
                "page": page,
            }

            data = await self._make_request(
                f"{self._api_base}/container/getIndex",
                params,
            )

            if not data or data.get("ok") != 1:
                break

            cards = data.get("data", {}).get("cards", [])
            if not cards:
                break

            for card in cards:
                if count >= limit:
                    break

                mblog = card.get("mblog")
                if mblog:
                    try:
                        post = self.parser.parse_weibo(mblog)
                        yield post
                        count += 1
                        self._stats["posts_collected"] += 1
                    except ParseException as e:
                        logger.warning(f"Failed to parse post: {e}")

            page += 1
            await asyncio.sleep(self._calculate_delay())

    async def _user_posts_browser(
        self,
        user_id: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """浏览器模式用户帖子采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        user_url = f"https://weibo.com/u/{user_id}"
        await self._page.goto(user_url, wait_until="networkidle")

        count = 0
        last_height = 0

        while count < limit:
            posts = await self._page.query_selector_all(".WB_feed_type")

            for post_elem in posts[count:]:
                if count >= limit:
                    break
                try:
                    post = await self._parse_weibo_browser_post(post_elem)
                    if post:
                        yield post
                        count += 1
                        self._stats["posts_collected"] += 1
                except Exception as e:
                    logger.warning(f"Failed to parse post: {e}")

            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(self._calculate_delay())

            new_height = await self._page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    async def crawl_topic(
        self,
        topic_id: str,
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        超话内容采集

        Args:
            topic_id: 超话ID或名称
            limit: 最大采集数量

        Yields:
            Post对象
        """
        if self.config.use_mobile_api:
            async for post in self._topic_mobile_api(topic_id, limit):
                yield post
        else:
            async for post in self._topic_browser(topic_id, limit):
                yield post

    async def _topic_mobile_api(
        self,
        topic_id: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """移动端API超话采集"""
        if topic_id.startswith("100808"):
            containerid = f"{topic_id}_-_feed"
        else:
            containerid = f"100808{topic_id}_-_feed"

        page = 1
        count = 0

        while count < limit:
            params = {
                "containerid": containerid,
                "page": page,
            }

            data = await self._make_request(
                f"{self._api_base}/container/getIndex",
                params,
            )

            if not data or data.get("ok") != 1:
                break

            cards = data.get("data", {}).get("cards", [])
            if not cards:
                break

            for card in cards:
                if count >= limit:
                    break

                card_group = card.get("card_group", [])
                for item in card_group:
                    if count >= limit:
                        break
                    mblog = item.get("mblog")
                    if mblog:
                        try:
                            post = self.parser.parse_weibo(mblog)
                            yield post
                            count += 1
                            self._stats["posts_collected"] += 1
                        except ParseException as e:
                            logger.warning(f"Failed to parse post: {e}")

            page += 1
            await asyncio.sleep(self._calculate_delay())

    async def _topic_browser(
        self,
        topic_id: str,
        limit: int,
    ) -> AsyncIterator[Post]:
        """浏览器模式超话采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        if topic_id.startswith("100808"):
            topic_url = f"https://weibo.com/p/{topic_id}"
        else:
            topic_url = f"https://weibo.com/p/100808{topic_id}"

        await self._page.goto(topic_url, wait_until="networkidle")

        count = 0
        last_height = 0

        while count < limit:
            posts = await self._page.query_selector_all(".WB_feed_type")

            for post_elem in posts[count:]:
                if count >= limit:
                    break
                try:
                    post = await self._parse_weibo_browser_post(post_elem)
                    if post:
                        yield post
                        count += 1
                        self._stats["posts_collected"] += 1
                except Exception as e:
                    logger.warning(f"Failed to parse post: {e}")

            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(self._calculate_delay())

            new_height = await self._page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    async def crawl_hot_search(self) -> list[dict[str, Any]]:
        """
        热搜榜单采集

        Returns:
            热搜列表，每项包含排名、话题、热度等信息
        """
        if self.config.use_mobile_api:
            return await self._hot_search_mobile_api()
        else:
            return await self._hot_search_browser()

    async def _hot_search_mobile_api(self) -> list[dict[str, Any]]:
        """移动端API热搜采集"""
        params = {
            "containerid": "106003type=25&t=3&disable_hot=1&filter_type=realtimehot",
        }

        data = await self._make_request(
            f"{self._api_base}/container/getIndex",
            params,
        )

        if not data or data.get("ok") != 1:
            return []

        hot_list: list[dict[str, Any]] = []
        cards = data.get("data", {}).get("cards", [])

        for card in cards:
            card_group = card.get("card_group", [])
            for item in card_group:
                desc = item.get("desc", "")
                if not desc:
                    continue

                hot_list.append({
                    "rank": len(hot_list) + 1,
                    "topic": desc,
                    "hot_value": item.get("desc_extr", ""),
                    "url": f"https://s.weibo.com/weibo?q={quote(desc)}",
                    "label": item.get("icon_desc", ""),
                })

        return hot_list

    async def _hot_search_browser(self) -> list[dict[str, Any]]:
        """浏览器模式热搜采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        await self._page.goto("https://s.weibo.com/top/summary", wait_until="networkidle")

        hot_list: list[dict[str, Any]] = []
        rows = await self._page.query_selector_all("#pl_top_realtimehot tbody tr")

        for i, row in enumerate(rows):
            if i == 0:
                continue

            try:
                topic_elem = await row.query_selector("td:nth-child(2) a")
                if not topic_elem:
                    continue

                topic = await topic_elem.inner_text()
                href = await topic_elem.get_attribute("href")

                hot_elem = await row.query_selector("td:nth-child(3)")
                hot_value = await hot_elem.inner_text() if hot_elem else ""

                hot_list.append({
                    "rank": i,
                    "topic": topic.strip(),
                    "hot_value": hot_value.strip(),
                    "url": f"https://s.weibo.com{href}" if href else "",
                })
            except Exception as e:
                logger.warning(f"Failed to parse hot search row: {e}")

        return hot_list

    async def crawl_comments(
        self,
        post_id: str,
        limit: int = 100,
    ) -> AsyncIterator[Interaction]:
        """
        评论采集

        Args:
            post_id: 微博ID
            limit: 最大采集数量

        Yields:
            Interaction对象
        """
        if self.config.use_mobile_api:
            async for comment in self._comments_mobile_api(post_id, limit):
                yield comment
        else:
            async for comment in self._comments_browser(post_id, limit):
                yield comment

    async def _comments_mobile_api(
        self,
        post_id: str,
        limit: int,
    ) -> AsyncIterator[Interaction]:
        """移动端API评论采集"""
        page = 1
        count = 0

        while count < limit:
            params = {
                "id": post_id,
                "page": page,
            }

            data = await self._make_request(
                f"{self._api_base}/comments/show",
                params,
            )

            if not data or data.get("ok") != 1:
                break

            comments = data.get("data", {}).get("data", [])
            if not comments:
                break

            for comment_data in comments:
                if count >= limit:
                    break

                try:
                    interactions = self.parser.parse_comments([comment_data])
                    for interaction in interactions:
                        yield interaction
                        count += 1
                except ParseException as e:
                    logger.warning(f"Failed to parse comment: {e}")

            page += 1
            await asyncio.sleep(self._calculate_delay())

    async def _comments_browser(
        self,
        post_id: str,
        limit: int,
    ) -> AsyncIterator[Interaction]:
        """浏览器模式评论采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        post_url = f"https://weibo.com/{post_id}"
        await self._page.goto(post_url, wait_until="networkidle")

        count = 0
        last_height = 0

        while count < limit:
            comments = await self._page.query_selector_all(".WB_text")

            for comment_elem in comments[count:]:
                if count >= limit:
                    break
                try:
                    text = await comment_elem.inner_text()
                    user_link = await comment_elem.query_selector("a")
                    user_id = ""
                    if user_link:
                        href = await user_link.get_attribute("href")
                        if href:
                            user_id = href.split("/")[-1]

                    interaction = Interaction(
                        interaction_id=f"weibo_comment_{post_id}_{count}",
                        interaction_type=InteractionType.COMMENT,
                        source_user_id=user_id,
                        target_post_id=post_id,
                        content=text.strip(),
                    )
                    yield interaction
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse comment: {e}")

            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(self._calculate_delay())

            new_height = await self._page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    async def crawl_user(self, user_id: str) -> Optional[User]:
        """
        爬取用户信息

        Args:
            user_id: 用户ID

        Returns:
            User对象
        """
        if self.config.use_mobile_api:
            return await self._user_mobile_api(user_id)
        else:
            return await self._user_browser(user_id)

    async def _user_mobile_api(self, user_id: str) -> Optional[User]:
        """移动端API用户信息采集"""
        params = {
            "containerid": f"100505{user_id}",
        }

        data = await self._make_request(
            f"{self._api_base}/container/getIndex",
            params,
        )

        if not data or data.get("ok") != 1:
            return None

        user_info = data.get("data", {}).get("userInfo", {})
        if not user_info:
            return None

        try:
            user = self.parser.parse_user(user_info)
            self._stats["users_collected"] += 1
            return user
        except ParseException as e:
            logger.warning(f"Failed to parse user: {e}")
            return None

    async def _user_browser(self, user_id: str) -> Optional[User]:
        """浏览器模式用户信息采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        user_url = f"https://weibo.com/u/{user_id}"
        await self._page.goto(user_url, wait_until="networkidle")

        try:
            name_elem = await self._page.query_selector(".username")
            if not name_elem:
                return None

            username = await name_elem.inner_text()

            bio_elem = await self._page.query_selector(".bio")
            bio = await bio_elem.inner_text() if bio_elem else ""

            followers_elem = await self._page.query_selector('[node-type="followers"] strong')
            followers_count = 0
            if followers_elem:
                followers_text = await followers_elem.inner_text()
                followers_count = self._parse_count(followers_text)

            friends_elem = await self._page.query_selector('[node-type="follows"] strong')
            friends_count = 0
            if friends_elem:
                friends_text = await friends_elem.inner_text()
                friends_count = self._parse_count(friends_text)

            user = User(
                user_id=user_id,
                platform=Platform.WEIBO,
                username=username.strip(),
                display_name=username.strip(),
                bio=bio.strip(),
                followers_count=followers_count,
                friends_count=friends_count,
            )
            self._stats["users_collected"] += 1
            return user
        except Exception as e:
            logger.warning(f"Failed to parse user page: {e}")
            return None

    async def _crawl_single_post(self, post_id: str) -> Optional[Post]:
        """爬取单条微博"""
        if self.config.use_mobile_api:
            return await self._post_mobile_api(post_id)
        else:
            return await self._post_browser(post_id)

    async def _post_mobile_api(self, post_id: str) -> Optional[Post]:
        """移动端API单条微博采集"""
        params = {
            "id": post_id,
        }

        data = await self._make_request(
            f"{self._api_base}/statuses/show",
            params,
        )

        if not data or data.get("ok") != 1:
            return None

        try:
            post = self.parser.parse_weibo(data.get("data", {}))
            self._stats["posts_collected"] += 1
            return post
        except ParseException as e:
            logger.warning(f"Failed to parse post: {e}")
            return None

    async def _post_browser(self, post_id: str) -> Optional[Post]:
        """浏览器模式单条微博采集"""
        if not self._page:
            raise RuntimeError("Browser not initialized")

        post_url = f"https://weibo.com/{post_id}"
        await self._page.goto(post_url, wait_until="networkidle")

        try:
            post_elem = await self._page.query_selector(".WB_feed_type")
            if not post_elem:
                return None

            post = await self._parse_weibo_browser_post(post_elem)
            if post:
                self._stats["posts_collected"] += 1
            return post
        except Exception as e:
            logger.warning(f"Failed to parse post page: {e}")
            return None

    async def _make_request(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        发送HTTP请求

        Args:
            url: 请求URL
            params: 请求参数

        Returns:
            JSON响应数据

        Raises:
            WeiboAuthException: 认证失败
            WeiboRateLimitException: 速率限制
            WeiboBlockedException: 账号被封
        """
        if not self._session:
            raise RuntimeError("Session not initialized")

        retry_count = 0

        while retry_count < self.config.max_retries:
            try:
                context = await self.get_request_context()

                proxy = context.get("proxy")
                proxy_kwargs = {"proxy": proxy} if proxy else {}

                async with self._session.get(url, params=params, **proxy_kwargs) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self.release_request_context(success=True)

                        if data.get("ok") == 0:
                            msg = data.get("msg", "")
                            if "登录" in msg or "auth" in msg.lower():
                                raise WeiboAuthException(f"Authentication required: {msg}")
                            if "限制" in msg or "block" in msg.lower():
                                raise WeiboBlockedException(f"Account blocked: {msg}")

                        return data

                    elif response.status == 401 or response.status == 403:
                        await self.release_request_context(success=False)
                        raise WeiboAuthException(f"Authentication failed: {response.status}")

                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        await self.release_request_context(success=False)
                        raise WeiboRateLimitException(
                            "Rate limit exceeded",
                            retry_after=float(retry_after),
                        )

                    elif response.status == 404:
                        await self.release_request_context(success=True)
                        return None

                    else:
                        await self.release_request_context(success=False)
                        logger.warning(f"Request failed with status {response.status}")
                        retry_count += 1

            except WeiboAuthException:
                raise
            except WeiboRateLimitException:
                raise
            except WeiboBlockedException:
                raise
            except Exception as e:
                logger.error(f"Request error: {e}")
                await self.release_request_context(success=False)
                retry_count += 1

            if retry_count < self.config.max_retries:
                await asyncio.sleep(2**retry_count)

        return None

    async def _parse_weibo_card(self, card_element: Any) -> Optional[Post]:
        """解析微博卡片元素（浏览器模式）"""
        try:
            content_elem = await card_element.query_selector(".node-content")
            if not content_elem:
                return None

            text = await content_elem.inner_text()

            mid_elem = await card_element.query_selector("[mid]")
            post_id = ""
            if mid_elem:
                post_id = await mid_elem.get_attribute("mid") or ""

            user_elem = await card_element.query_selector(".name")
            user_id = ""
            if user_elem:
                href = await user_elem.get_attribute("href")
                if href:
                    user_id = href.split("/")[-1]

            return Post(
                post_id=post_id,
                user_id=user_id,
                platform=Platform.WEIBO,
                content=text.strip(),
                language="zh",
            )
        except Exception as e:
            logger.warning(f"Failed to parse weibo card: {e}")
            return None

    async def _parse_weibo_browser_post(self, post_element: Any) -> Optional[Post]:
        """解析微博元素（浏览器模式）"""
        try:
            text_elem = await post_element.query_selector(".WB_text")
            if not text_elem:
                return None

            text = await text_elem.inner_text()

            mid = await post_element.get_attribute("mid")
            post_id = mid or ""

            user_id = ""
            user_info = await post_element.query_selector(".WB_info a")
            if user_info:
                href = await user_info.get_attribute("href")
                if href:
                    user_id = href.split("/")[-1]

            likes_count = 0
            likes_elem = await post_element.query_selector('[action-type="feed_list_like"] em')
            if likes_elem:
                likes_text = await likes_elem.inner_text()
                likes_count = self._parse_count(likes_text)

            reposts_count = 0
            reposts_elem = await post_element.query_selector('[action-type="feed_list_forward"] em')
            if reposts_elem:
                reposts_text = await reposts_elem.inner_text()
                reposts_count = self._parse_count(reposts_text)

            comments_count = 0
            comments_elem = await post_element.query_selector('[action-type="feed_list_comment"] em')
            if comments_elem:
                comments_text = await comments_elem.inner_text()
                comments_count = self._parse_count(comments_text)

            return Post(
                post_id=post_id,
                user_id=user_id,
                platform=Platform.WEIBO,
                content=text.strip(),
                language="zh",
                likes_count=likes_count,
                shares_count=reposts_count,
                comments_count=comments_count,
            )
        except Exception as e:
            logger.warning(f"Failed to parse browser post: {e}")
            return None

    def _calculate_delay(self) -> float:
        """计算请求延迟"""
        min_delay, max_delay = self.config.request_delay
        return random.uniform(min_delay, max_delay)

    @staticmethod
    def _parse_count(text: str) -> int:
        """解析数量文本（如：1万 -> 10000）"""
        if not text:
            return 0

        text = text.strip().lower()

        if "万" in text or "w" in text:
            num_str = text.replace("万", "").replace("w", "").strip()
            try:
                return int(float(num_str) * 10000)
            except ValueError:
                return 0
        elif "亿" in text or "b" in text:
            num_str = text.replace("亿", "").replace("b", "").strip()
            try:
                return int(float(num_str) * 100000000)
            except ValueError:
                return 0
        else:
            try:
                return int(text)
            except ValueError:
                return 0
