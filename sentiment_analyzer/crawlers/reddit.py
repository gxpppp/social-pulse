"""
Reddit 平台适配器

实现Reddit平台的爬虫功能，包括OAuth认证、帖子采集、评论采集和实时流监听。
"""

import asyncio
import base64
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from loguru import logger

from .base import (
    AuthenticationException,
    BaseCrawler,
    CrawlerException,
    NetworkException,
    ParseException,
    RateLimitConfig,
)
from storage.models import Platform, Post, User


@dataclass
class RedditConfig:
    """
    Reddit 配置类

    Attributes:
        client_id: OAuth 应用客户端ID
        client_secret: OAuth 应用客户端密钥
        user_agent: 用户代理字符串
        username: Reddit 用户名（可选，用于OAuth密码授权）
        password: Reddit 密码（可选，用于OAuth密码授权）
    """

    client_id: str
    client_secret: str
    user_agent: str = "python:multi-social-crawler:v1.0 (by /u/crawler)"
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class RedditComment:
    """Reddit 评论数据结构"""

    comment_id: str
    post_id: str
    author: str
    body: str
    created_at: Optional[datetime]
    score: int = 0
    parent_id: Optional[str] = None
    replies: list["RedditComment"] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


class RedditParser:
    """
    Reddit 数据解析器

    负责将Reddit API返回的原始数据转换为统一的数据模型。
    """

    @staticmethod
    def parse_post(data: dict[str, Any]) -> Post:
        """
        解析Reddit帖子数据

        Args:
            data: Reddit API返回的帖子数据

        Returns:
            解析后的Post对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            post_data = data.get("data", data)

            media: list[dict[str, Any]] = []
            if post_data.get("url") and not post_data.get("is_self"):
                media.append({
                    "type": "link",
                    "url": post_data["url"],
                })

            preview = post_data.get("preview", {}).get("images", [])
            for img in preview:
                source = img.get("source", {})
                if source.get("url"):
                    media.append({
                        "type": "image",
                        "url": source["url"],
                        "width": source.get("width"),
                        "height": source.get("height"),
                    })

            content = post_data.get("selftext", "")
            title = post_data.get("title", "")
            full_content = f"{title}\n\n{content}" if content else title

            hashtags = []
            flair = post_data.get("link_flair_text")
            if flair:
                hashtags.append(flair)

            return Post(
                post_id=post_data.get("id", ""),
                user_id=post_data.get("author", "[deleted]"),
                platform=Platform.REDDIT,
                content=full_content.strip() or None,
                language=post_data.get("lang", "en"),
                posted_at=RedditParser._parse_timestamp(post_data.get("created_utc")),
                likes_count=post_data.get("ups", 0),
                shares_count=0,
                comments_count=post_data.get("num_comments", 0),
                hashtags=hashtags,
                mentions=[],
                urls=[post_data.get("url")] if post_data.get("url") and not post_data.get("is_self") else [],
                media=media,
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Reddit post: {e}",
                raw_data=data,
            )

    @staticmethod
    def parse_comment(data: dict[str, Any]) -> RedditComment:
        """
        解析Reddit评论数据

        Args:
            data: Reddit API返回的评论数据

        Returns:
            解析后的RedditComment对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            comment_data = data.get("data", data)

            replies: list[RedditComment] = []
            replies_data = comment_data.get("replies", {})
            if isinstance(replies_data, dict):
                children = replies_data.get("data", {}).get("children", [])
                for child in children:
                    if child.get("kind") == "t1":
                        try:
                            replies.append(RedditParser.parse_comment(child))
                        except ParseException:
                            pass

            return RedditComment(
                comment_id=comment_data.get("id", ""),
                post_id=comment_data.get("link_id", "").replace("t3_", ""),
                author=comment_data.get("author", "[deleted]"),
                body=comment_data.get("body", ""),
                created_at=RedditParser._parse_timestamp(comment_data.get("created_utc")),
                score=comment_data.get("score", 0),
                parent_id=comment_data.get("parent_id"),
                replies=replies,
                raw_data=comment_data,
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Reddit comment: {e}",
                raw_data=data,
            )

    @staticmethod
    def parse_user(data: dict[str, Any]) -> User:
        """
        解析Reddit用户数据

        Args:
            data: Reddit API返回的用户数据

        Returns:
            解析后的User对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            user_data = data.get("data", data)
            subreddit = user_data.get("subreddit", {})

            return User(
                user_id=user_data.get("name", ""),
                platform=Platform.REDDIT,
                username=user_data.get("name", ""),
                display_name=subreddit.get("title") if isinstance(subreddit, dict) else None,
                bio=subreddit.get("public_description") if isinstance(subreddit, dict) else None,
                avatar_url=user_data.get("icon_img") or user_data.get("snoovatar_img"),
                created_at=RedditParser._parse_timestamp(user_data.get("created_utc")),
                followers_count=0,
                friends_count=0,
                posts_count=user_data.get("link_karma", 0) + user_data.get("comment_karma", 0),
                verified=user_data.get("verified", False),
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Reddit user: {e}",
                raw_data=data,
            )

    @staticmethod
    def _parse_timestamp(timestamp: Optional[float]) -> Optional[datetime]:
        """解析Unix时间戳"""
        if timestamp is None:
            return None
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            return None


class RedditCrawler(BaseCrawler):
    """
    Reddit 爬虫类

    实现Reddit平台的数据采集功能，支持OAuth认证、帖子采集、评论采集和实时流监听。

    Attributes:
        platform: 平台标识（REDDIT）
        config: Reddit配置
        parser: 数据解析器
    """

    platform = Platform.REDDIT

    def __init__(
        self,
        config: RedditConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """
        初始化Reddit爬虫

        Args:
            config: Reddit配置
            rate_limit_config: 限流配置（可选）
        """
        super().__init__(rate_limit_config=rate_limit_config)
        self.config = config
        self.parser = RedditParser()
        self._session: Optional[Any] = None
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._api_base = "https://oauth.reddit.com"
        self._streaming = False

    async def _ensure_session(self) -> Any:
        """确保aiohttp会话已创建"""
        if self._session is None:
            import aiohttp

            headers = {
                "User-Agent": self.config.user_agent,
            }
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _authenticate(self) -> None:
        """
        执行OAuth认证

        Raises:
            AuthenticationException: 认证失败时抛出
        """
        session = await self._ensure_session()

        auth_str = f"{self.config.client_id}:{self.config.client_secret}"
        auth_bytes = base64.b64encode(auth_str.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_bytes}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}

        if self.config.username and self.config.password:
            data = {
                "grant_type": "password",
                "username": self.config.username,
                "password": self.config.password,
            }

        try:
            async with session.post(
                "https://www.reddit.com/api/v1/access_token",
                headers=headers,
                data=data,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self._token = result.get("access_token")
                    expires_in = result.get("expires_in", 3600)
                    self._token_expires = datetime.now(timezone.utc).replace(
                        tzinfo=None
                    ) + __import__("datetime").timedelta(seconds=expires_in - 60)
                    session.headers["Authorization"] = f"Bearer {self._token}"
                    logger.info("Reddit authentication successful")
                else:
                    error_text = await response.text()
                    raise AuthenticationException(
                        f"Reddit authentication failed: {response.status} - {error_text}",
                        platform=Platform.REDDIT,
                    )
        except aiohttp.ClientError as e:
            raise AuthenticationException(
                f"Reddit authentication network error: {e}",
                platform=Platform.REDDIT,
            )

    async def _ensure_authenticated(self) -> None:
        """确保已认证且token有效"""
        if self._token is None or (
            self._token_expires and datetime.now(timezone.utc).replace(tzinfo=None) >= self._token_expires
        ):
            await self._authenticate()

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        method: str = "GET",
    ) -> Optional[dict[str, Any]]:
        """
        发送API请求

        Args:
            endpoint: API端点
            params: 请求参数
            method: HTTP方法

        Returns:
            API响应数据，失败返回None

        Raises:
            NetworkException: 网络错误时抛出
        """
        await self._ensure_authenticated()
        session = await self._ensure_session()

        url = f"{self._api_base}/{endpoint.lstrip('/')}"

        context = await self.get_request_context()
        proxy = context.get("proxy")

        try:
            kwargs: dict[str, Any] = {"params": params} if method == "GET" else {"json": params}

            async with session.request(method, url, proxy=proxy, **kwargs) as response:
                if response.status == 200:
                    result = await response.json()
                    await self.release_request_context(success=True)
                    return result
                elif response.status == 401:
                    self._token = None
                    await self._ensure_authenticated()
                    return await self._make_request(endpoint, params, method)
                elif response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Reddit rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    await self.release_request_context(success=False)
                    return await self._make_request(endpoint, params, method)
                else:
                    error_text = await response.text()
                    logger.error(f"Reddit request failed: {response.status} - {error_text}")
                    await self.release_request_context(success=False)
                    return None
        except asyncio.TimeoutError:
            await self.release_request_context(success=False)
            raise NetworkException(
                "Reddit request timeout",
                platform=Platform.REDDIT,
            )
        except aiohttp.ClientError as e:
            await self.release_request_context(success=False)
            raise NetworkException(
                f"Reddit network error: {e}",
                platform=Platform.REDDIT,
            )

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
            **kwargs: 其他参数（如subreddit, limit等）

        Yields:
            爬取到的用户或帖子对象
        """
        await self.before_crawl()

        try:
            if user_ids:
                for user_id in user_ids:
                    user = await self.crawl_user(user_id)
                    if user:
                        yield user

            if post_ids:
                for post_id in post_ids:
                    post = await self._crawl_single_post(post_id)
                    if post:
                        yield post

            if query:
                subreddit = kwargs.get("subreddit")
                limit = kwargs.get("limit", 100)
                async for post in self.search_posts(query, limit, subreddit):
                    yield post
                    self._stats["posts_collected"] += 1

        finally:
            await self.after_crawl()

    async def parse_user(self, data: dict[str, Any]) -> User:
        """解析用户数据"""
        return self.parser.parse_user(data)

    async def parse_post(self, data: dict[str, Any]) -> Post:
        """解析帖子数据"""
        return self.parser.parse_post(data)

    async def crawl_user(self, username: str) -> Optional[User]:
        """
        爬取用户信息

        Args:
            username: Reddit用户名

        Returns:
            用户对象，失败返回None
        """
        data = await self._make_request(f"user/{username}/about")

        if not data or "data" not in data:
            return None

        user = self.parser.parse_user(data)
        self._stats["users_collected"] += 1
        return user

    async def crawl_subreddit(
        self,
        subreddit: str,
        limit: int = 100,
        sort: str = "hot",
        time_filter: str = "day",
    ) -> AsyncIterator[Post]:
        """
        爬取Subreddit帖子

        Args:
            subreddit: Subreddit名称
            limit: 最大帖子数
            sort: 排序方式（hot, new, top, rising, controversial）
            time_filter: 时间过滤（hour, day, week, month, year, all）

        Yields:
            帖子对象
        """
        params: dict[str, Any] = {"limit": min(100, limit)}

        if sort in ("top", "controversial"):
            params["t"] = time_filter

        count = 0
        after = None

        while count < limit:
            if after:
                params["after"] = after

            data = await self._make_request(f"r/{subreddit}/{sort}", params)

            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            for child in children:
                if count >= limit:
                    break
                if child.get("kind") == "t3":
                    try:
                        post = self.parser.parse_post(child)
                        yield post
                        self._stats["posts_collected"] += 1
                        count += 1
                    except ParseException as e:
                        logger.warning(f"Failed to parse post: {e}")

            after = data["data"].get("after")
            if not after:
                break

    async def crawl_post_comments(
        self,
        post_id: str,
        limit: int = 100,
        depth: Optional[int] = None,
    ) -> AsyncIterator[RedditComment]:
        """
        爬取帖子评论树

        Args:
            post_id: 帖子ID
            limit: 最大评论数
            depth: 评论深度限制

        Yields:
            评论对象
        """
        params: dict[str, Any] = {
            "limit": limit,
            "sort": "best",
        }
        if depth is not None:
            params["depth"] = depth

        data = await self._make_request(f"comments/{post_id}", params)

        if not data or not isinstance(data, list) or len(data) < 2:
            return

        comments_data = data[1].get("data", {}).get("children", [])

        count = 0

        def extract_comments(children: list[dict[str, Any]], current_depth: int = 0) -> list[RedditComment]:
            nonlocal count
            comments = []
            for child in children:
                if count >= limit:
                    break
                if child.get("kind") == "t1":
                    try:
                        comment = self.parser.parse_comment(child)
                        count += 1
                        comments.append(comment)
                        if depth is None or current_depth < depth:
                            replies_data = child.get("data", {}).get("replies", {})
                            if isinstance(replies_data, dict):
                                reply_children = replies_data.get("data", {}).get("children", [])
                                comment.replies = extract_comments(reply_children, current_depth + 1)
                    except ParseException as e:
                        logger.warning(f"Failed to parse comment: {e}")
            return comments

        for comment in extract_comments(comments_data):
            yield comment

    async def crawl_user_posts(
        self,
        username: str,
        limit: int = 100,
        sort: str = "new",
    ) -> AsyncIterator[Post]:
        """
        爬取用户帖子

        Args:
            username: Reddit用户名
            limit: 最大帖子数
            sort: 排序方式（new, hot, top, controversial）

        Yields:
            帖子对象
        """
        params: dict[str, Any] = {
            "limit": min(100, limit),
            "sort": sort,
        }

        count = 0
        after = None

        while count < limit:
            if after:
                params["after"] = after

            data = await self._make_request(f"user/{username}/submitted", params)

            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            for child in children:
                if count >= limit:
                    break
                if child.get("kind") == "t3":
                    try:
                        post = self.parser.parse_post(child)
                        yield post
                        self._stats["posts_collected"] += 1
                        count += 1
                    except ParseException as e:
                        logger.warning(f"Failed to parse post: {e}")

            after = data["data"].get("after")
            if not after:
                break

    async def stream_subreddit(
        self,
        subreddit: str,
        interval: float = 5.0,
    ) -> AsyncIterator[Post]:
        """
        实时流监听Subreddit新帖子

        Args:
            subreddit: Subreddit名称
            interval: 轮询间隔（秒）

        Yields:
            新帖子对象
        """
        self._streaming = True
        seen_ids: set[str] = set()

        logger.info(f"Starting Reddit stream for r/{subreddit}")

        try:
            while self._streaming:
                try:
                    params = {"limit": 25}
                    data = await self._make_request(f"r/{subreddit}/new", params)

                    if data and "data" in data:
                        children = data["data"].get("children", [])
                        for child in reversed(children):
                            if child.get("kind") == "t3":
                                post_data = child.get("data", {})
                                post_id = post_data.get("id", "")

                                if post_id and post_id not in seen_ids:
                                    seen_ids.add(post_id)
                                    try:
                                        post = self.parser.parse_post(child)
                                        yield post
                                        self._stats["posts_collected"] += 1
                                    except ParseException as e:
                                        logger.warning(f"Failed to parse streamed post: {e}")

                    await asyncio.sleep(interval)

                except (NetworkException, CrawlerException) as e:
                    logger.error(f"Stream error: {e}")
                    await asyncio.sleep(interval * 2)

        finally:
            self._streaming = False
            logger.info(f"Stopped Reddit stream for r/{subreddit}")

    def stop_stream(self) -> None:
        """停止实时流监听"""
        self._streaming = False

    async def search_posts(
        self,
        query: str,
        limit: int = 100,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        time_filter: str = "all",
    ) -> AsyncIterator[Post]:
        """
        搜索帖子

        Args:
            query: 搜索查询
            limit: 最大结果数
            subreddit: 限定Subreddit（可选）
            sort: 排序方式（relevance, hot, top, new, comments）
            time_filter: 时间过滤

        Yields:
            帖子对象
        """
        params: dict[str, Any] = {
            "q": query,
            "limit": min(100, limit),
            "sort": sort,
            "t": time_filter,
        }

        if subreddit:
            params["restrict_sr"] = True
            endpoint = f"r/{subreddit}/search"
        else:
            endpoint = "search"

        count = 0
        after = None

        while count < limit:
            if after:
                params["after"] = after

            data = await self._make_request(endpoint, params)

            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            for child in children:
                if count >= limit:
                    break
                if child.get("kind") == "t3":
                    try:
                        post = self.parser.parse_post(child)
                        yield post
                        self._stats["posts_collected"] += 1
                        count += 1
                    except ParseException as e:
                        logger.warning(f"Failed to parse search result: {e}")

            after = data["data"].get("after")
            if not after:
                break

    async def _crawl_single_post(self, post_id: str) -> Optional[Post]:
        """爬取单个帖子"""
        data = await self._make_request(f"comments/{post_id}")

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        post_data = data[0].get("data", {}).get("children", [])
        if not post_data:
            return None

        try:
            post = self.parser.parse_post(post_data[0])
            self._stats["posts_collected"] += 1
            return post
        except ParseException as e:
            logger.warning(f"Failed to parse post {post_id}: {e}")
            return None

    async def close(self) -> None:
        """关闭爬虫，释放资源"""
        self._streaming = False
        if self._session:
            await self._session.close()
            self._session = None
        self._token = None
        logger.info("Reddit crawler closed")
