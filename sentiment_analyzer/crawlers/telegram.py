"""
Telegram 平台适配器

实现Telegram平台的爬虫功能，包括MTProto认证、频道消息采集、群组消息采集和搜索功能。
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING

from loguru import logger

from .base import (
    AuthenticationException,
    BaseCrawler,
    CrawlerException,
    NetworkException,
    ParseException,
    RateLimitConfig,
)
from ..storage.models import Platform, Post, User

if TYPE_CHECKING:
    from telethon import TelegramClient
    from telethon.tl.types import (
        Channel,
        Chat,
        Message,
        User as TelethonUser,
    )


@dataclass
class TelegramConfig:
    """
    Telegram 配置类

    Attributes:
        api_id: Telegram API ID（从my.telegram.org获取）
        api_hash: Telegram API Hash（从my.telegram.org获取）
        phone: 手机号码（国际格式，如+8613800138000）
        session_name: 会话名称/文件路径
        bot_token: Bot Token（可选，用于Bot模式）
    """

    api_id: int
    api_hash: str
    phone: Optional[str] = None
    session_name: str = "telegram_session"
    bot_token: Optional[str] = None


@dataclass
class TelegramChannel:
    """Telegram 频道数据结构"""

    channel_id: str
    title: str
    username: Optional[str] = None
    description: Optional[str] = None
    subscribers_count: int = 0
    is_verified: bool = False
    is_scam: bool = False
    created_at: Optional[datetime] = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TelegramChat:
    """Telegram 群组数据结构"""

    chat_id: str
    title: str
    username: Optional[str] = None
    description: Optional[str] = None
    members_count: int = 0
    is_verified: bool = False
    raw_data: dict[str, Any] = field(default_factory=dict)


class TelegramParser:
    """
    Telegram 数据解析器

    负责将Telethon返回的原始数据转换为统一的数据模型。
    """

    @staticmethod
    def parse_message(message: "Message", chat_id: Optional[int] = None) -> Post:
        """
        解析Telegram消息

        Args:
            message: Telethon Message对象
            chat_id: 聊天ID（可选）

        Returns:
            解析后的Post对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            media: list[dict[str, Any]] = []
            if message.media:
                media_type = type(message.media).__name__
                media_info: dict[str, Any] = {"type": media_type.lower()}

                if hasattr(message.media, "photo"):
                    media_info["media_type"] = "photo"
                elif hasattr(message.media, "document"):
                    media_info["media_type"] = "document"
                    if hasattr(message.media.document, "mime_type"):
                        media_info["mime_type"] = message.media.document.mime_type

                media.append(media_info)

            hashtags: list[str] = []
            mentions: list[str] = []
            urls: list[str] = []

            text = message.text or ""
            hashtags = re.findall(r"#(\w+)", text)
            mentions = re.findall(r"@(\w+)", text)
            urls = re.findall(r"https?://[^\s]+", text)

            sender_id = ""
            sender_name = ""
            if message.sender:
                sender_id = str(message.sender.id)
                if hasattr(message.sender, "username") and message.sender.username:
                    sender_name = message.sender.username
                elif hasattr(message.sender, "first_name"):
                    parts = [message.sender.first_name]
                    if hasattr(message.sender, "last_name") and message.sender.last_name:
                        parts.append(message.sender.last_name)
                    sender_name = " ".join(parts)

            views = getattr(message, "views", 0) or 0
            forwards = getattr(message, "forwards", 0) or 0
            replies_count = 0
            if hasattr(message, "replies") and message.replies:
                replies_count = message.replies.replies or 0

            chat_id_str = str(chat_id or message.chat_id or message.peer_id)

            return Post(
                post_id=f"{chat_id_str}:{message.id}",
                user_id=sender_id,
                platform=Platform.TELEGRAM,
                content=text or None,
                language=None,
                posted_at=message.date,
                likes_count=views,
                shares_count=forwards,
                comments_count=replies_count,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                media=media,
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Telegram message: {e}",
                raw_data={"message_id": message.id},
            )

    @staticmethod
    def parse_user(user: "TelethonUser") -> User:
        """
        解析Telegram用户

        Args:
            user: Telethon User对象

        Returns:
            解析后的User对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            display_name = ""
            if hasattr(user, "first_name"):
                display_name = user.first_name or ""
                if hasattr(user, "last_name") and user.last_name:
                    display_name = f"{display_name} {user.last_name}".strip()

            bio = None
            if hasattr(user, "about"):
                bio = user.about

            return User(
                user_id=str(user.id),
                platform=Platform.TELEGRAM,
                username=getattr(user, "username", None),
                display_name=display_name or None,
                bio=bio,
                avatar_url=None,
                created_at=None,
                followers_count=0,
                friends_count=0,
                posts_count=0,
                verified=getattr(user, "verified", False),
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Telegram user: {e}",
                raw_data={"user_id": user.id if hasattr(user, "id") else None},
            )

    @staticmethod
    def parse_channel(channel: "Channel") -> TelegramChannel:
        """
        解析Telegram频道

        Args:
            channel: Telethon Channel对象

        Returns:
            解析后的TelegramChannel对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            return TelegramChannel(
                channel_id=str(channel.id),
                title=channel.title or "",
                username=getattr(channel, "username", None),
                description=None,
                subscribers_count=getattr(channel, "participants_count", 0) or 0,
                is_verified=getattr(channel, "verified", False),
                is_scam=getattr(channel, "scam", False),
                created_at=getattr(channel, "date", None),
                raw_data=channel.to_dict() if hasattr(channel, "to_dict") else {},
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Telegram channel: {e}",
                raw_data={"channel_id": channel.id if hasattr(channel, "id") else None},
            )

    @staticmethod
    def parse_chat(chat: "Chat") -> TelegramChat:
        """
        解析Telegram群组

        Args:
            chat: Telethon Chat对象

        Returns:
            解析后的TelegramChat对象

        Raises:
            ParseException: 解析失败时抛出
        """
        try:
            return TelegramChat(
                chat_id=str(chat.id),
                title=chat.title or "",
                username=getattr(chat, "username", None),
                description=None,
                members_count=getattr(chat, "participants_count", 0) or 0,
                is_verified=getattr(chat, "verified", False),
                raw_data=chat.to_dict() if hasattr(chat, "to_dict") else {},
            )
        except Exception as e:
            raise ParseException(
                f"Failed to parse Telegram chat: {e}",
                raw_data={"chat_id": chat.id if hasattr(chat, "id") else None},
            )


class TelegramCrawler(BaseCrawler):
    """
    Telegram 爬虫类

    实现Telegram平台的数据采集功能，支持MTProto认证、频道消息采集、群组消息采集和搜索功能。

    Attributes:
        platform: 平台标识（TELEGRAM）
        config: Telegram配置
        parser: 数据解析器
    """

    platform = Platform.TELEGRAM

    def __init__(
        self,
        config: TelegramConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """
        初始化Telegram爬虫

        Args:
            config: Telegram配置
            rate_limit_config: 限流配置（可选）
        """
        super().__init__(rate_limit_config=rate_limit_config)
        self.config = config
        self.parser = TelegramParser()
        self._client: Optional["TelegramClient"] = None
        self._connected = False

    async def _ensure_client(self) -> "TelegramClient":
        """
        确保Telegram客户端已连接

        Returns:
            TelegramClient实例

        Raises:
            AuthenticationException: 认证失败时抛出
        """
        if self._client is None or not self._connected:
            try:
                from telethon import TelegramClient
                from telethon.errors import SessionPasswordNeededError
            except ImportError as e:
                raise CrawlerException(
                    "telethon library not installed. Install with: pip install telethon",
                    platform=Platform.TELEGRAM,
                )

            session_path = Path(self.config.session_name)
            if not session_path.is_absolute():
                session_path = Path("data") / "sessions" / self.config.session_name
                session_path.parent.mkdir(parents=True, exist_ok=True)

            self._client = TelegramClient(
                str(session_path),
                self.config.api_id,
                self.config.api_hash,
            )

            try:
                await self._client.connect()

                if self.config.bot_token:
                    await self._client.sign_in(bot_token=self.config.bot_token)
                elif not await self._client.is_user_authorized():
                    if self.config.phone:
                        await self._client.send_code_request(self.config.phone)
                        logger.warning(
                            "Telegram requires authentication. Please run interactive login first."
                        )
                    else:
                        raise AuthenticationException(
                            "Telegram client not authorized. Provide phone number or bot token.",
                            platform=Platform.TELEGRAM,
                        )

                self._connected = True
                logger.info("Telegram client connected successfully")

            except SessionPasswordNeededError:
                raise AuthenticationException(
                    "Telegram account has 2FA enabled. Please provide password.",
                    platform=Platform.TELEGRAM,
                )
            except Exception as e:
                if self._client:
                    await self._client.disconnect()
                self._client = None
                raise AuthenticationException(
                    f"Telegram authentication failed: {e}",
                    platform=Platform.TELEGRAM,
                )

        return self._client

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
            post_ids: 帖子ID列表（格式：channel_id:message_id）
            **kwargs: 其他参数（如channel_id, group_id, limit等）

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
                    post = await self._crawl_single_message(post_id)
                    if post:
                        yield post

            if query:
                channel_id = kwargs.get("channel_id")
                limit = kwargs.get("limit", 100)
                async for post in self.search_messages(query, limit, channel_id):
                    yield post
                    self._stats["posts_collected"] += 1

            channel_id = kwargs.get("channel_id")
            if channel_id and not query:
                limit = kwargs.get("limit", 100)
                async for post in self.crawl_channel(channel_id, limit):
                    yield post

            group_id = kwargs.get("group_id")
            if group_id and not query and not channel_id:
                limit = kwargs.get("limit", 100)
                async for post in self.crawl_group(group_id, limit):
                    yield post

        finally:
            await self.after_crawl()

    async def parse_user(self, data: dict[str, Any]) -> User:
        """解析用户数据"""
        from telethon.tl.types import User as TelethonUser

        user = TelethonUser(**data)
        return self.parser.parse_user(user)

    async def parse_post(self, data: dict[str, Any]) -> Post:
        """解析帖子数据"""
        from telethon.tl.types import Message

        message = Message(**data)
        return self.parser.parse_message(message)

    async def crawl_user(self, user_id: str) -> Optional[User]:
        """
        爬取用户信息

        Args:
            user_id: Telegram用户ID或用户名

        Returns:
            用户对象，失败返回None
        """
        client = await self._ensure_client()

        try:
            entity = await client.get_entity(user_id)

            if hasattr(entity, "id"):
                user = self.parser.parse_user(entity)
                self._stats["users_collected"] += 1
                return user

            return None
        except Exception as e:
            logger.error(f"Failed to get Telegram user {user_id}: {e}")
            return None

    async def crawl_channel(
        self,
        channel_id: str,
        limit: int = 100,
        offset_id: int = 0,
        min_id: int = 0,
        max_id: int = 0,
    ) -> AsyncIterator[Post]:
        """
        爬取频道消息

        Args:
            channel_id: 频道ID或用户名
            limit: 最大消息数
            offset_id: 偏移消息ID
            min_id: 最小消息ID
            max_id: 最大消息ID

        Yields:
            帖子对象
        """
        client = await self._ensure_client()

        try:
            entity = await client.get_entity(channel_id)
            chat_id = entity.id

            kwargs: dict[str, Any] = {"limit": limit}
            if offset_id:
                kwargs["offset_id"] = offset_id
            if min_id:
                kwargs["min_id"] = min_id
            if max_id:
                kwargs["max_id"] = max_id

            async for message in client.iter_messages(entity, **kwargs):
                if message.text or message.media:
                    try:
                        post = self.parser.parse_message(message, chat_id)
                        yield post
                        self._stats["posts_collected"] += 1
                    except ParseException as e:
                        logger.warning(f"Failed to parse message: {e}")

        except Exception as e:
            logger.error(f"Failed to crawl Telegram channel {channel_id}: {e}")

    async def crawl_group(
        self,
        group_id: str,
        limit: int = 100,
        offset_id: int = 0,
    ) -> AsyncIterator[Post]:
        """
        爬取群组消息

        Args:
            group_id: 群组ID或用户名
            limit: 最大消息数
            offset_id: 偏移消息ID

        Yields:
            帖子对象
        """
        async for post in self.crawl_channel(group_id, limit, offset_id):
            yield post

    async def crawl_user_messages(
        self,
        user_id: str,
        limit: int = 100,
    ) -> AsyncIterator[Post]:
        """
        爬取用户消息（需要用户在共同群组中）

        Args:
            user_id: 用户ID或用户名
            limit: 最大消息数

        Yields:
            帖子对象
        """
        client = await self._ensure_client()

        try:
            entity = await client.get_entity(user_id)

            async for message in client.iter_messages(entity, limit=limit):
                if message.text or message.media:
                    try:
                        post = self.parser.parse_message(message, entity.id)
                        yield post
                        self._stats["posts_collected"] += 1
                    except ParseException as e:
                        logger.warning(f"Failed to parse message: {e}")

        except Exception as e:
            logger.error(f"Failed to get messages from user {user_id}: {e}")

    async def get_channel_info(self, channel_id: str) -> Optional[TelegramChannel]:
        """
        获取频道信息

        Args:
            channel_id: 频道ID或用户名

        Returns:
            频道信息对象，失败返回None
        """
        client = await self._ensure_client()

        try:
            entity = await client.get_entity(channel_id)

            if hasattr(entity, "title"):
                return self.parser.parse_channel(entity)

            return None
        except Exception as e:
            logger.error(f"Failed to get channel info for {channel_id}: {e}")
            return None

    async def get_chat_info(self, chat_id: str) -> Optional[TelegramChat]:
        """
        获取群组信息

        Args:
            chat_id: 群组ID或用户名

        Returns:
            群组信息对象，失败返回None
        """
        client = await self._ensure_client()

        try:
            entity = await client.get_entity(chat_id)

            if hasattr(entity, "title") and not hasattr(entity, "broadcast"):
                return self.parser.parse_chat(entity)

            return None
        except Exception as e:
            logger.error(f"Failed to get chat info for {chat_id}: {e}")
            return None

    async def search_messages(
        self,
        query: str,
        limit: int = 100,
        channel_id: Optional[str] = None,
    ) -> AsyncIterator[Post]:
        """
        搜索消息

        Args:
            query: 搜索查询
            limit: 最大结果数
            channel_id: 限定频道/群组ID（可选）

        Yields:
            帖子对象
        """
        client = await self._ensure_client()

        try:
            if channel_id:
                entity = await client.get_entity(channel_id)
                chat_id = entity.id

                async for message in client.iter_messages(
                    entity,
                    search=query,
                    limit=limit,
                ):
                    if message.text or message.media:
                        try:
                            post = self.parser.parse_message(message, chat_id)
                            yield post
                            self._stats["posts_collected"] += 1
                        except ParseException as e:
                            logger.warning(f"Failed to parse search result: {e}")
            else:
                logger.warning(
                    "Telegram global search is limited. Consider specifying a channel_id."
                )
                return

        except Exception as e:
            logger.error(f"Failed to search messages: {e}")

    async def get_message_by_id(
        self,
        channel_id: str,
        message_id: int,
    ) -> Optional[Post]:
        """
        根据ID获取消息

        Args:
            channel_id: 频道/群组ID
            message_id: 消息ID

        Returns:
            帖子对象，失败返回None
        """
        client = await self._ensure_client()

        try:
            entity = await client.get_entity(channel_id)
            message = await client.get_messages(entity, ids=message_id)

            if message:
                return self.parser.parse_message(message, entity.id)

            return None
        except Exception as e:
            logger.error(f"Failed to get message {message_id} from {channel_id}: {e}")
            return None

    async def get_dialogs(
        self,
        limit: int = 100,
    ) -> AsyncIterator[TelegramChannel | TelegramChat]:
        """
        获取用户的对话列表

        Args:
            limit: 最大数量

        Yields:
            频道或群组对象
        """
        client = await self._ensure_client()

        try:
            count = 0
            async for dialog in client.iter_dialogs(limit=limit):
                if count >= limit:
                    break

                entity = dialog.entity
                if hasattr(entity, "broadcast") and entity.broadcast:
                    yield self.parser.parse_channel(entity)
                    count += 1
                elif hasattr(entity, "title"):
                    yield self.parser.parse_chat(entity)
                    count += 1

        except Exception as e:
            logger.error(f"Failed to get dialogs: {e}")

    async def _crawl_single_message(self, post_id: str) -> Optional[Post]:
        """
        爬取单条消息

        Args:
            post_id: 消息ID（格式：channel_id:message_id）

        Returns:
            帖子对象，失败返回None
        """
        parts = post_id.split(":")
        if len(parts) != 2:
            logger.error(f"Invalid post_id format: {post_id}, expected channel_id:message_id")
            return None

        channel_id, message_id_str = parts
        try:
            message_id = int(message_id_str)
            post = await self.get_message_by_id(channel_id, message_id)
            if post:
                self._stats["posts_collected"] += 1
            return post
        except ValueError:
            logger.error(f"Invalid message_id: {message_id_str}")
            return None

    async def close(self) -> None:
        """关闭爬虫，释放资源"""
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
        self._client = None
        logger.info("Telegram crawler closed")
