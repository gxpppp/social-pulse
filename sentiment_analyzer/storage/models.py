"""
Pydantic 数据模型定义

包含用户、帖子和互动的数据模型，用于数据验证和序列化。
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Platform(str, Enum):
    """支持的社交平台枚举"""

    TWITTER = "twitter"
    WEIBO = "weibo"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"


class InteractionType(str, Enum):
    """互动类型枚举"""

    LIKE = "like"
    SHARE = "share"
    COMMENT = "comment"
    REPLY = "reply"
    MENTION = "mention"
    RETWEET = "retweet"
    QUOTE = "quote"
    FOLLOW = "follow"
    BOOKMARK = "bookmark"


class User(BaseModel):
    """
    用户数据模型

    表示从各社交平台采集的用户信息。

    Attributes:
        user_id: 平台唯一的用户标识符
        platform: 用户所属平台
        username: 用户名（@句柄）
        display_name: 显示名称
        bio: 用户简介
        avatar_url: 头像URL
        avatar_hash: 头像内容的哈希值，用于检测变更
        created_at: 账户创建时间
        followers_count: 粉丝数
        friends_count: 关注数
        posts_count: 发帖数
        verified: 是否认证
        first_seen: 首次采集时间
        last_updated: 最后更新时间
    """

    user_id: str = Field(..., min_length=1, max_length=256, description="平台唯一的用户标识符")
    platform: Platform = Field(..., description="用户所属平台")
    username: Optional[str] = Field(None, max_length=256, description="用户名（@句柄）")
    display_name: Optional[str] = Field(None, max_length=512, description="显示名称")
    bio: Optional[str] = Field(None, max_length=4096, description="用户简介")
    avatar_url: Optional[str] = Field(None, max_length=2048, description="头像URL")
    avatar_hash: Optional[str] = Field(None, max_length=128, description="头像内容的哈希值")
    created_at: Optional[datetime] = Field(None, description="账户创建时间")
    followers_count: int = Field(default=0, ge=0, description="粉丝数")
    friends_count: int = Field(default=0, ge=0, description="关注数")
    posts_count: int = Field(default=0, ge=0, description="发帖数")
    verified: bool = Field(default=False, description="是否认证")
    first_seen: datetime = Field(default_factory=datetime.utcnow, description="首次采集时间")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新时间")

    @field_validator("username", "display_name", "bio", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Optional[str]:
        """去除字符串首尾空白"""
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return str(v)

    @field_validator("avatar_url", mode="before")
    @classmethod
    def validate_url(cls, v: Any) -> Optional[str]:
        """验证URL格式"""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return None
            if not v.startswith(("http://", "https://")):
                return None
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123456789",
                "platform": "twitter",
                "username": "example_user",
                "display_name": "Example User",
                "bio": "This is an example bio",
                "avatar_url": "https://example.com/avatar.png",
                "avatar_hash": "a1b2c3d4e5f6",
                "created_at": "2020-01-01T00:00:00",
                "followers_count": 1000,
                "friends_count": 500,
                "posts_count": 200,
                "verified": True,
                "first_seen": "2024-01-01T00:00:00",
                "last_updated": "2024-01-15T12:00:00",
            }
        }


class Post(BaseModel):
    """
    帖子数据模型

    表示从各社交平台采集的帖子内容。

    Attributes:
        post_id: 平台唯一的帖子标识符
        user_id: 发布者用户ID
        platform: 帖子所属平台
        content: 帖子文本内容
        language: 内容语言代码
        posted_at: 发布时间
        likes_count: 点赞数
        shares_count: 转发/分享数
        comments_count: 评论数
        hashtags: 话题标签列表
        mentions: 提及用户列表
        urls: 包含的URL列表
        media: 媒体附件信息
        collected_at: 采集时间
    """

    post_id: str = Field(..., min_length=1, max_length=256, description="平台唯一的帖子标识符")
    user_id: str = Field(..., min_length=1, max_length=256, description="发布者用户ID")
    platform: Platform = Field(..., description="帖子所属平台")
    content: Optional[str] = Field(None, max_length=65535, description="帖子文本内容")
    language: Optional[str] = Field(None, max_length=16, description="内容语言代码（ISO 639-1）")
    posted_at: Optional[datetime] = Field(None, description="发布时间")
    likes_count: int = Field(default=0, ge=0, description="点赞数")
    shares_count: int = Field(default=0, ge=0, description="转发/分享数")
    comments_count: int = Field(default=0, ge=0, description="评论数")
    hashtags: list[str] = Field(default_factory=list, description="话题标签列表")
    mentions: list[str] = Field(default_factory=list, description="提及用户列表")
    urls: list[str] = Field(default_factory=list, description="包含的URL列表")
    media: list[dict[str, Any]] = Field(default_factory=list, description="媒体附件信息")
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="采集时间")

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> Optional[str]:
        """验证并清理内容"""
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip() or None
        return str(v)

    @field_validator("hashtags", "mentions", mode="before")
    @classmethod
    def validate_string_list(cls, v: Any) -> list[str]:
        """验证字符串列表，去除空白和重复项"""
        if v is None:
            return []
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped and stripped not in result:
                        result.append(stripped)
            return result
        return []

    @field_validator("urls", mode="before")
    @classmethod
    def validate_urls(cls, v: Any) -> list[str]:
        """验证URL列表"""
        if v is None:
            return []
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    url = item.strip()
                    if url.startswith(("http://", "https://")) and url not in result:
                        result.append(url)
            return result
        return []

    @field_validator("language", mode="before")
    @classmethod
    def validate_language_code(cls, v: Any) -> Optional[str]:
        """验证语言代码格式"""
        if v is None:
            return None
        if isinstance(v, str):
            code = v.strip().lower()
            if len(code) <= 16:
                return code
        return None

    class Config:
        json_schema_extra = {
            "example": {
                "post_id": "987654321",
                "user_id": "123456789",
                "platform": "twitter",
                "content": "This is an example post #example @mention",
                "language": "en",
                "posted_at": "2024-01-15T10:30:00",
                "likes_count": 100,
                "shares_count": 50,
                "comments_count": 25,
                "hashtags": ["example"],
                "mentions": ["mention"],
                "urls": ["https://example.com"],
                "media": [{"type": "image", "url": "https://example.com/image.png"}],
                "collected_at": "2024-01-15T12:00:00",
            }
        }


class Interaction(BaseModel):
    """
    互动数据模型

    表示用户之间的互动行为。

    Attributes:
        interaction_id: 互动唯一标识符
        interaction_type: 互动类型
        source_user_id: 发起互动的用户ID
        target_user_id: 目标用户ID
        source_post_id: 源帖子ID（可选）
        target_post_id: 目标帖子ID（可选）
        timestamp: 互动时间
        content: 互动内容（如评论文本）
    """

    interaction_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="互动唯一标识符",
    )
    interaction_type: InteractionType = Field(..., description="互动类型")
    source_user_id: str = Field(..., min_length=1, max_length=256, description="发起互动的用户ID")
    target_user_id: Optional[str] = Field(None, max_length=256, description="目标用户ID")
    source_post_id: Optional[str] = Field(None, max_length=256, description="源帖子ID")
    target_post_id: Optional[str] = Field(None, max_length=256, description="目标帖子ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="互动时间")
    content: Optional[str] = Field(None, max_length=65535, description="互动内容")

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Any) -> Optional[str]:
        """验证并清理内容"""
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip() or None
        return str(v)

    class Config:
        json_schema_extra = {
            "example": {
                "interaction_id": "int_001",
                "interaction_type": "comment",
                "source_user_id": "123456789",
                "target_user_id": "987654321",
                "source_post_id": None,
                "target_post_id": "post_001",
                "timestamp": "2024-01-15T12:00:00",
                "content": "This is a comment",
            }
        }
