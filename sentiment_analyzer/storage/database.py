"""
数据库连接和 ORM 模型定义

使用 SQLAlchemy 定义数据库表结构和异步连接管理。
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import NullPool, QueuePool

from .models import InteractionType, Platform


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""

    pass


class UserORM(Base):
    """
    用户 ORM 模型

    对应数据库中的 users 表。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    avatar_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    followers_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    friends_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    posts_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    posts: Mapped[list["PostORM"]] = relationship(
        "PostORM",
        back_populates="user",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_users_platform_user_id", "platform", "user_id", unique=True),
        Index("ix_users_last_updated", "last_updated"),
    )

    def to_pydantic(self) -> "storage.models.User":
        """转换为 Pydantic 模型"""
        from storage.models import User

        return User(
            user_id=self.user_id,
            platform=Platform(self.platform),
            username=self.username,
            display_name=self.display_name,
            bio=self.bio,
            avatar_url=self.avatar_url,
            avatar_hash=self.avatar_hash,
            created_at=self.created_at,
            followers_count=self.followers_count,
            friends_count=self.friends_count,
            posts_count=self.posts_count,
            verified=self.verified,
            first_seen=self.first_seen,
            last_updated=self.last_updated,
        )

    @classmethod
    def from_pydantic(cls, user: "storage.models.User") -> "UserORM":
        """从 Pydantic 模型创建 ORM 实例"""
        return cls(
            user_id=user.user_id,
            platform=user.platform.value,
            username=user.username,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            avatar_hash=user.avatar_hash,
            created_at=user.created_at,
            followers_count=user.followers_count,
            friends_count=user.friends_count,
            posts_count=user.posts_count,
            verified=user.verified,
            first_seen=user.first_seen,
            last_updated=user.last_updated,
        )

    def __repr__(self) -> str:
        return f"<UserORM(user_id={self.user_id}, platform={self.platform})>"


class PostORM(Base):
    """
    帖子 ORM 模型

    对应数据库中的 posts 表。
    """

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    posted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    likes_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    shares_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    comments_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    hashtags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)
    mentions: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)
    urls: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True, default=list)
    media: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True, default=list)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
    )

    user: Mapped["UserORM"] = relationship(
        "UserORM",
        back_populates="posts",
        lazy="selectin",
        primaryjoin=(
            "and_(foreign(PostORM.user_id) == UserORM.user_id, "
            "foreign(PostORM.platform) == UserORM.platform)"
        ),
    )

    __table_args__ = (
        Index("ix_posts_platform_post_id", "platform", "post_id", unique=True),
        Index("ix_posts_platform_user_id", "platform", "user_id"),
        Index("ix_posts_posted_at", "posted_at"),
        Index("ix_posts_collected_at", "collected_at"),
    )

    def to_pydantic(self) -> "storage.models.Post":
        """转换为 Pydantic 模型"""
        from storage.models import Post

        return Post(
            post_id=self.post_id,
            user_id=self.user_id,
            platform=Platform(self.platform),
            content=self.content,
            language=self.language,
            posted_at=self.posted_at,
            likes_count=self.likes_count,
            shares_count=self.shares_count,
            comments_count=self.comments_count,
            hashtags=self.hashtags or [],
            mentions=self.mentions or [],
            urls=self.urls or [],
            media=self.media or [],
            collected_at=self.collected_at,
        )

    @classmethod
    def from_pydantic(cls, post: "storage.models.Post") -> "PostORM":
        """从 Pydantic 模型创建 ORM 实例"""
        return cls(
            post_id=post.post_id,
            user_id=post.user_id,
            platform=post.platform.value,
            content=post.content,
            language=post.language,
            posted_at=post.posted_at,
            likes_count=post.likes_count,
            shares_count=post.shares_count,
            comments_count=post.comments_count,
            hashtags=post.hashtags,
            mentions=post.mentions,
            urls=post.urls,
            media=post.media,
            collected_at=post.collected_at,
        )

    def __repr__(self) -> str:
        return f"<PostORM(post_id={self.post_id}, platform={self.platform})>"


class InteractionORM(Base):
    """
    互动 ORM 模型

    对应数据库中的 interactions 表。
    """

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    interaction_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    target_user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    source_post_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    target_post_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
    )
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_interactions_interaction_id", "interaction_id", unique=True),
        Index("ix_interactions_type_timestamp", "interaction_type", "timestamp"),
        Index("ix_interactions_source_user", "source_user_id"),
        Index("ix_interactions_target_user", "target_user_id"),
    )

    def to_pydantic(self) -> "storage.models.Interaction":
        """转换为 Pydantic 模型"""
        from storage.models import Interaction

        return Interaction(
            interaction_id=self.interaction_id,
            interaction_type=InteractionType(self.interaction_type),
            source_user_id=self.source_user_id,
            target_user_id=self.target_user_id,
            source_post_id=self.source_post_id,
            target_post_id=self.target_post_id,
            timestamp=self.timestamp,
            content=self.content,
        )

    @classmethod
    def from_pydantic(cls, interaction: "storage.models.Interaction") -> "InteractionORM":
        """从 Pydantic 模型创建 ORM 实例"""
        return cls(
            interaction_id=interaction.interaction_id,
            interaction_type=interaction.interaction_type.value,
            source_user_id=interaction.source_user_id,
            target_user_id=interaction.target_user_id,
            source_post_id=interaction.source_post_id,
            target_post_id=interaction.target_post_id,
            timestamp=interaction.timestamp,
            content=interaction.content,
        )

    def __repr__(self) -> str:
        return f"<InteractionORM(id={self.interaction_id}, type={self.interaction_type})>"


class DatabaseConfig:
    """
    数据库配置

    Attributes:
        url: 数据库连接URL
        echo: 是否输出SQL日志
        pool_size: 连接池大小
        max_overflow: 最大溢出连接数
        pool_timeout: 连接池超时时间（秒）
        pool_recycle: 连接回收时间（秒）
    """

    def __init__(
        self,
        url: str = "sqlite+aiosqlite:///./sentiment_analyzer.db",
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: float = 30.0,
        pool_recycle: int = 3600,
    ):
        self.url = url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        从环境变量创建配置

        支持以下环境变量:
        - DATABASE_URL: 数据库连接URL
        - DB_ECHO: 是否输出SQL日志
        - DB_POOL_SIZE: 连接池大小
        - DB_MAX_OVERFLOW: 最大溢出连接数
        """
        import os

        url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./sentiment_analyzer.db")
        echo = os.getenv("DB_ECHO", "false").lower() == "true"
        pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

        return cls(
            url=url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
        )


class DatabaseManager:
    """
    数据库连接管理器

    提供异步数据库连接和会话管理功能。

    Attributes:
        config: 数据库配置
        engine: 异步引擎实例
        session_factory: 会话工厂
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        初始化数据库管理器

        Args:
            config: 数据库配置，如果为None则使用默认配置
        """
        self.config = config or DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    @property
    def engine(self) -> AsyncEngine:
        """获取异步引擎实例"""
        if self._engine is None:
            raise RuntimeError("Database engine not initialized. Call init() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """获取会话工厂"""
        if self._session_factory is None:
            raise RuntimeError("Session factory not initialized. Call init() first.")
        return self._session_factory

    async def init(self) -> None:
        """
        初始化数据库连接

        创建异步引擎和会话工厂，创建所有表。
        """
        pool_class = QueuePool
        pool_kwargs = {
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "pool_timeout": self.config.pool_timeout,
            "pool_recycle": self.config.pool_recycle,
        }

        if "sqlite" in self.config.url:
            pool_class = NullPool
            pool_kwargs = {}

        self._engine = create_async_engine(
            self.config.url,
            echo=self.config.echo,
            poolclass=pool_class,
            **pool_kwargs,
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        获取数据库会话上下文管理器

        Yields:
            异步会话实例

        Example:
            async with db.session() as session:
                result = await session.execute(select(UserORM))
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        获取数据库会话生成器（用于依赖注入）

        Yields:
            异步会话实例
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def add_user(self, user: "storage.models.User") -> UserORM:
        """
        添加用户

        Args:
            user: Pydantic 用户模型

        Returns:
            创建的 ORM 用户实例
        """
        async with self.session() as session:
            user_orm = UserORM.from_pydantic(user)
            session.add(user_orm)
            await session.flush()
            await session.refresh(user_orm)
            return user_orm

    async def add_post(self, post: "storage.models.Post") -> PostORM:
        """
        添加帖子

        Args:
            post: Pydantic 帖子模型

        Returns:
            创建的 ORM 帖子实例
        """
        async with self.session() as session:
            post_orm = PostORM.from_pydantic(post)
            session.add(post_orm)
            await session.flush()
            await session.refresh(post_orm)
            return post_orm

    async def add_interaction(self, interaction: "storage.models.Interaction") -> InteractionORM:
        """
        添加互动

        Args:
            interaction: Pydantic 互动模型

        Returns:
            创建的 ORM 互动实例
        """
        async with self.session() as session:
            interaction_orm = InteractionORM.from_pydantic(interaction)
            session.add(interaction_orm)
            await session.flush()
            await session.refresh(interaction_orm)
            return interaction_orm

    async def get_user_by_id(self, platform: Platform, user_id: str) -> Optional[UserORM]:
        """
        根据平台和用户ID获取用户

        Args:
            platform: 平台
            user_id: 用户ID

        Returns:
            用户 ORM 实例，如果不存在则返回 None
        """
        async with self.session() as session:
            stmt = select(UserORM).where(
                UserORM.platform == platform.value,
                UserORM.user_id == user_id,
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_post_by_id(self, platform: Platform, post_id: str) -> Optional[PostORM]:
        """
        根据平台和帖子ID获取帖子

        Args:
            platform: 平台
            post_id: 帖子ID

        Returns:
            帖子 ORM 实例，如果不存在则返回 None
        """
        async with self.session() as session:
            stmt = select(PostORM).where(
                PostORM.platform == platform.value,
                PostORM.post_id == post_id,
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def upsert_user(self, user: "storage.models.User") -> UserORM:
        """
        插入或更新用户

        如果用户已存在则更新，否则插入新用户。

        Args:
            user: Pydantic 用户模型

        Returns:
            创建或更新的 ORM 用户实例
        """
        existing = await self.get_user_by_id(user.platform, user.user_id)
        if existing:
            async with self.session() as session:
                stmt = select(UserORM).where(UserORM.id == existing.id)
                result = await session.execute(stmt)
                user_orm = result.scalar_one()
                user_orm.username = user.username
                user_orm.display_name = user.display_name
                user_orm.bio = user.bio
                user_orm.avatar_url = user.avatar_url
                user_orm.avatar_hash = user.avatar_hash
                user_orm.created_at = user.created_at
                user_orm.followers_count = user.followers_count
                user_orm.friends_count = user.friends_count
                user_orm.posts_count = user.posts_count
                user_orm.verified = user.verified
                user_orm.last_updated = datetime.utcnow()
                await session.flush()
                await session.refresh(user_orm)
                return user_orm
        else:
            return await self.add_user(user)

    async def upsert_post(self, post: "storage.models.Post") -> PostORM:
        """
        插入或更新帖子

        如果帖子已存在则更新，否则插入新帖子。

        Args:
            post: Pydantic 帖子模型

        Returns:
            创建或更新的 ORM 帖子实例
        """
        existing = await self.get_post_by_id(post.platform, post.post_id)
        if existing:
            async with self.session() as session:
                stmt = select(PostORM).where(PostORM.id == existing.id)
                result = await session.execute(stmt)
                post_orm = result.scalar_one()
                post_orm.content = post.content
                post_orm.language = post.language
                post_orm.posted_at = post.posted_at
                post_orm.likes_count = post.likes_count
                post_orm.shares_count = post.shares_count
                post_orm.comments_count = post.comments_count
                post_orm.hashtags = post.hashtags
                post_orm.mentions = post.mentions
                post_orm.urls = post.urls
                post_orm.media = post.media
                await session.flush()
                await session.refresh(post_orm)
                return post_orm
        else:
            return await self.add_post(post)


db_manager: Optional[DatabaseManager] = None


async def get_db() -> DatabaseManager:
    """
    获取全局数据库管理器实例

    Returns:
        数据库管理器实例
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.init()
    return db_manager


async def close_db() -> None:
    """关闭全局数据库连接"""
    global db_manager
    if db_manager:
        await db_manager.close()
        db_manager = None
