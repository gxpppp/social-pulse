"""
SQLite 数据存储模块
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..crawlers.base import Platform, PostData, UserData

Base = declarative_base()


class PostModel(Base):
    """帖子数据表模型"""
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(20), nullable=False, index=True)
    post_id = Column(String(100), nullable=False, index=True)
    author_id = Column(String(100), nullable=False, index=True)
    author_name = Column(String(200))
    content = Column(Text)
    created_at = Column(DateTime, index=True)
    language = Column(String(10))
    likes = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    hashtags = Column(Text)
    mentions = Column(Text)
    urls = Column(Text)
    media_urls = Column(Text)
    location = Column(String(200))
    is_retweet = Column(Boolean, default=False)
    parent_id = Column(String(100))
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    raw_data = Column(Text)
    crawled_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        {"unique_constraint": ("platform", "post_id")},
    )


class UserModel(Base):
    """用户数据表模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    platform = Column(String(20), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    username = Column(String(200))
    display_name = Column(String(200))
    bio = Column(Text)
    followers_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)
    posts_count = Column(Integer, default=0)
    created_at = Column(DateTime)
    verified = Column(Boolean, default=False)
    avatar_url = Column(String(500))
    location = Column(String(200))
    raw_data = Column(Text)
    crawled_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        {"unique_constraint": ("platform", "user_id")},
    )


class SQLiteStore:
    """SQLite 数据存储类"""

    def __init__(self, db_path: str = "./data/sentiment.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: Any = None
        self._async_session: Any = None

    async def initialize(self) -> None:
        """初始化数据库连接和表结构"""
        db_url = f"sqlite+aiosqlite:///{self.db_path}"
        self._engine = create_async_engine(db_url, echo=False)
        self._async_session = sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info(f"SQLite database initialized at {self.db_path}")

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._engine:
            await self._engine.dispose()
        logger.info("SQLite database connection closed")

    async def save_post(self, post: PostData) -> bool:
        """保存帖子数据"""
        async with self._async_session() as session:
            try:
                existing = await session.execute(
                    f"SELECT id FROM posts WHERE platform = ? AND post_id = ?",
                    (post.platform, post.post_id)
                )
                if existing.fetchone():
                    logger.debug(f"Post {post.post_id} already exists")
                    return False

                await session.execute(
                    """
                    INSERT INTO posts (
                        platform, post_id, author_id, author_name, content,
                        created_at, language, likes, shares, comments,
                        hashtags, mentions, urls, media_urls, location,
                        is_retweet, parent_id, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post.platform,
                        post.post_id,
                        post.author_id,
                        post.author_name,
                        post.content,
                        post.created_at,
                        post.language,
                        post.likes,
                        post.shares,
                        post.comments,
                        json.dumps(post.hashtags),
                        json.dumps(post.mentions),
                        json.dumps(post.urls),
                        json.dumps(post.media_urls),
                        post.location,
                        post.is_retweet,
                        post.parent_id,
                        json.dumps(post.raw_data)
                    )
                )
                await session.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save post: {e}")
                await session.rollback()
                return False

    async def save_user(self, user: UserData) -> bool:
        """保存用户数据"""
        async with self._async_session() as session:
            try:
                await session.execute(
                    """
                    INSERT OR REPLACE INTO users (
                        platform, user_id, username, display_name, bio,
                        followers_count, following_count, posts_count,
                        created_at, verified, avatar_url, location, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user.platform,
                        user.user_id,
                        user.username,
                        user.display_name,
                        user.bio,
                        user.followers_count,
                        user.following_count,
                        user.posts_count,
                        user.created_at,
                        user.verified,
                        user.avatar_url,
                        user.location,
                        json.dumps(user.raw_data)
                    )
                )
                await session.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save user: {e}")
                await session.rollback()
                return False

    async def get_post(self, platform: str, post_id: str) -> Optional[dict[str, Any]]:
        """获取帖子数据"""
        async with self._async_session() as session:
            result = await session.execute(
                "SELECT * FROM posts WHERE platform = ? AND post_id = ?",
                (platform, post_id)
            )
            row = result.fetchone()
            return dict(row) if row else None

    async def get_user(self, platform: str, user_id: str) -> Optional[dict[str, Any]]:
        """获取用户数据"""
        async with self._async_session() as session:
            result = await session.execute(
                "SELECT * FROM users WHERE platform = ? AND user_id = ?",
                (platform, user_id)
            )
            row = result.fetchone()
            return dict(row) if row else None

    async def get_posts_by_user(
        self,
        platform: str,
        user_id: str,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """获取用户的所有帖子"""
        async with self._async_session() as session:
            result = await session.execute(
                """
                SELECT * FROM posts
                WHERE platform = ? AND author_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (platform, user_id, limit)
            )
            return [dict(row) for row in result.fetchall()]

    async def search_posts(
        self,
        query: str,
        platform: Optional[str] = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """搜索帖子"""
        async with self._async_session() as session:
            sql = "SELECT * FROM posts WHERE content LIKE ?"
            params = [f"%{query}%"]

            if platform:
                sql += " AND platform = ?"
                params.append(platform)

            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            result = await session.execute(sql, params)
            return [dict(row) for row in result.fetchall()]

    async def update_sentiment(
        self,
        platform: str,
        post_id: str,
        score: float,
        label: str
    ) -> bool:
        """更新帖子情感分析结果"""
        async with self._async_session() as session:
            try:
                await session.execute(
                    """
                    UPDATE posts
                    SET sentiment_score = ?, sentiment_label = ?
                    WHERE platform = ? AND post_id = ?
                    """,
                    (score, label, platform, post_id)
                )
                await session.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to update sentiment: {e}")
                await session.rollback()
                return False

    async def get_statistics(self) -> dict[str, Any]:
        """获取数据库统计信息"""
        async with self._async_session() as session:
            posts_count = await session.execute("SELECT COUNT(*) FROM posts")
            users_count = await session.execute("SELECT COUNT(*) FROM users")
            platforms = await session.execute(
                "SELECT platform, COUNT(*) as count FROM posts GROUP BY platform"
            )

            return {
                "total_posts": posts_count.fetchone()[0],
                "total_users": users_count.fetchone()[0],
                "posts_by_platform": {row[0]: row[1] for row in platforms.fetchall()}
            }
