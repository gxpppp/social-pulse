"""
数据库Schema定义模块

使用SQLAlchemy定义所有数据库表结构，包括：
- 用户表 (users)
- 帖子表 (posts)
- 互动表 (interactions)
- 采集任务表 (crawl_tasks)
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """SQLAlchemy基类，支持异步属性加载"""

    pass


class TimestampMixin:
    """时间戳混入类"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )


class User(Base, TimestampMixin):
    """用户表

    存储多平台用户信息，支持跨平台用户关联分析。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(512))
    avatar_hash: Mapped[Optional[str]] = mapped_column(String(64))
    registered_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    followers_count: Mapped[int] = mapped_column(Integer, default=0)
    friends_count: Mapped[int] = mapped_column(Integer, default=0)
    posts_count: Mapped[int] = mapped_column(Integer, default=0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    first_seen: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    is_suspicious: Mapped[bool] = mapped_column(Boolean, default=False)
    suspicious_score: Mapped[Optional[float]] = mapped_column(Float)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    posts: Mapped[list["Post"]] = relationship(
        "Post", back_populates="user", lazy="selectin"
    )
    interactions: Mapped[list["Interaction"]] = relationship(
        "Interaction", back_populates="user", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_users_platform_username", "platform", "username"),
        Index("idx_users_platform", "platform"),
        Index("idx_users_suspicious", "is_suspicious"),
        Index("idx_users_first_seen", "first_seen"),
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, platform={self.platform}, username={self.username})>"


class Post(Base, TimestampMixin):
    """帖子表

    存储各平台的帖子/推文内容，支持内容分析和传播追踪。
    """

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    posted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    likes_count: Mapped[int] = mapped_column(Integer, default=0)
    shares_count: Mapped[int] = mapped_column(Integer, default=0)
    comments_count: Mapped[int] = mapped_column(Integer, default=0)
    views_count: Mapped[int] = mapped_column(Integer, default=0)
    hashtags: Mapped[Optional[list]] = mapped_column(JSON)
    mentions: Mapped[Optional[list]] = mapped_column(JSON)
    urls: Mapped[Optional[list]] = mapped_column(JSON)
    media: Mapped[Optional[list]] = mapped_column(JSON)
    parent_post_id: Mapped[Optional[str]] = mapped_column(String(64))
    is_retweet: Mapped[bool] = mapped_column(Boolean, default=False)
    is_reply: Mapped[bool] = mapped_column(Boolean, default=False)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    user: Mapped["User"] = relationship("User", back_populates="posts")
    interactions: Mapped[list["Interaction"]] = relationship(
        "Interaction", back_populates="post", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_posts_user_id", "user_id"),
        Index("idx_posts_platform", "platform"),
        Index("idx_posts_posted_at", "posted_at"),
        Index("idx_posts_collected_at", "collected_at"),
        Index("idx_posts_parent", "parent_post_id"),
        Index("idx_posts_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<Post(post_id={self.post_id}, platform={self.platform}, user_id={self.user_id})>"


class Interaction(Base, TimestampMixin):
    """互动表

    记录用户间的互动行为（转发、评论、点赞等），用于社交网络分析。
    """

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), nullable=False
    )
    post_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("posts.post_id"), nullable=False
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    interaction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    interacted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    collected_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    user: Mapped["User"] = relationship("User", back_populates="interactions")
    post: Mapped["Post"] = relationship("Post", back_populates="interactions")

    __table_args__ = (
        Index("idx_interactions_user_id", "user_id"),
        Index("idx_interactions_post_id", "post_id"),
        Index("idx_interactions_type", "interaction_type"),
        Index("idx_interactions_platform", "platform"),
        Index("idx_interactions_interacted_at", "interacted_at"),
        Index(
            "idx_interactions_user_post_type", "user_id", "post_id", "interaction_type"
        ),
    )

    def __repr__(self) -> str:
        return f"<Interaction(interaction_id={self.interaction_id}, type={self.interaction_type})>"


class CrawlTask(Base, TimestampMixin):
    """采集任务表

    管理数据采集任务的状态、进度和配置。
    """

    __tablename__ = "crawl_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=5)
    config: Mapped[Optional[dict]] = mapped_column(JSON)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    __table_args__ = (
        Index("idx_crawl_tasks_platform", "platform"),
        Index("idx_crawl_tasks_status", "status"),
        Index("idx_crawl_tasks_task_type", "task_type"),
        Index("idx_crawl_tasks_priority", "priority"),
        Index("idx_crawl_tasks_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<CrawlTask(task_id={self.task_id}, platform={self.platform}, status={self.status})>"


class UserFeature(Base):
    """用户特征表

    存储用户行为特征分析结果，用于异常检测和分类。
    """

    __tablename__ = "user_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.user_id"), unique=True, nullable=False
    )
    platform: Mapped[str] = mapped_column(String(32), nullable=False)

    daily_post_avg: Mapped[Optional[float]] = mapped_column(Float)
    daily_post_std: Mapped[Optional[float]] = mapped_column(Float)
    hour_entropy: Mapped[Optional[float]] = mapped_column(Float)
    night_activity_ratio: Mapped[Optional[float]] = mapped_column(Float)
    weekend_activity_ratio: Mapped[Optional[float]] = mapped_column(Float)

    content_similarity_avg: Mapped[Optional[float]] = mapped_column(Float)
    topic_entropy: Mapped[Optional[float]] = mapped_column(Float)
    sentiment_variance: Mapped[Optional[float]] = mapped_column(Float)
    avg_text_length: Mapped[Optional[float]] = mapped_column(Float)
    url_ratio: Mapped[Optional[float]] = mapped_column(Float)
    mention_ratio: Mapped[Optional[float]] = mapped_column(Float)

    follower_ratio: Mapped[Optional[float]] = mapped_column(Float)
    account_age_days: Mapped[Optional[int]] = mapped_column(Integer)
    profile_completeness: Mapped[Optional[float]] = mapped_column(Float)

    degree_centrality: Mapped[Optional[float]] = mapped_column(Float)
    betweenness_centrality: Mapped[Optional[float]] = mapped_column(Float)
    clustering_coefficient: Mapped[Optional[float]] = mapped_column(Float)

    anomaly_score: Mapped[Optional[float]] = mapped_column(Float)
    predicted_label: Mapped[Optional[str]] = mapped_column(String(32))
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)

    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )
    feature_version: Mapped[str] = mapped_column(String(32), default="1.0")

    __table_args__ = (
        Index("idx_user_features_user_id", "user_id"),
        Index("idx_user_features_platform", "platform"),
        Index("idx_user_features_anomaly_score", "anomaly_score"),
        Index("idx_user_features_predicted_label", "predicted_label"),
    )

    def __repr__(self) -> str:
        return f"<UserFeature(user_id={self.user_id}, anomaly_score={self.anomaly_score})>"


class SystemLog(Base):
    """系统日志表

    记录系统运行日志，用于监控和审计。
    """

    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_level: Mapped[str] = mapped_column(String(16), nullable=False)
    module: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_system_logs_log_level", "log_level"),
        Index("idx_system_logs_module", "module"),
        Index("idx_system_logs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<SystemLog(level={self.log_level}, module={self.module})>"


def get_all_models() -> list[type[Base]]:
    """获取所有模型类列表"""
    return [User, Post, Interaction, CrawlTask, UserFeature, SystemLog]
