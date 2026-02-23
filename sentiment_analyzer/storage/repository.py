"""
数据访问层模块

提供异步的数据访问接口，包括：
- UserRepository: 用户数据CRUD操作
- PostRepository: 帖子数据CRUD操作
- InteractionRepository: 互动数据CRUD操作
- CrawlTaskRepository: 采集任务CRUD操作
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .schema import (
    Base,
    CrawlTask,
    Interaction,
    Post,
    SystemLog,
    User,
    UserFeature,
)


class DatabaseError(Exception):
    """数据库操作异常"""

    pass


class NotFoundError(DatabaseError):
    """数据未找到异常"""

    pass


class DuplicateError(DatabaseError):
    """数据重复异常"""

    pass


class Database:
    """数据库管理类

    提供数据库连接、会话管理和初始化功能。
    """

    def __init__(self, db_path: str = "sqlite+aiosqlite:///./data/sentiment.db"):
        self.db_path = db_path
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise DatabaseError("Database not initialized. Call init() first.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise DatabaseError("Database not initialized. Call init() first.")
        return self._session_factory

    async def init(self) -> None:
        """初始化数据库连接"""
        db_dir = Path(self.db_path.replace("sqlite+aiosqlite:///", ""))
        db_dir = db_dir.parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(
            self.db_path,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        self._session_factory = async_sessionmaker(
            self._engine,
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
    async def session(self):
        """获取数据库会话上下文管理器"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise DatabaseError(f"Database operation failed: {e}") from e

    async def execute_raw_sql(self, sql: str, params: Optional[dict] = None) -> Any:
        """执行原始SQL语句"""
        async with self.session() as session:
            result = await session.execute(sql, params or {})
            return result


class UserRepository:
    """用户数据访问类

    提供用户数据的CRUD操作和查询功能。
    """

    def __init__(self, db: Database):
        self.db = db

    async def create(self, user_data: dict) -> User:
        """创建用户"""
        async with self.db.session() as session:
            user = User(**user_data)
            session.add(user)
            await session.flush()
            await session.refresh(user)
            return user

    async def get_by_id(self, user_id: str) -> Optional[User]:
        """根据user_id获取用户"""
        async with self.db.session() as session:
            result = await session.execute(
                select(User).where(User.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_by_platform_username(
        self, platform: str, username: str
    ) -> Optional[User]:
        """根据平台和用户名获取用户"""
        async with self.db.session() as session:
            result = await session.execute(
                select(User).where(
                    User.platform == platform, User.username == username
                )
            )
            return result.scalar_one_or_none()

    async def update(self, user_id: str, update_data: dict) -> Optional[User]:
        """更新用户信息"""
        async with self.db.session() as session:
            result = await session.execute(
                update(User).where(User.user_id == user_id).values(**update_data)
            )
            if result.rowcount == 0:
                return None
            return await self.get_by_id(user_id)

    async def delete(self, user_id: str) -> bool:
        """删除用户"""
        async with self.db.session() as session:
            result = await session.execute(
                delete(User).where(User.user_id == user_id)
            )
            return result.rowcount > 0

    async def upsert(self, user_data: dict) -> User:
        """创建或更新用户（存在则更新，不存在则创建）"""
        user_id = user_data.get("user_id")
        existing = await self.get_by_id(user_id)
        if existing:
            return await self.update(user_id, user_data)
        return await self.create(user_data)

    async def batch_insert(self, users: list[dict]) -> int:
        """批量插入用户"""
        async with self.db.session() as session:
            stmt = sqlite_insert(User).values(users)
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "username": stmt.excluded.username,
                    "display_name": stmt.excluded.display_name,
                    "bio": stmt.excluded.bio,
                    "avatar_url": stmt.excluded.avatar_url,
                    "followers_count": stmt.excluded.followers_count,
                    "friends_count": stmt.excluded.friends_count,
                    "posts_count": stmt.excluded.posts_count,
                    "verified": stmt.excluded.verified,
                    "last_updated": datetime.now(),
                },
            )
            result = await session.execute(stmt)
            return result.rowcount

    async def list_users(
        self,
        platform: Optional[str] = None,
        is_suspicious: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[User]:
        """列出用户"""
        async with self.db.session() as session:
            query = select(User)
            if platform:
                query = query.where(User.platform == platform)
            if is_suspicious is not None:
                query = query.where(User.is_suspicious == is_suspicious)
            query = query.order_by(User.first_seen.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def count_users(
        self, platform: Optional[str] = None, is_suspicious: Optional[bool] = None
    ) -> int:
        """统计用户数量"""
        async with self.db.session() as session:
            query = select(func.count(User.id))
            if platform:
                query = query.where(User.platform == platform)
            if is_suspicious is not None:
                query = query.where(User.is_suspicious == is_suspicious)
            result = await session.execute(query)
            return result.scalar() or 0

    async def mark_suspicious(
        self, user_id: str, score: float, is_suspicious: bool = True
    ) -> bool:
        """标记用户为可疑"""
        async with self.db.session() as session:
            result = await session.execute(
                update(User)
                .where(User.user_id == user_id)
                .values(is_suspicious=is_suspicious, suspicious_score=score)
            )
            return result.rowcount > 0


class PostRepository:
    """帖子数据访问类

    提供帖子数据的CRUD操作和查询功能。
    """

    def __init__(self, db: Database):
        self.db = db

    async def create(self, post_data: dict) -> Post:
        """创建帖子"""
        async with self.db.session() as session:
            if isinstance(post_data.get("hashtags"), list):
                post_data["hashtags"] = json.dumps(post_data["hashtags"])
            if isinstance(post_data.get("mentions"), list):
                post_data["mentions"] = json.dumps(post_data["mentions"])
            if isinstance(post_data.get("urls"), list):
                post_data["urls"] = json.dumps(post_data["urls"])
            if isinstance(post_data.get("media"), list):
                post_data["media"] = json.dumps(post_data["media"])

            post = Post(**post_data)
            session.add(post)
            await session.flush()
            await session.refresh(post)
            return post

    async def get_by_id(self, post_id: str) -> Optional[Post]:
        """根据post_id获取帖子"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Post).where(Post.post_id == post_id)
            )
            return result.scalar_one_or_none()

    async def update(self, post_id: str, update_data: dict) -> Optional[Post]:
        """更新帖子信息"""
        async with self.db.session() as session:
            result = await session.execute(
                update(Post).where(Post.post_id == post_id).values(**update_data)
            )
            if result.rowcount == 0:
                return None
            return await self.get_by_id(post_id)

    async def delete(self, post_id: str) -> bool:
        """删除帖子"""
        async with self.db.session() as session:
            result = await session.execute(
                delete(Post).where(Post.post_id == post_id)
            )
            return result.rowcount > 0

    async def upsert(self, post_data: dict) -> Post:
        """创建或更新帖子"""
        post_id = post_data.get("post_id")
        existing = await self.get_by_id(post_id)
        if existing:
            return await self.update(post_id, post_data)
        return await self.create(post_data)

    async def batch_insert(self, posts: list[dict]) -> int:
        """批量插入帖子"""
        processed_posts = []
        for post in posts:
            processed = post.copy()
            if isinstance(processed.get("hashtags"), list):
                processed["hashtags"] = json.dumps(processed["hashtags"])
            if isinstance(processed.get("mentions"), list):
                processed["mentions"] = json.dumps(processed["mentions"])
            if isinstance(processed.get("urls"), list):
                processed["urls"] = json.dumps(processed["urls"])
            if isinstance(processed.get("media"), list):
                processed["media"] = json.dumps(processed["media"])
            processed_posts.append(processed)

        async with self.db.session() as session:
            stmt = sqlite_insert(Post).values(processed_posts)
            stmt = stmt.on_conflict_do_update(
                index_elements=["post_id"],
                set_={
                    "content": stmt.excluded.content,
                    "likes_count": stmt.excluded.likes_count,
                    "shares_count": stmt.excluded.shares_count,
                    "comments_count": stmt.excluded.comments_count,
                    "views_count": stmt.excluded.views_count,
                    "updated_at": datetime.now(),
                },
            )
            result = await session.execute(stmt)
            return result.rowcount

    async def list_posts(
        self,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Post]:
        """列出帖子"""
        async with self.db.session() as session:
            query = select(Post)
            if user_id:
                query = query.where(Post.user_id == user_id)
            if platform:
                query = query.where(Post.platform == platform)
            if start_time:
                query = query.where(Post.posted_at >= start_time)
            if end_time:
                query = query.where(Post.posted_at <= end_time)
            query = query.order_by(Post.posted_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def count_posts(
        self,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """统计帖子数量"""
        async with self.db.session() as session:
            query = select(func.count(Post.id))
            if user_id:
                query = query.where(Post.user_id == user_id)
            if platform:
                query = query.where(Post.platform == platform)
            if start_time:
                query = query.where(Post.posted_at >= start_time)
            if end_time:
                query = query.where(Post.posted_at <= end_time)
            result = await session.execute(query)
            return result.scalar() or 0

    async def get_user_posts(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> list[Post]:
        """获取用户的所有帖子"""
        return await self.list_posts(user_id=user_id, limit=limit, offset=offset)

    async def get_posts_by_hashtag(
        self, hashtag: str, platform: Optional[str] = None, limit: int = 100
    ) -> list[Post]:
        """根据话题标签获取帖子"""
        async with self.db.session() as session:
            query = select(Post).where(Post.hashtags.contains(hashtag))
            if platform:
                query = query.where(Post.platform == platform)
            query = query.order_by(Post.posted_at.desc()).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())


class InteractionRepository:
    """互动数据访问类

    提供互动数据的CRUD操作和查询功能。
    """

    def __init__(self, db: Database):
        self.db = db

    async def create(self, interaction_data: dict) -> Interaction:
        """创建互动记录"""
        async with self.db.session() as session:
            interaction = Interaction(**interaction_data)
            session.add(interaction)
            await session.flush()
            await session.refresh(interaction)
            return interaction

    async def get_by_id(self, interaction_id: str) -> Optional[Interaction]:
        """根据interaction_id获取互动记录"""
        async with self.db.session() as session:
            result = await session.execute(
                select(Interaction).where(Interaction.interaction_id == interaction_id)
            )
            return result.scalar_one_or_none()

    async def delete(self, interaction_id: str) -> bool:
        """删除互动记录"""
        async with self.db.session() as session:
            result = await session.execute(
                delete(Interaction).where(Interaction.interaction_id == interaction_id)
            )
            return result.rowcount > 0

    async def batch_insert(self, interactions: list[dict]) -> int:
        """批量插入互动记录"""
        async with self.db.session() as session:
            stmt = sqlite_insert(Interaction).values(interactions)
            stmt = stmt.on_conflict_do_nothing(index_elements=["interaction_id"])
            result = await session.execute(stmt)
            return result.rowcount

    async def list_interactions(
        self,
        user_id: Optional[str] = None,
        post_id: Optional[str] = None,
        interaction_type: Optional[str] = None,
        platform: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Interaction]:
        """列出互动记录"""
        async with self.db.session() as session:
            query = select(Interaction)
            if user_id:
                query = query.where(Interaction.user_id == user_id)
            if post_id:
                query = query.where(Interaction.post_id == post_id)
            if interaction_type:
                query = query.where(Interaction.interaction_type == interaction_type)
            if platform:
                query = query.where(Interaction.platform == platform)
            query = (
                query.order_by(Interaction.interacted_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            return list(result.scalars().all())

    async def count_interactions(
        self,
        user_id: Optional[str] = None,
        post_id: Optional[str] = None,
        interaction_type: Optional[str] = None,
    ) -> int:
        """统计互动数量"""
        async with self.db.session() as session:
            query = select(func.count(Interaction.id))
            if user_id:
                query = query.where(Interaction.user_id == user_id)
            if post_id:
                query = query.where(Interaction.post_id == post_id)
            if interaction_type:
                query = query.where(Interaction.interaction_type == interaction_type)
            result = await session.execute(query)
            return result.scalar() or 0

    async def get_user_interactions(
        self, user_id: str, interaction_type: Optional[str] = None, limit: int = 100
    ) -> list[Interaction]:
        """获取用户的互动记录"""
        return await self.list_interactions(
            user_id=user_id, interaction_type=interaction_type, limit=limit
        )

    async def get_post_interactions(
        self, post_id: str, interaction_type: Optional[str] = None, limit: int = 100
    ) -> list[Interaction]:
        """获取帖子的互动记录"""
        return await self.list_interactions(
            post_id=post_id, interaction_type=interaction_type, limit=limit
        )


class CrawlTaskRepository:
    """采集任务数据访问类

    提供采集任务的CRUD操作和状态管理。
    """

    def __init__(self, db: Database):
        self.db = db

    async def create(self, task_data: dict) -> CrawlTask:
        """创建采集任务"""
        async with self.db.session() as session:
            if isinstance(task_data.get("config"), dict):
                task_data["config"] = json.dumps(task_data["config"])

            task = CrawlTask(**task_data)
            session.add(task)
            await session.flush()
            await session.refresh(task)
            return task

    async def get_by_id(self, task_id: str) -> Optional[CrawlTask]:
        """根据task_id获取任务"""
        async with self.db.session() as session:
            result = await session.execute(
                select(CrawlTask).where(CrawlTask.task_id == task_id)
            )
            return result.scalar_one_or_none()

    async def update_status(
        self,
        task_id: str,
        status: str,
        result_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """更新任务状态"""
        async with self.db.session() as session:
            update_data: dict[str, Any] = {"status": status}

            if status == "running":
                update_data["started_at"] = datetime.now()
            elif status in ("completed", "failed"):
                update_data["completed_at"] = datetime.now()

            if result_count is not None:
                update_data["result_count"] = result_count
            if error_message is not None:
                update_data["error_message"] = error_message

            result = await session.execute(
                update(CrawlTask).where(CrawlTask.task_id == task_id).values(update_data)
            )
            return result.rowcount > 0

    async def increment_retry(self, task_id: str) -> bool:
        """增加重试计数"""
        async with self.db.session() as session:
            result = await session.execute(
                update(CrawlTask)
                .where(CrawlTask.task_id == task_id)
                .values(retry_count=CrawlTask.retry_count + 1)
            )
            return result.rowcount > 0

    async def get_pending_tasks(
        self, platform: Optional[str] = None, limit: int = 10
    ) -> list[CrawlTask]:
        """获取待处理的任务"""
        async with self.db.session() as session:
            query = (
                select(CrawlTask)
                .where(CrawlTask.status == "pending")
                .where(CrawlTask.retry_count < CrawlTask.max_retries)
                .order_by(CrawlTask.priority.desc(), CrawlTask.created_at.asc())
            )
            if platform:
                query = query.where(CrawlTask.platform == platform)
            query = query.limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def list_tasks(
        self,
        platform: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CrawlTask]:
        """列出任务"""
        async with self.db.session() as session:
            query = select(CrawlTask)
            if platform:
                query = query.where(CrawlTask.platform == platform)
            if status:
                query = query.where(CrawlTask.status == status)
            query = query.order_by(CrawlTask.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_old_tasks(self, days: int = 30) -> int:
        """删除旧任务"""
        async with self.db.session() as session:
            cutoff = datetime.now() - __import__("datetime").timedelta(days=days)
            result = await session.execute(
                delete(CrawlTask)
                .where(CrawlTask.status.in_(["completed", "failed"]))
                .where(CrawlTask.completed_at < cutoff)
            )
            return result.rowcount


class UserFeatureRepository:
    """用户特征数据访问类"""

    def __init__(self, db: Database):
        self.db = db

    async def upsert(self, feature_data: dict) -> UserFeature:
        """创建或更新用户特征"""
        async with self.db.session() as session:
            user_id = feature_data.get("user_id")
            existing = await session.execute(
                select(UserFeature).where(UserFeature.user_id == user_id)
            )
            existing_feature = existing.scalar_one_or_none()

            if existing_feature:
                await session.execute(
                    update(UserFeature)
                    .where(UserFeature.user_id == user_id)
                    .values(**feature_data)
                )
                await session.flush()
                result = await session.execute(
                    select(UserFeature).where(UserFeature.user_id == user_id)
                )
                return result.scalar_one()
            else:
                feature = UserFeature(**feature_data)
                session.add(feature)
                await session.flush()
                await session.refresh(feature)
                return feature

    async def get_by_user_id(self, user_id: str) -> Optional[UserFeature]:
        """根据user_id获取用户特征"""
        async with self.db.session() as session:
            result = await session.execute(
                select(UserFeature).where(UserFeature.user_id == user_id)
            )
            return result.scalar_one_or_none()

    async def get_suspicious_users(
        self, threshold: float = 0.7, limit: int = 100
    ) -> list[UserFeature]:
        """获取可疑用户列表"""
        async with self.db.session() as session:
            result = await session.execute(
                select(UserFeature)
                .where(UserFeature.anomaly_score >= threshold)
                .order_by(UserFeature.anomaly_score.desc())
                .limit(limit)
            )
            return list(result.scalars().all())


class SystemLogRepository:
    """系统日志数据访问类"""

    def __init__(self, db: Database):
        self.db = db

    async def log(
        self, level: str, module: str, message: str, details: Optional[dict] = None
    ) -> SystemLog:
        """记录系统日志"""
        async with self.db.session() as session:
            log_entry = SystemLog(
                log_level=level,
                module=module,
                message=message,
                details=json.dumps(details) if details else None,
            )
            session.add(log_entry)
            await session.flush()
            await session.refresh(log_entry)
            return log_entry

    async def list_logs(
        self,
        level: Optional[str] = None,
        module: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SystemLog]:
        """列出日志"""
        async with self.db.session() as session:
            query = select(SystemLog)
            if level:
                query = query.where(SystemLog.log_level == level)
            if module:
                query = query.where(SystemLog.module == module)
            query = query.order_by(SystemLog.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def delete_old_logs(self, days: int = 30) -> int:
        """删除旧日志"""
        async with self.db.session() as session:
            cutoff = datetime.now() - __import__("datetime").timedelta(days=days)
            result = await session.execute(
                delete(SystemLog).where(SystemLog.created_at < cutoff)
            )
            return result.rowcount
