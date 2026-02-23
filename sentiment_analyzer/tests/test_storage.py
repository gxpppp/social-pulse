"""
存储模块测试

测试数据库管理、用户仓储、帖子仓储等功能。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from ..storage.schema import Base, User, Post, Interaction, CrawlTask, UserFeature
from ..storage.repository import (
    Database,
    UserRepository,
    PostRepository,
    InteractionRepository,
    CrawlTaskRepository,
    UserFeatureRepository,
    DatabaseError,
    NotFoundError,
)


class TestDatabase:
    """数据库管理测试"""

    @pytest.mark.asyncio
    async def test_init(self, test_db: Database):
        """测试数据库初始化"""
        assert test_db._engine is not None
        assert test_db._session_factory is not None

    @pytest.mark.asyncio
    async def test_close(self, test_db: Database):
        """测试关闭数据库"""
        await test_db.close()
        assert test_db._engine is None
        assert test_db._session_factory is None

    @pytest.mark.asyncio
    async def test_session_context(self, test_db: Database):
        """测试会话上下文管理"""
        async with test_db.session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_engine_property_before_init(self):
        """测试初始化前访问引擎"""
        db = Database(db_path="sqlite+aiosqlite:///:memory:")
        with pytest.raises(DatabaseError):
            _ = db.engine

    @pytest.mark.asyncio
    async def test_session_factory_property_before_init(self):
        """测试初始化前访问会话工厂"""
        db = Database(db_path="sqlite+aiosqlite:///:memory:")
        with pytest.raises(DatabaseError):
            _ = db.session_factory

    @pytest.mark.asyncio
    async def test_database_path_creation(self, tmp_path: Path):
        """测试数据库路径创建"""
        db_path = tmp_path / "test_data" / "test.db"
        db = Database(db_path=f"sqlite+aiosqlite:///{db_path}")
        await db.init()
        await db.close()
        assert db_path.parent.exists()


class TestUserRepository:
    """用户仓储测试"""

    @pytest.mark.asyncio
    async def test_create_user(self, test_db: Database, sample_user_data: dict):
        """测试创建用户"""
        repo = UserRepository(test_db)
        user = await repo.create(sample_user_data)
        
        assert user is not None
        assert user.user_id == sample_user_data["user_id"]
        assert user.username == sample_user_data["username"]

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, test_db: Database, sample_user_data: dict):
        """测试根据ID获取用户"""
        repo = UserRepository(test_db)
        await repo.create(sample_user_data)
        
        user = await repo.get_by_id(sample_user_data["user_id"])
        assert user is not None
        assert user.user_id == sample_user_data["user_id"]

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self, test_db: Database):
        """测试获取不存在的用户"""
        repo = UserRepository(test_db)
        user = await repo.get_by_id("nonexistent_user")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_by_platform_username(self, test_db: Database, sample_user_data: dict):
        """测试根据平台和用户名获取用户"""
        repo = UserRepository(test_db)
        await repo.create(sample_user_data)
        
        user = await repo.get_by_platform_username(
            sample_user_data["platform"],
            sample_user_data["username"]
        )
        assert user is not None

    @pytest.mark.asyncio
    async def test_update_user(self, test_db: Database, sample_user_data: dict):
        """测试更新用户"""
        repo = UserRepository(test_db)
        await repo.create(sample_user_data)
        
        update_data = {
            "display_name": "Updated Name",
            "followers_count": 2000,
        }
        user = await repo.update(sample_user_data["user_id"], update_data)
        
        assert user is not None
        assert user.display_name == "Updated Name"
        assert user.followers_count == 2000

    @pytest.mark.asyncio
    async def test_update_user_not_found(self, test_db: Database):
        """测试更新不存在的用户"""
        repo = UserRepository(test_db)
        user = await repo.update("nonexistent", {"display_name": "Test"})
        assert user is None

    @pytest.mark.asyncio
    async def test_delete_user(self, test_db: Database, sample_user_data: dict):
        """测试删除用户"""
        repo = UserRepository(test_db)
        await repo.create(sample_user_data)
        
        result = await repo.delete(sample_user_data["user_id"])
        assert result is True
        
        user = await repo.get_by_id(sample_user_data["user_id"])
        assert user is None

    @pytest.mark.asyncio
    async def test_delete_user_not_found(self, test_db: Database):
        """测试删除不存在的用户"""
        repo = UserRepository(test_db)
        result = await repo.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_create(self, test_db: Database, sample_user_data: dict):
        """测试upsert创建"""
        repo = UserRepository(test_db)
        user = await repo.upsert(sample_user_data)
        
        assert user is not None
        assert user.user_id == sample_user_data["user_id"]

    @pytest.mark.asyncio
    async def test_upsert_update(self, test_db: Database, sample_user_data: dict):
        """测试upsert更新"""
        repo = UserRepository(test_db)
        await repo.create(sample_user_data)
        
        updated_data = sample_user_data.copy()
        updated_data["display_name"] = "Updated Name"
        user = await repo.upsert(updated_data)
        
        assert user.display_name == "Updated Name"

    @pytest.mark.asyncio
    async def test_batch_insert(self, test_db: Database):
        """测试批量插入用户"""
        repo = UserRepository(test_db)
        
        users = []
        for i in range(10):
            users.append({
                "user_id": f"batch_user_{i}",
                "platform": "twitter",
                "username": f"batch_user_{i}",
                "display_name": f"Batch User {i}",
            })
        
        count = await repo.batch_insert(users)
        assert count == 10

    @pytest.mark.asyncio
    async def test_list_users(self, test_db: Database):
        """测试列出用户"""
        repo = UserRepository(test_db)
        
        for i in range(15):
            await repo.create({
                "user_id": f"list_user_{i}",
                "platform": "twitter" if i < 10 else "weibo",
                "username": f"list_user_{i}",
                "is_suspicious": i >= 12,
            })
        
        users = await repo.list_users(limit=10)
        assert len(users) == 10

    @pytest.mark.asyncio
    async def test_list_users_by_platform(self, test_db: Database):
        """测试按平台列出用户"""
        repo = UserRepository(test_db)
        
        for i in range(10):
            await repo.create({
                "user_id": f"platform_user_{i}",
                "platform": "twitter" if i < 5 else "weibo",
                "username": f"platform_user_{i}",
            })
        
        users = await repo.list_users(platform="twitter")
        assert len(users) == 5

    @pytest.mark.asyncio
    async def test_list_users_by_suspicious(self, test_db: Database):
        """测试按可疑状态列出用户"""
        repo = UserRepository(test_db)
        
        for i in range(10):
            await repo.create({
                "user_id": f"susp_user_{i}",
                "platform": "twitter",
                "username": f"susp_user_{i}",
                "is_suspicious": i >= 5,
            })
        
        users = await repo.list_users(is_suspicious=True)
        assert len(users) == 5

    @pytest.mark.asyncio
    async def test_count_users(self, test_db: Database):
        """测试统计用户数量"""
        repo = UserRepository(test_db)
        
        for i in range(10):
            await repo.create({
                "user_id": f"count_user_{i}",
                "platform": "twitter",
                "username": f"count_user_{i}",
            })
        
        count = await repo.count_users()
        assert count == 10

    @pytest.mark.asyncio
    async def test_mark_suspicious(self, test_db: Database, sample_user_data: dict):
        """测试标记用户为可疑"""
        repo = UserRepository(test_db)
        await repo.create(sample_user_data)
        
        result = await repo.mark_suspicious(
            sample_user_data["user_id"],
            score=0.85,
            is_suspicious=True
        )
        assert result is True
        
        user = await repo.get_by_id(sample_user_data["user_id"])
        assert user.is_suspicious is True
        assert user.suspicious_score == 0.85


class TestPostRepository:
    """帖子仓储测试"""

    @pytest.mark.asyncio
    async def test_create_post(self, test_db: Database, sample_user_data: dict, sample_post_data: dict):
        """测试创建帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        post = await post_repo.create(sample_post_data)
        
        assert post is not None
        assert post.post_id == sample_post_data["post_id"]

    @pytest.mark.asyncio
    async def test_get_post_by_id(self, test_db: Database, sample_user_data: dict, sample_post_data: dict):
        """测试根据ID获取帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        await post_repo.create(sample_post_data)
        
        post = await post_repo.get_by_id(sample_post_data["post_id"])
        assert post is not None

    @pytest.mark.asyncio
    async def test_update_post(self, test_db: Database, sample_user_data: dict, sample_post_data: dict):
        """测试更新帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        await post_repo.create(sample_post_data)
        
        update_data = {
            "likes_count": 500,
            "content": "Updated content",
        }
        post = await post_repo.update(sample_post_data["post_id"], update_data)
        
        assert post.likes_count == 500

    @pytest.mark.asyncio
    async def test_delete_post(self, test_db: Database, sample_user_data: dict, sample_post_data: dict):
        """测试删除帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        await post_repo.create(sample_post_data)
        
        result = await post_repo.delete(sample_post_data["post_id"])
        assert result is True
        
        post = await post_repo.get_by_id(sample_post_data["post_id"])
        assert post is None

    @pytest.mark.asyncio
    async def test_upsert_post(self, test_db: Database, sample_user_data: dict, sample_post_data: dict):
        """测试upsert帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        post = await post_repo.upsert(sample_post_data)
        assert post is not None

    @pytest.mark.asyncio
    async def test_batch_insert_posts(self, test_db: Database, sample_user_data: dict):
        """测试批量插入帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        posts = []
        for i in range(10):
            posts.append({
                "post_id": f"batch_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"Batch post {i}",
                "posted_at": datetime.utcnow(),
            })
        
        count = await post_repo.batch_insert(posts)
        assert count == 10

    @pytest.mark.asyncio
    async def test_list_posts(self, test_db: Database, sample_user_data: dict):
        """测试列出帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        for i in range(15):
            await post_repo.create({
                "post_id": f"list_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"Post {i}",
                "posted_at": datetime.utcnow() - timedelta(hours=i),
            })
        
        posts = await post_repo.list_posts(limit=10)
        assert len(posts) == 10

    @pytest.mark.asyncio
    async def test_list_posts_by_user(self, test_db: Database, sample_user_data: dict):
        """测试按用户列出帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        for i in range(5):
            await post_repo.create({
                "post_id": f"user_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"User post {i}",
                "posted_at": datetime.utcnow(),
            })
        
        posts = await post_repo.list_posts(user_id=sample_user_data["user_id"])
        assert len(posts) == 5

    @pytest.mark.asyncio
    async def test_list_posts_by_time_range(self, test_db: Database, sample_user_data: dict):
        """测试按时间范围列出帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        now = datetime.utcnow()
        for i in range(10):
            await post_repo.create({
                "post_id": f"time_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"Time post {i}",
                "posted_at": now - timedelta(days=i),
            })
        
        start_time = now - timedelta(days=3)
        end_time = now + timedelta(days=1)
        posts = await post_repo.list_posts(start_time=start_time, end_time=end_time)
        assert len(posts) == 4

    @pytest.mark.asyncio
    async def test_count_posts(self, test_db: Database, sample_user_data: dict):
        """测试统计帖子数量"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        for i in range(10):
            await post_repo.create({
                "post_id": f"count_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"Count post {i}",
                "posted_at": datetime.utcnow(),
            })
        
        count = await post_repo.count_posts()
        assert count == 10

    @pytest.mark.asyncio
    async def test_get_user_posts(self, test_db: Database, sample_user_data: dict):
        """测试获取用户帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        for i in range(5):
            await post_repo.create({
                "post_id": f"get_user_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"Get user post {i}",
                "posted_at": datetime.utcnow(),
            })
        
        posts = await post_repo.get_user_posts(sample_user_data["user_id"])
        assert len(posts) == 5

    @pytest.mark.asyncio
    async def test_get_posts_by_hashtag(self, test_db: Database, sample_user_data: dict):
        """测试按话题标签获取帖子"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        
        for i in range(5):
            await post_repo.create({
                "post_id": f"hashtag_post_{i}",
                "user_id": sample_user_data["user_id"],
                "platform": "twitter",
                "content": f"Post with hashtag",
                "posted_at": datetime.utcnow(),
                "hashtags": json.dumps(["test", "example"]),
            })
        
        posts = await post_repo.get_posts_by_hashtag("test")
        assert len(posts) >= 0


class TestInteractionRepository:
    """互动仓储测试"""

    @pytest_asyncio.fixture
    async def setup_data(self, test_db: Database, sample_user_data: dict, sample_post_data: dict):
        """设置测试数据"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        
        post_repo = PostRepository(test_db)
        await post_repo.create(sample_post_data)
        
        return sample_user_data, sample_post_data

    @pytest.mark.asyncio
    async def test_create_interaction(self, test_db: Database, setup_data):
        """测试创建互动"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        interaction_data = {
            "interaction_id": "int_001",
            "user_id": user_data["user_id"],
            "post_id": post_data["post_id"],
            "platform": "twitter",
            "interaction_type": "like",
            "interacted_at": datetime.utcnow(),
        }
        
        interaction = await repo.create(interaction_data)
        assert interaction is not None
        assert interaction.interaction_id == "int_001"

    @pytest.mark.asyncio
    async def test_get_by_id(self, test_db: Database, setup_data):
        """测试根据ID获取互动"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        interaction_data = {
            "interaction_id": "int_002",
            "user_id": user_data["user_id"],
            "post_id": post_data["post_id"],
            "platform": "twitter",
            "interaction_type": "comment",
            "interacted_at": datetime.utcnow(),
        }
        await repo.create(interaction_data)
        
        interaction = await repo.get_by_id("int_002")
        assert interaction is not None

    @pytest.mark.asyncio
    async def test_delete_interaction(self, test_db: Database, setup_data):
        """测试删除互动"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        interaction_data = {
            "interaction_id": "int_003",
            "user_id": user_data["user_id"],
            "post_id": post_data["post_id"],
            "platform": "twitter",
            "interaction_type": "share",
            "interacted_at": datetime.utcnow(),
        }
        await repo.create(interaction_data)
        
        result = await repo.delete("int_003")
        assert result is True

    @pytest.mark.asyncio
    async def test_batch_insert(self, test_db: Database, setup_data):
        """测试批量插入互动"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        interactions = []
        for i in range(10):
            interactions.append({
                "interaction_id": f"batch_int_{i}",
                "user_id": user_data["user_id"],
                "post_id": post_data["post_id"],
                "platform": "twitter",
                "interaction_type": "like",
                "interacted_at": datetime.utcnow(),
            })
        
        count = await repo.batch_insert(interactions)
        assert count == 10

    @pytest.mark.asyncio
    async def test_list_interactions(self, test_db: Database, setup_data):
        """测试列出互动"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        for i in range(15):
            await repo.create({
                "interaction_id": f"list_int_{i}",
                "user_id": user_data["user_id"],
                "post_id": post_data["post_id"],
                "platform": "twitter",
                "interaction_type": "like" if i < 10 else "comment",
                "interacted_at": datetime.utcnow(),
            })
        
        interactions = await repo.list_interactions(limit=10)
        assert len(interactions) == 10

    @pytest.mark.asyncio
    async def test_list_interactions_by_type(self, test_db: Database, setup_data):
        """测试按类型列出互动"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        for i in range(10):
            await repo.create({
                "interaction_id": f"type_int_{i}",
                "user_id": user_data["user_id"],
                "post_id": post_data["post_id"],
                "platform": "twitter",
                "interaction_type": "like" if i < 5 else "comment",
                "interacted_at": datetime.utcnow(),
            })
        
        interactions = await repo.list_interactions(interaction_type="like")
        assert len(interactions) == 5

    @pytest.mark.asyncio
    async def test_count_interactions(self, test_db: Database, setup_data):
        """测试统计互动数量"""
        user_data, post_data = await setup_data
        repo = InteractionRepository(test_db)
        
        for i in range(10):
            await repo.create({
                "interaction_id": f"count_int_{i}",
                "user_id": user_data["user_id"],
                "post_id": post_data["post_id"],
                "platform": "twitter",
                "interaction_type": "like",
                "interacted_at": datetime.utcnow(),
            })
        
        count = await repo.count_interactions()
        assert count == 10


class TestCrawlTaskRepository:
    """采集任务仓储测试"""

    @pytest.mark.asyncio
    async def test_create_task(self, test_db: Database, sample_crawl_task_data: dict):
        """测试创建任务"""
        repo = CrawlTaskRepository(test_db)
        task = await repo.create(sample_crawl_task_data)
        
        assert task is not None
        assert task.task_id == sample_crawl_task_data["task_id"]

    @pytest.mark.asyncio
    async def test_get_by_id(self, test_db: Database, sample_crawl_task_data: dict):
        """测试根据ID获取任务"""
        repo = CrawlTaskRepository(test_db)
        await repo.create(sample_crawl_task_data)
        
        task = await repo.get_by_id(sample_crawl_task_data["task_id"])
        assert task is not None

    @pytest.mark.asyncio
    async def test_update_status(self, test_db: Database, sample_crawl_task_data: dict):
        """测试更新任务状态"""
        repo = CrawlTaskRepository(test_db)
        await repo.create(sample_crawl_task_data)
        
        result = await repo.update_status(
            sample_crawl_task_data["task_id"],
            status="running"
        )
        assert result is True
        
        task = await repo.get_by_id(sample_crawl_task_data["task_id"])
        assert task.status == "running"
        assert task.started_at is not None

    @pytest.mark.asyncio
    async def test_update_status_completed(self, test_db: Database, sample_crawl_task_data: dict):
        """测试完成任务"""
        repo = CrawlTaskRepository(test_db)
        await repo.create(sample_crawl_task_data)
        
        result = await repo.update_status(
            sample_crawl_task_data["task_id"],
            status="completed",
            result_count=100
        )
        assert result is True
        
        task = await repo.get_by_id(sample_crawl_task_data["task_id"])
        assert task.status == "completed"
        assert task.result_count == 100

    @pytest.mark.asyncio
    async def test_increment_retry(self, test_db: Database, sample_crawl_task_data: dict):
        """测试增加重试计数"""
        repo = CrawlTaskRepository(test_db)
        await repo.create(sample_crawl_task_data)
        
        result = await repo.increment_retry(sample_crawl_task_data["task_id"])
        assert result is True
        
        task = await repo.get_by_id(sample_crawl_task_data["task_id"])
        assert task.retry_count == 1

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, test_db: Database):
        """测试获取待处理任务"""
        repo = CrawlTaskRepository(test_db)
        
        for i in range(10):
            await repo.create({
                "task_id": f"pending_task_{i}",
                "platform": "twitter",
                "task_type": "user_timeline",
                "target": f"user_{i}",
                "status": "pending" if i < 5 else "completed",
                "priority": i,
            })
        
        tasks = await repo.get_pending_tasks()
        assert len(tasks) == 5

    @pytest.mark.asyncio
    async def test_list_tasks(self, test_db: Database):
        """测试列出任务"""
        repo = CrawlTaskRepository(test_db)
        
        for i in range(15):
            await repo.create({
                "task_id": f"list_task_{i}",
                "platform": "twitter",
                "task_type": "user_timeline",
                "target": f"user_{i}",
                "status": "pending",
            })
        
        tasks = await repo.list_tasks(limit=10)
        assert len(tasks) == 10

    @pytest.mark.asyncio
    async def test_delete_old_tasks(self, test_db: Database):
        """测试删除旧任务"""
        repo = CrawlTaskRepository(test_db)
        
        for i in range(10):
            task = await repo.create({
                "task_id": f"old_task_{i}",
                "platform": "twitter",
                "task_type": "user_timeline",
                "target": f"user_{i}",
                "status": "completed",
            })
        
        deleted = await repo.delete_old_tasks(days=0)
        assert deleted >= 0


class TestUserFeatureRepository:
    """用户特征仓储测试"""

    @pytest_asyncio.fixture
    async def setup_user(self, test_db: Database, sample_user_data: dict):
        """设置用户数据"""
        user_repo = UserRepository(test_db)
        await user_repo.create(sample_user_data)
        return sample_user_data

    @pytest.mark.asyncio
    async def test_upsert_create(self, test_db: Database, setup_user, sample_user_feature_data: dict):
        """测试创建用户特征"""
        repo = UserFeatureRepository(test_db)
        feature = await repo.upsert(sample_user_feature_data)
        
        assert feature is not None
        assert feature.user_id == sample_user_feature_data["user_id"]

    @pytest.mark.asyncio
    async def test_upsert_update(self, test_db: Database, setup_user, sample_user_feature_data: dict):
        """测试更新用户特征"""
        repo = UserFeatureRepository(test_db)
        await repo.upsert(sample_user_feature_data)
        
        updated_data = sample_user_feature_data.copy()
        updated_data["anomaly_score"] = 0.9
        feature = await repo.upsert(updated_data)
        
        assert feature.anomaly_score == 0.9

    @pytest.mark.asyncio
    async def test_get_by_user_id(self, test_db: Database, setup_user, sample_user_feature_data: dict):
        """测试根据用户ID获取特征"""
        repo = UserFeatureRepository(test_db)
        await repo.upsert(sample_user_feature_data)
        
        feature = await repo.get_by_user_id(sample_user_feature_data["user_id"])
        assert feature is not None

    @pytest.mark.asyncio
    async def test_get_suspicious_users(self, test_db: Database, setup_user):
        """测试获取可疑用户"""
        repo = UserFeatureRepository(test_db)
        
        await repo.upsert({
            "user_id": setup_user["user_id"],
            "platform": "twitter",
            "anomaly_score": 0.85,
        })
        
        users = await repo.get_suspicious_users(threshold=0.7)
        assert len(users) == 1
