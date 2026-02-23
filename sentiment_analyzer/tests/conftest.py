"""
pytest配置和共享fixtures

提供测试所需的数据库连接、测试数据生成器和清理函数。
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentiment_analyzer.storage.schema import Base, User, Post, Interaction, CrawlTask, UserFeature
from sentiment_analyzer.storage.repository import Database


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db() -> AsyncGenerator[Database, None]:
    """创建内存SQLite数据库用于测试"""
    db = Database(db_path="sqlite+aiosqlite:///:memory:")
    await db.init()
    yield db
    await db.close()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_db: Database):
    """获取数据库会话"""
    async with test_db.session() as session:
        yield session


@pytest.fixture
def sample_user_data() -> dict:
    """生成示例用户数据"""
    return {
        "user_id": f"user_{uuid4().hex[:8]}",
        "platform": "twitter",
        "username": f"test_user_{uuid4().hex[:6]}",
        "display_name": "Test User",
        "bio": "This is a test user bio",
        "avatar_url": "https://example.com/avatar.png",
        "followers_count": 1000,
        "friends_count": 500,
        "posts_count": 100,
        "verified": False,
        "is_suspicious": False,
    }


@pytest.fixture
def sample_post_data(sample_user_data: dict) -> dict:
    """生成示例帖子数据"""
    return {
        "post_id": f"post_{uuid4().hex[:8]}",
        "user_id": sample_user_data["user_id"],
        "platform": "twitter",
        "content": "This is a test post content #test #example",
        "language": "en",
        "posted_at": datetime.utcnow(),
        "likes_count": 100,
        "shares_count": 50,
        "comments_count": 25,
        "hashtags": ["test", "example"],
        "mentions": ["user1", "user2"],
        "urls": ["https://example.com"],
    }


@pytest.fixture
def sample_interaction_data(sample_user_data: dict, sample_post_data: dict) -> dict:
    """生成示例互动数据"""
    return {
        "interaction_id": f"int_{uuid4().hex[:8]}",
        "user_id": sample_user_data["user_id"],
        "post_id": sample_post_data["post_id"],
        "platform": "twitter",
        "interaction_type": "like",
        "content": None,
        "interacted_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_crawl_task_data() -> dict:
    """生成示例采集任务数据"""
    return {
        "task_id": f"task_{uuid4().hex[:8]}",
        "platform": "twitter",
        "task_type": "user_timeline",
        "target": "test_user",
        "status": "pending",
        "priority": 5,
        "config": {"count": 100},
    }


@pytest.fixture
def sample_user_feature_data(sample_user_data: dict) -> dict:
    """生成示例用户特征数据"""
    return {
        "user_id": sample_user_data["user_id"],
        "platform": "twitter",
        "daily_post_avg": 5.5,
        "daily_post_std": 2.3,
        "hour_entropy": 3.2,
        "night_activity_ratio": 0.15,
        "weekend_activity_ratio": 0.25,
        "content_similarity_avg": 0.3,
        "topic_entropy": 2.5,
        "anomaly_score": 0.25,
    }


@pytest.fixture
def sample_posts_batch(sample_user_data: dict) -> list[dict]:
    """生成批量帖子数据用于特征提取测试"""
    posts = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(50):
        post = {
            "post_id": f"post_{uuid4().hex[:8]}_{i}",
            "user_id": sample_user_data["user_id"],
            "platform": "twitter",
            "content": f"Test post content number {i}. This is a sample post for testing.",
            "posted_at": base_time + timedelta(hours=i*12),
            "likes_count": np.random.randint(10, 500),
            "shares_count": np.random.randint(5, 100),
            "comments_count": np.random.randint(0, 50),
        }
        posts.append(post)
    
    return posts


@pytest.fixture
def sample_temporal_posts() -> list[dict]:
    """生成用于时序特征测试的帖子数据"""
    posts = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    for day in range(30):
        for hour in [9, 10, 11, 14, 15, 16, 20, 21]:
            post = {
                "post_id": f"temp_post_{day}_{hour}",
                "user_id": "temp_user_001",
                "platform": "twitter",
                "content": f"Temporal test post day {day} hour {hour}",
                "posted_at": base_time + timedelta(days=day, hours=hour),
            }
            posts.append(post)
    
    return posts


@pytest.fixture
def sample_anomaly_data() -> np.ndarray:
    """生成用于异常检测测试的数据"""
    np.random.seed(42)
    normal_data = np.random.randn(100, 5)
    anomaly_data = np.random.randn(10, 5) * 3 + 5
    return np.vstack([normal_data, anomaly_data])


@pytest.fixture
def sample_text_data() -> list[str]:
    """生成用于文本处理测试的数据"""
    return [
        "This is a normal English text for testing purposes.",
        "这是一段中文测试文本，用于测试中文处理功能。",
        "Mixed content: English and 中文 mixed together!",
        "<html><body>HTML content should be cleaned</body></html>",
        "Visit https://example.com for more info #test @user",
        "Multiple   spaces   should   be   normalized.",
        "UPPERCASE AND lowercase MiXeD cOnTeNt",
        "Text with emojis 😀 🎉 🚀 and special chars!@#$%",
        "",
        "   ",
    ]


@pytest.fixture
def mock_rate_limiter():
    """创建模拟的限流器"""
    mock = MagicMock()
    mock.acquire = MagicMock(return_value=True)
    mock.try_acquire = MagicMock(return_value=True)
    mock.wait_time = MagicMock(return_value=0.0)
    return mock


@pytest.fixture
def mock_proxy_pool():
    """创建模拟的代理池"""
    mock = MagicMock()
    mock.get_proxy = MagicMock(return_value=MagicMock(url="http://proxy:8080"))
    mock.report_success = MagicMock()
    mock.report_failure = MagicMock()
    return mock


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """创建临时数据目录"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """创建临时日志目录"""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def generate_random_user_data(count: int = 10) -> list[dict]:
    """生成随机用户数据"""
    users = []
    for i in range(count):
        user = {
            "user_id": f"user_{uuid4().hex[:8]}",
            "platform": np.random.choice(["twitter", "weibo", "reddit"]),
            "username": f"user_{i}_{uuid4().hex[:6]}",
            "display_name": f"User {i}",
            "bio": f"Bio for user {i}",
            "followers_count": np.random.randint(0, 10000),
            "friends_count": np.random.randint(0, 5000),
            "posts_count": np.random.randint(0, 1000),
            "verified": np.random.random() > 0.9,
        }
        users.append(user)
    return users


def generate_random_post_data(user_ids: list[str], count: int = 50) -> list[dict]:
    """生成随机帖子数据"""
    posts = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(count):
        post = {
            "post_id": f"post_{uuid4().hex[:8]}",
            "user_id": np.random.choice(user_ids),
            "platform": np.random.choice(["twitter", "weibo", "reddit"]),
            "content": f"Random post content {i} with some text.",
            "posted_at": base_time + timedelta(hours=np.random.randint(0, 720)),
            "likes_count": np.random.randint(0, 1000),
            "shares_count": np.random.randint(0, 500),
            "comments_count": np.random.randint(0, 100),
        }
        posts.append(post)
    
    return posts


def assert_valid_user(user: User, expected_data: dict) -> None:
    """验证用户对象"""
    assert user.user_id == expected_data["user_id"]
    assert user.platform == expected_data["platform"]
    assert user.username == expected_data["username"]


def assert_valid_post(post: Post, expected_data: dict) -> None:
    """验证帖子对象"""
    assert post.post_id == expected_data["post_id"]
    assert post.user_id == expected_data["user_id"]
    assert post.platform == expected_data["platform"]


@pytest_asyncio.fixture
async def populated_db(test_db: Database, sample_user_data: dict, sample_post_data: dict) -> Database:
    """创建已填充测试数据的数据库"""
    from storage.repository import UserRepository, PostRepository
    
    user_repo = UserRepository(test_db)
    post_repo = PostRepository(test_db)
    
    await user_repo.create(sample_user_data)
    await post_repo.create(sample_post_data)
    
    return test_db


def cleanup_test_files(paths: list[Path]) -> None:
    """清理测试文件"""
    for path in paths:
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                for file in path.rglob("*"):
                    if file.is_file():
                        file.unlink()
                path.rmdir()


pytest_plugins = ["pytest_asyncio"]
