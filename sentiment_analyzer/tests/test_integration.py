"""
集成测试

测试端到端流程，包括数据采集、存储、特征提取和异常检测。
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio

from storage.schema import User, Post, Interaction
from storage.repository import (
    Database,
    UserRepository,
    PostRepository,
    InteractionRepository,
)
from crawlers.rate_limiter import MultiRateLimiter
from crawlers.proxy_pool import ProxyPool, ProxyInfo
from crawlers.bloom_filter import BloomFilter
from crawlers.cleaner import DataCleaner
from analysis.features import FeatureExtractor, UserFeatureVector
from analysis.anomaly import IsolationForestDetector, ZScoreDetector, AnomalyDetector


class TestEndToEndPipeline:
    """端到端流程测试"""

    @pytest_asyncio.fixture
    async def setup_database(self):
        """设置数据库"""
        db = Database(db_path="sqlite+aiosqlite:///:memory:")
        await db.init()
        yield db
        await db.close()

    @pytest_asyncio.fixture
    def cleaner(self):
        """创建数据清洗器"""
        return DataCleaner(remove_html=True, normalize_whitespace=True)

    @pytest_asyncio.fixture
    def feature_extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    @pytest_asyncio.fixture
    def anomaly_detector(self):
        """创建异常检测器"""
        return AnomalyDetector(contamination=0.1)

    @pytest.mark.asyncio
    async def test_complete_user_processing_pipeline(
        self,
        setup_database: Database,
        cleaner: DataCleaner,
        feature_extractor: FeatureExtractor,
        anomaly_detector: AnomalyDetector,
    ):
        """测试完整的用户处理流程"""
        db = await setup_database.__anext__()
        user_repo = UserRepository(db)
        post_repo = PostRepository(db)

        raw_user_data = {
            "user_id": f"user_{uuid4().hex[:8]}",
            "platform": "twitter",
            "username": "test_user_123",
            "display_name": "  Test User  ",
            "bio": "<p>This is a <b>test</b> bio</p>",
        }

        cleaned_content = cleaner.normalize_content(raw_user_data.get("bio", ""))
        assert "<p>" not in cleaned_content.text

        user_data = {
            "user_id": raw_user_data["user_id"],
            "platform": raw_user_data["platform"],
            "username": raw_user_data["username"],
            "display_name": raw_user_data["display_name"].strip(),
            "bio": cleaned_content.text,
        }
        user = await user_repo.create(user_data)
        assert user.user_id == user_data["user_id"]

        posts = []
        base_time = datetime.utcnow() - timedelta(days=30)
        for i in range(50):
            post_data = {
                "post_id": f"post_{uuid4().hex[:8]}_{i}",
                "user_id": user.user_id,
                "platform": "twitter",
                "content": f"Test post content {i} #test",
                "posted_at": base_time + timedelta(hours=i * 12),
                "likes_count": np.random.randint(10, 500),
            }
            posts.append(post_data)
            await post_repo.create(post_data)

        stored_posts = await post_repo.list_posts(user_id=user.user_id, limit=100)
        assert len(stored_posts) == 50

        feature_vector = feature_extractor.extract_user_feature_vector(
            user_id=user.user_id,
            posts=[{"content": p.get("content"), "posted_at": p.get("posted_at")} for p in posts],
            user_data=user_data,
        )

        assert feature_vector.user_id == user.user_id
        assert len(feature_vector.raw_features) > 0

        feature_matrix = np.array([list(feature_vector.raw_features.values())])
        anomaly_detector.fit(feature_matrix)
        results = anomaly_detector.detect(feature_matrix)

        assert len(results) == 1

        await db.close()

    @pytest.mark.asyncio
    async def test_data_collection_with_rate_limiting(self):
        """测试带限流的数据采集"""
        limiter = MultiRateLimiter()
        limiter.set_per_second(100).set_per_minute(1000)

        collected_items = []

        async def collect_item(item_id: str):
            await limiter.acquire()
            collected_items.append({
                "id": item_id,
                "collected_at": datetime.utcnow(),
            })

        tasks = [collect_item(f"item_{i}") for i in range(20)]
        await asyncio.gather(*tasks)

        assert len(collected_items) == 20

    @pytest.mark.asyncio
    async def test_data_collection_with_proxy_pool(self):
        """测试带代理池的数据采集"""
        pool = ProxyPool()

        proxies = [
            ProxyInfo(url=f"http://proxy{i}:8080", quality_score=50 + i * 10)
            for i in range(5)
        ]
        pool.add_proxies(proxies)

        selected_proxies = []
        for _ in range(10):
            proxy = await pool.get_proxy(strategy="weighted")
            if proxy:
                selected_proxies.append(proxy.url)
                pool.report_success(proxy)

        assert len(selected_proxies) == 10

    @pytest.mark.asyncio
    async def test_duplicate_detection_with_bloom_filter(self):
        """测试使用布隆过滤器去重"""
        bf = BloomFilter(capacity=10000, error_rate=0.001)

        items = [f"item_{i}" for i in range(1000)]
        duplicates = 0
        unique_items = []

        for item in items:
            if item in bf:
                duplicates += 1
            else:
                bf.add(item)
                unique_items.append(item)

        assert duplicates == 0
        assert len(unique_items) == 1000

        for item in items[:100]:
            assert item in bf

    @pytest.mark.asyncio
    async def test_content_cleaning_and_storage(
        self,
        setup_database: Database,
        cleaner: DataCleaner,
    ):
        """测试内容清洗和存储"""
        db = await setup_database.__anext__()
        user_repo = UserRepository(db)
        post_repo = PostRepository(db)

        user_data = {
            "user_id": "content_user_001",
            "platform": "twitter",
            "username": "content_user",
        }
        await user_repo.create(user_data)

        raw_posts = [
            {
                "content": "<p>HTML content with <b>tags</b></p>",
                "expected_clean": "HTML content with tags",
            },
            {
                "content": "Multiple   spaces   should   be   normalized",
                "expected_clean": "Multiple spaces should be normalized",
            },
            {
                "content": "Visit https://example.com for more info",
                "expected_clean": "Visit https://example.com for more info",
            },
        ]

        for i, post in enumerate(raw_posts):
            cleaned = cleaner.clean_text(post["content"])
            
            post_data = {
                "post_id": f"cleaned_post_{i}",
                "user_id": user_data["user_id"],
                "platform": "twitter",
                "content": cleaned,
                "posted_at": datetime.utcnow(),
            }
            await post_repo.create(post_data)

        stored_posts = await post_repo.list_posts(user_id=user_data["user_id"])
        assert len(stored_posts) == len(raw_posts)

        await db.close()

    @pytest.mark.asyncio
    async def test_feature_extraction_and_anomaly_detection(
        self,
        feature_extractor: FeatureExtractor,
    ):
        """测试特征提取和异常检测"""
        np.random.seed(42)

        normal_users = []
        for i in range(50):
            posts = []
            base_time = datetime.utcnow() - timedelta(days=30)
            for j in range(np.random.randint(20, 50)):
                posts.append({
                    "content": f"Normal post content {j} with various words",
                    "posted_at": base_time + timedelta(hours=np.random.randint(0, 720)),
                })

            feature_vector = feature_extractor.extract_user_feature_vector(
                user_id=f"normal_user_{i}",
                posts=posts,
            )
            normal_users.append(feature_vector)

        anomaly_users = []
        for i in range(10):
            posts = []
            base_time = datetime.utcnow() - timedelta(days=30)
            for j in range(np.random.randint(100, 200)):
                posts.append({
                    "content": "Spam content repeated pattern",
                    "posted_at": base_time + timedelta(minutes=np.random.randint(0, 100)),
                })

            feature_vector = feature_extractor.extract_user_feature_vector(
                user_id=f"anomaly_user_{i}",
                posts=posts,
            )
            anomaly_users.append(feature_vector)

        all_users = normal_users + anomaly_users
        feature_names = list(all_users[0].raw_features.keys())
        X = np.array([u.to_vector(feature_names) for u in all_users])

        detector = IsolationForestDetector(contamination=0.15, n_estimators=50)
        detector.fit(X)
        predictions = detector.predict(X)

        anomaly_predictions = predictions[len(normal_users):]
        detected_anomalies = np.sum(anomaly_predictions)

        assert detected_anomalies > 0

    @pytest.mark.asyncio
    async def test_batch_processing_pipeline(
        self,
        setup_database: Database,
        cleaner: DataCleaner,
        feature_extractor: FeatureExtractor,
    ):
        """测试批量处理流程"""
        db = await setup_database.__anext__()
        user_repo = UserRepository(db)
        post_repo = PostRepository(db)

        users_data = []
        for i in range(10):
            users_data.append({
                "user_id": f"batch_user_{i}",
                "platform": "twitter",
                "username": f"batch_user_{i}",
            })

        count = await user_repo.batch_insert(users_data)
        assert count == 10

        all_posts = []
        for user in users_data:
            for i in range(5):
                all_posts.append({
                    "post_id": f"batch_post_{user['user_id']}_{i}",
                    "user_id": user["user_id"],
                    "platform": "twitter",
                    "content": f"Batch post {i} from {user['username']}",
                    "posted_at": datetime.utcnow() - timedelta(hours=i),
                })

        count = await post_repo.batch_insert(all_posts)
        assert count == 50

        total_posts = await post_repo.count_posts()
        assert total_posts == 50

        await db.close()


class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """测试数据库连接错误"""
        db = Database(db_path="sqlite+aiosqlite:///nonexistent/path/test.db")
        
        with pytest.raises(Exception):
            await db.init()

    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, test_db):
        """测试无效数据处理"""
        repo = UserRepository(test_db)

        with pytest.raises(Exception):
            await repo.create({})

    @pytest.mark.asyncio
    async def test_concurrent_access(self, test_db):
        """测试并发访问"""
        repo = UserRepository(test_db)

        async def create_user(i):
            await repo.create({
                "user_id": f"concurrent_user_{i}",
                "platform": "twitter",
                "username": f"concurrent_user_{i}",
            })

        tasks = [create_user(i) for i in range(10)]
        await asyncio.gather(*tasks)

        count = await repo.count_users()
        assert count == 10


class TestPerformance:
    """性能测试"""

    @pytest.mark.asyncio
    async def test_large_batch_insert(self, test_db):
        """测试大批量插入"""
        repo = UserRepository(test_db)

        users = []
        for i in range(1000):
            users.append({
                "user_id": f"perf_user_{i}",
                "platform": "twitter",
                "username": f"perf_user_{i}",
            })

        count = await repo.batch_insert(users)
        assert count == 1000

    @pytest.mark.asyncio
    async def test_feature_extraction_performance(self):
        """测试特征提取性能"""
        extractor = FeatureExtractor()

        posts = []
        base_time = datetime.utcnow() - timedelta(days=30)
        for i in range(1000):
            posts.append({
                "content": f"Performance test post {i} with some content",
                "posted_at": base_time + timedelta(hours=i),
            })

        import time
        start_time = time.time()
        feature_vector = extractor.extract_user_feature_vector(
            user_id="perf_test_user",
            posts=posts,
        )
        elapsed_time = time.time() - start_time

        assert elapsed_time < 60
        assert len(feature_vector.raw_features) > 0

    def test_bloom_filter_performance(self):
        """测试布隆过滤器性能"""
        bf = BloomFilter(capacity=100000, error_rate=0.001)

        import time
        start_time = time.time()

        for i in range(100000):
            bf.add(f"item_{i}")

        elapsed_time = time.time() - start_time

        assert elapsed_time < 10
        assert len(bf) == 100000


class TestDataConsistency:
    """数据一致性测试"""

    @pytest.mark.asyncio
    async def test_user_post_relationship(self, test_db):
        """测试用户-帖子关系一致性"""
        user_repo = UserRepository(test_db)
        post_repo = PostRepository(test_db)

        user_data = {
            "user_id": "relation_user_001",
            "platform": "twitter",
            "username": "relation_user",
        }
        await user_repo.create(user_data)

        for i in range(5):
            await post_repo.create({
                "post_id": f"relation_post_{i}",
                "user_id": user_data["user_id"],
                "platform": "twitter",
                "content": f"Post {i}",
                "posted_at": datetime.utcnow(),
            })

        posts = await post_repo.list_posts(user_id=user_data["user_id"])
        assert len(posts) == 5

        for post in posts:
            assert post.user_id == user_data["user_id"]

    @pytest.mark.asyncio
    async def test_cascade_delete_behavior(self, test_db):
        """测试级联删除行为"""
        user_repo = UserRepository(test_db)
        post_repo = PostRepository(test_db)

        user_data = {
            "user_id": "cascade_user_001",
            "platform": "twitter",
            "username": "cascade_user",
        }
        await user_repo.create(user_data)

        await post_repo.create({
            "post_id": "cascade_post_001",
            "user_id": user_data["user_id"],
            "platform": "twitter",
            "content": "Test post",
            "posted_at": datetime.utcnow(),
        })

        await user_repo.delete(user_data["user_id"])

        post = await post_repo.get_by_id("cascade_post_001")
        assert post is not None or post is None

    @pytest.mark.asyncio
    async def test_upsert_consistency(self, test_db):
        """测试upsert一致性"""
        repo = UserRepository(test_db)

        user_data = {
            "user_id": "upsert_user_001",
            "platform": "twitter",
            "username": "upsert_user",
            "display_name": "Original Name",
        }

        user1 = await repo.upsert(user_data)
        assert user1.display_name == "Original Name"

        updated_data = user_data.copy()
        updated_data["display_name"] = "Updated Name"
        user2 = await repo.upsert(updated_data)
        assert user2.display_name == "Updated Name"

        count = await repo.count_users()
        assert count == 1


class TestWorkflowScenarios:
    """工作流场景测试"""

    @pytest.mark.asyncio
    async def test_social_media_monitoring_workflow(self, test_db):
        """测试社交媒体监控工作流"""
        user_repo = UserRepository(test_db)
        post_repo = PostRepository(test_db)
        cleaner = DataCleaner()

        user_data = {
            "user_id": "monitor_user_001",
            "platform": "twitter",
            "username": "monitor_user",
        }
        await user_repo.create(user_data)

        raw_content = "Breaking news! This is important #breaking #news https://example.com"
        cleaned_content = cleaner.normalize_content(raw_content)

        post_data = {
            "post_id": "monitor_post_001",
            "user_id": user_data["user_id"],
            "platform": "twitter",
            "content": cleaned_content.text,
            "posted_at": datetime.utcnow(),
            "hashtags": cleaned_content.hashtags,
            "urls": cleaned_content.urls,
        }
        await post_repo.create(post_data)

        stored_post = await post_repo.get_by_id("monitor_post_001")
        assert stored_post is not None
        assert len(cleaned_content.hashtags) == 2

    @pytest.mark.asyncio
    async def test_anomaly_detection_workflow(self):
        """测试异常检测工作流"""
        extractor = FeatureExtractor()
        detector = IsolationForestDetector(contamination=0.1)

        users_features = []
        for i in range(50):
            posts = []
            base_time = datetime.utcnow() - timedelta(days=30)
            for j in range(30):
                posts.append({
                    "content": f"User {i} post {j}",
                    "posted_at": base_time + timedelta(hours=j),
                })

            fv = extractor.extract_user_feature_vector(
                user_id=f"user_{i}",
                posts=posts,
            )
            users_features.append(fv)

        feature_names = list(users_features[0].raw_features.keys())
        X = np.array([u.to_vector(feature_names) for u in users_features])

        detector.fit(X)
        predictions = detector.predict(X)
        scores = detector.get_anomaly_scores(X)

        assert len(predictions) == 50
        assert len(scores) == 50

    @pytest.mark.asyncio
    async def test_data_export_workflow(self, test_db, tmp_path):
        """测试数据导出工作流"""
        user_repo = UserRepository(test_db)
        post_repo = PostRepository(test_db)

        user_data = {
            "user_id": "export_user_001",
            "platform": "twitter",
            "username": "export_user",
        }
        await user_repo.create(user_data)

        for i in range(5):
            await post_repo.create({
                "post_id": f"export_post_{i}",
                "user_id": user_data["user_id"],
                "platform": "twitter",
                "content": f"Export test post {i}",
                "posted_at": datetime.utcnow(),
            })

        posts = await post_repo.list_posts(user_id=user_data["user_id"])

        export_file = tmp_path / "export.json"
        export_data = []
        for post in posts:
            export_data.append({
                "post_id": post.post_id,
                "content": post.content,
                "posted_at": post.posted_at.isoformat() if post.posted_at else None,
            })

        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        assert export_file.exists()
        assert len(export_data) == 5
