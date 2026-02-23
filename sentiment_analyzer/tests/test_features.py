"""
特征提取模块测试

测试时序特征、内容特征、特征向量等功能。
"""

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ..analysis.features import (
    FeatureExtractor,
    TextFeatures,
    SentimentResult,
    TemporalFeatures,
    ContentFeatures,
    NetworkFeatures,
    MetadataFeatures,
    UserFeatureVector,
)


class TestTextFeatures:
    """文本特征测试"""

    def test_text_features_creation(self):
        """测试创建文本特征"""
        features = TextFeatures(
            word_count=10,
            char_count=50,
            avg_word_length=5.0,
            hashtag_count=2,
            mention_count=1,
            url_count=1,
            emoji_count=3,
            exclamation_count=1,
            question_count=0,
            uppercase_ratio=0.1,
            language="en",
            keywords=["test", "example"],
        )
        
        assert features.word_count == 10
        assert features.char_count == 50
        assert features.language == "en"


class TestSentimentResult:
    """情感分析结果测试"""

    def test_sentiment_result_creation(self):
        """测试创建情感分析结果"""
        result = SentimentResult(
            score=0.5,
            label="positive",
            confidence=0.85,
            positive_words=["good", "great"],
            negative_words=[],
        )
        
        assert result.score == 0.5
        assert result.label == "positive"
        assert result.confidence == 0.85


class TestTemporalFeatures:
    """时序特征测试"""

    def test_temporal_features_default_values(self):
        """测试默认值"""
        features = TemporalFeatures()
        assert features.daily_post_mean == 0.0
        assert features.daily_post_std == 0.0
        assert features.burst_score == 0.0
        assert features.hour_entropy == 0.0

    def test_temporal_features_custom_values(self):
        """测试自定义值"""
        features = TemporalFeatures(
            daily_post_mean=5.5,
            daily_post_std=2.3,
            burst_score=1.5,
            hour_entropy=3.2,
            work_hours_ratio=0.6,
            night_activity_ratio=0.1,
            weekend_activity_ratio=0.2,
            autocorrelation=[0.5, 0.3, 0.1],
            avg_response_delay=120.0,
            response_delay_std=60.0,
        )
        
        assert features.daily_post_mean == 5.5
        assert features.autocorrelation == [0.5, 0.3, 0.1]


class TestContentFeatures:
    """内容特征测试"""

    def test_content_features_default_values(self):
        """测试默认值"""
        features = ContentFeatures()
        assert features.text_similarity_mean == 0.0
        assert features.topic_entropy == 0.0
        assert features.template_match_ratio == 0.0

    def test_content_features_custom_values(self):
        """测试自定义值"""
        features = ContentFeatures(
            text_similarity_mean=0.3,
            text_similarity_std=0.1,
            text_similarity_max=0.8,
            topic_entropy=2.5,
            topic_consistency=0.7,
            sentiment_polarity_mean=0.2,
            sentiment_polarity_std=0.15,
            sentiment_consistency=0.8,
            template_match_ratio=0.1,
            unique_template_count=5,
        )
        
        assert features.text_similarity_mean == 0.3
        assert features.unique_template_count == 5


class TestNetworkFeatures:
    """网络特征测试"""

    def test_network_features_default_values(self):
        """测试默认值"""
        features = NetworkFeatures()
        assert features.degree_centrality == 0.0
        assert features.betweenness_centrality == 0.0
        assert features.community_id == -1

    def test_network_features_custom_values(self):
        """测试自定义值"""
        features = NetworkFeatures(
            degree_centrality=0.5,
            betweenness_centrality=0.3,
            eigenvector_centrality=0.2,
            pagerank=0.01,
            clustering_coefficient=0.6,
            community_id=2,
            modularity_contribution=0.05,
        )
        
        assert features.degree_centrality == 0.5
        assert features.community_id == 2


class TestMetadataFeatures:
    """元数据特征测试"""

    def test_metadata_features_default_values(self):
        """测试默认值"""
        features = MetadataFeatures()
        assert features.registration_cluster_score == 0.0
        assert features.profile_completeness == 0.0
        assert features.avatar_hash == ""

    def test_metadata_features_custom_values(self):
        """测试自定义值"""
        features = MetadataFeatures(
            registration_cluster_score=0.8,
            profile_completeness=0.6,
            username_pattern_score=0.4,
            username_digit_ratio=0.3,
            avatar_similarity_count=5,
            avatar_hash="abc123",
        )
        
        assert features.registration_cluster_score == 0.8
        assert features.avatar_hash == "abc123"


class TestUserFeatureVector:
    """用户特征向量测试"""

    def test_user_feature_vector_creation(self):
        """测试创建用户特征向量"""
        vector = UserFeatureVector(user_id="test_user")
        assert vector.user_id == "test_user"
        assert isinstance(vector.temporal_features, TemporalFeatures)
        assert isinstance(vector.content_features, ContentFeatures)

    def test_to_vector_all_features(self):
        """测试转换为向量（所有特征）"""
        vector = UserFeatureVector(
            user_id="test_user",
            raw_features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
        )
        arr = vector.to_vector()
        
        assert len(arr) == 3
        assert list(arr) == [1.0, 2.0, 3.0]

    def test_to_vector_selected_features(self):
        """测试转换为向量（选择特征）"""
        vector = UserFeatureVector(
            user_id="test_user",
            raw_features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
        )
        arr = vector.to_vector(feature_names=["f2", "f3"])
        
        assert len(arr) == 2
        assert list(arr) == [2.0, 3.0]

    def test_to_vector_missing_features(self):
        """测试转换为向量（缺失特征）"""
        vector = UserFeatureVector(
            user_id="test_user",
            raw_features={"f1": 1.0},
        )
        arr = vector.to_vector(feature_names=["f1", "f2", "f3"])
        
        assert len(arr) == 3
        assert list(arr) == [1.0, 0.0, 0.0]

    def test_normalize_no_scaler(self):
        """测试标准化（无scaler）"""
        vector = UserFeatureVector(
            user_id="test_user",
            raw_features={"f1": 10.0, "f2": 20.0, "f3": 30.0},
        )
        normalized = vector.normalize()
        
        assert len(normalized) == 3
        assert all(isinstance(v, float) for v in normalized.values())

    def test_normalize_empty_features(self):
        """测试标准化（空特征）"""
        vector = UserFeatureVector(user_id="test_user")
        normalized = vector.normalize()
        
        assert normalized == {}

    def test_select_features(self):
        """测试特征选择"""
        vector = UserFeatureVector(
            user_id="test_user",
            raw_features={"f1": 1.0, "f2": 2.0, "f3": 3.0},
        )
        selected = vector.select_features(["f1", "f3"])
        
        assert selected == {"f1": 1.0, "f3": 3.0}

    def test_get_all_features(self):
        """测试获取所有特征"""
        vector = UserFeatureVector(
            user_id="test_user",
            temporal_features=TemporalFeatures(daily_post_mean=5.0),
            content_features=ContentFeatures(text_similarity_mean=0.5),
        )
        features = vector.get_all_features()
        
        assert "temporal_features.daily_post_mean" in features
        assert "content_features.text_similarity_mean" in features
        assert features["temporal_features.daily_post_mean"] == 5.0


class TestFeatureExtractor:
    """特征提取器测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_features_english(self, extractor: FeatureExtractor):
        """测试提取英文文本特征"""
        text = "Hello World! This is a test. #example @user https://example.com"
        features = extractor.extract_features(text)
        
        assert features.word_count > 0
        assert features.char_count == len(text)
        assert features.hashtag_count == 1
        assert features.mention_count == 1
        assert features.url_count == 1

    def test_extract_features_chinese(self, extractor: FeatureExtractor):
        """测试提取中文文本特征"""
        text = "这是一段中文测试文本。#测试 @用户"
        features = extractor.extract_features(text)
        
        assert features.word_count > 0
        assert features.language in ["zh", "zh-cn", "unknown"]

    def test_extract_features_empty(self, extractor: FeatureExtractor):
        """测试提取空文本特征"""
        features = extractor.extract_features("")
        assert features.word_count == 0
        assert features.char_count == 0

    def test_extract_features_with_emoji(self, extractor: FeatureExtractor):
        """测试提取包含表情符号的文本特征"""
        text = "Hello 😀 World 🎉 Test 🚀"
        features = extractor.extract_features(text)
        
        assert features.emoji_count >= 3

    def test_analyze_sentiment_positive(self, extractor: FeatureExtractor):
        """测试分析积极情感"""
        text = "This is great! I love it! Amazing and wonderful!"
        result = extractor.analyze_sentiment(text)
        
        assert result.label in ["positive", "neutral"]
        assert isinstance(result.confidence, float)

    def test_analyze_sentiment_negative(self, extractor: FeatureExtractor):
        """测试分析消极情感"""
        text = "This is terrible! I hate it! Bad and awful!"
        result = extractor.analyze_sentiment(text)
        
        assert result.label in ["negative", "neutral"]
        assert isinstance(result.confidence, float)

    def test_analyze_sentiment_neutral(self, extractor: FeatureExtractor):
        """测试分析中性情感"""
        text = "The weather is normal today."
        result = extractor.analyze_sentiment(text)
        
        assert result.label in ["positive", "negative", "neutral"]

    def test_extract_batch_features(self, extractor: FeatureExtractor):
        """测试批量提取特征"""
        texts = ["Hello World", "Test content", "Another text"]
        features = extractor.extract_batch_features(texts)
        
        assert len(features) == 3
        assert all(isinstance(f, TextFeatures) for f in features)

    def test_analyze_batch_sentiment(self, extractor: FeatureExtractor):
        """测试批量分析情感"""
        texts = ["Great!", "Terrible!", "Normal."]
        results = extractor.analyze_batch_sentiment(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, SentimentResult) for r in results)


class TestPostFrequencyExtraction:
    """发布频率特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_post_frequency_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_post_frequency([])
        assert result["daily_post_mean"] == 0.0
        assert result["daily_post_std"] == 0.0
        assert result["burst_score"] == 0.0

    def test_extract_post_frequency_single_post(self, extractor: FeatureExtractor):
        """测试单个帖子"""
        posts = [{"posted_at": datetime.utcnow()}]
        result = extractor.extract_post_frequency(posts)
        
        assert result["daily_post_mean"] == 1
        assert result["daily_post_std"] == 0.0

    def test_extract_post_frequency_multiple_posts(self, extractor: FeatureExtractor):
        """测试多个帖子"""
        base_time = datetime.utcnow() - timedelta(days=10)
        posts = []
        for i in range(30):
            posts.append({
                "posted_at": base_time + timedelta(days=i // 3),
            })
        
        result = extractor.extract_post_frequency(posts)
        
        assert result["daily_post_mean"] > 0
        assert result["daily_post_std"] >= 0

    def test_extract_post_frequency_string_timestamp(self, extractor: FeatureExtractor):
        """测试字符串时间戳"""
        posts = [
            {"posted_at": "2024-01-01T10:00:00"},
            {"posted_at": "2024-01-01T12:00:00"},
        ]
        result = extractor.extract_post_frequency(posts)
        
        assert result["daily_post_mean"] > 0


class TestTimeDistributionExtraction:
    """时间分布特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_time_distribution_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_time_distribution([])
        assert result["hour_entropy"] == 0.0
        assert result["work_hours_ratio"] == 0.0

    def test_extract_time_distribution_work_hours(self, extractor: FeatureExtractor):
        """测试工作时间分布"""
        posts = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(10):
            posts.append({
                "posted_at": base_time + timedelta(days=i),
            })
        
        result = extractor.extract_time_distribution(posts)
        
        assert result["work_hours_ratio"] > 0

    def test_extract_time_distribution_night_hours(self, extractor: FeatureExtractor):
        """测试夜间分布"""
        posts = []
        base_time = datetime(2024, 1, 1, 2, 0, 0)
        for i in range(10):
            posts.append({
                "posted_at": base_time + timedelta(days=i),
            })
        
        result = extractor.extract_time_distribution(posts)
        
        assert result["night_activity_ratio"] > 0

    def test_extract_time_distribution_weekend(self, extractor: FeatureExtractor):
        """测试周末分布"""
        saturday = datetime(2024, 1, 6, 10, 0, 0)
        posts = []
        for i in range(10):
            posts.append({
                "posted_at": saturday + timedelta(days=i * 7),
            })
        
        result = extractor.extract_time_distribution(posts)
        
        assert result["weekend_activity_ratio"] > 0

    def test_extract_time_distribution_entropy(self, extractor: FeatureExtractor):
        """测试时间熵"""
        posts = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        for hour in range(24):
            posts.append({
                "posted_at": base_time + timedelta(hours=hour),
            })
        
        result = extractor.extract_time_distribution(posts)
        
        assert result["hour_entropy"] > 0


class TestAutocorrelationExtraction:
    """自相关特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_autocorrelation_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_autocorrelation([])
        assert result == []

    def test_extract_autocorrelation_single_post(self, extractor: FeatureExtractor):
        """测试单个帖子"""
        posts = [{"posted_at": datetime.utcnow()}]
        result = extractor.extract_autocorrelation(posts)
        assert result == []

    def test_extract_autocorrelation_regular_pattern(self, extractor: FeatureExtractor):
        """测试规律模式"""
        posts = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(100):
            posts.append({
                "posted_at": base_time + timedelta(hours=i * 24),
            })
        
        result = extractor.extract_autocorrelation(posts, max_lag=50)
        
        assert len(result) > 0

    def test_extract_autocorrelation_random_pattern(self, extractor: FeatureExtractor):
        """测试随机模式"""
        np.random.seed(42)
        posts = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(100):
            random_hours = np.random.randint(0, 24)
            posts.append({
                "posted_at": base_time + timedelta(hours=i * 24 + random_hours),
            })
        
        result = extractor.extract_autocorrelation(posts, max_lag=50)
        
        assert len(result) > 0


class TestTextSimilarityExtraction:
    """文本相似度特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_text_similarity_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_text_similarity([])
        assert result["text_similarity_mean"] == 0.0

    def test_extract_text_similarity_single_post(self, extractor: FeatureExtractor):
        """测试单个帖子"""
        posts = [{"content": "Single post"}]
        result = extractor.extract_text_similarity(posts)
        assert result["text_similarity_mean"] == 0.0

    def test_extract_text_similarity_similar_content(self, extractor: FeatureExtractor):
        """测试相似内容"""
        posts = [
            {"content": "This is a test post about Python"},
            {"content": "This is a test post about Python"},
            {"content": "This is a test post about Python"},
        ]
        result = extractor.extract_text_similarity(posts)
        
        assert result["text_similarity_mean"] > 0.9

    def test_extract_text_similarity_different_content(self, extractor: FeatureExtractor):
        """测试不同内容"""
        posts = [
            {"content": "Python is a programming language"},
            {"content": "The weather is nice today"},
            {"content": "I like to eat pizza"},
        ]
        result = extractor.extract_text_similarity(posts)
        
        assert result["text_similarity_mean"] < 0.5


class TestTopicFeaturesExtraction:
    """话题特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_topic_features_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_topic_features([])
        assert result["topic_entropy"] == 0.0
        assert result["topic_consistency"] == 0.0

    def test_extract_topic_features_single_post(self, extractor: FeatureExtractor):
        """测试单个帖子"""
        posts = [{"content": "Single post about Python"}]
        result = extractor.extract_topic_features(posts)
        assert result["topic_entropy"] == 0.0

    def test_extract_topic_features_consistent_topics(self, extractor: FeatureExtractor):
        """测试一致话题"""
        posts = [
            {"content": "Python programming is fun"},
            {"content": "Python development tips"},
            {"content": "Learning Python basics"},
        ]
        result = extractor.extract_topic_features(posts, n_topics=2)
        
        assert result["topic_consistency"] > 0


class TestSentimentFeaturesExtraction:
    """情感特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_sentiment_features_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_sentiment_features([])
        assert result["sentiment_polarity_mean"] == 0.0

    def test_extract_sentiment_features_positive(self, extractor: FeatureExtractor):
        """测试积极情感"""
        posts = [
            {"content": "This is great! I love it!"},
            {"content": "Amazing and wonderful!"},
            {"content": "Best experience ever!"},
        ]
        result = extractor.extract_sentiment_features(posts)
        
        assert result["sentiment_polarity_mean"] > 0

    def test_extract_sentiment_features_negative(self, extractor: FeatureExtractor):
        """测试消极情感"""
        posts = [
            {"content": "This is terrible! I hate it!"},
            {"content": "Awful and bad!"},
            {"content": "Worst experience ever!"},
        ]
        result = extractor.extract_sentiment_features(posts)
        
        assert result["sentiment_polarity_mean"] < 0


class TestTemplateFeaturesExtraction:
    """模板特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_template_features_empty(self, extractor: FeatureExtractor):
        """测试空帖子列表"""
        result = extractor.extract_template_features([])
        assert result["template_match_ratio"] == 0.0
        assert result["unique_template_count"] == 0

    def test_extract_template_features_no_template(self, extractor: FeatureExtractor):
        """测试无模板"""
        posts = [
            {"content": "Random content about weather"},
            {"content": "Different topic entirely"},
            {"content": "Something completely new"},
        ]
        result = extractor.extract_template_features(posts)
        
        assert result["unique_template_count"] == 3

    def test_extract_template_features_with_template(self, extractor: FeatureExtractor):
        """测试有模板"""
        template = "Check out this amazing product at"
        posts = [
            {"content": f"{template} https://example1.com"},
            {"content": f"{template} https://example2.com"},
            {"content": f"{template} https://example3.com"},
        ]
        result = extractor.extract_template_features(posts)
        
        assert result["template_match_ratio"] > 0


class TestUsernameFeaturesExtraction:
    """用户名特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_username_features_empty(self, extractor: FeatureExtractor):
        """测试空用户名列表"""
        result = extractor.extract_username_features([])
        assert result["username_pattern_score"] == 0.0
        assert result["username_digit_ratio"] == 0.0

    def test_extract_username_features_normal(self, extractor: FeatureExtractor):
        """测试正常用户名"""
        usernames = ["john_doe", "alice_smith", "bob_jones"]
        result = extractor.extract_username_features(usernames)
        
        assert result["username_digit_ratio"] == 0.0

    def test_extract_username_features_bot_like(self, extractor: FeatureExtractor):
        """测试机器人风格用户名"""
        usernames = ["user12345", "user67890", "user11111"]
        result = extractor.extract_username_features(usernames)
        
        assert result["username_pattern_score"] > 0
        assert result["username_digit_ratio"] > 0


class TestTemporalFeaturesExtraction:
    """完整时序特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_temporal_features_empty(self, extractor: FeatureExtractor):
        """测试空数据"""
        result = extractor.extract_temporal_features([])
        assert isinstance(result, TemporalFeatures)
        assert result.daily_post_mean == 0.0

    def test_extract_temporal_features_complete(self, extractor: FeatureExtractor):
        """测试完整数据"""
        posts = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(50):
            posts.append({
                "posted_at": base_time + timedelta(hours=i * 6),
            })
        
        interactions = [
            {"response_time": 60},
            {"response_time": 120},
            {"response_time": 90},
        ]
        
        result = extractor.extract_temporal_features(posts, interactions)
        
        assert isinstance(result, TemporalFeatures)
        assert result.daily_post_mean > 0
        assert result.avg_response_delay > 0


class TestContentFeaturesExtraction:
    """完整内容特征提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_content_features_empty(self, extractor: FeatureExtractor):
        """测试空数据"""
        result = extractor.extract_content_features([])
        assert isinstance(result, ContentFeatures)
        assert result.text_similarity_mean == 0.0

    def test_extract_content_features_complete(self, extractor: FeatureExtractor):
        """测试完整数据"""
        posts = [
            {"content": "Python is great for programming"},
            {"content": "Python has many libraries"},
            {"content": "Learning Python is fun"},
        ]
        
        result = extractor.extract_content_features(posts)
        
        assert isinstance(result, ContentFeatures)
        assert result.text_similarity_mean > 0


class TestUserFeatureVectorExtraction:
    """完整用户特征向量提取测试"""

    @pytest.fixture
    def extractor(self):
        """创建特征提取器"""
        return FeatureExtractor()

    def test_extract_user_feature_vector_minimal(self, extractor: FeatureExtractor):
        """测试最小数据"""
        result = extractor.extract_user_feature_vector(
            user_id="test_user",
            posts=[],
        )
        
        assert isinstance(result, UserFeatureVector)
        assert result.user_id == "test_user"

    def test_extract_user_feature_vector_complete(self, extractor: FeatureExtractor):
        """测试完整数据"""
        posts = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for i in range(30):
            posts.append({
                "content": f"Test post {i}",
                "posted_at": base_time + timedelta(hours=i * 12),
            })
        
        user_data = {
            "username": "test_user123",
            "display_name": "Test User",
            "bio": "Test bio",
            "avatar_url": "https://example.com/avatar.png",
        }
        
        result = extractor.extract_user_feature_vector(
            user_id="test_user",
            posts=posts,
            user_data=user_data,
        )
        
        assert isinstance(result, UserFeatureVector)
        assert result.user_id == "test_user"
        assert len(result.raw_features) > 0
