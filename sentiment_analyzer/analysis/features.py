"""
特征提取模块 - 文本特征、情感分析和用户行为特征工程

该模块提供全面的特征提取功能，包括：
- 时序行为特征 (F_t): 发布频率、时间分布、自相关系数、响应延迟
- 内容生成特征 (F_c): 文本相似度、话题一致性、情感分析、模板检测
- 社交网络特征 (F_s): 中心性、社群特征
- 账号元数据特征 (F_n): 注册信息、用户名模式、头像特征
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Optional

import jieba
import numpy as np
from langdetect import detect
from loguru import logger

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class TextFeatures:
    """文本特征"""
    word_count: int
    char_count: int
    avg_word_length: float
    hashtag_count: int
    mention_count: int
    url_count: int
    emoji_count: int
    exclamation_count: int
    question_count: int
    uppercase_ratio: float
    language: str
    keywords: list[str]


@dataclass
class SentimentResult:
    """情感分析结果"""
    score: float
    label: str
    confidence: float
    positive_words: list[str]
    negative_words: list[str]


@dataclass
class TemporalFeatures:
    """时序行为特征 (F_t)"""
    daily_post_mean: float = 0.0
    daily_post_std: float = 0.0
    burst_score: float = 0.0
    hour_entropy: float = 0.0
    work_hours_ratio: float = 0.0
    night_activity_ratio: float = 0.0
    weekend_activity_ratio: float = 0.0
    autocorrelation: list[float] = field(default_factory=list)
    avg_response_delay: float = 0.0
    response_delay_std: float = 0.0


@dataclass
class ContentFeatures:
    """内容生成特征 (F_c)"""
    text_similarity_mean: float = 0.0
    text_similarity_std: float = 0.0
    text_similarity_max: float = 0.0
    topic_entropy: float = 0.0
    topic_consistency: float = 0.0
    sentiment_polarity_mean: float = 0.0
    sentiment_polarity_std: float = 0.0
    sentiment_consistency: float = 0.0
    template_match_ratio: float = 0.0
    unique_template_count: int = 0


@dataclass
class NetworkFeatures:
    """社交网络特征 (F_s)"""
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    clustering_coefficient: float = 0.0
    community_id: int = -1
    modularity_contribution: float = 0.0


@dataclass
class MetadataFeatures:
    """账号元数据特征 (F_n)"""
    registration_cluster_score: float = 0.0
    profile_completeness: float = 0.0
    username_pattern_score: float = 0.0
    username_digit_ratio: float = 0.0
    avatar_similarity_count: int = 0
    avatar_hash: str = ""


@dataclass
class UserFeatureVector:
    """
    用户特征向量
    
    存储用户的所有特征向量，支持特征标准化和特征选择。
    
    Attributes:
        user_id: 用户唯一标识符
        temporal_features: 时序行为特征
        content_features: 内容生成特征
        network_features: 社交网络特征
        metadata_features: 账号元数据特征
    """
    user_id: str
    temporal_features: TemporalFeatures = field(default_factory=TemporalFeatures)
    content_features: ContentFeatures = field(default_factory=ContentFeatures)
    network_features: NetworkFeatures = field(default_factory=NetworkFeatures)
    metadata_features: MetadataFeatures = field(default_factory=MetadataFeatures)
    raw_features: dict[str, float] = field(default_factory=dict)
    normalized_features: dict[str, float] = field(default_factory=dict)
    
    def to_vector(self, feature_names: Optional[list[str]] = None) -> np.ndarray:
        """
        将特征转换为向量
        
        Args:
            feature_names: 指定要提取的特征名称列表，如果为None则提取所有特征
            
        Returns:
            特征向量
        """
        if feature_names is None:
            return np.array(list(self.raw_features.values()))
        
        return np.array([self.raw_features.get(name, 0.0) for name in feature_names])
    
    def normalize(self, scaler: Optional[Any] = None) -> dict[str, float]:
        """
        标准化特征
        
        Args:
            scaler: sklearn的StandardScaler实例，如果为None则使用z-score标准化
            
        Returns:
            标准化后的特征字典
        """
        if not self.raw_features:
            return {}
        
        feature_names = list(self.raw_features.keys())
        feature_values = np.array([self.raw_features[name] for name in feature_names]).reshape(1, -1)
        
        if scaler is not None and SKLEARN_AVAILABLE:
            normalized_values = scaler.transform(feature_values)[0]
        else:
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            if std > 0:
                normalized_values = (feature_values[0] - mean) / std
            else:
                normalized_values = np.zeros_like(feature_values[0])
        
        self.normalized_features = dict(zip(feature_names, normalized_values))
        return self.normalized_features
    
    def select_features(self, feature_names: list[str]) -> dict[str, float]:
        """
        特征选择
        
        Args:
            feature_names: 要选择的特征名称列表
            
        Returns:
            选择的特征字典
        """
        return {name: self.raw_features.get(name, 0.0) for name in feature_names}
    
    def get_all_features(self) -> dict[str, float]:
        """
        获取所有特征的扁平化字典
        
        Returns:
            所有特征的字典
        """
        features = {}
        
        for attr_name in ['temporal_features', 'content_features', 'network_features', 'metadata_features']:
            attr = getattr(self, attr_name)
            if attr is not None:
                for field_name, value in attr.__dict__.items():
                    if isinstance(value, (int, float)):
                        features[f"{attr_name}.{field_name}"] = value
                    elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                        for i, v in enumerate(value):
                            features[f"{attr_name}.{field_name}_{i}"] = v
        
        self.raw_features = features
        return features


class FeatureExtractor:
    """
    特征提取器
    
    提供全面的特征提取功能，包括文本特征、时序行为特征、
    内容生成特征、社交网络特征和账号元数据特征。
    """
    
    WORK_HOURS = set(range(9, 18))
    NIGHT_HOURS = set(range(0, 6)) | set(range(22, 24))
    WEEKEND_DAYS = {5, 6}
    
    def __init__(self) -> None:
        self._sentiment_model: Any = None
        self._sentiment_tokenizer: Any = None
        self._is_initialized: bool = False
        self._lda_model: Any = None
        self._tfidf_vectorizer: Any = None
    
    def initialize(self) -> None:
        """初始化模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self._sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._is_initialized = True
            logger.info("Sentiment model initialized")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
            self._is_initialized = False
    
    def extract_features(self, text: str) -> TextFeatures:
        """提取文本特征"""
        words = self._tokenize(text)
        word_count = len(words)
        char_count = len(text)
        
        return TextFeatures(
            word_count=word_count,
            char_count=char_count,
            avg_word_length=sum(len(w) for w in words) / max(word_count, 1),
            hashtag_count=len(re.findall(r'#\w+', text)),
            mention_count=len(re.findall(r'@[\w\u4e00-\u9fff]+', text)),
            url_count=len(re.findall(r'https?://\S+', text)),
            emoji_count=self._count_emojis(text),
            exclamation_count=text.count('!'),
            question_count=text.count('?'),
            uppercase_ratio=sum(1 for c in text if c.isupper()) / max(char_count, 1),
            language=self._detect_language(text),
            keywords=self._extract_keywords(text)
        )
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """分析情感"""
        if self._is_initialized and self._sentiment_model and self._sentiment_tokenizer:
            return self._analyze_with_model(text)
        else:
            return self._analyze_with_lexicon(text)
    
    def _analyze_with_model(self, text: str) -> SentimentResult:
        """使用深度学习模型分析情感"""
        import torch
        
        inputs = self._sentiment_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self._sentiment_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]
        
        labels = ["negative", "neutral", "positive"]
        max_idx = torch.argmax(scores).item()
        
        return SentimentResult(
            score=float(scores[2] - scores[0]),
            label=labels[max_idx],
            confidence=float(scores[max_idx]),
            positive_words=[],
            negative_words=[]
        )
    
    def _analyze_with_lexicon(self, text: str) -> SentimentResult:
        """使用词典方法分析情感"""
        positive_words = self._get_positive_words()
        negative_words = self._get_negative_words()
        
        words = self._tokenize(text.lower())
        found_positive = [w for w in words if w in positive_words]
        found_negative = [w for w in words if w in negative_words]
        
        pos_count = len(found_positive)
        neg_count = len(found_negative)
        total = max(pos_count + neg_count, 1)
        
        score = (pos_count - neg_count) / total
        confidence = (pos_count + neg_count) / max(len(words), 1)
        
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(
            score=score,
            label=label,
            confidence=confidence,
            positive_words=found_positive,
            negative_words=found_negative
        )
    
    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[@#]\w+', '', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        try:
            lang = detect(text)
        except:
            lang = "en"
        
        if lang == "zh":
            return list(jieba.cut(text))
        else:
            return text.split()
    
    def _detect_language(self, text: str) -> str:
        """检测语言"""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def _count_emojis(self, text: str) -> int:
        """计算表情符号数量"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """提取关键词"""
        words = self._tokenize(text)
        stopwords = self._get_stopwords()
        filtered = [w for w in words if w not in stopwords and len(w) > 1]
        
        word_counts = Counter(filtered)
        return [w for w, _ in word_counts.most_common(top_n)]
    
    def _get_stopwords(self) -> set[str]:
        """获取停用词"""
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "的", "是", "在", "了", "和", "与", "或", "这", "那", "有",
            "我", "你", "他", "她", "它", "们", "就", "也", "都", "而",
            "及", "着", "或", "但", "如果", "因为", "所以", "虽然", "但是"
        }
    
    def _get_positive_words(self) -> set[str]:
        """获取积极词汇"""
        return {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "awesome", "love", "happy", "joy", "beautiful", "perfect",
            "best", "nice", "brilliant", "superb", "outstanding",
            "好", "棒", "优秀", "喜欢", "爱", "开心", "快乐", "美好",
            "完美", "精彩", "出色", "赞", "厉害", "优秀"
        }
    
    def _get_negative_words(self) -> set[str]:
        """获取消极词汇"""
        return {
            "bad", "terrible", "awful", "horrible", "worst", "hate",
            "sad", "angry", "disappointed", "poor", "negative", "wrong",
            "fail", "failure", "problem", "issue", "disaster",
            "坏", "差", "糟糕", "讨厌", "恨", "难过", "伤心", "愤怒",
            "失望", "问题", "错误", "失败", "垃圾"
        }
    
    def extract_batch_features(self, texts: list[str]) -> list[TextFeatures]:
        """批量提取特征"""
        return [self.extract_features(text) for text in texts]
    
    def analyze_batch_sentiment(self, texts: list[str]) -> list[SentimentResult]:
        """批量分析情感"""
        return [self.analyze_sentiment(text) for text in texts]
    
    def extract_post_frequency(self, posts: list[dict[str, Any]]) -> dict[str, float]:
        """
        提取发布频率特征
        
        计算日均发帖量、发帖频率标准差和突发发帖检测。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'posted_at' 字段
            
        Returns:
            包含发布频率特征的字典:
            - daily_post_mean: 日均发帖量
            - daily_post_std: 发帖频率标准差
            - burst_score: 突发发帖得分 (基于泊松分布的离群程度)
        """
        if not posts:
            return {'daily_post_mean': 0.0, 'daily_post_std': 0.0, 'burst_score': 0.0}
        
        timestamps = []
        for post in posts:
            posted_at = post.get('posted_at')
            if posted_at is not None:
                if isinstance(posted_at, str):
                    try:
                        posted_at = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                timestamps.append(posted_at)
        
        if len(timestamps) < 2:
            return {'daily_post_mean': len(timestamps), 'daily_post_std': 0.0, 'burst_score': 0.0}
        
        timestamps.sort()
        
        daily_counts: dict[str, int] = Counter()
        for ts in timestamps:
            date_key = ts.strftime('%Y-%m-%d')
            daily_counts[date_key] += 1
        
        counts = np.array(list(daily_counts.values()))
        daily_mean = float(np.mean(counts))
        daily_std = float(np.std(counts))
        
        burst_score = 0.0
        if daily_mean > 0 and SCIPY_AVAILABLE:
            max_count = max(counts)
            if max_count > daily_mean:
                burst_score = float((max_count - daily_mean) / max(daily_std, 1.0))
        
        return {
            'daily_post_mean': daily_mean,
            'daily_post_std': daily_std,
            'burst_score': burst_score
        }
    
    def extract_time_distribution(self, posts: list[dict[str, Any]]) -> dict[str, float]:
        """
        提取时间分布特征
        
        计算小时分布熵、工作时间占比、夜间活跃度和周末活跃度。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'posted_at' 字段
            
        Returns:
            包含时间分布特征的字典:
            - hour_entropy: 小时分布熵 H = -Σp_i * log(p_i)
            - work_hours_ratio: 工作时间(9-18点)占比
            - night_activity_ratio: 夜间(0-6点, 22-24点)活跃度
            - weekend_activity_ratio: 周末活跃度
        """
        if not posts:
            return {
                'hour_entropy': 0.0,
                'work_hours_ratio': 0.0,
                'night_activity_ratio': 0.0,
                'weekend_activity_ratio': 0.0
            }
        
        hour_counts = np.zeros(24)
        work_hours_count = 0
        night_hours_count = 0
        weekend_count = 0
        total_count = 0
        
        for post in posts:
            posted_at = post.get('posted_at')
            if posted_at is None:
                continue
            
            if isinstance(posted_at, str):
                try:
                    posted_at = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
                except ValueError:
                    continue
            
            hour = posted_at.hour
            weekday = posted_at.weekday()
            
            hour_counts[hour] += 1
            total_count += 1
            
            if hour in self.WORK_HOURS:
                work_hours_count += 1
            if hour in self.NIGHT_HOURS:
                night_hours_count += 1
            if weekday in self.WEEKEND_DAYS:
                weekend_count += 1
        
        if total_count == 0:
            return {
                'hour_entropy': 0.0,
                'work_hours_ratio': 0.0,
                'night_activity_ratio': 0.0,
                'weekend_activity_ratio': 0.0
            }
        
        hour_probs = hour_counts / total_count
        hour_probs = hour_probs[hour_probs > 0]
        hour_entropy = float(-np.sum(hour_probs * np.log2(hour_probs + 1e-10)))
        
        return {
            'hour_entropy': hour_entropy,
            'work_hours_ratio': work_hours_count / total_count,
            'night_activity_ratio': night_hours_count / total_count,
            'weekend_activity_ratio': weekend_count / total_count
        }
    
    def extract_autocorrelation(
        self, 
        posts: list[dict[str, Any]], 
        max_lag: int = 168
    ) -> list[float]:
        """
        计算自相关系数
        
        分析发帖时间序列的自相关性，用于检测周期性行为模式。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'posted_at' 字段
            max_lag: 最大滞后阶数，默认168（一周的小时数）
            
        Returns:
            自相关系数列表，长度为 min(max_lag, 序列长度-1)
        """
        if not posts:
            return []
        
        timestamps = []
        for post in posts:
            posted_at = post.get('posted_at')
            if posted_at is not None:
                if isinstance(posted_at, str):
                    try:
                        posted_at = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                timestamps.append(posted_at)
        
        if len(timestamps) < 2:
            return []
        
        timestamps.sort()
        
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        total_hours = int((max_ts - min_ts).total_seconds() / 3600) + 1
        
        if total_hours < 2:
            return []
        
        hourly_counts = np.zeros(total_hours)
        for ts in timestamps:
            hour_idx = int((ts - min_ts).total_seconds() / 3600)
            if 0 <= hour_idx < total_hours:
                hourly_counts[hour_idx] += 1
        
        if np.std(hourly_counts) == 0:
            return [0.0] * min(max_lag, len(hourly_counts) - 1)
        
        autocorr = []
        n = len(hourly_counts)
        mean = np.mean(hourly_counts)
        var = np.var(hourly_counts)
        
        if var == 0:
            return [0.0] * min(max_lag, n - 1)
        
        for lag in range(1, min(max_lag + 1, n)):
            if lag >= n:
                break
            cov = np.mean((hourly_counts[:-lag] - mean) * (hourly_counts[lag:] - mean))
            autocorr.append(float(cov / var))
        
        return autocorr
    
    def extract_response_delay(
        self, 
        interactions: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        提取响应延迟特征
        
        分析用户对互动的响应时间特征。
        
        Args:
            interactions: 互动列表，每个互动应包含 'timestamp' 和 'response_time' 字段，
                         或包含原始互动和响应的时间戳
            
        Returns:
            包含响应延迟特征的字典:
            - avg_response_delay: 平均响应延迟（秒）
            - response_delay_std: 响应延迟标准差
        """
        if not interactions:
            return {'avg_response_delay': 0.0, 'response_delay_std': 0.0}
        
        delays = []
        
        for interaction in interactions:
            response_time = interaction.get('response_time')
            if response_time is not None:
                delays.append(float(response_time))
                continue
            
            original_ts = interaction.get('original_timestamp')
            response_ts = interaction.get('response_timestamp')
            
            if original_ts is not None and response_ts is not None:
                if isinstance(original_ts, str):
                    try:
                        original_ts = datetime.fromisoformat(original_ts.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                if isinstance(response_ts, str):
                    try:
                        response_ts = datetime.fromisoformat(response_ts.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                
                delay = (response_ts - original_ts).total_seconds()
                if delay >= 0:
                    delays.append(delay)
        
        if not delays:
            return {'avg_response_delay': 0.0, 'response_delay_std': 0.0}
        
        return {
            'avg_response_delay': float(np.mean(delays)),
            'response_delay_std': float(np.std(delays))
        }
    
    def extract_text_similarity(
        self, 
        posts: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        提取文本相似度特征
        
        使用TF-IDF和余弦相似度计算帖子间的文本相似度分布。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'content' 字段
            
        Returns:
            包含文本相似度特征的字典:
            - text_similarity_mean: 平均相似度
            - text_similarity_std: 相似度标准差
            - text_similarity_max: 最大相似度
        """
        if not posts:
            return {
                'text_similarity_mean': 0.0,
                'text_similarity_std': 0.0,
                'text_similarity_max': 0.0
            }
        
        contents = []
        for post in posts:
            content = post.get('content')
            if content and isinstance(content, str) and content.strip():
                contents.append(content.strip())
        
        if len(contents) < 2:
            return {
                'text_similarity_mean': 0.0,
                'text_similarity_std': 0.0,
                'text_similarity_max': 0.0
            }
        
        if not SKLEARN_AVAILABLE:
            return self._compute_simple_similarity(contents)
        
        try:
            if self._tfidf_vectorizer is None:
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(contents)
            
            if SCIPY_AVAILABLE:
                similarity_matrix = 1 - squareform(pdist(tfidf_matrix.toarray(), 'cosine'))
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
            
            np.fill_diagonal(similarity_matrix, 0)
            
            upper_triangle = similarity_matrix[np.triu_indices(len(contents), k=1)]
            
            if len(upper_triangle) == 0:
                return {
                    'text_similarity_mean': 0.0,
                    'text_similarity_std': 0.0,
                    'text_similarity_max': 0.0
                }
            
            return {
                'text_similarity_mean': float(np.mean(upper_triangle)),
                'text_similarity_std': float(np.std(upper_triangle)),
                'text_similarity_max': float(np.max(upper_triangle))
            }
        except Exception as e:
            logger.warning(f"TF-IDF similarity computation failed: {e}")
            return self._compute_simple_similarity(contents)
    
    def _compute_simple_similarity(self, contents: list[str]) -> dict[str, float]:
        """使用简单的序列匹配计算相似度"""
        similarities = []
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                sim = SequenceMatcher(None, contents[i], contents[j]).ratio()
                similarities.append(sim)
        
        if not similarities:
            return {
                'text_similarity_mean': 0.0,
                'text_similarity_std': 0.0,
                'text_similarity_max': 0.0
            }
        
        return {
            'text_similarity_mean': float(np.mean(similarities)),
            'text_similarity_std': float(np.std(similarities)),
            'text_similarity_max': float(np.max(similarities))
        }
    
    def extract_topic_features(
        self, 
        posts: list[dict[str, Any]],
        n_topics: int = 5
    ) -> dict[str, float]:
        """
        提取话题一致性特征
        
        使用LDA主题模型分析帖子的话题分布和一致性。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'content' 字段
            n_topics: 主题数量
            
        Returns:
            包含话题特征的字典:
            - topic_entropy: 话题分布熵
            - topic_consistency: 话题一致性（主要话题占比）
        """
        if not posts:
            return {'topic_entropy': 0.0, 'topic_consistency': 0.0}
        
        contents = []
        for post in posts:
            content = post.get('content')
            if content and isinstance(content, str) and content.strip():
                contents.append(content.strip())
        
        if len(contents) < 2:
            return {'topic_entropy': 0.0, 'topic_consistency': 0.0}
        
        if not SKLEARN_AVAILABLE:
            return self._compute_simple_topic_features(contents)
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            if self._lda_model is None:
                self._lda_model = LatentDirichletAllocation(
                    n_components=min(n_topics, len(contents)),
                    random_state=42,
                    max_iter=10
                )
            
            topic_distributions = self._lda_model.fit_transform(tfidf_matrix)
            
            avg_topic_probs = np.mean(topic_distributions, axis=0)
            avg_topic_probs = avg_topic_probs[avg_topic_probs > 0]
            
            topic_entropy = float(-np.sum(avg_topic_probs * np.log2(avg_topic_probs + 1e-10)))
            
            main_topic_ratios = np.max(topic_distributions, axis=1)
            topic_consistency = float(np.mean(main_topic_ratios))
            
            return {
                'topic_entropy': topic_entropy,
                'topic_consistency': topic_consistency
            }
        except Exception as e:
            logger.warning(f"LDA topic modeling failed: {e}")
            return self._compute_simple_topic_features(contents)
    
    def _compute_simple_topic_features(self, contents: list[str]) -> dict[str, float]:
        """使用简单的关键词频率计算话题特征"""
        all_words = []
        for content in contents:
            words = self._tokenize(content.lower())
            stopwords = self._get_stopwords()
            words = [w for w in words if w not in stopwords and len(w) > 2]
            all_words.extend(words)
        
        if not all_words:
            return {'topic_entropy': 0.0, 'topic_consistency': 0.0}
        
        word_counts = Counter(all_words)
        total = sum(word_counts.values())
        
        if total == 0:
            return {'topic_entropy': 0.0, 'topic_consistency': 0.0}
        
        probs = np.array([c / total for c in word_counts.values()])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        
        top_word_ratio = word_counts.most_common(1)[0][1] / total if word_counts else 0
        
        return {
            'topic_entropy': entropy,
            'topic_consistency': float(top_word_ratio)
        }
    
    def extract_sentiment_features(
        self, 
        posts: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        提取情感分析特征
        
        分析帖子情感极性分布和一致性。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'content' 字段
            
        Returns:
            包含情感特征的字典:
            - sentiment_polarity_mean: 平均情感极性
            - sentiment_polarity_std: 情感极性标准差
            - sentiment_consistency: 情感一致性（主导情感占比）
        """
        if not posts:
            return {
                'sentiment_polarity_mean': 0.0,
                'sentiment_polarity_std': 0.0,
                'sentiment_consistency': 0.0
            }
        
        polarities = []
        labels = []
        
        for post in posts:
            content = post.get('content')
            if content and isinstance(content, str) and content.strip():
                result = self.analyze_sentiment(content)
                polarities.append(result.score)
                labels.append(result.label)
        
        if not polarities:
            return {
                'sentiment_polarity_mean': 0.0,
                'sentiment_polarity_std': 0.0,
                'sentiment_consistency': 0.0
            }
        
        label_counts = Counter(labels)
        dominant_ratio = max(label_counts.values()) / len(labels) if labels else 0
        
        return {
            'sentiment_polarity_mean': float(np.mean(polarities)),
            'sentiment_polarity_std': float(np.std(polarities)),
            'sentiment_consistency': float(dominant_ratio)
        }
    
    def extract_template_features(
        self, 
        posts: list[dict[str, Any]],
        similarity_threshold: float = 0.7
    ) -> dict[str, Any]:
        """
        检测模板化内容
        
        使用最长公共子序列(LCS)检测模板化发帖行为。
        
        Args:
            posts: 帖子列表，每个帖子应包含 'content' 字段
            similarity_threshold: 相似度阈值，超过此值视为模板匹配
            
        Returns:
            包含模板特征的字典:
            - template_match_ratio: 模板匹配率
            - unique_template_count: 唯一模板数量
        """
        if not posts:
            return {'template_match_ratio': 0.0, 'unique_template_count': 0}
        
        contents = []
        for post in posts:
            content = post.get('content')
            if content and isinstance(content, str) and content.strip():
                contents.append(content.strip())
        
        if len(contents) < 2:
            return {'template_match_ratio': 0.0, 'unique_template_count': 0}
        
        templates: list[str] = []
        matched_count = 0
        
        for content in contents:
            is_match = False
            for template in templates:
                similarity = self._lcs_similarity(content, template)
                if similarity >= similarity_threshold:
                    is_match = True
                    break
            
            if is_match:
                matched_count += 1
            else:
                templates.append(content)
        
        return {
            'template_match_ratio': matched_count / len(contents),
            'unique_template_count': len(templates)
        }
    
    def _lcs_similarity(self, text1: str, text2: str) -> float:
        """
        计算基于最长公共子序列的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度得分 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        max_length = max(m, n)
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def extract_centrality_features(
        self, 
        user_id: str, 
        graph: Any
    ) -> dict[str, float]:
        """
        提取中心性特征
        
        计算用户在社交网络中的各种中心性指标。
        
        Args:
            user_id: 用户ID
            graph: NetworkX图对象
            
        Returns:
            包含中心性特征的字典:
            - degree_centrality: 度中心性
            - betweenness_centrality: 介数中心性
            - eigenvector_centrality: 特征向量中心性
            - pagerank: PageRank值
        """
        result = {
            'degree_centrality': 0.0,
            'betweenness_centrality': 0.0,
            'eigenvector_centrality': 0.0,
            'pagerank': 0.0
        }
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for centrality computation")
            return result
        
        if graph is None or not isinstance(graph, nx.Graph):
            return result
        
        if user_id not in graph:
            return result
        
        try:
            degree_cent = nx.degree_centrality(graph)
            result['degree_centrality'] = float(degree_cent.get(user_id, 0.0))
        except Exception as e:
            logger.warning(f"Degree centrality computation failed: {e}")
        
        try:
            betweenness_cent = nx.betweenness_centrality(graph, normalized=True)
            result['betweenness_centrality'] = float(betweenness_cent.get(user_id, 0.0))
        except Exception as e:
            logger.warning(f"Betweenness centrality computation failed: {e}")
        
        try:
            eigenvector_cent = nx.eigenvector_centrality(graph, max_iter=1000)
            result['eigenvector_centrality'] = float(eigenvector_cent.get(user_id, 0.0))
        except Exception as e:
            logger.warning(f"Eigenvector centrality computation failed: {e}")
        
        try:
            pagerank = nx.pagerank(graph)
            result['pagerank'] = float(pagerank.get(user_id, 0.0))
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}")
        
        return result
    
    def extract_community_features(
        self, 
        user_id: str, 
        graph: Any
    ) -> dict[str, Any]:
        """
        提取社群特征
        
        分析用户的社群归属和聚类特征。
        
        Args:
            user_id: 用户ID
            graph: NetworkX图对象
            
        Returns:
            包含社群特征的字典:
            - clustering_coefficient: 聚类系数
            - community_id: 社区归属ID
            - modularity_contribution: 模块度贡献
        """
        result = {
            'clustering_coefficient': 0.0,
            'community_id': -1,
            'modularity_contribution': 0.0
        }
        
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available for community computation")
            return result
        
        if graph is None or not isinstance(graph, nx.Graph):
            return result
        
        if user_id not in graph:
            return result
        
        try:
            result['clustering_coefficient'] = float(nx.clustering(graph, user_id))
        except Exception as e:
            logger.warning(f"Clustering coefficient computation failed: {e}")
        
        try:
            if hasattr(nx, 'community') and hasattr(nx.community, 'greedy_modularity_communities'):
                communities = list(nx.community.greedy_modularity_communities(graph))
                for i, community in enumerate(communities):
                    if user_id in community:
                        result['community_id'] = i
                        
                        m = graph.number_of_edges()
                        if m > 0:
                            degree = graph.degree(user_id)
                            result['modularity_contribution'] = float(degree / (2 * m))
                        break
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
        
        return result
    
    def extract_registration_features(
        self, 
        users: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        提取注册信息特征
        
        分析用户注册时间的聚集性和资料完整度。
        
        Args:
            users: 用户列表，每个用户应包含 'created_at' 和其他元数据字段
            
        Returns:
            包含注册特征的字典:
            - registration_cluster_score: 注册时间聚集性得分
            - profile_completeness: 平均资料完整度
        """
        if not users:
            return {'registration_cluster_score': 0.0, 'profile_completeness': 0.0}
        
        registration_times = []
        completeness_scores = []
        
        for user in users:
            created_at = user.get('created_at')
            if created_at is not None:
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except ValueError:
                        created_at = None
                if created_at is not None:
                    registration_times.append(created_at.timestamp())
            
            completeness = self._compute_profile_completeness(user)
            completeness_scores.append(completeness)
        
        cluster_score = 0.0
        if len(registration_times) >= 2 and SCIPY_AVAILABLE:
            try:
                times_array = np.array(registration_times)
                if np.std(times_array) > 0:
                    _, p_value = stats.normaltest(times_array)
                    cluster_score = 1.0 - p_value
            except Exception:
                pass
        
        return {
            'registration_cluster_score': float(cluster_score),
            'profile_completeness': float(np.mean(completeness_scores)) if completeness_scores else 0.0
        }
    
    def _compute_profile_completeness(self, user: dict[str, Any]) -> float:
        """
        计算资料完整度
        
        Args:
            user: 用户数据字典
            
        Returns:
            资料完整度得分 (0-1)
        """
        fields = ['username', 'display_name', 'bio', 'avatar_url', 'created_at']
        filled = 0
        
        for field in fields:
            value = user.get(field)
            if value is not None and str(value).strip():
                filled += 1
        
        return filled / len(fields)
    
    def extract_username_features(
        self, 
        usernames: list[str]
    ) -> dict[str, float]:
        """
        分析用户名模式
        
        检测用户名的模式特征，如数字比例、命名规则等。
        
        Args:
            usernames: 用户名列表
            
        Returns:
            包含用户名特征的字典:
            - username_pattern_score: 模式匹配得分
            - username_digit_ratio: 平均数字比例
        """
        if not usernames:
            return {'username_pattern_score': 0.0, 'username_digit_ratio': 0.0}
        
        pattern_scores = []
        digit_ratios = []
        
        patterns = [
            re.compile(r'^[a-z]+[0-9]+$', re.IGNORECASE),
            re.compile(r'^[a-z]+_[0-9]+$', re.IGNORECASE),
            re.compile(r'^user[0-9]+$', re.IGNORECASE),
            re.compile(r'^[a-z]{8,12}[0-9]{3,6}$', re.IGNORECASE),
            re.compile(r'^[a-z]+[0-9]{4,}$', re.IGNORECASE),
        ]
        
        for username in usernames:
            if not username:
                continue
            
            pattern_match = any(p.match(username) for p in patterns)
            pattern_scores.append(1.0 if pattern_match else 0.0)
            
            digits = sum(1 for c in username if c.isdigit())
            digit_ratios.append(digits / len(username))
        
        return {
            'username_pattern_score': float(np.mean(pattern_scores)) if pattern_scores else 0.0,
            'username_digit_ratio': float(np.mean(digit_ratios)) if digit_ratios else 0.0
        }
    
    def extract_avatar_features(
        self, 
        avatar_urls: list[str]
    ) -> dict[str, Any]:
        """
        提取头像特征
        
        使用感知哈希计算头像相似度，检测相似头像。
        
        Args:
            avatar_urls: 头像URL列表
            
        Returns:
            包含头像特征的字典:
            - avatar_similarity_count: 相似头像对数量
            - avatar_hash: 头像哈希列表（用于后续比对）
        """
        if not avatar_urls:
            return {'avatar_similarity_count': 0, 'avatar_hashes': []}
        
        hashes = []
        valid_urls = []
        
        for url in avatar_urls:
            if url and isinstance(url, str) and url.strip():
                url_hash = self._compute_url_hash(url.strip())
                hashes.append(url_hash)
                valid_urls.append(url.strip())
        
        if len(hashes) < 2:
            return {'avatar_similarity_count': 0, 'avatar_hashes': hashes}
        
        similarity_count = 0
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if self._hamming_distance(hashes[i], hashes[j]) <= 5:
                    similarity_count += 1
        
        return {
            'avatar_similarity_count': similarity_count,
            'avatar_hashes': hashes
        }
    
    def _compute_url_hash(self, url: str) -> str:
        """
        计算URL的哈希值
        
        Args:
            url: 头像URL
            
        Returns:
            哈希字符串
        """
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        计算两个哈希之间的汉明距离
        
        Args:
            hash1: 第一个哈希
            hash2: 第二个哈希
            
        Returns:
            汉明距离
        """
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def extract_temporal_features(
        self, 
        posts: list[dict[str, Any]],
        interactions: Optional[list[dict[str, Any]]] = None,
        max_lag: int = 168
    ) -> TemporalFeatures:
        """
        提取完整的时序行为特征
        
        Args:
            posts: 帖子列表
            interactions: 互动列表（可选）
            max_lag: 自相关最大滞后阶数
            
        Returns:
            TemporalFeatures 对象
        """
        freq_features = self.extract_post_frequency(posts)
        time_features = self.extract_time_distribution(posts)
        autocorr = self.extract_autocorrelation(posts, max_lag)
        
        response_features = {'avg_response_delay': 0.0, 'response_delay_std': 0.0}
        if interactions:
            response_features = self.extract_response_delay(interactions)
        
        return TemporalFeatures(
            daily_post_mean=freq_features['daily_post_mean'],
            daily_post_std=freq_features['daily_post_std'],
            burst_score=freq_features['burst_score'],
            hour_entropy=time_features['hour_entropy'],
            work_hours_ratio=time_features['work_hours_ratio'],
            night_activity_ratio=time_features['night_activity_ratio'],
            weekend_activity_ratio=time_features['weekend_activity_ratio'],
            autocorrelation=autocorr,
            avg_response_delay=response_features['avg_response_delay'],
            response_delay_std=response_features['response_delay_std']
        )
    
    def extract_content_features(
        self, 
        posts: list[dict[str, Any]],
        n_topics: int = 5,
        similarity_threshold: float = 0.7
    ) -> ContentFeatures:
        """
        提取完整的内容生成特征
        
        Args:
            posts: 帖子列表
            n_topics: 主题数量
            similarity_threshold: 模板检测相似度阈值
            
        Returns:
            ContentFeatures 对象
        """
        similarity_features = self.extract_text_similarity(posts)
        topic_features = self.extract_topic_features(posts, n_topics)
        sentiment_features = self.extract_sentiment_features(posts)
        template_features = self.extract_template_features(posts, similarity_threshold)
        
        return ContentFeatures(
            text_similarity_mean=similarity_features['text_similarity_mean'],
            text_similarity_std=similarity_features['text_similarity_std'],
            text_similarity_max=similarity_features['text_similarity_max'],
            topic_entropy=topic_features['topic_entropy'],
            topic_consistency=topic_features['topic_consistency'],
            sentiment_polarity_mean=sentiment_features['sentiment_polarity_mean'],
            sentiment_polarity_std=sentiment_features['sentiment_polarity_std'],
            sentiment_consistency=sentiment_features['sentiment_consistency'],
            template_match_ratio=template_features['template_match_ratio'],
            unique_template_count=template_features['unique_template_count']
        )
    
    def extract_network_features(
        self, 
        user_id: str, 
        graph: Any
    ) -> NetworkFeatures:
        """
        提取完整的社交网络特征
        
        Args:
            user_id: 用户ID
            graph: NetworkX图对象
            
        Returns:
            NetworkFeatures 对象
        """
        centrality_features = self.extract_centrality_features(user_id, graph)
        community_features = self.extract_community_features(user_id, graph)
        
        return NetworkFeatures(
            degree_centrality=centrality_features['degree_centrality'],
            betweenness_centrality=centrality_features['betweenness_centrality'],
            eigenvector_centrality=centrality_features['eigenvector_centrality'],
            pagerank=centrality_features['pagerank'],
            clustering_coefficient=community_features['clustering_coefficient'],
            community_id=community_features['community_id'],
            modularity_contribution=community_features['modularity_contribution']
        )
    
    def extract_metadata_features(
        self, 
        user: dict[str, Any],
        all_users: Optional[list[dict[str, Any]]] = None
    ) -> MetadataFeatures:
        """
        提取完整的账号元数据特征
        
        Args:
            user: 目标用户数据
            all_users: 所有用户数据（用于聚集性分析）
            
        Returns:
            MetadataFeatures 对象
        """
        reg_features = {'registration_cluster_score': 0.0, 'profile_completeness': 0.0}
        if all_users:
            reg_features = self.extract_registration_features(all_users)
        
        username_features = self.extract_username_features(
            [user.get('username', '')] if user.get('username') else []
        )
        
        avatar_features = self.extract_avatar_features(
            [user.get('avatar_url')] if user.get('avatar_url') else []
        )
        
        profile_completeness = self._compute_profile_completeness(user)
        
        return MetadataFeatures(
            registration_cluster_score=reg_features['registration_cluster_score'],
            profile_completeness=profile_completeness,
            username_pattern_score=username_features['username_pattern_score'],
            username_digit_ratio=username_features['username_digit_ratio'],
            avatar_similarity_count=avatar_features['avatar_similarity_count'],
            avatar_hash=avatar_features['avatar_hashes'][0] if avatar_features['avatar_hashes'] else ""
        )
    
    def extract_user_feature_vector(
        self,
        user_id: str,
        posts: list[dict[str, Any]],
        graph: Optional[Any] = None,
        interactions: Optional[list[dict[str, Any]]] = None,
        all_users: Optional[list[dict[str, Any]]] = None,
        user_data: Optional[dict[str, Any]] = None
    ) -> UserFeatureVector:
        """
        提取用户的完整特征向量
        
        这是主要的特征提取入口方法，整合所有特征类型。
        
        Args:
            user_id: 用户ID
            posts: 用户帖子列表
            graph: 社交网络图（可选）
            interactions: 用户互动列表（可选）
            all_users: 所有用户数据（可选，用于聚集性分析）
            user_data: 用户自身数据（可选）
            
        Returns:
            UserFeatureVector 对象
        """
        temporal_features = self.extract_temporal_features(posts, interactions)
        content_features = self.extract_content_features(posts)
        
        network_features = NetworkFeatures()
        if graph is not None:
            network_features = self.extract_network_features(user_id, graph)
        
        metadata_features = MetadataFeatures()
        if user_data:
            metadata_features = self.extract_metadata_features(user_data, all_users)
        
        feature_vector = UserFeatureVector(
            user_id=user_id,
            temporal_features=temporal_features,
            content_features=content_features,
            network_features=network_features,
            metadata_features=metadata_features
        )
        
        feature_vector.get_all_features()
        
        return feature_vector
