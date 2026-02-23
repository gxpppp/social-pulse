"""
跨事件关联分析模块 - 分析多个事件之间的关联关系

提供跨事件的账号复用检测、行为演化分析、实体对齐等功能。
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np
from loguru import logger

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class EventSnapshot:
    """
    事件图快照类
    
    存储单个事件的图快照信息，用于跨事件比较分析。
    
    Attributes:
        event_id: 事件唯一标识符
        event_name: 事件名称
        time_window: 时间窗口 (start_time, end_time)
        graph: NetworkX图对象
        active_users: 活跃用户列表
        active_posts: 活跃帖子列表
    """
    event_id: str
    event_name: str
    time_window: tuple[datetime, datetime]
    graph: Optional[Any] = None
    active_users: list[str] = field(default_factory=list)
    active_posts: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_time_window(
        cls,
        graph: Any,
        start_time: datetime,
        end_time: datetime,
        event_id: Optional[str] = None,
        event_name: Optional[str] = None
    ) -> "EventSnapshot":
        """
        从时间窗口创建快照
        
        Args:
            graph: 完整的NetworkX图对象
            start_time: 时间窗口起始时间
            end_time: 时间窗口结束时间
            event_id: 事件ID (可选，自动生成)
            event_name: 事件名称 (可选)
            
        Returns:
            EventSnapshot 实例
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, creating empty snapshot")
            return cls(
                event_id=event_id or f"event_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                event_name=event_name or "Unknown Event",
                time_window=(start_time, end_time),
                active_users=[],
                active_posts=[]
            )
        
        subgraph = nx.DiGraph()
        active_users = set()
        active_posts = set()
        
        for node, node_data in graph.nodes(data=True):
            node_time = node_data.get("timestamp") or node_data.get("created_at")
            if node_time is not None:
                if isinstance(node_time, str):
                    try:
                        node_time = datetime.fromisoformat(node_time.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                
                if start_time <= node_time <= end_time:
                    subgraph.add_node(node, **node_data)
                    node_type = node_data.get("type", "unknown")
                    if node_type == "user":
                        active_users.add(node)
                    elif node_type == "post":
                        active_posts.add(node)
        
        for u, v, edge_data in graph.edges(data=True):
            if subgraph.has_node(u) and subgraph.has_node(v):
                subgraph.add_edge(u, v, **edge_data)
        
        event_id = event_id or f"event_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"
        event_name = event_name or f"Event {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
        
        return cls(
            event_id=event_id,
            event_name=event_name,
            time_window=(start_time, end_time),
            graph=subgraph,
            active_users=list(active_users),
            active_posts=list(active_posts)
        )
    
    @property
    def duration(self) -> timedelta:
        """事件持续时间"""
        return self.time_window[1] - self.time_window[0]
    
    @property
    def node_count(self) -> int:
        """节点数量"""
        if self.graph is None:
            return 0
        return self.graph.number_of_nodes()
    
    @property
    def edge_count(self) -> int:
        """边数量"""
        if self.graph is None:
            return 0
        return self.graph.number_of_edges()
    
    def get_user_nodes(self) -> list[str]:
        """获取用户节点列表"""
        if self.graph is None:
            return []
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "user"]
    
    def get_post_nodes(self) -> list[str]:
        """获取帖子节点列表"""
        if self.graph is None:
            return []
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "post"]
    
    def get_node_attributes(self, node_id: str) -> dict[str, Any]:
        """获取节点属性"""
        if self.graph is None or not self.graph.has_node(node_id):
            return {}
        return dict(self.graph.nodes[node_id])
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_name": self.event_name,
            "time_window": [
                self.time_window[0].isoformat(),
                self.time_window[1].isoformat()
            ],
            "duration_seconds": self.duration.total_seconds(),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "active_users_count": len(self.active_users),
            "active_posts_count": len(self.active_posts),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class GraphAlignmentResult:
    """图对齐结果"""
    node_mapping: dict[str, str] = field(default_factory=dict)
    node_intersection: set[str] = field(default_factory=set)
    node_union: set[str] = field(default_factory=set)
    edge_intersection: set[tuple[str, str]] = field(default_factory=set)
    edge_union: set[tuple[str, str]] = field(default_factory=set)
    similarity_score: float = 0.0
    node_similarity: float = 0.0
    edge_similarity: float = 0.0


class GraphAligner:
    """
    图对齐器
    
    对齐两个图快照并计算相似度。
    """
    
    def __init__(self) -> None:
        self._alignment_cache: dict[str, GraphAlignmentResult] = {}
    
    def align_snapshots(
        self,
        snapshot1: EventSnapshot,
        snapshot2: EventSnapshot
    ) -> GraphAlignmentResult:
        """
        对齐两个图快照
        
        Args:
            snapshot1: 第一个图快照
            snapshot2: 第二个图快照
            
        Returns:
            GraphAlignmentResult 对齐结果
        """
        cache_key = f"{snapshot1.event_id}_{snapshot2.event_id}"
        if cache_key in self._alignment_cache:
            return self._alignment_cache[cache_key]
        
        if snapshot1.graph is None or snapshot2.graph is None:
            return GraphAlignmentResult()
        
        nodes1 = set(snapshot1.graph.nodes())
        nodes2 = set(snapshot2.graph.nodes())
        
        node_intersection = nodes1 & nodes2
        node_union = nodes1 | nodes2
        
        edges1 = set(snapshot1.graph.edges())
        edges2 = set(snapshot2.graph.edges())
        
        edge_intersection = edges1 & edges2
        edge_union = edges1 | edges2
        
        node_mapping = {n: n for n in node_intersection}
        
        node_similarity = len(node_intersection) / len(node_union) if node_union else 0.0
        edge_similarity = len(edge_intersection) / len(edge_union) if edge_union else 0.0
        
        similarity_score = 0.6 * node_similarity + 0.4 * edge_similarity
        
        result = GraphAlignmentResult(
            node_mapping=node_mapping,
            node_intersection=node_intersection,
            node_union=node_union,
            edge_intersection=edge_intersection,
            edge_union=edge_union,
            similarity_score=similarity_score,
            node_similarity=node_similarity,
            edge_similarity=edge_similarity
        )
        
        self._alignment_cache[cache_key] = result
        return result
    
    def calculate_similarity(
        self,
        snapshot1: EventSnapshot,
        snapshot2: EventSnapshot
    ) -> dict[str, float]:
        """
        计算图相似度
        
        Args:
            snapshot1: 第一个图快照
            snapshot2: 第二个图快照
            
        Returns:
            包含各种相似度指标的字典
        """
        alignment = self.align_snapshots(snapshot1, snapshot2)
        
        structural_similarity = self._calculate_structural_similarity(
            snapshot1, snapshot2, alignment
        )
        
        return {
            "node_intersection": len(alignment.node_intersection),
            "node_union": len(alignment.node_union),
            "node_similarity": alignment.node_similarity,
            "edge_intersection": len(alignment.edge_intersection),
            "edge_union": len(alignment.edge_union),
            "edge_similarity": alignment.edge_similarity,
            "overall_similarity": alignment.similarity_score,
            "structural_similarity": structural_similarity
        }
    
    def _calculate_structural_similarity(
        self,
        snapshot1: EventSnapshot,
        snapshot2: EventSnapshot,
        alignment: GraphAlignmentResult
    ) -> float:
        """
        计算结构相似度
        
        基于图的拓扑特征计算相似度。
        """
        if snapshot1.graph is None or snapshot2.graph is None:
            return 0.0
        
        if not NETWORKX_AVAILABLE:
            return 0.0
        
        try:
            density1 = nx.density(snapshot1.graph)
            density2 = nx.density(snapshot2.graph)
            
            density_sim = 1.0 - abs(density1 - density2)
            
            clustering1 = nx.average_clustering(snapshot1.graph.to_undirected()) if snapshot1.node_count > 0 else 0
            clustering2 = nx.average_clustering(snapshot2.graph.to_undirected()) if snapshot2.node_count > 0 else 0
            
            clustering_sim = 1.0 - abs(clustering1 - clustering2)
            
            return 0.5 * density_sim + 0.5 * clustering_sim
        except Exception as e:
            logger.warning(f"Structural similarity calculation failed: {e}")
            return 0.0
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._alignment_cache.clear()


@dataclass
class EntityAlignment:
    """实体对齐结果"""
    user1_id: str
    user2_id: str
    attribute_score: float = 0.0
    behavior_score: float = 0.0
    relation_score: float = 0.0
    overall_score: float = 0.0
    alignment_details: dict[str, Any] = field(default_factory=dict)


class EntityAligner:
    """
    实体对齐器
    
    通过属性、行为和关系匹配对齐用户实体。
    """
    
    def __init__(self) -> None:
        self._username_weight: float = 0.3
        self._bio_weight: float = 0.3
        self._avatar_weight: float = 0.4
    
    def align_by_attribute(
        self,
        users1: list[dict[str, Any]],
        users2: list[dict[str, Any]]
    ) -> list[EntityAlignment]:
        """
        属性匹配
        
        通过用户名编辑距离、简介文本相似度和头像哈希匹配用户。
        
        Args:
            users1: 第一组用户列表
            users2: 第二组用户列表
            
        Returns:
            实体对齐结果列表
        """
        alignments = []
        
        for u1 in users1:
            u1_id = u1.get("user_id", u1.get("id", ""))
            u1_username = u1.get("username", u1.get("screen_name", ""))
            u1_bio = u1.get("bio", u1.get("description", ""))
            u1_avatar = u1.get("avatar_url", u1.get("profile_image_url", ""))
            
            for u2 in users2:
                u2_id = u2.get("user_id", u2.get("id", ""))
                u2_username = u2.get("username", u2.get("screen_name", ""))
                u2_bio = u2.get("bio", u2.get("description", ""))
                u2_avatar = u2.get("avatar_url", u2.get("profile_image_url", ""))
                
                username_sim = self._edit_distance_similarity(u1_username, u2_username)
                bio_sim = self._text_similarity(u1_bio, u2_bio)
                avatar_sim = self._avatar_hash_similarity(u1_avatar, u2_avatar)
                
                attribute_score = (
                    self._username_weight * username_sim +
                    self._bio_weight * bio_sim +
                    self._avatar_weight * avatar_sim
                )
                
                if attribute_score > 0.3:
                    alignments.append(EntityAlignment(
                        user1_id=u1_id,
                        user2_id=u2_id,
                        attribute_score=attribute_score,
                        alignment_details={
                            "username_similarity": username_sim,
                            "bio_similarity": bio_sim,
                            "avatar_similarity": avatar_sim
                        }
                    ))
        
        return alignments
    
    def align_by_behavior(
        self,
        users1: list[dict[str, Any]],
        users2: list[dict[str, Any]],
        posts1: Optional[dict[str, list[dict[str, Any]]]] = None,
        posts2: Optional[dict[str, list[dict[str, Any]]]] = None
    ) -> list[EntityAlignment]:
        """
        行为指纹匹配
        
        通过发帖时间分布JS散度、文本风格向量余弦相似度和话题偏好相似度匹配用户。
        
        Args:
            users1: 第一组用户列表
            users2: 第二组用户列表
            posts1: 第一组用户的帖子 (user_id -> posts)
            posts2: 第二组用户的帖子 (user_id -> posts)
            
        Returns:
            实体对齐结果列表
        """
        alignments = []
        
        posts1 = posts1 or {}
        posts2 = posts2 or {}
        
        for u1 in users1:
            u1_id = u1.get("user_id", u1.get("id", ""))
            u1_posts = posts1.get(u1_id, [])
            
            for u2 in users2:
                u2_id = u2.get("user_id", u2.get("id", ""))
                u2_posts = posts2.get(u2_id, [])
                
                time_dist_sim = self._time_distribution_similarity(u1_posts, u2_posts)
                style_sim = self._text_style_similarity(u1_posts, u2_posts)
                topic_sim = self._topic_preference_similarity(u1_posts, u2_posts)
                
                behavior_score = 0.4 * time_dist_sim + 0.3 * style_sim + 0.3 * topic_sim
                
                if behavior_score > 0.3:
                    alignments.append(EntityAlignment(
                        user1_id=u1_id,
                        user2_id=u2_id,
                        behavior_score=behavior_score,
                        alignment_details={
                            "time_distribution_similarity": time_dist_sim,
                            "text_style_similarity": style_sim,
                            "topic_preference_similarity": topic_sim
                        }
                    ))
        
        return alignments
    
    def align_by_relation(
        self,
        users1: list[dict[str, Any]],
        users2: list[dict[str, Any]],
        graph1: Optional[Any] = None,
        graph2: Optional[Any] = None
    ) -> list[EntityAlignment]:
        """
        关系匹配
        
        通过共同邻居Jaccard相似度和社区归属一致性匹配用户。
        
        Args:
            users1: 第一组用户列表
            users2: 第二组用户列表
            graph1: 第一个事件的图
            graph2: 第二个事件的图
            
        Returns:
            实体对齐结果列表
        """
        alignments = []
        
        if not NETWORKX_AVAILABLE or graph1 is None or graph2 is None:
            return alignments
        
        for u1 in users1:
            u1_id = u1.get("user_id", u1.get("id", ""))
            
            if not graph1.has_node(u1_id):
                continue
            
            neighbors1 = set(graph1.neighbors(u1_id))
            community1 = self._get_user_community(u1_id, graph1)
            
            for u2 in users2:
                u2_id = u2.get("user_id", u2.get("id", ""))
                
                if not graph2.has_node(u2_id):
                    continue
                
                neighbors2 = set(graph2.neighbors(u2_id))
                community2 = self._get_user_community(u2_id, graph2)
                
                neighbor_sim = self._jaccard_similarity(neighbors1, neighbors2)
                community_sim = 1.0 if community1 == community2 else 0.0
                
                relation_score = 0.7 * neighbor_sim + 0.3 * community_sim
                
                if relation_score > 0.2:
                    alignments.append(EntityAlignment(
                        user1_id=u1_id,
                        user2_id=u2_id,
                        relation_score=relation_score,
                        alignment_details={
                            "neighbor_similarity": neighbor_sim,
                            "community_consistency": community_sim,
                            "neighbors1_count": len(neighbors1),
                            "neighbors2_count": len(neighbors2)
                        }
                    ))
        
        return alignments
    
    def compute_overall_alignment(
        self,
        attribute_alignments: list[EntityAlignment],
        behavior_alignments: list[EntityAlignment],
        relation_alignments: list[EntityAlignment],
        weights: Optional[dict[str, float]] = None
    ) -> list[EntityAlignment]:
        """
        计算综合对齐结果
        
        Args:
            attribute_alignments: 属性对齐结果
            behavior_alignments: 行为对齐结果
            relation_alignments: 关系对齐结果
            weights: 各类型权重 (默认: attribute=0.4, behavior=0.35, relation=0.25)
            
        Returns:
            综合对齐结果列表
        """
        weights = weights or {"attribute": 0.4, "behavior": 0.35, "relation": 0.25}
        
        alignment_map: dict[tuple[str, str], EntityAlignment] = {}
        
        for a in attribute_alignments:
            key = (a.user1_id, a.user2_id)
            if key not in alignment_map:
                alignment_map[key] = EntityAlignment(
                    user1_id=a.user1_id,
                    user2_id=a.user2_id
                )
            alignment_map[key].attribute_score = a.attribute_score
            alignment_map[key].alignment_details.update(a.alignment_details)
        
        for a in behavior_alignments:
            key = (a.user1_id, a.user2_id)
            if key not in alignment_map:
                alignment_map[key] = EntityAlignment(
                    user1_id=a.user1_id,
                    user2_id=a.user2_id
                )
            alignment_map[key].behavior_score = a.behavior_score
            alignment_map[key].alignment_details.update(a.alignment_details)
        
        for a in relation_alignments:
            key = (a.user1_id, a.user2_id)
            if key not in alignment_map:
                alignment_map[key] = EntityAlignment(
                    user1_id=a.user1_id,
                    user2_id=a.user2_id
                )
            alignment_map[key].relation_score = a.relation_score
            alignment_map[key].alignment_details.update(a.alignment_details)
        
        results = []
        for alignment in alignment_map.values():
            alignment.overall_score = (
                weights["attribute"] * alignment.attribute_score +
                weights["behavior"] * alignment.behavior_score +
                weights["relation"] * alignment.relation_score
            )
            results.append(alignment)
        
        return sorted(results, key=lambda x: x.overall_score, reverse=True)
    
    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        """计算编辑距离相似度"""
        if not s1 or not s2:
            return 0.0
        
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
        
        return SequenceMatcher(None, s1_lower, s2_lower).ratio()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        if text1_lower == text2_lower:
            return 1.0
        
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        return self._jaccard_similarity(words1, words2)
    
    def _avatar_hash_similarity(self, url1: str, url2: str) -> float:
        """计算头像URL相似度"""
        if not url1 or not url2:
            return 0.0
        
        if url1 == url2:
            return 1.0
        
        hash1 = hashlib.md5(url1.encode()).hexdigest()
        hash2 = hashlib.md5(url2.encode()).hexdigest()
        
        hamming_dist = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        
        return 1.0 - (hamming_dist / len(hash1))
    
    def _time_distribution_similarity(
        self,
        posts1: list[dict[str, Any]],
        posts2: list[dict[str, Any]]
    ) -> float:
        """计算发帖时间分布相似度 (JS散度)"""
        if not posts1 or not posts2:
            return 0.0
        
        hour_dist1 = np.zeros(24)
        hour_dist2 = np.zeros(24)
        
        for post in posts1:
            posted_at = post.get("posted_at") or post.get("created_at")
            if posted_at:
                if isinstance(posted_at, str):
                    try:
                        posted_at = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                hour_dist1[posted_at.hour] += 1
        
        for post in posts2:
            posted_at = post.get("posted_at") or post.get("created_at")
            if posted_at:
                if isinstance(posted_at, str):
                    try:
                        posted_at = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
                    except ValueError:
                        continue
                hour_dist2[posted_at.hour] += 1
        
        total1 = hour_dist1.sum()
        total2 = hour_dist2.sum()
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        hour_dist1 = hour_dist1 / total1
        hour_dist2 = hour_dist2 / total2
        
        if SCIPY_AVAILABLE:
            js_divergence = jensenshannon(hour_dist1, hour_dist2)
            return 1.0 - js_divergence
        else:
            return self._jaccard_similarity(
                set(np.where(hour_dist1 > 0)[0]),
                set(np.where(hour_dist2 > 0)[0])
            )
    
    def _text_style_similarity(
        self,
        posts1: list[dict[str, Any]],
        posts2: list[dict[str, Any]]
    ) -> float:
        """计算文本风格相似度"""
        if not posts1 or not posts2:
            return 0.0
        
        style1 = self._extract_text_style(posts1)
        style2 = self._extract_text_style(posts2)
        
        vec1 = np.array([
            style1["avg_length"],
            style1["exclamation_ratio"],
            style1["question_ratio"],
            style1["uppercase_ratio"],
            style1["emoji_ratio"]
        ])
        
        vec2 = np.array([
            style2["avg_length"],
            style2["exclamation_ratio"],
            style2["question_ratio"],
            style2["uppercase_ratio"],
            style2["emoji_ratio"]
        ])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return max(0, cosine_sim)
    
    def _extract_text_style(self, posts: list[dict[str, Any]]) -> dict[str, float]:
        """提取文本风格特征"""
        if not posts:
            return {
                "avg_length": 0,
                "exclamation_ratio": 0,
                "question_ratio": 0,
                "uppercase_ratio": 0,
                "emoji_ratio": 0
            }
        
        lengths = []
        exclamation_count = 0
        question_count = 0
        uppercase_count = 0
        emoji_count = 0
        total_chars = 0
        
        for post in posts:
            content = post.get("content", post.get("text", ""))
            if not content:
                continue
            
            lengths.append(len(content))
            total_chars += len(content)
            exclamation_count += content.count('!')
            question_count += content.count('?')
            uppercase_count += sum(1 for c in content if c.isupper())
            emoji_count += self._count_emojis(content)
        
        total_chars = max(total_chars, 1)
        
        return {
            "avg_length": np.mean(lengths) if lengths else 0,
            "exclamation_ratio": exclamation_count / total_chars,
            "question_ratio": question_count / total_chars,
            "uppercase_ratio": uppercase_count / total_chars,
            "emoji_ratio": emoji_count / total_chars
        }
    
    def _count_emojis(self, text: str) -> int:
        """计算表情符号数量"""
        import re
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
    
    def _topic_preference_similarity(
        self,
        posts1: list[dict[str, Any]],
        posts2: list[dict[str, Any]]
    ) -> float:
        """计算话题偏好相似度"""
        if not posts1 or not posts2:
            return 0.0
        
        topics1 = self._extract_topics(posts1)
        topics2 = self._extract_topics(posts2)
        
        return self._jaccard_similarity(topics1, topics2)
    
    def _extract_topics(self, posts: list[dict[str, Any]]) -> set[str]:
        """提取话题集合"""
        topics = set()
        for post in posts:
            content = post.get("content", post.get("text", ""))
            if content:
                import re
                hashtags = re.findall(r'#\w+', content.lower())
                topics.update(hashtags)
        return topics
    
    def _get_user_community(self, user_id: str, graph: Any) -> int:
        """获取用户所属社区"""
        try:
            node_data = graph.nodes[user_id]
            return node_data.get("community_id", -1)
        except Exception:
            return -1
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """计算Jaccard相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


@dataclass
class AccountReuseMatch:
    """账号复用匹配结果"""
    user1_id: str
    user2_id: str
    event1_id: str
    event2_id: str
    reuse_score: float
    match_type: str
    evidence: dict[str, Any] = field(default_factory=dict)


class AccountReuseDetector:
    """
    账号复用检测器
    
    检测跨事件的账号复用行为。
    """
    
    def __init__(self, reuse_threshold: float = 0.7) -> None:
        self._reuse_threshold = reuse_threshold
        self._entity_aligner = EntityAligner()
    
    def detect_reuse(
        self,
        event1_users: list[dict[str, Any]],
        event2_users: list[dict[str, Any]],
        event1_id: str = "event1",
        event2_id: str = "event2",
        posts1: Optional[dict[str, list[dict[str, Any]]]] = None,
        posts2: Optional[dict[str, list[dict[str, Any]]]] = None
    ) -> list[AccountReuseMatch]:
        """
        检测复用账号
        
        Args:
            event1_users: 事件1的用户列表
            event2_users: 事件2的用户列表
            event1_id: 事件1的ID
            event2_id: 事件2的ID
            posts1: 事件1的用户帖子
            posts2: 事件2的用户帖子
            
        Returns:
            复用账号匹配列表
        """
        reuse_matches = []
        
        attribute_alignments = self._entity_aligner.align_by_attribute(
            event1_users, event2_users
        )
        
        behavior_alignments = self._entity_aligner.align_by_behavior(
            event1_users, event2_users, posts1, posts2
        )
        
        combined_alignments = self._entity_aligner.compute_overall_alignment(
            attribute_alignments, behavior_alignments, []
        )
        
        for alignment in combined_alignments:
            reuse_score = self.calculate_reuse_score(alignment)
            
            if reuse_score >= self._reuse_threshold:
                match_type = self._determine_match_type(alignment)
                
                reuse_matches.append(AccountReuseMatch(
                    user1_id=alignment.user1_id,
                    user2_id=alignment.user2_id,
                    event1_id=event1_id,
                    event2_id=event2_id,
                    reuse_score=reuse_score,
                    match_type=match_type,
                    evidence={
                        "attribute_score": alignment.attribute_score,
                        "behavior_score": alignment.behavior_score,
                        "alignment_details": alignment.alignment_details
                    }
                ))
        
        return sorted(reuse_matches, key=lambda x: x.reuse_score, reverse=True)
    
    def calculate_reuse_score(self, alignment: EntityAlignment) -> float:
        """
        计算复用分数
        
        Args:
            alignment: 实体对齐结果
            
        Returns:
            复用分数 (0-1)
        """
        attribute_weight = 0.45
        behavior_weight = 0.35
        relation_weight = 0.20
        
        score = (
            attribute_weight * alignment.attribute_score +
            behavior_weight * alignment.behavior_score +
            relation_weight * alignment.relation_score
        )
        
        details = alignment.alignment_details
        
        if details.get("username_similarity", 0) > 0.9:
            score = min(score * 1.1, 1.0)
        
        if details.get("avatar_similarity", 0) > 0.95:
            score = min(score * 1.05, 1.0)
        
        return score
    
    def identify_account_assets(
        self,
        events: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        识别账号资产库
        
        分析多个事件，识别可能属于同一资产库的账号群组。
        
        Args:
            events: 事件列表，每个事件包含 event_id, users, posts
            
        Returns:
            账号资产库字典 (asset_id -> 账号列表)
        """
        if len(events) < 2:
            return {}
        
        all_matches: list[AccountReuseMatch] = []
        
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1 = events[i]
                event2 = events[j]
                
                matches = self.detect_reuse(
                    event1.get("users", []),
                    event2.get("users", []),
                    event1.get("event_id", f"event_{i}"),
                    event2.get("event_id", f"event_{j}"),
                    event1.get("posts"),
                    event2.get("posts")
                )
                
                all_matches.extend(matches)
        
        account_clusters = self._cluster_accounts(all_matches)
        
        return account_clusters
    
    def _determine_match_type(self, alignment: EntityAlignment) -> str:
        """确定匹配类型"""
        details = alignment.alignment_details
        
        if details.get("username_similarity", 0) > 0.95:
            return "same_username"
        
        if details.get("avatar_similarity", 0) > 0.95:
            return "same_avatar"
        
        if alignment.behavior_score > 0.8:
            return "behavioral_match"
        
        if alignment.attribute_score > 0.7:
            return "attribute_match"
        
        return "probable_match"
    
    def _cluster_accounts(
        self,
        matches: list[AccountReuseMatch]
    ) -> dict[str, list[dict[str, Any]]]:
        """聚类账号形成资产库"""
        clusters: dict[str, set[str]] = defaultdict(set)
        user_to_cluster: dict[str, str] = {}
        cluster_counter = 0
        
        for match in matches:
            u1_key = f"{match.event1_id}:{match.user1_id}"
            u2_key = f"{match.event2_id}:{match.user2_id}"
            
            if u1_key in user_to_cluster and u2_key in user_to_cluster:
                c1 = user_to_cluster[u1_key]
                c2 = user_to_cluster[u2_key]
                if c1 != c2:
                    clusters[c1].update(clusters[c2])
                    for u in clusters[c2]:
                        user_to_cluster[u] = c1
                    del clusters[c2]
            elif u1_key in user_to_cluster:
                c = user_to_cluster[u1_key]
                clusters[c].add(u2_key)
                user_to_cluster[u2_key] = c
            elif u2_key in user_to_cluster:
                c = user_to_cluster[u2_key]
                clusters[c].add(u1_key)
                user_to_cluster[u1_key] = c
            else:
                cluster_id = f"asset_{cluster_counter}"
                cluster_counter += 1
                clusters[cluster_id].add(u1_key)
                clusters[cluster_id].add(u2_key)
                user_to_cluster[u1_key] = cluster_id
                user_to_cluster[u2_key] = cluster_id
        
        result: dict[str, list[dict[str, Any]]] = {}
        for cluster_id, accounts in clusters.items():
            if len(accounts) >= 2:
                result[cluster_id] = [
                    {"account_key": acc, "event_id": acc.split(":")[0], "user_id": acc.split(":")[1]}
                    for acc in accounts
                ]
        
        return result


@dataclass
class BehaviorEvolution:
    """行为演化结果"""
    user_id: str
    events: list[str]
    activity_trend: list[float] = field(default_factory=list)
    sentiment_trend: list[float] = field(default_factory=list)
    influence_trend: list[float] = field(default_factory=list)
    strategy_changes: list[dict[str, Any]] = field(default_factory=list)
    role_transitions: list[dict[str, Any]] = field(default_factory=list)


class BehaviorEvolutionAnalyzer:
    """
    行为演化分析器
    
    分析用户跨事件的行为演化。
    """
    
    def __init__(self) -> None:
        self._strategy_change_threshold: float = 0.3
        self._role_transition_threshold: float = 0.4
    
    def analyze_evolution(
        self,
        user_id: str,
        events: list[dict[str, Any]]
    ) -> BehaviorEvolution:
        """
        分析用户行为演化
        
        Args:
            user_id: 用户ID
            events: 事件列表，每个事件包含 event_id, posts, graph, user_data
            
        Returns:
            BehaviorEvolution 对象
        """
        if len(events) < 2:
            return BehaviorEvolution(
                user_id=user_id,
                events=[e.get("event_id", "") for e in events]
            )
        
        activity_trend = []
        sentiment_trend = []
        influence_trend = []
        
        for event in events:
            posts = event.get("posts", {}).get(user_id, [])
            graph = event.get("graph")
            
            activity = self._calculate_activity_level(posts)
            activity_trend.append(activity)
            
            sentiment = self._calculate_sentiment(posts)
            sentiment_trend.append(sentiment)
            
            influence = self._calculate_influence(user_id, graph)
            influence_trend.append(influence)
        
        strategy_changes = []
        for i in range(1, len(events)):
            change = self.detect_strategy_change(
                user_id,
                events[i-1],
                events[i]
            )
            if change:
                strategy_changes.append(change)
        
        role_transitions = self.identify_role_transition(user_id, events)
        
        return BehaviorEvolution(
            user_id=user_id,
            events=[e.get("event_id", "") for e in events],
            activity_trend=activity_trend,
            sentiment_trend=sentiment_trend,
            influence_trend=influence_trend,
            strategy_changes=strategy_changes,
            role_transitions=role_transitions
        )
    
    def detect_strategy_change(
        self,
        user_id: str,
        event1: dict[str, Any],
        event2: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        检测策略变化
        
        Args:
            user_id: 用户ID
            event1: 第一个事件
            event2: 第二个事件
            
        Returns:
            策略变化信息，如果没有显著变化则返回None
        """
        posts1 = event1.get("posts", {}).get(user_id, [])
        posts2 = event2.get("posts", {}).get(user_id, [])
        
        if not posts1 or not posts2:
            return None
        
        strategy1 = self._extract_strategy_features(posts1)
        strategy2 = self._extract_strategy_features(posts2)
        
        changes = {}
        significant_change = False
        
        for key in strategy1:
            if key in strategy2:
                diff = abs(strategy1[key] - strategy2[key])
                if diff > self._strategy_change_threshold:
                    changes[key] = {
                        "before": strategy1[key],
                        "after": strategy2[key],
                        "change": strategy2[key] - strategy1[key]
                    }
                    significant_change = True
        
        if significant_change:
            return {
                "event1_id": event1.get("event_id", ""),
                "event2_id": event2.get("event_id", ""),
                "changes": changes,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return None
    
    def identify_role_transition(
        self,
        user_id: str,
        events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        识别角色转换
        
        Args:
            user_id: 用户ID
            events: 事件列表
            
        Returns:
            角色转换列表
        """
        transitions = []
        
        if len(events) < 2:
            return transitions
        
        roles = []
        for event in events:
            role = self._identify_user_role(user_id, event)
            roles.append({
                "event_id": event.get("event_id", ""),
                "role": role
            })
        
        for i in range(1, len(roles)):
            prev_role = roles[i-1]["role"]
            curr_role = roles[i]["role"]
            
            if prev_role != curr_role:
                transitions.append({
                    "from_event": roles[i-1]["event_id"],
                    "to_event": roles[i]["event_id"],
                    "from_role": prev_role,
                    "to_role": curr_role,
                    "transition_type": self._classify_role_transition(prev_role, curr_role)
                })
        
        return transitions
    
    def _calculate_activity_level(self, posts: list[dict[str, Any]]) -> float:
        """计算活跃度"""
        if not posts:
            return 0.0
        
        post_count = len(posts)
        
        total_interactions = 0
        for post in posts:
            total_interactions += post.get("likes", 0)
            total_interactions += post.get("comments", 0)
            total_interactions += post.get("shares", 0)
        
        return min(1.0, (post_count / 10.0) * 0.5 + (total_interactions / 100.0) * 0.5)
    
    def _calculate_sentiment(self, posts: list[dict[str, Any]]) -> float:
        """计算情感倾向"""
        if not posts:
            return 0.0
        
        sentiments = []
        for post in posts:
            sentiment = post.get("sentiment_score")
            if sentiment is not None:
                sentiments.append(sentiment)
        
        return np.mean(sentiments) if sentiments else 0.0
    
    def _calculate_influence(self, user_id: str, graph: Optional[Any]) -> float:
        """计算影响力"""
        if graph is None or not NETWORKX_AVAILABLE:
            return 0.0
        
        if not graph.has_node(user_id):
            return 0.0
        
        try:
            degree = graph.degree(user_id)
            max_degree = max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 1
            
            return degree / max_degree if max_degree > 0 else 0.0
        except Exception:
            return 0.0
    
    def _extract_strategy_features(self, posts: list[dict[str, Any]]) -> dict[str, float]:
        """提取策略特征"""
        if not posts:
            return {
                "posting_frequency": 0.0,
                "content_length_avg": 0.0,
                "hashtag_usage": 0.0,
                "mention_usage": 0.0,
                "url_usage": 0.0
            }
        
        import re
        
        total_length = 0
        hashtag_count = 0
        mention_count = 0
        url_count = 0
        
        for post in posts:
            content = post.get("content", post.get("text", ""))
            if content:
                total_length += len(content)
                hashtag_count += len(re.findall(r'#\w+', content))
                mention_count += len(re.findall(r'@[\w\u4e00-\u9fff]+', content))
                url_count += len(re.findall(r'https?://\S+', content))
        
        n_posts = len(posts)
        
        return {
            "posting_frequency": n_posts / max(n_posts, 1),
            "content_length_avg": total_length / n_posts / 1000.0,
            "hashtag_usage": hashtag_count / n_posts,
            "mention_usage": mention_count / n_posts,
            "url_usage": url_count / n_posts
        }
    
    def _identify_user_role(self, user_id: str, event: dict[str, Any]) -> str:
        """识别用户角色"""
        posts = event.get("posts", {}).get(user_id, [])
        graph = event.get("graph")
        
        activity = self._calculate_activity_level(posts)
        influence = self._calculate_influence(user_id, graph)
        
        if activity > 0.7 and influence > 0.6:
            return "leader"
        elif activity > 0.5 and influence > 0.4:
            return "amplifier"
        elif activity > 0.3:
            return "participant"
        elif activity > 0.1:
            return "observer"
        else:
            return "inactive"
    
    def _classify_role_transition(self, from_role: str, to_role: str) -> str:
        """分类角色转换类型"""
        role_hierarchy = {
            "inactive": 0,
            "observer": 1,
            "participant": 2,
            "amplifier": 3,
            "leader": 4
        }
        
        from_level = role_hierarchy.get(from_role, 0)
        to_level = role_hierarchy.get(to_role, 0)
        
        if to_level > from_level:
            return "promotion"
        elif to_level < from_level:
            return "demotion"
        else:
            return "lateral"


@dataclass
class CrossEventReport:
    """
    跨事件分析报告
    
    包含复用账号列表、行为演化分析和账号资产库信息。
    """
    report_id: str
    generated_at: datetime
    events_analyzed: list[str]
    reused_accounts: list[dict[str, Any]] = field(default_factory=list)
    behavior_evolutions: list[dict[str, Any]] = field(default_factory=list)
    account_assets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    graph_similarities: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "events_analyzed": self.events_analyzed,
            "reused_accounts": self.reused_accounts,
            "behavior_evolutions": self.behavior_evolutions,
            "account_assets": self.account_assets,
            "graph_similarities": self.graph_similarities,
            "summary": self.summary
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def get_high_risk_accounts(self, threshold: float = 0.8) -> list[dict[str, Any]]:
        """获取高风险账号"""
        return [
            acc for acc in self.reused_accounts
            if acc.get("reuse_score", 0) >= threshold
        ]
    
    def get_account_asset_summary(self) -> dict[str, int]:
        """获取账号资产库摘要"""
        return {
            asset_id: len(accounts)
            for asset_id, accounts in self.account_assets.items()
        }
    
    def get_evolution_summary(self) -> dict[str, Any]:
        """获取行为演化摘要"""
        if not self.behavior_evolutions:
            return {}
        
        total_users = len(self.behavior_evolutions)
        users_with_changes = sum(
            1 for e in self.behavior_evolutions
            if e.get("strategy_changes")
        )
        users_with_transitions = sum(
            1 for e in self.behavior_evolutions
            if e.get("role_transitions")
        )
        
        return {
            "total_users_analyzed": total_users,
            "users_with_strategy_changes": users_with_changes,
            "users_with_role_transitions": users_with_transitions,
            "strategy_change_rate": users_with_changes / total_users if total_users > 0 else 0,
            "role_transition_rate": users_with_transitions / total_users if total_users > 0 else 0
        }
    
    def generate_summary(self) -> None:
        """生成报告摘要"""
        self.summary = {
            "total_events": len(self.events_analyzed),
            "total_reused_accounts": len(self.reused_accounts),
            "high_risk_accounts_count": len(self.get_high_risk_accounts()),
            "total_account_assets": len(self.account_assets),
            "total_accounts_in_assets": sum(
                len(accounts) for accounts in self.account_assets.values()
            ),
            "evolution_summary": self.get_evolution_summary(),
            "avg_graph_similarity": np.mean([
                s.get("overall_similarity", 0) for s in self.graph_similarities
            ]) if self.graph_similarities else 0
        }


class CrossEventAnalyzer:
    """
    跨事件分析器
    
    整合所有跨事件分析功能的主类。
    """
    
    def __init__(self) -> None:
        self._graph_aligner = GraphAligner()
        self._entity_aligner = EntityAligner()
        self._reuse_detector = AccountReuseDetector()
        self._evolution_analyzer = BehaviorEvolutionAnalyzer()
    
    def analyze(
        self,
        events: list[dict[str, Any]],
        analyze_reuse: bool = True,
        analyze_evolution: bool = True,
        analyze_assets: bool = True
    ) -> CrossEventReport:
        """
        执行跨事件分析
        
        Args:
            events: 事件列表，每个事件应包含:
                   - event_id: 事件ID
                   - event_name: 事件名称
                   - users: 用户列表
                   - posts: 用户帖子字典 (user_id -> posts)
                   - graph: NetworkX图对象 (可选)
            analyze_reuse: 是否分析账号复用
            analyze_evolution: 是否分析行为演化
            analyze_assets: 是否识别账号资产库
            
        Returns:
            CrossEventReport 分析报告
        """
        if len(events) < 2:
            logger.warning("需要至少两个事件才能进行跨事件分析")
        
        report_id = f"cross_event_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        reused_accounts = []
        if analyze_reuse and len(events) >= 2:
            reused_accounts = self._analyze_all_reuse(events)
        
        behavior_evolutions = []
        if analyze_evolution:
            behavior_evolutions = self._analyze_all_evolutions(events)
        
        account_assets = {}
        if analyze_assets and len(events) >= 2:
            account_assets = self._reuse_detector.identify_account_assets(events)
        
        graph_similarities = self._analyze_graph_similarities(events)
        
        report = CrossEventReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            events_analyzed=[e.get("event_id", "") for e in events],
            reused_accounts=reused_accounts,
            behavior_evolutions=behavior_evolutions,
            account_assets=account_assets,
            graph_similarities=graph_similarities
        )
        
        report.generate_summary()
        
        return report
    
    def _analyze_all_reuse(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """分析所有事件对之间的账号复用"""
        all_matches: list[AccountReuseMatch] = []
        
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1 = events[i]
                event2 = events[j]
                
                matches = self._reuse_detector.detect_reuse(
                    event1.get("users", []),
                    event2.get("users", []),
                    event1.get("event_id", f"event_{i}"),
                    event2.get("event_id", f"event_{j}"),
                    event1.get("posts"),
                    event2.get("posts")
                )
                
                all_matches.extend(matches)
        
        unique_matches: dict[tuple[str, str], AccountReuseMatch] = {}
        for match in all_matches:
            key = (match.user1_id, match.user2_id)
            if key not in unique_matches or match.reuse_score > unique_matches[key].reuse_score:
                unique_matches[key] = match
        
        return [
            {
                "user1_id": m.user1_id,
                "user2_id": m.user2_id,
                "event1_id": m.event1_id,
                "event2_id": m.event2_id,
                "reuse_score": m.reuse_score,
                "match_type": m.match_type,
                "evidence": m.evidence
            }
            for m in sorted(unique_matches.values(), key=lambda x: x.reuse_score, reverse=True)
        ]
    
    def _analyze_all_evolutions(
        self,
        events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """分析所有用户的行为演化"""
        all_user_ids = set()
        for event in events:
            for user in event.get("users", []):
                user_id = user.get("user_id", user.get("id", ""))
                if user_id:
                    all_user_ids.add(user_id)
        
        evolutions = []
        for user_id in all_user_ids:
            evolution = self._evolution_analyzer.analyze_evolution(user_id, events)
            
            if evolution.strategy_changes or evolution.role_transitions:
                evolutions.append({
                    "user_id": evolution.user_id,
                    "events": evolution.events,
                    "activity_trend": evolution.activity_trend,
                    "sentiment_trend": evolution.sentiment_trend,
                    "influence_trend": evolution.influence_trend,
                    "strategy_changes": evolution.strategy_changes,
                    "role_transitions": evolution.role_transitions
                })
        
        return evolutions
    
    def _analyze_graph_similarities(
        self,
        events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """分析事件图之间的相似度"""
        similarities = []
        
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1 = events[i]
                event2 = events[j]
                
                graph1 = event1.get("graph")
                graph2 = event2.get("graph")
                
                if graph1 is not None and graph2 is not None:
                    snapshot1 = EventSnapshot(
                        event_id=event1.get("event_id", f"event_{i}"),
                        event_name=event1.get("event_name", ""),
                        time_window=(datetime.min, datetime.max),
                        graph=graph1
                    )
                    
                    snapshot2 = EventSnapshot(
                        event_id=event2.get("event_id", f"event_{j}"),
                        event_name=event2.get("event_name", ""),
                        time_window=(datetime.min, datetime.max),
                        graph=graph2
                    )
                    
                    sim = self._graph_aligner.calculate_similarity(snapshot1, snapshot2)
                    
                    similarities.append({
                        "event1_id": event1.get("event_id", f"event_{i}"),
                        "event2_id": event2.get("event_id", f"event_{j}"),
                        **sim
                    })
        
        return similarities
