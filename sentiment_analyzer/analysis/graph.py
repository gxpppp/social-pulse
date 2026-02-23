"""
图分析模块 - 社交网络图分析

提供社区发现、中心性分析、传播分析和同质性分析功能。
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from loguru import logger


@dataclass
class CommunityResult:
    """社区检测结果"""
    community_id: int
    nodes: list[str]
    size: int
    density: float
    top_users: list[str]


@dataclass
class InfluenceResult:
    """影响力分析结果"""
    node_id: str
    pagerank: float
    betweenness: float
    closeness: float
    degree: int
    influence_score: float


@dataclass
class Community:
    """
    社区类 - 存储社区信息
    
    Attributes:
        community_id: 社区唯一标识符
        members: 社区成员节点列表
        features: 社区特征字典
        created_at: 社区创建时间
        modularity: 社区模块度贡献
    """
    community_id: int
    members: list[str]
    features: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    modularity: float = 0.0
    
    @property
    def size(self) -> int:
        """社区大小"""
        return len(self.members)
    
    def add_member(self, node_id: str) -> None:
        """添加成员"""
        if node_id not in self.members:
            self.members.append(node_id)
    
    def remove_member(self, node_id: str) -> None:
        """移除成员"""
        if node_id in self.members:
            self.members.remove(node_id)
    
    def has_member(self, node_id: str) -> bool:
        """检查是否包含成员"""
        return node_id in self.members
    
    def get_feature(self, key: str, default: Any = None) -> Any:
        """获取社区特征"""
        return self.features.get(key, default)
    
    def set_feature(self, key: str, value: Any) -> None:
        """设置社区特征"""
        self.features[key] = value
    
    def similarity_to(self, other: "Community") -> float:
        """
        计算与另一个社区的相似度 (Jaccard系数)
        
        Args:
            other: 另一个社区对象
            
        Returns:
            Jaccard相似度系数 (0-1)
        """
        if not self.members or not other.members:
            return 0.0
        intersection = len(set(self.members) & set(other.members))
        union = len(set(self.members) | set(other.members))
        return intersection / union if union > 0 else 0.0
    
    def overlap_with(self, other: "Community") -> list[str]:
        """
        获取与另一个社区的重叠成员
        
        Args:
            other: 另一个社区对象
            
        Returns:
            重叠成员列表
        """
        return list(set(self.members) & set(other.members))
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "community_id": self.community_id,
            "members": self.members,
            "size": self.size,
            "features": self.features,
            "created_at": self.created_at.isoformat(),
            "modularity": self.modularity
        }


@dataclass
class PropagationPath:
    """
    传播路径类 - 存储传播路径信息
    
    Attributes:
        path_id: 路径唯一标识符
        nodes: 节点序列 (按传播顺序)
        timestamps: 时间戳序列 (与节点一一对应)
        source_id: 源节点ID
        target_id: 目标节点ID
        path_length: 路径长度
    """
    path_id: str
    nodes: list[str]
    timestamps: list[datetime] = field(default_factory=list)
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    
    @property
    def path_length(self) -> int:
        """路径长度 (边数)"""
        return max(0, len(self.nodes) - 1)
    
    @property
    def duration(self) -> float:
        """传播持续时间 (秒)"""
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]).total_seconds()
    
    def add_node(self, node_id: str, timestamp: Optional[datetime] = None) -> None:
        """
        添加节点到路径
        
        Args:
            node_id: 节点ID
            timestamp: 时间戳 (可选)
        """
        self.nodes.append(node_id)
        if timestamp:
            self.timestamps.append(timestamp)
        if self.source_id is None:
            self.source_id = node_id
        self.target_id = node_id
    
    def get_node_at(self, index: int) -> Optional[str]:
        """获取指定位置的节点"""
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None
    
    def get_timestamp_at(self, index: int) -> Optional[datetime]:
        """获取指定位置的时间戳"""
        if 0 <= index < len(self.timestamps):
            return self.timestamps[index]
        return None
    
    def contains_node(self, node_id: str) -> bool:
        """检查路径是否包含指定节点"""
        return node_id in self.nodes
    
    def subpath(self, start: int, end: int) -> "PropagationPath":
        """
        获取子路径
        
        Args:
            start: 起始索引
            end: 结束索引
            
        Returns:
            新的传播路径对象
        """
        sub_nodes = self.nodes[start:end]
        sub_timestamps = self.timestamps[start:end] if self.timestamps else []
        return PropagationPath(
            path_id=f"{self.path_id}_sub_{start}_{end}",
            nodes=sub_nodes,
            timestamps=sub_timestamps,
            source_id=sub_nodes[0] if sub_nodes else None,
            target_id=sub_nodes[-1] if sub_nodes else None
        )
    
    def to_visualization_data(self) -> dict[str, Any]:
        """
        转换为可视化数据格式
        
        Returns:
            可视化数据字典
        """
        edges = []
        for i in range(len(self.nodes) - 1):
            edge = {
                "source": self.nodes[i],
                "target": self.nodes[i + 1],
                "order": i
            }
            if i < len(self.timestamps) - 1:
                edge["time_diff"] = (
                    self.timestamps[i + 1] - self.timestamps[i]
                ).total_seconds() if self.timestamps[i + 1] and self.timestamps[i] else None
            edges.append(edge)
        
        return {
            "path_id": self.path_id,
            "nodes": self.nodes,
            "edges": edges,
            "source": self.source_id,
            "target": self.target_id,
            "length": self.path_length,
            "duration": self.duration,
            "timestamps": [ts.isoformat() if ts else None for ts in self.timestamps]
        }
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "path_id": self.path_id,
            "nodes": self.nodes,
            "timestamps": [ts.isoformat() if ts else None for ts in self.timestamps],
            "source_id": self.source_id,
            "target_id": self.target_id,
            "path_length": self.path_length,
            "duration": self.duration
        }


class GraphAnalyzer:
    """
    图分析器 - 提供社区发现、中心性分析、传播分析和同质性分析功能
    
    Attributes:
        _graph: 当前加载的图数据
        _communities: 检测到的社区列表
        _centrality_cache: 中心性计算缓存
    """
    
    def __init__(self) -> None:
        self._graph: Optional[nx.DiGraph] = None
        self._communities: list[Community] = []
        self._centrality_cache: dict[str, dict[str, float]] = {}
    
    def load_graph(self, graph: nx.DiGraph) -> None:
        """
        加载图数据
        
        Args:
            graph: NetworkX有向图对象
        """
        self._graph = graph
        self._centrality_cache.clear()
        logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    def build_interaction_graph(
        self,
        posts: list[dict[str, Any]],
        min_interactions: int = 1
    ) -> nx.DiGraph:
        """
        构建交互图
        
        Args:
            posts: 帖子数据列表
            min_interactions: 最小交互次数阈值
            
        Returns:
            构建的有向图
        """
        graph = nx.DiGraph()
        interactions: dict[tuple[str, str], int] = defaultdict(int)
        
        for post in posts:
            author = post.get("author_id")
            if not author:
                continue
            
            mentions = post.get("mentions", [])
            for mention in mentions:
                if author != mention:
                    interactions[(author, mention)] += 1
            
            parent_id = post.get("parent_id")
            if parent_id:
                for other_post in posts:
                    if other_post.get("post_id") == parent_id:
                        parent_author = other_post.get("author_id")
                        if parent_author and author != parent_author:
                            interactions[(author, parent_author)] += 1
        
        for (source, target), weight in interactions.items():
            if weight >= min_interactions:
                graph.add_edge(source, target, weight=weight)
        
        self._graph = graph
        return graph
    
    def detect_communities(
        self,
        min_size: int = 3
    ) -> list[CommunityResult]:
        """
        社区检测 (基础方法)
        
        Args:
            min_size: 最小社区大小
            
        Returns:
            社区检测结果列表
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return []
        
        undirected = self._graph.to_undirected()
        
        try:
            communities = nx.community.greedy_modularity_communities(undirected)
        except Exception:
            communities = [set(self._graph.nodes())]
        
        results = []
        for i, community in enumerate(communities):
            if len(community) < min_size:
                continue
            
            subgraph = self._graph.subgraph(community)
            density = nx.density(subgraph) if len(community) > 1 else 0
            
            degrees = dict(subgraph.degree())
            top_users = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:5]
            
            results.append(CommunityResult(
                community_id=i,
                nodes=list(community),
                size=len(community),
                density=density,
                top_users=top_users
            ))
        
        return sorted(results, key=lambda x: x.size, reverse=True)
    
    def detect_communities_louvain(
        self,
        graph: Optional[nx.DiGraph] = None,
        resolution: float = 1.0,
        random_state: Optional[int] = None
    ) -> list[Community]:
        """
        Louvain算法社区发现
        
        基于模块度优化的两阶段迭代算法:
        1. 局部移动阶段: 将每个节点移动到能最大增加模块度的社区
        2. 凝聚阶段: 将社区合并为超级节点，构建新图
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            resolution: 分辨率参数，值越大社区越小
            random_state: 随机种子
            
        Returns:
            检测到的社区列表
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            logger.warning("Graph is empty or not loaded")
            return []
        
        undirected = target_graph.to_undirected()
        
        try:
            partition = nx.community.louvain_communities(
                undirected,
                resolution=resolution,
                seed=random_state
            )
        except Exception as e:
            logger.error(f"Louvain community detection failed: {e}")
            return []
        
        communities = []
        total_nodes = target_graph.number_of_nodes()
        
        for i, members in enumerate(partition):
            if len(members) == 0:
                continue
            
            community = Community(
                community_id=i,
                members=list(members)
            )
            
            subgraph = target_graph.subgraph(members)
            community.set_feature("size", len(members))
            community.set_feature("density", nx.density(subgraph) if len(members) > 1 else 0)
            community.set_feature("internal_edges", subgraph.number_of_edges())
            
            degrees = dict(subgraph.degree())
            if degrees:
                community.set_feature("avg_degree", np.mean(list(degrees.values())))
                community.set_feature("max_degree", max(degrees.values()))
            
            modularity_contribution = len(members) / total_nodes
            community.modularity = modularity_contribution
            
            communities.append(community)
        
        self._communities = communities
        logger.info(f"Louvain detected {len(communities)} communities")
        return communities
    
    def detect_communities_lpa(
        self,
        graph: Optional[nx.DiGraph] = None,
        max_iterations: int = 100,
        async_update: bool = True
    ) -> list[Community]:
        """
        标签传播算法 (Label Propagation Algorithm) 社区发现
        
        基于标签传播的快速社区检测:
        - 异步更新: 每个节点立即使用邻居的最新标签
        - 快速收敛: 通常在几次迭代内收敛
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            max_iterations: 最大迭代次数
            async_update: 是否使用异步更新
            
        Returns:
            检测到的社区列表
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            logger.warning("Graph is empty or not loaded")
            return []
        
        undirected = target_graph.to_undirected()
        nodes = list(undirected.nodes())
        
        labels = {node: i for i, node in enumerate(nodes)}
        
        for iteration in range(max_iterations):
            changed = False
            np.random.shuffle(nodes)
            
            for node in nodes:
                neighbors = list(undirected.neighbors(node))
                if not neighbors:
                    continue
                
                neighbor_labels = defaultdict(int)
                for neighbor in neighbors:
                    neighbor_labels[labels[neighbor]] += 1
                
                if not neighbor_labels:
                    continue
                
                max_count = max(neighbor_labels.values())
                candidates = [label for label, count in neighbor_labels.items() if count == max_count]
                
                new_label = np.random.choice(candidates)
                
                if labels[node] != new_label:
                    labels[node] = new_label
                    changed = True
            
            if not changed:
                logger.info(f"LPA converged after {iteration + 1} iterations")
                break
        
        label_to_nodes: dict[int, list[str]] = defaultdict(list)
        for node, label in labels.items():
            label_to_nodes[label].append(node)
        
        communities = []
        for i, (label, members) in enumerate(label_to_nodes.items()):
            if len(members) == 0:
                continue
            
            community = Community(
                community_id=i,
                members=members
            )
            
            subgraph = target_graph.subgraph(members)
            community.set_feature("label", label)
            community.set_feature("size", len(members))
            community.set_feature("density", nx.density(subgraph) if len(members) > 1 else 0)
            
            communities.append(community)
        
        self._communities = communities
        logger.info(f"LPA detected {len(communities)} communities")
        return communities
    
    def detect_communities_gnn(
        self,
        graph: Optional[nx.DiGraph] = None,
        model: Optional[Any] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        overlap_threshold: float = 0.5
    ) -> list[Community]:
        """
        GNN社区发现
        
        基于图神经网络的端到端社区检测:
        - 支持端到端学习
        - 支持重叠社区检测
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            model: 预训练的GNN模型 (可选)
            hidden_dim: 隐藏层维度
            num_layers: GNN层数
            overlap_threshold: 重叠社区阈值
            
        Returns:
            检测到的社区列表
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            logger.warning("Graph is empty or not loaded")
            return []
        
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv
            from torch_geometric.utils import from_networkx
        except ImportError:
            logger.warning("PyTorch Geometric not available, falling back to Louvain")
            return self.detect_communities_louvain(target_graph)
        
        undirected = target_graph.to_undirected()
        
        node_list = list(undirected.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        for node in undirected.nodes():
            undirected.nodes[node]['x'] = [undirected.degree(node)]
        
        try:
            pyg_data = from_networkx(undirected)
        except Exception as e:
            logger.error(f"Failed to convert graph: {e}")
            return self.detect_communities_louvain(target_graph)
        
        num_nodes = len(node_list)
        num_communities = max(2, int(np.sqrt(num_nodes)))
        
        class GCNCommunityDetector(torch.nn.Module):
            def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
                super().__init__()
                self.convs = torch.nn.ModuleList()
                self.convs.append(GCNConv(in_channels, hidden_channels))
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.convs.append(GCNConv(hidden_channels, out_channels))
            
            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                for conv in self.convs[:-1]:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
                x = self.convs[-1](x, edge_index)
                return F.log_softmax(x, dim=1)
        
        if model is None:
            in_channels = pyg_data.x.shape[1] if pyg_data.x is not None else 1
            model = GCNCommunityDetector(in_channels, hidden_dim, num_communities, num_layers)
            model.eval()
        
        with torch.no_grad():
            if pyg_data.x is None:
                pyg_data.x = torch.ones((num_nodes, 1))
            logits = model(pyg_data.x, pyg_data.edge_index)
            probs = torch.exp(logits).numpy()
        
        communities = []
        node_communities: dict[str, list[int]] = defaultdict(list)
        
        for i, node in enumerate(node_list):
            for community_id in range(num_communities):
                if probs[i, community_id] > overlap_threshold:
                    node_communities[node].append(community_id)
        
        community_members: dict[int, list[str]] = defaultdict(list)
        for node, comm_ids in node_communities.items():
            for comm_id in comm_ids:
                community_members[comm_id].append(node)
        
        for community_id, members in community_members.items():
            if len(members) == 0:
                continue
            
            community = Community(
                community_id=community_id,
                members=members
            )
            
            subgraph = target_graph.subgraph(members)
            community.set_feature("size", len(members))
            community.set_feature("density", nx.density(subgraph) if len(members) > 1 else 0)
            community.set_feature("avg_membership", np.mean([len(node_communities[m]) for m in members]))
            community.set_feature("is_overlapping", True)
            
            communities.append(community)
        
        self._communities = communities
        logger.info(f"GNN detected {len(communities)} communities (with overlap support)")
        return communities
    
    def calculate_pagerank(
        self,
        graph: Optional[nx.DiGraph] = None,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> dict[str, float]:
        """
        计算PageRank值
        
        PageRank算法通过模拟随机游走来衡量节点的重要性。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            alpha: 阻尼系数 (0-1)
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            节点到PageRank值的映射
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return {}
        
        try:
            pagerank = nx.pagerank(
                target_graph,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol
            )
            
            if "pagerank" not in self._centrality_cache:
                self._centrality_cache["pagerank"] = {}
            self._centrality_cache["pagerank"].update(pagerank)
            
            return pagerank
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
            return {node: 0.0 for node in target_graph.nodes()}
    
    def calculate_betweenness(
        self,
        graph: Optional[nx.DiGraph] = None,
        normalized: bool = True,
        k: Optional[int] = None
    ) -> dict[str, float]:
        """
        计算介数中心性
        
        介数中心性衡量节点在最短路径上的重要性。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            normalized: 是否归一化
            k: 采样节点数 (用于大规模图)
            
        Returns:
            节点到介数中心性的映射
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return {}
        
        try:
            if k is not None and k < target_graph.number_of_nodes():
                betweenness = nx.betweenness_centrality_subset(
                    target_graph,
                    sources=np.random.choice(list(target_graph.nodes()), k, replace=False).tolist(),
                    targets=list(target_graph.nodes()),
                    normalized=normalized
                )
            else:
                betweenness = nx.betweenness_centrality(target_graph, normalized=normalized)
            
            if "betweenness" not in self._centrality_cache:
                self._centrality_cache["betweenness"] = {}
            self._centrality_cache["betweenness"].update(betweenness)
            
            return betweenness
        except Exception as e:
            logger.error(f"Betweenness calculation failed: {e}")
            return {node: 0.0 for node in target_graph.nodes()}
    
    def calculate_closeness(
        self,
        graph: Optional[nx.DiGraph] = None,
        wf_improved: bool = True
    ) -> dict[str, float]:
        """
        计算接近中心性
        
        接近中心性衡量节点到其他节点的平均距离。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            wf_improved: 是否使用Wasserman-Faust改进公式
            
        Returns:
            节点到接近中心性的映射
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return {}
        
        try:
            closeness = nx.closeness_centrality(target_graph, wf_improved=wf_improved)
            
            if "closeness" not in self._centrality_cache:
                self._centrality_cache["closeness"] = {}
            self._centrality_cache["closeness"].update(closeness)
            
            return closeness
        except Exception as e:
            logger.error(f"Closeness calculation failed: {e}")
            return {node: 0.0 for node in target_graph.nodes()}
    
    def calculate_eigenvector(
        self,
        graph: Optional[nx.DiGraph] = None,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> dict[str, float]:
        """
        计算特征向量中心性
        
        特征向量中心性衡量节点连接其他重要节点的能力。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            节点到特征向量中心性的映射
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return {}
        
        try:
            eigenvector = nx.eigenvector_centrality(
                target_graph,
                max_iter=max_iter,
                tol=tol
            )
            
            if "eigenvector" not in self._centrality_cache:
                self._centrality_cache["eigenvector"] = {}
            self._centrality_cache["eigenvector"].update(eigenvector)
            
            return eigenvector
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality did not converge, using approximation")
            try:
                undirected = target_graph.to_undirected()
                eigenvector = nx.eigenvector_centrality_numpy(undirected)
                return eigenvector
            except Exception as e:
                logger.error(f"Eigenvector approximation failed: {e}")
                return {node: 0.0 for node in target_graph.nodes()}
        except Exception as e:
            logger.error(f"Eigenvector calculation failed: {e}")
            return {node: 0.0 for node in target_graph.nodes()}
    
    def identify_key_nodes(
        self,
        graph: Optional[nx.DiGraph] = None,
        top_k: int = 10,
        weights: Optional[dict[str, float]] = None
    ) -> list[dict[str, Any]]:
        """
        识别关键节点
        
        综合多种中心性指标识别网络中的关键节点。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            top_k: 返回的顶部节点数量
            weights: 各中心性指标的权重 (默认: pagerank=0.3, betweenness=0.25, 
                     closeness=0.2, eigenvector=0.15, degree=0.1)
            
        Returns:
            关键节点列表，包含综合得分和各中心性值
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return []
        
        if weights is None:
            weights = {
                "pagerank": 0.3,
                "betweenness": 0.25,
                "closeness": 0.2,
                "eigenvector": 0.15,
                "degree": 0.1
            }
        
        pagerank = self.calculate_pagerank(target_graph)
        betweenness = self.calculate_betweenness(target_graph)
        closeness = self.calculate_closeness(target_graph)
        eigenvector = self.calculate_eigenvector(target_graph)
        degrees = dict(target_graph.degree())
        
        max_degree = max(degrees.values()) if degrees else 1
        
        node_scores = []
        for node in target_graph.nodes():
            pr = pagerank.get(node, 0)
            bc = betweenness.get(node, 0)
            cc = closeness.get(node, 0)
            ec = eigenvector.get(node, 0)
            dc = degrees.get(node, 0) / max_degree if max_degree > 0 else 0
            
            composite_score = (
                weights.get("pagerank", 0.3) * pr +
                weights.get("betweenness", 0.25) * bc +
                weights.get("closeness", 0.2) * cc +
                weights.get("eigenvector", 0.15) * ec +
                weights.get("degree", 0.1) * dc
            )
            
            node_scores.append({
                "node_id": node,
                "composite_score": composite_score,
                "pagerank": pr,
                "betweenness": bc,
                "closeness": cc,
                "eigenvector": ec,
                "degree": degrees.get(node, 0)
            })
        
        node_scores.sort(key=lambda x: x["composite_score"], reverse=True)
        return node_scores[:top_k]
    
    def trace_propagation(
        self,
        graph: Optional[nx.DiGraph] = None,
        source_id: str = "",
        max_depth: int = 10,
        time_attr: str = "timestamp"
    ) -> PropagationPath:
        """
        追踪传播路径
        
        从源节点开始追踪信息传播路径。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            source_id: 源节点ID
            max_depth: 最大追踪深度
            time_attr: 边上的时间属性名
            
        Returns:
            传播路径对象
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or not target_graph.has_node(source_id):
            logger.warning(f"Source node {source_id} not found in graph")
            return PropagationPath(path_id="empty", nodes=[])
        
        visited = {source_id}
        path_nodes = [source_id]
        timestamps = []
        
        current_nodes = [source_id]
        
        for depth in range(max_depth):
            next_nodes = []
            for node in current_nodes:
                for successor in target_graph.successors(node):
                    if successor not in visited:
                        visited.add(successor)
                        next_nodes.append(successor)
                        path_nodes.append(successor)
                        
                        edge_data = target_graph.get_edge_data(node, successor)
                        if edge_data and time_attr in edge_data:
                            ts = edge_data[time_attr]
                            if isinstance(ts, datetime):
                                timestamps.append(ts)
                            elif isinstance(ts, str):
                                try:
                                    timestamps.append(datetime.fromisoformat(ts))
                                except ValueError:
                                    pass
            
            if not next_nodes:
                break
            current_nodes = next_nodes
        
        path_id = f"prop_{source_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        return PropagationPath(
            path_id=path_id,
            nodes=path_nodes,
            timestamps=timestamps,
            source_id=source_id,
            target_id=path_nodes[-1] if path_nodes else None
        )
    
    def calculate_propagation_speed(
        self,
        graph: Optional[nx.DiGraph] = None,
        time_attr: str = "timestamp"
    ) -> dict[str, Any]:
        """
        计算传播速度
        
        分析网络中信息传播的速度特征。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            time_attr: 边上的时间属性名
            
        Returns:
            传播速度统计信息
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_edges() == 0:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "min_speed": 0.0,
                "total_propagations": 0
            }
        
        time_diffs = []
        edge_count = 0
        
        for u, v, data in target_graph.edges(data=True):
            if time_attr in data:
                ts = data[time_attr]
                if isinstance(ts, datetime):
                    time_diffs.append(ts.timestamp())
                    edge_count += 1
        
        if len(time_diffs) < 2:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "min_speed": 0.0,
                "total_propagations": target_graph.number_of_edges()
            }
        
        time_diffs.sort()
        intervals = []
        for i in range(1, len(time_diffs)):
            interval = time_diffs[i] - time_diffs[i-1]
            if interval > 0:
                intervals.append(interval)
        
        if not intervals:
            return {
                "avg_speed": 0.0,
                "max_speed": 0.0,
                "min_speed": 0.0,
                "total_propagations": edge_count
            }
        
        speeds = [1.0 / interval for interval in intervals]
        
        return {
            "avg_speed": np.mean(speeds),
            "max_speed": max(speeds),
            "min_speed": min(speeds),
            "median_speed": np.median(speeds),
            "std_speed": np.std(speeds),
            "total_propagations": edge_count,
            "avg_interval_seconds": np.mean(intervals)
        }
    
    def identify_influencers(
        self,
        graph: Optional[nx.DiGraph] = None,
        top_k: int = 20,
        method: str = "composite"
    ) -> list[dict[str, Any]]:
        """
        识别影响力节点
        
        识别网络中最具影响力的节点。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            top_k: 返回的顶部节点数量
            method: 识别方法 ("composite", "pagerank", "kshell", "degree")
            
        Returns:
            影响力节点列表
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return []
        
        influencers = []
        
        if method == "pagerank":
            scores = self.calculate_pagerank(target_graph)
            for node, score in scores.items():
                influencers.append({
                    "node_id": node,
                    "influence_score": score,
                    "method": "pagerank"
                })
        
        elif method == "kshell":
            try:
                undirected = target_graph.to_undirected()
                kshell = nx.core_number(undirected)
                for node, score in kshell.items():
                    influencers.append({
                        "node_id": node,
                        "influence_score": float(score),
                        "method": "kshell"
                    })
            except Exception as e:
                logger.error(f"K-shell calculation failed: {e}")
                return self.identify_influencers(target_graph, top_k, "pagerank")
        
        elif method == "degree":
            degrees = dict(target_graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            for node, degree in degrees.items():
                influencers.append({
                    "node_id": node,
                    "influence_score": degree / max_degree if max_degree > 0 else 0,
                    "method": "degree"
                })
        
        else:
            pagerank = self.calculate_pagerank(target_graph)
            betweenness = self.calculate_betweenness(target_graph)
            degrees = dict(target_graph.degree())
            
            try:
                undirected = target_graph.to_undirected()
                kshell = nx.core_number(undirected)
            except Exception:
                kshell = {node: 0 for node in target_graph.nodes()}
            
            max_degree = max(degrees.values()) if degrees else 1
            max_kshell = max(kshell.values()) if kshell else 1
            
            for node in target_graph.nodes():
                pr = pagerank.get(node, 0)
                bc = betweenness.get(node, 0)
                ks = kshell.get(node, 0) / max_kshell if max_kshell > 0 else 0
                dc = degrees.get(node, 0) / max_degree if max_degree > 0 else 0
                
                composite = 0.35 * pr + 0.25 * bc + 0.25 * ks + 0.15 * dc
                
                influencers.append({
                    "node_id": node,
                    "influence_score": composite,
                    "pagerank": pr,
                    "betweenness": bc,
                    "kshell": kshell.get(node, 0),
                    "degree": degrees.get(node, 0),
                    "method": "composite"
                })
        
        influencers.sort(key=lambda x: x["influence_score"], reverse=True)
        return influencers[:top_k]
    
    def build_propagation_tree(
        self,
        graph: Optional[nx.DiGraph] = None,
        post_id: str = "",
        time_attr: str = "timestamp"
    ) -> dict[str, Any]:
        """
        构建传播树
        
        以指定帖子为根节点构建传播树结构。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            post_id: 根帖子ID
            time_attr: 边上的时间属性名
            
        Returns:
            传播树结构
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or not target_graph.has_node(post_id):
            logger.warning(f"Post {post_id} not found in graph")
            return {
                "root": post_id,
                "nodes": [],
                "edges": [],
                "depth": 0,
                "total_nodes": 0
            }
        
        tree_nodes = {post_id}
        tree_edges = []
        levels = {post_id: 0}
        
        queue = [post_id]
        
        while queue:
            current = queue.pop(0)
            current_level = levels[current]
            
            for successor in target_graph.successors(current):
                if successor not in tree_nodes:
                    tree_nodes.add(successor)
                    levels[successor] = current_level + 1
                    queue.append(successor)
                    
                    edge_data = target_graph.get_edge_data(current, successor)
                    tree_edges.append({
                        "source": current,
                        "target": successor,
                        "level": current_level + 1,
                        "data": edge_data
                    })
        
        node_list = []
        for node in tree_nodes:
            node_data = dict(target_graph.nodes[node]) if target_graph.nodes[node] else {}
            node_list.append({
                "id": node,
                "level": levels[node],
                "data": node_data
            })
        
        return {
            "root": post_id,
            "nodes": node_list,
            "edges": tree_edges,
            "depth": max(levels.values()) if levels else 0,
            "total_nodes": len(tree_nodes),
            "total_edges": len(tree_edges),
            "levels": levels
        }
    
    def calculate_assortativity(
        self,
        graph: Optional[nx.DiGraph] = None,
        attribute: str = "degree"
    ) -> float:
        """
        计算同质性系数
        
        衡量网络中节点倾向于连接相似属性节点的程度。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            attribute: 属性名 ("degree" 表示度同质性)
            
        Returns:
            同质性系数 (-1到1，正值表示同质，负值表示异质)
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_edges() == 0:
            return 0.0
        
        try:
            if attribute == "degree":
                return nx.degree_assortativity_coefficient(target_graph)
            
            node_attrs = {}
            for node in target_graph.nodes():
                node_data = target_graph.nodes[node]
                if attribute in node_data:
                    node_attrs[node] = node_data[attribute]
            
            if len(node_attrs) < 2:
                logger.warning(f"Insufficient nodes with attribute '{attribute}'")
                return 0.0
            
            attr_values = set(node_attrs.values())
            if len(attr_values) < 2:
                logger.warning(f"All nodes have same value for attribute '{attribute}'")
                return 0.0
            
            if isinstance(list(attr_values)[0], (int, float)):
                return nx.numeric_assortativity_coefficient(
                    target_graph, attribute, nodes=node_attrs.keys()
                )
            else:
                return nx.attribute_assortativity_coefficient(
                    target_graph, attribute, nodes=node_attrs.keys()
                )
        except Exception as e:
            logger.error(f"Assortativity calculation failed: {e}")
            return 0.0
    
    def detect_attribute_clustering(
        self,
        graph: Optional[nx.DiGraph] = None,
        attribute: str = ""
    ) -> dict[str, Any]:
        """
        检测属性聚集
        
        分析特定属性在网络中的聚集模式。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            attribute: 要分析的属性名
            
        Returns:
            属性聚集分析结果
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return {
                "attribute": attribute,
                "clustering_coefficient": 0.0,
                "clusters": [],
                "homophily_ratio": 0.0
            }
        
        attr_groups: dict[Any, list[str]] = defaultdict(list)
        for node in target_graph.nodes():
            node_data = target_graph.nodes[node]
            if attribute in node_data:
                attr_groups[node_data[attribute]].append(node)
        
        if not attr_groups:
            return {
                "attribute": attribute,
                "clustering_coefficient": 0.0,
                "clusters": [],
                "homophily_ratio": 0.0
            }
        
        clusters = []
        total_internal_edges = 0
        total_external_edges = 0
        
        for attr_value, nodes in attr_groups.items():
            if len(nodes) < 2:
                continue
            
            subgraph = target_graph.subgraph(nodes)
            internal_edges = subgraph.number_of_edges()
            
            external_edges = 0
            for node in nodes:
                for neighbor in target_graph.neighbors(node):
                    if neighbor not in nodes:
                        external_edges += 1
            
            total_internal_edges += internal_edges
            total_external_edges += external_edges
            
            density = nx.density(subgraph) if len(nodes) > 1 else 0
            
            clusters.append({
                "attribute_value": attr_value,
                "nodes": nodes,
                "size": len(nodes),
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "density": density
            })
        
        total_edges = total_internal_edges + total_external_edges
        homophily_ratio = (
            total_internal_edges / total_edges if total_edges > 0 else 0.0
        )
        
        undirected = target_graph.to_undirected()
        avg_clustering = nx.average_clustering(undirected)
        
        return {
            "attribute": attribute,
            "clustering_coefficient": avg_clustering,
            "clusters": clusters,
            "homophily_ratio": homophily_ratio,
            "num_groups": len(attr_groups),
            "total_internal_edges": total_internal_edges,
            "total_external_edges": total_external_edges
        }
    
    def analyze_behavior_homophily(
        self,
        graph: Optional[nx.DiGraph] = None,
        behavior_attrs: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        分析行为同质性
        
        分析用户行为模式在网络中的同质性表现。
        
        Args:
            graph: 输入图 (可选，默认使用已加载的图)
            behavior_attrs: 行为属性列表 (默认: activity_level, sentiment, engagement)
            
        Returns:
            行为同质性分析结果
        """
        target_graph = graph if graph is not None else self._graph
        if target_graph is None or target_graph.number_of_nodes() == 0:
            return {
                "overall_homophily": 0.0,
                "attribute_homophily": {},
                "behavior_patterns": []
            }
        
        if behavior_attrs is None:
            behavior_attrs = ["activity_level", "sentiment", "engagement"]
        
        attribute_homophily = {}
        
        for attr in behavior_attrs:
            assortativity = self.calculate_assortativity(target_graph, attr)
            clustering = self.detect_attribute_clustering(target_graph, attr)
            
            attribute_homophily[attr] = {
                "assortativity": assortativity,
                "homophily_ratio": clustering.get("homophily_ratio", 0.0),
                "num_groups": clustering.get("num_groups", 0)
            }
        
        valid_values = [
            data["assortativity"] 
            for data in attribute_homophily.values() 
            if data["assortativity"] != 0.0
        ]
        overall_homophily = np.mean(valid_values) if valid_values else 0.0
        
        behavior_patterns = []
        for attr, data in attribute_homophily.items():
            if data["assortativity"] > 0.3:
                pattern = "strong_homophily"
            elif data["assortativity"] > 0.1:
                pattern = "moderate_homophily"
            elif data["assortativity"] < -0.1:
                pattern = "heterophily"
            else:
                pattern = "neutral"
            
            behavior_patterns.append({
                "attribute": attr,
                "pattern": pattern,
                "strength": abs(data["assortativity"])
            })
        
        return {
            "overall_homophily": overall_homophily,
            "attribute_homophily": attribute_homophily,
            "behavior_patterns": behavior_patterns
        }
    
    def calculate_influence_scores(
        self,
        top_n: int = 100
    ) -> list[InfluenceResult]:
        """
        计算影响力分数
        
        Args:
            top_n: 返回的顶部节点数量
            
        Returns:
            影响力分析结果列表
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return []
        
        try:
            pagerank = nx.pagerank(self._graph)
        except Exception:
            pagerank = {n: 0 for n in self._graph.nodes()}
        
        try:
            betweenness = nx.betweenness_centrality(self._graph)
        except Exception:
            betweenness = {n: 0 for n in self._graph.nodes()}
        
        try:
            closeness = nx.closeness_centrality(self._graph)
        except Exception:
            closeness = {n: 0 for n in self._graph.nodes()}
        
        degrees = dict(self._graph.degree())
        
        results = []
        for node in self._graph.nodes():
            influence_score = (
                pagerank.get(node, 0) * 0.4 +
                betweenness.get(node, 0) * 0.3 +
                closeness.get(node, 0) * 0.2 +
                (degrees.get(node, 0) / max(degrees.values()) if degrees else 0) * 0.1
            )
            
            results.append(InfluenceResult(
                node_id=node,
                pagerank=pagerank.get(node, 0),
                betweenness=betweenness.get(node, 0),
                closeness=closeness.get(node, 0),
                degree=degrees.get(node, 0),
                influence_score=influence_score
            ))
        
        return sorted(results, key=lambda x: x.influence_score, reverse=True)[:top_n]
    
    def find_key_spreaders(self, top_n: int = 10) -> list[dict[str, Any]]:
        """
        找出关键传播者
        
        Args:
            top_n: 返回的顶部节点数量
            
        Returns:
            关键传播者列表
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return []
        
        try:
            kshell = nx.core_number(self._graph.to_undirected())
        except Exception:
            kshell = {n: 0 for n in self._graph.nodes()}
        
        try:
            betweenness = nx.betweenness_centrality(self._graph)
        except Exception:
            betweenness = {n: 0 for n in self._graph.nodes()}
        
        spreader_scores = {}
        for node in self._graph.nodes():
            spreader_scores[node] = kshell.get(node, 0) * 0.5 + betweenness.get(node, 0) * 0.5
        
        top_spreaders = sorted(spreader_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [
            {
                "node_id": node,
                "spreader_score": score,
                "kshell": kshell.get(node, 0),
                "betweenness": betweenness.get(node, 0)
            }
            for node, score in top_spreaders
        ]
    
    def analyze_information_flow(
        self,
        source_node: str,
        max_depth: int = 3
    ) -> dict[str, Any]:
        """
        分析信息传播路径
        
        Args:
            source_node: 源节点ID
            max_depth: 最大分析深度
            
        Returns:
            信息流分析结果
        """
        if self._graph is None or not self._graph.has_node(source_node):
            return {"depth": 0, "nodes_reached": 0, "paths": []}
        
        visited = set([source_node])
        current_level = {source_node}
        all_paths = [[source_node]]
        
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                for neighbor in self._graph.successors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            
            if not next_level:
                break
            
            all_paths.append(list(next_level))
            current_level = next_level
        
        return {
            "source": source_node,
            "depth": len(all_paths) - 1,
            "nodes_reached": len(visited),
            "paths": all_paths
        }
    
    def calculate_network_metrics(self) -> dict[str, Any]:
        """
        计算网络指标
        
        Returns:
            网络指标字典
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return {}
        
        metrics = {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph),
        }
        
        try:
            metrics["average_clustering"] = nx.average_clustering(self._graph.to_undirected())
        except Exception:
            metrics["average_clustering"] = 0
        
        try:
            if nx.is_strongly_connected(self._graph):
                metrics["average_shortest_path"] = nx.average_shortest_path_length(self._graph)
            else:
                largest_scc = max(nx.strongly_connected_components(self._graph), key=len)
                subgraph = self._graph.subgraph(largest_scc)
                metrics["average_shortest_path"] = nx.average_shortest_path_length(subgraph)
        except Exception:
            metrics["average_shortest_path"] = float("inf")
        
        try:
            metrics["assortativity"] = nx.degree_assortativity_coefficient(self._graph)
        except Exception:
            metrics["assortativity"] = 0
        
        try:
            metrics["reciprocity"] = nx.reciprocity(self._graph)
        except Exception:
            metrics["reciprocity"] = 0
        
        return metrics
    
    def find_bridges(self) -> list[tuple[str, str]]:
        """
        找出桥接节点
        
        Returns:
            桥接边列表
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return []
        
        undirected = self._graph.to_undirected()
        
        try:
            bridges = list(nx.bridges(undirected))
            return bridges
        except Exception:
            return []
    
    def get_node_neighborhood(
        self,
        node: str,
        radius: int = 2
    ) -> dict[str, Any]:
        """
        获取节点邻域
        
        Args:
            node: 节点ID
            radius: 邻域半径
            
        Returns:
            节点邻域信息
        """
        if self._graph is None or not self._graph.has_node(node):
            return {"center": node, "nodes": [], "edges": []}
        
        try:
            ego_graph = nx.ego_graph(self._graph, node, radius=radius)
        except Exception:
            ego_graph = nx.DiGraph()
            ego_graph.add_node(node)
        
        return {
            "center": node,
            "nodes": list(ego_graph.nodes()),
            "edges": list(ego_graph.edges()),
            "node_count": ego_graph.number_of_nodes(),
            "edge_count": ego_graph.number_of_edges()
        }
    
    def get_communities(self) -> list[Community]:
        """
        获取已检测的社区
        
        Returns:
            社区列表
        """
        return self._communities
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._centrality_cache.clear()
        self._communities.clear()
        logger.info("Graph analyzer cache cleared")
