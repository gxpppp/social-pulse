"""
图神经网络模块 - 异构图神经网络模型

提供异构图数据结构、关系图卷积层、图注意力层、
异构图神经网络模型、模型训练器和推理功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import MessagePassing
    from torch_sparse import SparseTensor
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("PyTorch Geometric not fully available, some features may be limited")


class NodeType(str, Enum):
    """节点类型枚举"""
    USER = "user"
    POST = "post"
    HASHTAG = "hashtag"
    URL = "url"


class EdgeType(str, Enum):
    """边类型枚举"""
    FOLLOWS = "follows"
    POSTS = "posts"
    RETWEETS = "retweets"
    MENTIONS = "mentions"
    CONTAINS_HASHTAG = "contains_hashtag"
    CONTAINS_URL = "contains_url"
    SIMILAR_TO = "similar_to"
    COORDINATED_WITH = "coordinated_with"


class NodeLabel(int, Enum):
    """节点标签枚举"""
    NORMAL = 0
    TROLL = 1
    BOT = 2


@dataclass
class NodeInfo:
    """
    节点信息类
    
    Attributes:
        node_type: 节点类型
        node_id: 节点唯一标识符
        features: 节点特征向量
        label: 节点标签（可选）
        metadata: 节点元数据
    """
    node_type: NodeType
    node_id: str
    features: np.ndarray
    label: Optional[NodeLabel] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "node_type": self.node_type.value,
            "node_id": self.node_id,
            "features": self.features.tolist(),
            "label": self.label.value if self.label is not None else None,
            "metadata": self.metadata
        }


@dataclass
class EdgeInfo:
    """
    边信息类
    
    Attributes:
        edge_type: 边类型
        source_id: 源节点ID
        target_id: 目标节点ID
        features: 边特征向量（可选）
        weight: 边权重
        metadata: 边元数据
    """
    edge_type: EdgeType
    source_id: str
    target_id: str
    features: Optional[np.ndarray] = None
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "edge_type": self.edge_type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "features": self.features.tolist() if self.features is not None else None,
            "weight": self.weight,
            "metadata": self.metadata
        }


class HeteroGraph:
    """
    异构图数据结构
    
    支持多种节点类型和边类型的异构图。
    
    Attributes:
        nodes: 节点字典，按节点类型组织
        edges: 边字典，按边类型组织
        node_id_map: 节点ID到索引的映射
        feature_dim: 各节点类型的特征维度
    """
    
    def __init__(self) -> None:
        self._nodes: dict[NodeType, dict[str, NodeInfo]] = {
            node_type: {} for node_type in NodeType
        }
        self._edges: dict[EdgeType, list[EdgeInfo]] = {
            edge_type: [] for edge_type in EdgeType
        }
        self._node_id_map: dict[NodeType, dict[str, int]] = {
            node_type: {} for node_type in NodeType
        }
        self._feature_dim: dict[NodeType, int] = {}
        self._edge_feature_dim: dict[EdgeType, int] = {}
    
    @property
    def node_types(self) -> list[NodeType]:
        """获取所有节点类型"""
        return list(NodeType)
    
    @property
    def edge_types(self) -> list[EdgeType]:
        """获取所有边类型"""
        return list(EdgeType)
    
    def num_nodes(self, node_type: Optional[NodeType] = None) -> int:
        """
        获取节点数量
        
        Args:
            node_type: 节点类型，如果为None则返回所有节点数量
            
        Returns:
            节点数量
        """
        if node_type is not None:
            return len(self._nodes[node_type])
        return sum(len(nodes) for nodes in self._nodes.values())
    
    def num_edges(self, edge_type: Optional[EdgeType] = None) -> int:
        """
        获取边数量
        
        Args:
            edge_type: 边类型，如果为None则返回所有边数量
            
        Returns:
            边数量
        """
        if edge_type is not None:
            return len(self._edges[edge_type])
        return sum(len(edges) for edges in self._edges.values())
    
    def add_node(
        self,
        node_type: Union[NodeType, str],
        node_id: str,
        features: Union[np.ndarray, list[float]],
        label: Optional[Union[NodeLabel, int]] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        添加节点
        
        Args:
            node_type: 节点类型
            node_id: 节点唯一标识符
            features: 节点特征向量
            label: 节点标签（可选）
            metadata: 节点元数据（可选）
            
        Returns:
            是否成功添加
        """
        if isinstance(node_type, str):
            try:
                node_type = NodeType(node_type.lower())
            except ValueError:
                logger.error(f"Invalid node type: {node_type}")
                return False
        
        if isinstance(label, int):
            try:
                label = NodeLabel(label)
            except ValueError:
                label = None
        
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        elif not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)
        
        if node_id in self._nodes[node_type]:
            logger.warning(f"Node {node_id} already exists in {node_type.value}, updating features")
            self._nodes[node_type][node_id].features = features
            if label is not None:
                self._nodes[node_type][node_id].label = label
            return True
        
        idx = len(self._nodes[node_type])
        self._node_id_map[node_type][node_id] = idx
        
        if node_type not in self._feature_dim or self._feature_dim[node_type] == 0:
            self._feature_dim[node_type] = features.shape[0] if len(features.shape) > 0 else 1
        
        node_info = NodeInfo(
            node_type=node_type,
            node_id=node_id,
            features=features,
            label=label,
            metadata=metadata or {}
        )
        
        self._nodes[node_type][node_id] = node_info
        return True
    
    def add_edge(
        self,
        edge_type: Union[EdgeType, str],
        source_id: str,
        target_id: str,
        features: Optional[Union[np.ndarray, list[float]]] = None,
        weight: float = 1.0,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        添加边
        
        Args:
            edge_type: 边类型
            source_id: 源节点ID
            target_id: 目标节点ID
            features: 边特征向量（可选）
            weight: 边权重
            metadata: 边元数据（可选）
            
        Returns:
            是否成功添加
        """
        if isinstance(edge_type, str):
            try:
                edge_type = EdgeType(edge_type.lower())
            except ValueError:
                logger.error(f"Invalid edge type: {edge_type}")
                return False
        
        if features is not None:
            if isinstance(features, list):
                features = np.array(features, dtype=np.float32)
            elif not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            if edge_type not in self._edge_feature_dim or self._edge_feature_dim[edge_type] == 0:
                self._edge_feature_dim[edge_type] = features.shape[0] if len(features.shape) > 0 else 1
        
        edge_info = EdgeInfo(
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            features=features,
            weight=weight,
            metadata=metadata or {}
        )
        
        self._edges[edge_type].append(edge_info)
        return True
    
    def get_node(self, node_type: NodeType, node_id: str) -> Optional[NodeInfo]:
        """
        获取节点信息
        
        Args:
            node_type: 节点类型
            node_id: 节点ID
            
        Returns:
            节点信息，如果不存在则返回None
        """
        return self._nodes.get(node_type, {}).get(node_id)
    
    def get_node_index(self, node_type: NodeType, node_id: str) -> int:
        """
        获取节点索引
        
        Args:
            node_type: 节点类型
            node_id: 节点ID
            
        Returns:
            节点索引，如果不存在则返回-1
        """
        return self._node_id_map.get(node_type, {}).get(node_id, -1)
    
    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both"
    ) -> list[str]:
        """
        获取邻居节点
        
        Args:
            node_id: 节点ID
            edge_types: 边类型列表（可选）
            direction: 方向 ("in", "out", "both")
            
        Returns:
            邻居节点ID列表
        """
        neighbors = set()
        
        if edge_types is None:
            edge_types = list(EdgeType)
        
        for edge_type in edge_types:
            for edge in self._edges[edge_type]:
                if direction in ("out", "both") and edge.source_id == node_id:
                    neighbors.add(edge.target_id)
                if direction in ("in", "both") and edge.target_id == node_id:
                    neighbors.add(edge.source_id)
        
        return list(neighbors)
    
    def get_node_features(self, node_type: NodeType) -> Optional[np.ndarray]:
        """
        获取某类型所有节点的特征矩阵
        
        Args:
            node_type: 节点类型
            
        Returns:
            特征矩阵 [num_nodes, feature_dim]
        """
        nodes = self._nodes[node_type]
        if not nodes:
            return None
        
        feature_list = []
        for node_id in sorted(nodes.keys(), key=lambda x: self._node_id_map[node_type][x]):
            feature_list.append(nodes[node_id].features)
        
        return np.stack(feature_list, axis=0)
    
    def get_node_labels(self, node_type: NodeType) -> Optional[np.ndarray]:
        """
        获取某类型所有节点的标签
        
        Args:
            node_type: 节点类型
            
        Returns:
            标签数组 [num_nodes]
        """
        nodes = self._nodes[node_type]
        if not nodes:
            return None
        
        labels = []
        for node_id in sorted(nodes.keys(), key=lambda x: self._node_id_map[node_type][x]):
            label = nodes[node_id].label
            labels.append(label.value if label is not None else -1)
        
        return np.array(labels, dtype=np.int64)
    
    def get_edge_index(self, edge_type: EdgeType) -> Optional[np.ndarray]:
        """
        获取某类型所有边的索引矩阵
        
        Args:
            edge_type: 边类型
            
        Returns:
            边索引矩阵 [2, num_edges]
        """
        edges = self._edges[edge_type]
        if not edges:
            return None
        
        source_indices = []
        target_indices = []
        
        for edge in edges:
            source_idx = -1
            target_idx = -1
            
            for node_type in NodeType:
                if edge.source_id in self._node_id_map[node_type]:
                    source_idx = self._node_id_map[node_type][edge.source_id]
                    break
                if edge.target_id in self._node_id_map[node_type]:
                    target_idx = self._node_id_map[node_type][edge.target_id]
                    break
            
            if source_idx >= 0 and target_idx >= 0:
                source_indices.append(source_idx)
                target_indices.append(target_idx)
        
        if not source_indices:
            return None
        
        return np.array([source_indices, target_indices], dtype=np.int64)
    
    def to_pyg_data(self) -> Optional["HeteroData"]:
        """
        转换为PyTorch Geometric数据格式
        
        Returns:
            HeteroData对象，如果转换失败则返回None
        """
        if not PYG_AVAILABLE:
            logger.error("PyTorch Geometric not available")
            return None
        
        data = HeteroData()
        
        for node_type in NodeType:
            nodes = self._nodes[node_type]
            if not nodes:
                continue
            
            features = self.get_node_features(node_type)
            if features is not None:
                data[node_type.value].x = torch.from_numpy(features).float()
            
            labels = self.get_node_labels(node_type)
            if labels is not None and np.any(labels >= 0):
                data[node_type.value].y = torch.from_numpy(labels)
            
            node_ids = sorted(nodes.keys(), key=lambda x: self._node_id_map[node_type][x])
            data[node_type.value].node_id = node_ids
        
        for edge_type in EdgeType:
            edges = self._edges[edge_type]
            if not edges:
                continue
            
            edge_index = self.get_edge_index(edge_type)
            if edge_index is not None:
                source_type = self._get_node_type_for_edge(edge_type, "source")
                target_type = self._get_node_type_for_edge(edge_type, "target")
                
                if source_type and target_type:
                    edge_key = (source_type.value, edge_type.value, target_type.value)
                    data[edge_key].edge_index = torch.from_numpy(edge_index).long()
                    
                    weights = [e.weight for e in edges]
                    if weights:
                        data[edge_key].edge_weight = torch.tensor(weights, dtype=torch.float)
        
        return data
    
    def _get_node_type_for_edge(
        self,
        edge_type: EdgeType,
        position: str
    ) -> Optional[NodeType]:
        """
        根据边类型推断节点类型
        
        Args:
            edge_type: 边类型
            position: "source" 或 "target"
            
        Returns:
            节点类型
        """
        edge_node_map = {
            EdgeType.FOLLOWS: (NodeType.USER, NodeType.USER),
            EdgeType.POSTS: (NodeType.USER, NodeType.POST),
            EdgeType.RETWEETS: (NodeType.USER, NodeType.POST),
            EdgeType.MENTIONS: (NodeType.POST, NodeType.USER),
            EdgeType.CONTAINS_HASHTAG: (NodeType.POST, NodeType.HASHTAG),
            EdgeType.CONTAINS_URL: (NodeType.POST, NodeType.URL),
            EdgeType.SIMILAR_TO: (NodeType.POST, NodeType.POST),
            EdgeType.COORDINATED_WITH: (NodeType.USER, NodeType.USER),
        }
        
        if edge_type in edge_node_map:
            idx = 0 if position == "source" else 1
            return edge_node_map[edge_type][idx]
        
        return None
    
    def get_statistics(self) -> dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_nodes": self.num_nodes(),
            "total_edges": self.num_edges(),
            "nodes_by_type": {},
            "edges_by_type": {},
            "feature_dims": self._feature_dim,
            "edge_feature_dims": self._edge_feature_dim
        }
        
        for node_type in NodeType:
            stats["nodes_by_type"][node_type.value] = len(self._nodes[node_type])
        
        for edge_type in EdgeType:
            stats["edges_by_type"][edge_type.value] = len(self._edges[edge_type])
        
        return stats
    
    def clear(self) -> None:
        """清空图数据"""
        for node_type in NodeType:
            self._nodes[node_type].clear()
            self._node_id_map[node_type].clear()
        for edge_type in EdgeType:
            self._edges[edge_type].clear()
        self._feature_dim.clear()
        self._edge_feature_dim.clear()
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "nodes": {
                nt.value: {nid: node.to_dict() for nid, node in nodes.items()}
                for nt, nodes in self._nodes.items()
            },
            "edges": {
                et.value: [edge.to_dict() for edge in edges]
                for et, edges in self._edges.items()
            },
            "statistics": self.get_statistics()
        }


class RGCNLayer(MessagePassing if PYG_AVAILABLE else nn.Module):
    """
    关系图卷积层 (Relational Graph Convolutional Layer)
    
    支持多种关系类型的特定权重矩阵，实现关系感知的消息传递。
    
    公式: h_i^{(l+1)} = σ(Σ_{r∈R} Σ_{j∈N_r(i)} (1/c_{i,r}) W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)})
    
    Attributes:
        in_channels: 输入特征维度
        out_channels: 输出特征维度
        num_relations: 关系数量
        num_bases: 基矩阵数量（用于参数分解）
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        aggr: str = "mean",
        bias: bool = True,
        **kwargs
    ) -> None:
        if PYG_AVAILABLE:
            super().__init__(aggr=aggr, **kwargs)
        else:
            super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases is not None else num_relations
        
        if num_bases is not None and num_bases < num_relations:
            self.weight = nn.Parameter(
                torch.empty(num_bases, in_channels, out_channels)
            )
            self.comp = nn.Parameter(torch.empty(num_relations, num_bases))
        else:
            self.weight = nn.Parameter(
                torch.empty(num_relations, in_channels, out_channels)
            )
            self.comp = None
        
        self.root_weight = nn.Parameter(torch.empty(in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.root_weight)
        if self.comp is not None:
            nn.init.xavier_uniform_(self.comp)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            edge_weight: 边权重 [num_edges]（可选）
            
        Returns:
            更新后的节点特征 [num_nodes, out_channels]
        """
        if self.comp is not None:
            weight = torch.einsum("rb,bio->rio", self.comp, self.weight)
        else:
            weight = self.weight
        
        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not mask.any():
                continue
            
            rel_edge_index = edge_index[:, mask]
            rel_weight = edge_weight[mask] if edge_weight is not None else None
            
            rel_out = self._propagate(
                rel_edge_index, x=x, weight=weight[rel], edge_weight=rel_weight
            )
            out = out + rel_out
        
        out = out + torch.matmul(x, self.root_weight)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _propagate(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        消息传递
        
        Args:
            edge_index: 边索引
            x: 节点特征
            weight: 关系权重矩阵
            edge_weight: 边权重
            
        Returns:
            聚合后的特征
        """
        source, target = edge_index
        
        messages = torch.index_select(x, 0, source)
        messages = torch.matmul(messages, weight)
        
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(-1)
        
        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        out.index_add_(0, target, messages)
        
        degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        degree.index_add_(0, target, torch.ones(target.size(0), device=x.device))
        degree = degree.clamp(min=1).unsqueeze(-1)
        
        out = out / degree
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        weight: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        构造消息
        
        Args:
            x_j: 源节点特征
            weight: 权重矩阵
            edge_weight: 边权重
            
        Returns:
            消息向量
        """
        msg = torch.matmul(x_j, weight)
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
        return msg
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """
        更新节点特征
        
        Args:
            aggr_out: 聚合后的特征
            
        Returns:
            更新后的特征
        """
        return aggr_out
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"num_relations={self.num_relations}, "
                f"num_bases={self.num_bases})")


class GATLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    
    使用多头注意力机制聚合邻居信息。
    
    公式: h_i' = σ(Σ_{j∈N(i)} α_{ij} W h_j)
    其中 α_{ij} = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))
    
    Attributes:
        in_channels: 输入特征维度
        out_channels: 输出特征维度
        heads: 注意力头数量
        concat: 是否拼接多头输出
        negative_slope: LeakyReLU负斜率
        dropout: 注意力dropout率
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.weight = nn.Parameter(torch.empty(in_channels, heads * out_channels))
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            return_attention: 是否返回注意力权重
            
        Returns:
            更新后的节点特征 [num_nodes, heads * out_channels] 或 [num_nodes, out_channels]
            如果return_attention为True，还返回注意力权重
        """
        num_nodes = x.size(0)
        
        x = torch.matmul(x, self.weight).view(num_nodes, self.heads, self.out_channels)
        
        source, target = edge_index
        
        x_source = x[source]
        x_target = x[target]
        
        alpha = torch.cat([x_source, x_target], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        alpha = self._softmax(alpha, target, num_nodes)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        out = torch.zeros(num_nodes, self.heads, self.out_channels, 
                         device=x.device, dtype=x.dtype)
        out.index_add_(0, target, x_source * alpha.unsqueeze(-1))
        
        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention:
            return out, alpha
        return out
    
    def _softmax(
        self,
        alpha: torch.Tensor,
        target: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        对每个目标节点计算softmax
        
        Args:
            alpha: 注意力分数 [num_edges, heads]
            target: 目标节点索引 [num_edges]
            num_nodes: 节点数量
            
        Returns:
            归一化的注意力权重
        """
        alpha_max = torch.zeros(num_nodes, self.heads, device=alpha.device, dtype=alpha.dtype)
        alpha_max.index_reduce_(0, target, alpha, "amax", include_self=False)
        
        alpha = alpha - alpha_max[target]
        
        alpha_exp = torch.exp(alpha)
        
        alpha_sum = torch.zeros(num_nodes, self.heads, device=alpha.device, dtype=alpha.dtype)
        alpha_sum.index_add_(0, target, alpha_exp)
        
        alpha_sum = alpha_sum.clamp(min=1e-16)
        
        return alpha_exp / alpha_sum[target]
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"heads={self.heads}, "
                f"concat={self.concat})")


class HeteroGNN(nn.Module):
    """
    异构图神经网络模型
    
    架构: Input → RGCN Layer 1 → RGCN Layer 2 → GAT Layer → FC → Softmax
    
    支持节点分类任务（正常用户/水军/机器人）。
    
    Attributes:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出类别数
        num_relations: 关系数量
        num_heads: GAT注意力头数量
        dropout: Dropout率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_relations: int = 8,
        num_bases: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.rgcn1 = RGCNLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.rgcn2 = RGCNLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.gat = GATLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout
        )
        
        if use_batch_norm:
            self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化权重"""
        for module in [self.input_proj, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_type: 边类型 [num_edges]
            edge_weight: 边权重 [num_edges]（可选）
            return_embeddings: 是否返回节点嵌入
            
        Returns:
            分类logits [num_nodes, output_dim]
            如果return_embeddings为True，还返回节点嵌入
        """
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.rgcn1(x, edge_index, edge_type, edge_weight)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.rgcn2(x, edge_index, edge_type, edge_weight)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.gat(x, edge_index)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        embeddings = x
        
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        
        if return_embeddings:
            return x, embeddings
        
        return x
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重
            
        Returns:
            预测类别 [num_nodes]
        """
        logits = self.forward(x, edge_index, edge_type, edge_weight)
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测概率
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重
            
        Returns:
            预测概率 [num_nodes, output_dim]
        """
        logits = self.forward(x, edge_index, edge_type, edge_weight)
        return F.softmax(logits, dim=-1)
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"output_dim={self.output_dim}, "
                f"num_relations={self.num_relations}, "
                f"params={self.get_num_parameters():,})")


@dataclass
class TrainingConfig:
    """
    训练配置
    
    Attributes:
        learning_rate: 学习率
        weight_decay: 权重衰减
        epochs: 训练轮数
        patience: 早停耐心值
        batch_size: 批量大小（用于mini-batch训练）
        grad_clip: 梯度裁剪值
        lr_scheduler: 学习率调度器类型
        lr_decay_step: 学习率衰减步长
        lr_decay_gamma: 学习率衰减率
    """
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20
    batch_size: int = -1
    grad_clip: Optional[float] = 1.0
    lr_scheduler: str = "step"
    lr_decay_step: int = 50
    lr_decay_gamma: float = 0.5
    label_smoothing: float = 0.1


@dataclass
class TrainingResult:
    """
    训练结果
    
    Attributes:
        best_epoch: 最佳轮数
        best_val_loss: 最佳验证损失
        best_val_acc: 最佳验证准确率
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_accs: 验证准确率列表
        training_time: 训练时间（秒）
    """
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    training_time: float = 0.0


class GNNTrainer:
    """
    GNN模型训练器
    
    提供模型训练、评估、保存和加载功能。
    
    Attributes:
        model: GNN模型
        config: 训练配置
        device: 计算设备
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    
    def __init__(
        self,
        model: HeteroGNN,
        config: Optional[TrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = self._create_scheduler()
        
        self._best_state_dict: Optional[dict] = None
        self._training_result: Optional[TrainingResult] = None
    
    def _create_scheduler(self) -> Optional[Any]:
        """创建学习率调度器"""
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_gamma
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.lr_scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.lr_decay_gamma,
                patience=self.config.patience // 2
            )
        return None
    
    def train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> TrainingResult:
        """
        训练模型
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            labels: 节点标签
            train_mask: 训练集掩码
            val_mask: 验证集掩码（可选）
            edge_weight: 边权重（可选）
            verbose: 是否打印训练日志
            
        Returns:
            训练结果
        """
        import time
        
        result = TrainingResult()
        start_time = time.time()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        labels = labels.to(self.device)
        train_mask = train_mask.to(self.device)
        if val_mask is not None:
            val_mask = val_mask.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            logits = self.model(x, edge_index, edge_type, edge_weight)
            
            loss = criterion(logits[train_mask], labels[train_mask])
            
            loss.backward()
            
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss.item())
                else:
                    self.scheduler.step()
            
            train_loss = loss.item()
            result.train_losses.append(train_loss)
            
            if val_mask is not None:
                val_loss, val_acc = self._evaluate(
                    x, edge_index, edge_type, labels, val_mask, edge_weight, criterion
                )
                result.val_losses.append(val_loss)
                result.val_accs.append(val_acc)
                
                if val_loss < result.best_val_loss:
                    result.best_val_loss = val_loss
                    result.best_val_acc = val_acc
                    result.best_epoch = epoch
                    patience_counter = 0
                    self._best_state_dict = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Acc: {val_acc:.4f} | "
                        f"LR: {lr:.6f}"
                    )
                
                if patience_counter >= self.config.patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"LR: {lr:.6f}"
                    )
        
        result.training_time = time.time() - start_time
        self._training_result = result
        
        if self._best_state_dict is not None:
            self.model.load_state_dict({
                k: v.to(self.device) for k, v in self._best_state_dict.items()
            })
        
        if verbose:
            logger.info(
                f"Training completed in {result.training_time:.2f}s | "
                f"Best Epoch: {result.best_epoch} | "
                f"Best Val Loss: {result.best_val_loss:.4f} | "
                f"Best Val Acc: {result.best_val_acc:.4f}"
            )
        
        return result
    
    def _evaluate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        criterion: nn.Module
    ) -> tuple[float, float]:
        """
        评估模型
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            labels: 节点标签
            mask: 评估掩码
            edge_weight: 边权重
            criterion: 损失函数
            
        Returns:
            (损失, 准确率)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(x, edge_index, edge_type, edge_weight)
            loss = criterion(logits[mask], labels[mask])
            
            preds = torch.argmax(logits[mask], dim=-1)
            correct = (preds == labels[mask]).sum().item()
            acc = correct / mask.sum().item()
        
        return loss.item(), acc
    
    def evaluate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> dict[str, float]:
        """
        评估模型性能
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            labels: 节点标签
            mask: 评估掩码
            edge_weight: 边权重
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        labels = labels.to(self.device)
        mask = mask.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x, edge_index, edge_type, edge_weight)
            probs = F.softmax(logits[mask], dim=-1)
            preds = torch.argmax(logits[mask], dim=-1)
            
            correct = (preds == labels[mask]).sum().item()
            accuracy = correct / mask.sum().item()
            
            loss = F.cross_entropy(logits[mask], labels[mask])
            
            num_classes = logits.size(-1)
            if num_classes == 2:
                precision = self._compute_precision(preds, labels[mask], num_classes)
                recall = self._compute_recall(preds, labels[mask], num_classes)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
            else:
                precision = self._compute_macro_precision(preds, labels[mask], num_classes)
                recall = self._compute_macro_recall(preds, labels[mask], num_classes)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _compute_precision(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> float:
        """计算精确率（二分类）"""
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        return tp / (tp + fp + 1e-10)
    
    def _compute_recall(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> float:
        """计算召回率（二分类）"""
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        return tp / (tp + fn + 1e-10)
    
    def _compute_macro_precision(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> float:
        """计算宏平均精确率"""
        precisions = []
        for c in range(num_classes):
            tp = ((preds == c) & (labels == c)).sum().item()
            fp = ((preds == c) & (labels != c)).sum().item()
            precisions.append(tp / (tp + fp + 1e-10))
        return np.mean(precisions)
    
    def _compute_macro_recall(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> float:
        """计算宏平均召回率"""
        recalls = []
        for c in range(num_classes):
            tp = ((preds == c) & (labels == c)).sum().item()
            fn = ((preds != c) & (labels == c)).sum().item()
            recalls.append(tp / (tp + fn + 1e-10))
        return np.mean(recalls)
    
    def save_model(
        self,
        path: Union[str, Path],
        include_optimizer: bool = False,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        保存模型
        
        Args:
            path: 保存路径
            include_optimizer: 是否保存优化器状态
            metadata: 额外元数据
            
        Returns:
            是否成功保存
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "output_dim": self.model.output_dim,
                    "num_relations": self.model.num_relations,
                    "dropout": self.model.dropout,
                    "use_batch_norm": self.model.use_batch_norm
                },
                "training_config": {
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "epochs": self.config.epochs
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if include_optimizer:
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            
            if metadata:
                checkpoint["metadata"] = metadata
            
            if self._training_result is not None:
                checkpoint["training_result"] = {
                    "best_epoch": self._training_result.best_epoch,
                    "best_val_loss": self._training_result.best_val_loss,
                    "best_val_acc": self._training_result.best_val_acc,
                    "training_time": self._training_result.training_time
                }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(
        self,
        path: Union[str, Path],
        load_optimizer: bool = False,
        strict: bool = True
    ) -> bool:
        """
        加载模型
        
        Args:
            path: 模型路径
            load_optimizer: 是否加载优化器状态
            strict: 是否严格匹配参数
            
        Returns:
            是否成功加载
        """
        try:
            path = Path(path)
            
            if not path.exists():
                logger.error(f"Model file not found: {path}")
                return False
            
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            
            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if "training_result" in checkpoint:
                result = checkpoint["training_result"]
                self._training_result = TrainingResult(
                    best_epoch=result.get("best_epoch", 0),
                    best_val_loss=result.get("best_val_loss", float("inf")),
                    best_val_acc=result.get("best_val_acc", 0.0),
                    training_time=result.get("training_time", 0.0)
                )
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        extra_info: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        保存检查点
        
        Args:
            path: 检查点路径
            epoch: 当前轮数
            extra_info: 额外信息
            
        Returns:
            是否成功保存
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if extra_info:
            checkpoint["extra_info"] = extra_info
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved at epoch {epoch} to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(
        self,
        path: Union[str, Path]
    ) -> Optional[int]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            检查点的轮数，如果加载失败返回None
        """
        try:
            path = Path(path)
            
            if not path.exists():
                logger.error(f"Checkpoint file not found: {path}")
                return None
            
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            epoch = checkpoint.get("epoch", 0)
            logger.info(f"Checkpoint loaded from {path}, resuming from epoch {epoch}")
            return epoch
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_device_info(self) -> dict[str, Any]:
        """获取设备信息"""
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(self.device)
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(self.device)
            info["cuda_memory_reserved"] = torch.cuda.memory_reserved(self.device)
        
        return info


class GNNInference:
    """
    GNN模型推理器
    
    提供模型预测、嵌入提取和批量推理功能。
    
    Attributes:
        model: GNN模型
        device: 计算设备
    """
    
    def __init__(
        self,
        model: HeteroGNN,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None
    ) -> Optional["GNNInference"]:
        """
        从检查点加载推理器
        
        Args:
            path: 检查点路径
            device: 计算设备
            
        Returns:
            GNNInference实例，如果加载失败返回None
        """
        try:
            path = Path(path)
            checkpoint = torch.load(path, map_location="cpu")
            
            config = checkpoint.get("model_config", {})
            model = HeteroGNN(
                input_dim=config.get("input_dim", 128),
                hidden_dim=config.get("hidden_dim", 128),
                output_dim=config.get("output_dim", 3),
                num_relations=config.get("num_relations", 8),
                dropout=config.get("dropout", 0.5),
                use_batch_norm=config.get("use_batch_norm", True)
            )
            
            model.load_state_dict(checkpoint["model_state_dict"])
            
            return cls(model, device)
            
        except Exception as e:
            logger.error(f"Failed to load inference from checkpoint: {e}")
            return None
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        预测节点类别
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重
            
        Returns:
            预测类别数组 [num_nodes]
        """
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        preds = self.model.predict(x, edge_index, edge_type, edge_weight)
        
        return preds.cpu().numpy()
    
    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重
            
        Returns:
            预测概率数组 [num_nodes, num_classes]
        """
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        probs = self.model.predict_proba(x, edge_index, edge_type, edge_weight)
        
        return probs.cpu().numpy()
    
    @torch.no_grad()
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        获取节点嵌入
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            edge_weight: 边权重
            
        Returns:
            节点嵌入数组 [num_nodes, hidden_dim]
        """
        self.model.eval()
        
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        _, embeddings = self.model(x, edge_index, edge_type, edge_weight, return_embeddings=True)
        
        return embeddings.cpu().numpy()
    
    @torch.no_grad()
    def batch_predict(
        self,
        data_list: list[dict[str, torch.Tensor]],
        batch_size: int = 32
    ) -> list[np.ndarray]:
        """
        批量预测
        
        Args:
            data_list: 数据列表，每个元素包含 x, edge_index, edge_type 等张量
            batch_size: 批量大小
            
        Returns:
            预测结果列表
        """
        self.model.eval()
        predictions = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            for data in batch:
                pred = self.predict(
                    data["x"],
                    data["edge_index"],
                    data["edge_type"],
                    data.get("edge_weight")
                )
                predictions.append(pred)
        
        return predictions
    
    @torch.no_grad()
    def batch_get_embeddings(
        self,
        data_list: list[dict[str, torch.Tensor]],
        batch_size: int = 32
    ) -> list[np.ndarray]:
        """
        批量获取嵌入
        
        Args:
            data_list: 数据列表
            batch_size: 批量大小
            
        Returns:
            嵌入列表
        """
        self.model.eval()
        embeddings = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            for data in batch:
                emb = self.get_embeddings(
                    data["x"],
                    data["edge_index"],
                    data["edge_type"],
                    data.get("edge_weight")
                )
                embeddings.append(emb)
        
        return embeddings
    
    def predict_from_heterograph(
        self,
        graph: HeteroGraph,
        node_type: NodeType = NodeType.USER
    ) -> Optional[np.ndarray]:
        """
        从HeteroGraph预测
        
        Args:
            graph: 异构图
            node_type: 要预测的节点类型
            
        Returns:
            预测结果
        """
        pyg_data = graph.to_pyg_data()
        if pyg_data is None:
            logger.error("Failed to convert HeteroGraph to PyG data")
            return None
        
        if node_type.value not in pyg_data.node_types:
            logger.error(f"Node type {node_type.value} not found in graph")
            return None
        
        node_data = pyg_data[node_type.value]
        
        if not hasattr(node_data, "x"):
            logger.error(f"No features for node type {node_type.value}")
            return None
        
        all_edge_index = []
        all_edge_type = []
        all_edge_weight = []
        
        for edge_key in pyg_data.edge_types:
            edge_data = pyg_data[edge_key]
            if hasattr(edge_data, "edge_index"):
                edge_index = edge_data.edge_index
                all_edge_index.append(edge_index)
                
                edge_type_idx = list(EdgeType).index(EdgeType(edge_key[1]))
                all_edge_type.append(torch.full((edge_index.size(1),), edge_type_idx, dtype=torch.long))
                
                if hasattr(edge_data, "edge_weight"):
                    all_edge_weight.append(edge_data.edge_weight)
        
        if not all_edge_index:
            logger.error("No edges found in graph")
            return None
        
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_type = torch.cat(all_edge_type, dim=0)
        edge_weight = torch.cat(all_edge_weight) if all_edge_weight else None
        
        return self.predict(node_data.x, edge_index, edge_type, edge_weight)
    
    def get_label_name(self, label: int) -> str:
        """
        获取标签名称
        
        Args:
            label: 标签值
            
        Returns:
            标签名称
        """
        try:
            return NodeLabel(label).name
        except ValueError:
            return f"UNKNOWN_{label}"
    
    def get_prediction_report(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        labels: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> dict[str, Any]:
        """
        生成预测报告
        
        Args:
            x: 节点特征
            edge_index: 边索引
            edge_type: 边类型
            labels: 真实标签
            edge_weight: 边权重
            
        Returns:
            预测报告字典
        """
        preds = self.predict(x, edge_index, edge_type, edge_weight)
        probs = self.predict_proba(x, edge_index, edge_type, edge_weight)
        
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        
        correct = (preds == labels_np).sum()
        accuracy = correct / len(labels_np)
        
        report = {
            "total_samples": len(labels_np),
            "correct_predictions": int(correct),
            "accuracy": float(accuracy),
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "true_labels": labels_np.tolist(),
            "label_distribution": {}
        }
        
        for label in np.unique(np.concatenate([preds, labels_np])):
            label_name = self.get_label_name(int(label))
            pred_count = int((preds == label).sum())
            true_count = int((labels_np == label).sum())
            report["label_distribution"][label_name] = {
                "predicted": pred_count,
                "actual": true_count
            }
        
        return report
