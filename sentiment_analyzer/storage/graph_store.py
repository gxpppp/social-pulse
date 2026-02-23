"""
图数据库存储模块 - 用于社交网络关系分析

支持 Neo4j 和 NetworkX 双模式存储。
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
from loguru import logger
from pydantic import BaseModel, Field

from .models import Platform


class NodeType(str, Enum):
    """节点类型枚举"""

    USER = "user"
    POST = "post"
    HASHTAG = "hashtag"
    URL = "url"


class RelationType(str, Enum):
    """关系类型枚举"""

    FOLLOWS = "follows"
    POSTS = "posts"
    RETWEETS = "retweets"
    MENTIONS = "mentions"
    CONTAINS_HASHTAG = "contains_hashtag"
    CONTAINS_URL = "contains_url"
    SIMILAR_TO = "similar_to"
    COORDINATED_WITH = "coordinated_with"
    REPLY_TO = "reply_to"
    QUOTES = "quotes"


class GraphConfig(BaseModel):
    """图数据库配置"""

    use_neo4j: bool = Field(
        default=False,
        description="是否使用Neo4j（否则使用NetworkX内存图）"
    )
    neo4j_uri: Optional[str] = Field(
        default=None,
        description="Neo4j连接URI"
    )
    neo4j_user: Optional[str] = Field(
        default=None,
        description="Neo4j用户名"
    )
    neo4j_password: Optional[str] = Field(
        default=None,
        description="Neo4j密码"
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j数据库名称"
    )
    persist_path: str = Field(
        default="./data/graph_store.json",
        description="NetworkX图持久化路径"
    )
    auto_persist: bool = Field(
        default=True,
        description="是否自动持久化NetworkX图"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_neo4j": False,
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_user": "neo4j",
                "neo4j_password": "password",
                "neo4j_database": "neo4j",
                "persist_path": "./data/graph_store.json",
                "auto_persist": True
            }
        }


@dataclass
class NodeInfo:
    """节点信息"""

    node_id: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RelationInfo:
    """关系信息"""

    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CommunityResult:
    """社区检测结果"""

    community_id: int
    nodes: list[str]
    size: int
    density: float
    modularity: float = 0.0


@dataclass
class CentralityResult:
    """中心性分析结果"""

    node_id: str
    score: float
    rank: int


class GraphBackend(ABC):
    """图存储后端抽象基类"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化后端"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭后端"""
        pass

    @abstractmethod
    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        properties: dict[str, Any]
    ) -> bool:
        """添加节点"""
        pass

    @abstractmethod
    async def get_node(
        self,
        node_id: str,
        node_type: Optional[NodeType] = None
    ) -> Optional[NodeInfo]:
        """获取节点"""
        pass

    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """更新节点属性"""
        pass

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        pass

    @abstractmethod
    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: dict[str, Any]
    ) -> bool:
        """添加关系"""
        pass

    @abstractmethod
    async def get_relations(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both"
    ) -> list[RelationInfo]:
        """获取关系"""
        pass

    @abstractmethod
    async def get_graph(self) -> nx.DiGraph:
        """获取图对象"""
        pass

    @abstractmethod
    async def get_statistics(self) -> dict[str, Any]:
        """获取统计信息"""
        pass


class NetworkXBackend(GraphBackend):
    """NetworkX 内存图后端"""

    def __init__(self, persist_path: str, auto_persist: bool = True) -> None:
        self.persist_path = Path(persist_path)
        self.auto_persist = auto_persist
        self._graph: nx.DiGraph = nx.DiGraph()
        self._lock = asyncio.Lock()
        self._is_initialized = False

    async def initialize(self) -> None:
        """初始化后端"""
        async with self._lock:
            if self._is_initialized:
                return

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            if self.persist_path.exists():
                await self._load_graph()

            self._is_initialized = True
            logger.info(
                f"NetworkX backend initialized with "
                f"{self._graph.number_of_nodes()} nodes"
            )

    async def close(self) -> None:
        """关闭后端"""
        async with self._lock:
            if self.auto_persist:
                await self._save_graph()
            logger.info("NetworkX backend closed")

    async def _load_graph(self) -> None:
        """从文件加载图数据"""
        try:
            loop = asyncio.get_event_loop()
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._graph = await loop.run_in_executor(
                None,
                lambda: nx.node_link_graph(data, directed=True)
            )
            logger.info(
                f"Loaded graph with {self._graph.number_of_nodes()} nodes"
            )
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            self._graph = nx.DiGraph()

    async def _save_graph(self) -> None:
        """保存图数据到文件"""
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                nx.node_link_data,
                self._graph
            )

            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(
                f"Saved graph with {self._graph.number_of_nodes()} nodes"
            )
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        properties: dict[str, Any]
    ) -> bool:
        """添加节点"""
        async with self._lock:
            try:
                now = datetime.utcnow().isoformat()
                node_props = {
                    "node_type": node_type.value,
                    "created_at": now,
                    "updated_at": now,
                    **properties
                }
                self._graph.add_node(node_id, **node_props)
                return True
            except Exception as e:
                logger.error(f"Failed to add node {node_id}: {e}")
                return False

    async def get_node(
        self,
        node_id: str,
        node_type: Optional[NodeType] = None
    ) -> Optional[NodeInfo]:
        """获取节点"""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return None

            data = self._graph.nodes[node_id]
            node_type_val = data.get("node_type")

            if node_type and node_type_val != node_type.value:
                return None

            return NodeInfo(
                node_id=node_id,
                node_type=NodeType(node_type_val),
                properties={k: v for k, v in data.items() if k != "node_type"},
                created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
            )

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """更新节点属性"""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return False

            try:
                self._graph.nodes[node_id].update(properties)
                self._graph.nodes[node_id]["updated_at"] = datetime.utcnow().isoformat()
                return True
            except Exception as e:
                logger.error(f"Failed to update node {node_id}: {e}")
                return False

    async def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return False

            try:
                self._graph.remove_node(node_id)
                return True
            except Exception as e:
                logger.error(f"Failed to delete node {node_id}: {e}")
                return False

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: dict[str, Any]
    ) -> bool:
        """添加关系"""
        async with self._lock:
            try:
                now = datetime.utcnow().isoformat()
                edge_props = {
                    "relation": relation_type.value,
                    "created_at": now,
                    **properties
                }
                self._graph.add_edge(source_id, target_id, **edge_props)
                return True
            except Exception as e:
                logger.error(
                    f"Failed to add relation {source_id}->{target_id}: {e}"
                )
                return False

    async def get_relations(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both"
    ) -> list[RelationInfo]:
        """获取关系"""
        async with self._lock:
            if not self._graph.has_node(node_id):
                return []

            relations = []

            if direction in ("out", "both"):
                for target in self._graph.successors(node_id):
                    edge_data = self._graph.edges[node_id, target]
                    rel_type = RelationType(edge_data.get("relation", "unknown"))

                    if relation_type and rel_type != relation_type:
                        continue

                    relations.append(RelationInfo(
                        source_id=node_id,
                        target_id=target,
                        relation_type=rel_type,
                        properties={k: v for k, v in edge_data.items() if k != "relation"}
                    ))

            if direction in ("in", "both"):
                for source in self._graph.predecessors(node_id):
                    edge_data = self._graph.edges[source, node_id]
                    rel_type = RelationType(edge_data.get("relation", "unknown"))

                    if relation_type and rel_type != relation_type:
                        continue

                    relations.append(RelationInfo(
                        source_id=source,
                        target_id=node_id,
                        relation_type=rel_type,
                        properties={k: v for k, v in edge_data.items() if k != "relation"}
                    ))

            return relations

    async def get_graph(self) -> nx.DiGraph:
        """获取图对象"""
        return self._graph

    async def get_statistics(self) -> dict[str, Any]:
        """获取统计信息"""
        async with self._lock:
            node_types: dict[str, int] = {}
            for _, data in self._graph.nodes(data=True):
                node_type = data.get("node_type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1

            edge_types: dict[str, int] = {}
            for _, _, data in self._graph.edges(data=True):
                relation = data.get("relation", "unknown")
                edge_types[relation] = edge_types.get(relation, 0) + 1

            return {
                "total_nodes": self._graph.number_of_nodes(),
                "total_edges": self._graph.number_of_edges(),
                "node_types": node_types,
                "edge_types": edge_types,
                "density": nx.density(self._graph) if self._graph.number_of_nodes() > 0 else 0
            }


class Neo4jBackend(GraphBackend):
    """Neo4j 后端"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j"
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._lock = asyncio.Lock()
        self._is_initialized = False

    async def initialize(self) -> None:
        """初始化后端"""
        async with self._lock:
            if self._is_initialized:
                return

            try:
                from neo4j import AsyncGraphDatabase

                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )

                await self._create_constraints()
                self._is_initialized = True
                logger.info("Neo4j backend initialized")
            except ImportError:
                raise RuntimeError(
                    "neo4j package not installed. "
                    "Install it with: pip install neo4j"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j: {e}")
                raise

    async def close(self) -> None:
        """关闭后端"""
        async with self._lock:
            if self._driver:
                await self._driver.close()
                self._driver = None
            logger.info("Neo4j backend closed")

    async def _create_constraints(self) -> None:
        """创建约束和索引"""
        constraint_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:User) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Post) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Hashtag) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Url) REQUIRE n.node_id IS UNIQUE",
        ]

        async with self._driver.session(database=self.database) as session:
            for query in constraint_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")

    async def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        properties: dict[str, Any]
    ) -> bool:
        """添加节点"""
        async with self._lock:
            try:
                now = datetime.utcnow().isoformat()
                props = {
                    "node_id": node_id,
                    "created_at": now,
                    "updated_at": now,
                    **properties
                }

                label = node_type.value.capitalize()
                query = f"""
                MERGE (n:{label} {{node_id: $node_id}})
                SET n += $properties
                """

                async with self._driver.session(database=self.database) as session:
                    await session.run(query, node_id=node_id, properties=props)
                return True
            except Exception as e:
                logger.error(f"Failed to add node {node_id}: {e}")
                return False

    async def get_node(
        self,
        node_id: str,
        node_type: Optional[NodeType] = None
    ) -> Optional[NodeInfo]:
        """获取节点"""
        async with self._lock:
            try:
                if node_type:
                    label = node_type.value.capitalize()
                    query = f"""
                    MATCH (n:{label} {{node_id: $node_id}})
                    RETURN n
                    """
                else:
                    query = """
                    MATCH (n {node_id: $node_id})
                    RETURN n
                    """

                async with self._driver.session(database=self.database) as session:
                    result = await session.run(query, node_id=node_id)
                    record = await result.single()

                    if not record:
                        return None

                    node = record["n"]
                    props = dict(node)

                    return NodeInfo(
                        node_id=node_id,
                        node_type=NodeType(props.pop("node_type", "user")),
                        properties=props,
                        created_at=datetime.fromisoformat(
                            props.get("created_at", datetime.utcnow().isoformat())
                        ),
                        updated_at=datetime.fromisoformat(
                            props.get("updated_at", datetime.utcnow().isoformat())
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to get node {node_id}: {e}")
                return None

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """更新节点属性"""
        async with self._lock:
            try:
                properties["updated_at"] = datetime.utcnow().isoformat()
                query = """
                MATCH (n {node_id: $node_id})
                SET n += $properties
                """

                async with self._driver.session(database=self.database) as session:
                    await session.run(query, node_id=node_id, properties=properties)
                return True
            except Exception as e:
                logger.error(f"Failed to update node {node_id}: {e}")
                return False

    async def delete_node(self, node_id: str) -> bool:
        """删除节点"""
        async with self._lock:
            try:
                query = """
                MATCH (n {node_id: $node_id})
                DETACH DELETE n
                """

                async with self._driver.session(database=self.database) as session:
                    await session.run(query, node_id=node_id)
                return True
            except Exception as e:
                logger.error(f"Failed to delete node {node_id}: {e}")
                return False

    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: dict[str, Any]
    ) -> bool:
        """添加关系"""
        async with self._lock:
            try:
                now = datetime.utcnow().isoformat()
                props = {"created_at": now, **properties}
                rel_type = relation_type.value.upper()

                query = f"""
                MATCH (source {{node_id: $source_id}})
                MATCH (target {{node_id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += $properties
                """

                async with self._driver.session(database=self.database) as session:
                    await session.run(
                        query,
                        source_id=source_id,
                        target_id=target_id,
                        properties=props
                    )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to add relation {source_id}->{target_id}: {e}"
                )
                return False

    async def get_relations(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both"
    ) -> list[RelationInfo]:
        """获取关系"""
        async with self._lock:
            try:
                rel_filter = ""
                if relation_type:
                    rel_filter = f":{relation_type.value.upper()}"

                if direction == "out":
                    query = f"""
                    MATCH (n {{node_id: $node_id}})-[r{rel_filter}]->(target)
                    RETURN n.node_id as source, target.node_id as target,
                           type(r) as rel_type, properties(r) as props
                    """
                elif direction == "in":
                    query = f"""
                    MATCH (source)-[r{rel_filter}]->(n {{node_id: $node_id}})
                    RETURN source.node_id as source, n.node_id as target,
                           type(r) as rel_type, properties(r) as props
                    """
                else:
                    query = f"""
                    MATCH (n {{node_id: $node_id}})-[r{rel_filter}]-(other)
                    RETURN
                        CASE WHEN startNode(r) = n THEN n.node_id ELSE other.node_id END as source,
                        CASE WHEN endNode(r) = n THEN n.node_id ELSE other.node_id END as target,
                        type(r) as rel_type, properties(r) as props
                    """

                async with self._driver.session(database=self.database) as session:
                    result = await session.run(query, node_id=node_id)
                    records = await result.data()

                relations = []
                for record in records:
                    rel_type_str = record["rel_type"].lower()
                    try:
                        rel_type = RelationType(rel_type_str)
                    except ValueError:
                        continue

                    relations.append(RelationInfo(
                        source_id=record["source"],
                        target_id=record["target"],
                        relation_type=rel_type,
                        properties=record.get("props", {})
                    ))

                return relations
            except Exception as e:
                logger.error(f"Failed to get relations for {node_id}: {e}")
                return []

    async def get_graph(self) -> nx.DiGraph:
        """获取图对象"""
        graph = nx.DiGraph()

        try:
            async with self._driver.session(database=self.database) as session:
                nodes_result = await session.run("MATCH (n) RETURN n.node_id as id, labels(n) as labels, properties(n) as props")
                nodes = await nodes_result.data()

                for node in nodes:
                    node_id = node["id"]
                    props = node.get("props", {})
                    labels = node.get("labels", [])
                    props["node_type"] = labels[0].lower() if labels else "unknown"
                    graph.add_node(node_id, **props)

                edges_result = await session.run("""
                    MATCH (source)-[r]->(target)
                    RETURN source.node_id as source, target.node_id as target,
                           type(r) as rel_type, properties(r) as props
                """)
                edges = await edges_result.data()

                for edge in edges:
                    props = edge.get("props", {})
                    props["relation"] = edge["rel_type"].lower()
                    graph.add_edge(edge["source"], edge["target"], **props)

        except Exception as e:
            logger.error(f"Failed to get graph: {e}")

        return graph

    async def get_statistics(self) -> dict[str, Any]:
        """获取统计信息"""
        try:
            async with self._driver.session(database=self.database) as session:
                nodes_result = await session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(*) as count
                """)
                node_counts = await nodes_result.data()

                edges_result = await session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(*) as count
                """)
                edge_counts = await edges_result.data()

                total_nodes = sum(n["count"] for n in node_counts)
                total_edges = sum(e["count"] for e in edge_counts)

                return {
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "node_types": {n["label"].lower(): n["count"] for n in node_counts},
                    "edge_types": {e["type"].lower(): e["count"] for e in edge_counts},
                    "density": total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


class GraphStore:
    """
    图数据库存储类

    支持 Neo4j 和 NetworkX 双模式存储，提供节点管理、关系管理、
    图查询和图算法功能。
    """

    def __init__(self, config: Optional[GraphConfig] = None) -> None:
        self.config = config or GraphConfig()
        self._backend: Optional[GraphBackend] = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """初始化图存储"""
        if self._is_initialized:
            return

        if self.config.use_neo4j:
            if not all([self.config.neo4j_uri, self.config.neo4j_user, self.config.neo4j_password]):
                raise ValueError("Neo4j configuration incomplete")

            self._backend = Neo4jBackend(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
        else:
            self._backend = NetworkXBackend(
                persist_path=self.config.persist_path,
                auto_persist=self.config.auto_persist
            )

        await self._backend.initialize()
        self._is_initialized = True
        logger.info(f"GraphStore initialized (Neo4j: {self.config.use_neo4j})")

    async def close(self) -> None:
        """关闭图存储"""
        if self._backend:
            await self._backend.close()
            self._backend = None
        self._is_initialized = False
        logger.info("GraphStore closed")

    def _ensure_initialized(self) -> None:
        """确保已初始化"""
        if not self._is_initialized or not self._backend:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")

    def _make_user_node_id(self, user_id: str, platform: str) -> str:
        """生成用户节点ID"""
        return f"{platform}:user:{user_id}"

    def _make_post_node_id(self, post_id: str, platform: str) -> str:
        """生成帖子节点ID"""
        return f"{platform}:post:{post_id}"

    def _make_hashtag_node_id(self, tag_text: str) -> str:
        """生成话题节点ID"""
        normalized_tag = tag_text.lower().strip("#")
        return f"hashtag:{normalized_tag}"

    def _make_url_node_id(self, url: str) -> str:
        """生成URL节点ID"""
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"url:{url_hash}"

    async def add_user_node(
        self,
        user_id: str,
        platform: str,
        properties: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        添加用户节点

        Args:
            user_id: 用户ID
            platform: 平台名称
            properties: 节点属性

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        node_id = self._make_user_node_id(user_id, platform)
        props = {
            "user_id": user_id,
            "platform": platform,
            **(properties or {})
        }

        return await self._backend.add_node(node_id, NodeType.USER, props)

    async def add_post_node(
        self,
        post_id: str,
        platform: str,
        properties: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        添加帖子节点

        Args:
            post_id: 帖子ID
            platform: 平台名称
            properties: 节点属性

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        node_id = self._make_post_node_id(post_id, platform)
        props = {
            "post_id": post_id,
            "platform": platform,
            **(properties or {})
        }

        return await self._backend.add_node(node_id, NodeType.POST, props)

    async def add_hashtag_node(self, tag_text: str) -> bool:
        """
        添加话题节点

        Args:
            tag_text: 话题文本

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        node_id = self._make_hashtag_node_id(tag_text)
        normalized_tag = tag_text.lower().strip("#")

        return await self._backend.add_node(
            node_id,
            NodeType.HASHTAG,
            {"tag_text": normalized_tag}
        )

    async def add_url_node(self, url: str) -> bool:
        """
        添加URL节点

        Args:
            url: URL地址

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        node_id = self._make_url_node_id(url)

        return await self._backend.add_node(
            node_id,
            NodeType.URL,
            {"url": url}
        )

    async def get_node(
        self,
        node_id: str,
        node_type: Optional[NodeType] = None
    ) -> Optional[NodeInfo]:
        """
        获取节点

        Args:
            node_id: 节点ID
            node_type: 节点类型（可选）

        Returns:
            节点信息，如果不存在则返回None
        """
        self._ensure_initialized()
        return await self._backend.get_node(node_id, node_type)

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """
        更新节点属性

        Args:
            node_id: 节点ID
            properties: 要更新的属性

        Returns:
            是否更新成功
        """
        self._ensure_initialized()
        return await self._backend.update_node(node_id, properties)

    async def delete_node(self, node_id: str) -> bool:
        """
        删除节点

        Args:
            node_id: 节点ID

        Returns:
            是否删除成功
        """
        self._ensure_initialized()
        return await self._backend.delete_node(node_id)

    async def add_follows_relation(
        self,
        follower_id: str,
        followee_id: str,
        platform: str,
        properties: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        添加关注关系

        Args:
            follower_id: 关注者ID
            followee_id: 被关注者ID
            platform: 平台名称
            properties: 关系属性

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_user_node_id(follower_id, platform)
        target_id = self._make_user_node_id(followee_id, platform)

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.FOLLOWS,
            properties or {}
        )

    async def add_posts_relation(
        self,
        user_id: str,
        post_id: str,
        platform: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        添加发帖关系

        Args:
            user_id: 用户ID
            post_id: 帖子ID
            platform: 平台名称
            timestamp: 发帖时间

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_user_node_id(user_id, platform)
        target_id = self._make_post_node_id(post_id, platform)

        props = {}
        if timestamp:
            props["timestamp"] = timestamp.isoformat()

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.POSTS,
            props
        )

    async def add_retweets_relation(
        self,
        user_id: str,
        post_id: str,
        platform: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        添加转发关系

        Args:
            user_id: 转发者ID
            post_id: 原帖ID
            platform: 平台名称
            timestamp: 转发时间

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_user_node_id(user_id, platform)
        target_id = self._make_post_node_id(post_id, platform)

        props = {}
        if timestamp:
            props["timestamp"] = timestamp.isoformat()

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.RETWEETS,
            props
        )

    async def add_mentions_relation(
        self,
        post_id: str,
        user_id: str,
        platform: str,
        position: Optional[int] = None
    ) -> bool:
        """
        添加提及关系

        Args:
            post_id: 帖子ID
            user_id: 被提及用户ID
            platform: 平台名称
            position: 提及位置

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_post_node_id(post_id, platform)
        target_id = self._make_user_node_id(user_id, platform)

        props = {}
        if position is not None:
            props["position"] = position

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.MENTIONS,
            props
        )

    async def add_contains_hashtag_relation(
        self,
        post_id: str,
        platform: str,
        tag_text: str
    ) -> bool:
        """
        添加话题关系

        Args:
            post_id: 帖子ID
            platform: 平台名称
            tag_text: 话题文本

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_post_node_id(post_id, platform)
        target_id = self._make_hashtag_node_id(tag_text)

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.CONTAINS_HASHTAG,
            {}
        )

    async def add_contains_url_relation(
        self,
        post_id: str,
        platform: str,
        url: str
    ) -> bool:
        """
        添加URL关系

        Args:
            post_id: 帖子ID
            platform: 平台名称
            url: URL地址

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_post_node_id(post_id, platform)
        target_id = self._make_url_node_id(url)

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.CONTAINS_URL,
            {}
        )

    async def add_similar_to_relation(
        self,
        post_id1: str,
        post_id2: str,
        platform: str,
        similarity: float
    ) -> bool:
        """
        添加相似关系

        Args:
            post_id1: 帖子1 ID
            post_id2: 帖子2 ID
            platform: 平台名称
            similarity: 相似度分数

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_post_node_id(post_id1, platform)
        target_id = self._make_post_node_id(post_id2, platform)

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.SIMILAR_TO,
            {"similarity": similarity}
        )

    async def add_coordinated_with_relation(
        self,
        user_id1: str,
        user_id2: str,
        platform: str,
        event_id: str,
        score: float
    ) -> bool:
        """
        添加协同关系

        Args:
            user_id1: 用户1 ID
            user_id2: 用户2 ID
            platform: 平台名称
            event_id: 事件ID
            score: 协同分数

        Returns:
            是否添加成功
        """
        self._ensure_initialized()

        source_id = self._make_user_node_id(user_id1, platform)
        target_id = self._make_user_node_id(user_id2, platform)

        return await self._backend.add_relation(
            source_id,
            target_id,
            RelationType.COORDINATED_WITH,
            {"event_id": event_id, "score": score}
        )

    async def get_user_network(
        self,
        user_id: str,
        platform: str,
        depth: int = 2
    ) -> dict[str, Any]:
        """
        获取用户网络

        Args:
            user_id: 用户ID
            platform: 平台名称
            depth: 网络深度

        Returns:
            包含节点和边的网络数据
        """
        self._ensure_initialized()

        start_node = self._make_user_node_id(user_id, platform)
        graph = await self._backend.get_graph()

        if not graph.has_node(start_node):
            return {"nodes": [], "edges": []}

        nodes = {start_node}
        edges = set()

        current_level = {start_node}
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in graph.successors(node):
                    if neighbor not in nodes:
                        next_level.add(neighbor)
                        nodes.add(neighbor)
                    edges.add((node, neighbor))

                for neighbor in graph.predecessors(node):
                    if neighbor not in nodes:
                        next_level.add(neighbor)
                        nodes.add(neighbor)
                    edges.add((neighbor, node))

            current_level = next_level
            if not current_level:
                break

        return {
            "nodes": [
                {"id": n, **graph.nodes[n]}
                for n in nodes
            ],
            "edges": [
                {"source": s, "target": t, **graph.edges[s, t]}
                for s, t in edges
            ]
        }

    async def get_post_propagation(
        self,
        post_id: str,
        platform: str
    ) -> dict[str, Any]:
        """
        获取帖子传播路径

        Args:
            post_id: 帖子ID
            platform: 平台名称

        Returns:
            传播路径数据
        """
        self._ensure_initialized()

        post_node = self._make_post_node_id(post_id, platform)
        graph = await self._backend.get_graph()

        if not graph.has_node(post_node):
            return {"depth": 0, "nodes": [], "edges": []}

        nodes = {post_node}
        edges = set()
        depth = 0

        current_level = {post_node}
        while current_level:
            next_level = set()
            for node in current_level:
                for neighbor in graph.predecessors(node):
                    edge_data = graph.edges[neighbor, node]
                    if edge_data.get("relation") in ["retweets", "reply_to", "quotes"]:
                        if neighbor not in nodes:
                            next_level.add(neighbor)
                            nodes.add(neighbor)
                        edges.add((neighbor, node))

            if next_level:
                depth += 1
            current_level = next_level

        return {
            "depth": depth,
            "nodes": list(nodes),
            "edges": list(edges)
        }

    async def find_communities(
        self,
        algorithm: str = "louvain",
        min_size: int = 3
    ) -> list[CommunityResult]:
        """
        社区发现

        Args:
            algorithm: 算法类型 ('louvain' 或 'lpa')
            min_size: 最小社区大小

        Returns:
            社区列表
        """
        self._ensure_initialized()

        graph = await self._backend.get_graph()

        if graph.number_of_nodes() == 0:
            return []

        undirected = graph.to_undirected()

        try:
            if algorithm == "louvain":
                communities = nx.community.louvain_communities(undirected)
            elif algorithm == "lpa":
                communities = list(nx.community.asyn_lpa_communities(undirected))
            else:
                communities = list(nx.community.greedy_modularity_communities(undirected))
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []

        results = []
        for i, community in enumerate(communities):
            if len(community) < min_size:
                continue

            subgraph = graph.subgraph(community)
            density = nx.density(subgraph) if len(community) > 1 else 0

            results.append(CommunityResult(
                community_id=i,
                nodes=list(community),
                size=len(community),
                density=density
            ))

        return sorted(results, key=lambda x: x.size, reverse=True)

    async def get_central_nodes(
        self,
        metric: str = "pagerank",
        top_n: int = 10,
        node_type: Optional[NodeType] = None
    ) -> list[CentralityResult]:
        """
        获取中心节点

        Args:
            metric: 中心性度量 ('pagerank', 'betweenness', 'closeness', 'degree')
            top_n: 返回数量
            node_type: 节点类型过滤

        Returns:
            中心节点列表
        """
        self._ensure_initialized()

        graph = await self._backend.get_graph()

        if graph.number_of_nodes() == 0:
            return []

        if node_type:
            node_ids = [
                n for n, data in graph.nodes(data=True)
                if data.get("node_type") == node_type.value
            ]
            if node_ids:
                graph = graph.subgraph(node_ids)

        try:
            if metric == "pagerank":
                scores = nx.pagerank(graph)
            elif metric == "betweenness":
                scores = nx.betweenness_centrality(graph)
            elif metric == "closeness":
                scores = nx.closeness_centrality(graph)
            elif metric == "degree":
                scores = dict(graph.degree())
            else:
                scores = nx.pagerank(graph)
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            CentralityResult(
                node_id=node,
                score=score,
                rank=i + 1
            )
            for i, (node, score) in enumerate(ranked)
        ]

    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        node_type: Optional[NodeType] = None
    ) -> list[str]:
        """
        最短路径

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            node_type: 节点类型前缀

        Returns:
            路径节点列表
        """
        self._ensure_initialized()

        graph = await self._backend.get_graph()

        if not graph.has_node(source_id) or not graph.has_node(target_id):
            return []

        try:
            path = nx.shortest_path(graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Shortest path calculation failed: {e}")
            return []

    async def calculate_pagerank(
        self,
        node_type: Optional[NodeType] = None
    ) -> dict[str, float]:
        """
        计算PageRank

        Args:
            node_type: 节点类型过滤

        Returns:
            PageRank分数字典
        """
        self._ensure_initialized()

        graph = await self._backend.get_graph()

        if graph.number_of_nodes() == 0:
            return {}

        if node_type:
            node_ids = [
                n for n, data in graph.nodes(data=True)
                if data.get("node_type") == node_type.value
            ]
            if node_ids:
                graph = graph.subgraph(node_ids)

        try:
            return nx.pagerank(graph)
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
            return {}

    async def calculate_betweenness_centrality(
        self,
        node_type: Optional[NodeType] = None,
        normalized: bool = True
    ) -> dict[str, float]:
        """
        计算介数中心性

        Args:
            node_type: 节点类型过滤
            normalized: 是否归一化

        Returns:
            介数中心性字典
        """
        self._ensure_initialized()

        graph = await self._backend.get_graph()

        if graph.number_of_nodes() == 0:
            return {}

        if node_type:
            node_ids = [
                n for n, data in graph.nodes(data=True)
                if data.get("node_type") == node_type.value
            ]
            if node_ids:
                graph = graph.subgraph(node_ids)

        try:
            return nx.betweenness_centrality(graph, normalized=normalized)
        except Exception as e:
            logger.error(f"Betweenness centrality calculation failed: {e}")
            return {}

    async def calculate_clustering_coefficient(
        self,
        node_type: Optional[NodeType] = None
    ) -> dict[str, float]:
        """
        计算聚类系数

        Args:
            node_type: 节点类型过滤

        Returns:
            聚类系数字典
        """
        self._ensure_initialized()

        graph = await self._backend.get_graph()

        if graph.number_of_nodes() == 0:
            return {}

        if node_type:
            node_ids = [
                n for n, data in graph.nodes(data=True)
                if data.get("node_type") == node_type.value
            ]
            if node_ids:
                graph = graph.subgraph(node_ids)

        try:
            undirected = graph.to_undirected()
            return nx.clustering(undirected)
        except Exception as e:
            logger.error(f"Clustering coefficient calculation failed: {e}")
            return {}

    async def detect_communities_louvain(
        self,
        min_size: int = 3
    ) -> list[CommunityResult]:
        """
        Louvain社区发现

        Args:
            min_size: 最小社区大小

        Returns:
            社区列表
        """
        return await self.find_communities(algorithm="louvain", min_size=min_size)

    async def detect_communities_lpa(
        self,
        min_size: int = 3
    ) -> list[CommunityResult]:
        """
        标签传播算法社区发现

        Args:
            min_size: 最小社区大小

        Returns:
            社区列表
        """
        return await self.find_communities(algorithm="lpa", min_size=min_size)

    async def get_statistics(self) -> dict[str, Any]:
        """
        获取图统计信息

        Returns:
            统计信息字典
        """
        self._ensure_initialized()
        return await self._backend.get_statistics()

    async def get_graph(self) -> nx.DiGraph:
        """
        获取图对象

        Returns:
            NetworkX有向图
        """
        self._ensure_initialized()
        return await self._backend.get_graph()

    async def clear(self) -> bool:
        """
        清空图数据

        Returns:
            是否成功
        """
        self._ensure_initialized()

        if isinstance(self._backend, NetworkXBackend):
            async with self._backend._lock:
                self._backend._graph = nx.DiGraph()
            return True
        elif isinstance(self._backend, Neo4jBackend):
            try:
                async with self._backend._driver.session(database=self._backend.database) as session:
                    await session.run("MATCH (n) DETACH DELETE n")
                return True
            except Exception as e:
                logger.error(f"Failed to clear Neo4j: {e}")
                return False

        return False


_graph_store: Optional[GraphStore] = None


async def get_graph_store(config: Optional[GraphConfig] = None) -> GraphStore:
    """
    获取全局图存储实例

    Args:
        config: 图配置

    Returns:
        GraphStore实例
    """
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore(config)
        await _graph_store.initialize()
    return _graph_store


async def close_graph_store() -> None:
    """关闭全局图存储"""
    global _graph_store
    if _graph_store:
        await _graph_store.close()
        _graph_store = None
