"""
搜索引擎模块 - 全文搜索功能

支持 Elasticsearch 和 Whoosh 双模式的搜索引擎实现。
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import jieba
from loguru import logger

from ..storage.models import Post, User


class SearchBackend(str, Enum):
    """搜索引擎后端类型"""
    ELASTICSEARCH = "elasticsearch"
    WHOOSH = "whoosh"
    MEMORY = "memory"


class SortOrder(str, Enum):
    """排序方式"""
    RELEVANCE = "relevance"
    TIME_DESC = "time_desc"
    TIME_ASC = "time_asc"
    LIKES_DESC = "likes_desc"
    COMMENTS_DESC = "comments_desc"


@dataclass
class SearchConfig:
    """
    搜索引擎配置

    Attributes:
        use_elasticsearch: 是否使用 Elasticsearch
        es_hosts: ES 集群地址列表
        es_username: ES 用户名
        es_password: ES 密码
        es_timeout: ES 连接超时时间（秒）
        index_name: 索引名称前缀
        use_whoosh: 是否使用 Whoosh
        whoosh_index_dir: Whoosh 索引目录
        whoosh_analyzer: Whoosh 分析器类型
        enable_chinese_analyzer: 启用中文分词
        jieba_dict_path: jieba 自定义词典路径
        jieba_user_dict: jieba 用户词典路径
        auto_commit: 自动提交索引
        batch_size: 批量索引大小
        refresh_interval: 索引刷新间隔（秒）
    """
    use_elasticsearch: bool = False
    es_hosts: list[str] = field(default_factory=lambda: ["http://localhost:9200"])
    es_username: Optional[str] = None
    es_password: Optional[str] = None
    es_timeout: int = 30
    es_max_retries: int = 3
    es_retry_on_timeout: bool = True
    index_name: str = "social_posts"
    use_whoosh: bool = True
    whoosh_index_dir: str = "./data/search_index"
    whoosh_analyzer: str = "jieba"
    enable_chinese_analyzer: bool = True
    jieba_dict_path: Optional[str] = None
    jieba_user_dict: Optional[str] = None
    auto_commit: bool = True
    batch_size: int = 100
    refresh_interval: int = 1


@dataclass
class SearchFilter:
    """
    搜索过滤器

    Attributes:
        platforms: 平台过滤列表
        user_ids: 用户ID过滤列表
        start_time: 开始时间
        end_time: 结束时间
        min_likes: 最小点赞数
        max_likes: 最大点赞数
        min_comments: 最小评论数
        hashtags: 话题标签过滤列表
        mentions: 提及用户过滤列表
        language: 语言过滤
    """
    platforms: Optional[list[str]] = None
    user_ids: Optional[list[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_likes: Optional[int] = None
    max_likes: Optional[int] = None
    min_comments: Optional[int] = None
    hashtags: Optional[list[str]] = None
    mentions: Optional[list[str]] = None
    language: Optional[str] = None


@dataclass
class SearchResult:
    """
    搜索结果

    Attributes:
        total: 总结果数
        hits: 命中结果列表
        offset: 偏移量
        limit: 限制数
        took: 搜索耗时（毫秒）
        aggregations: 聚合结果
    """
    total: int
    hits: list[dict[str, Any]]
    offset: int = 0
    limit: int = 10
    took: float = 0.0
    aggregations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "total": self.total,
            "hits": self.hits,
            "offset": self.offset,
            "limit": self.limit,
            "took": self.took,
            "aggregations": self.aggregations,
        }


class BaseSearchBackend(ABC):
    """搜索引擎后端抽象基类"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化搜索引擎"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭搜索引擎"""
        pass

    @abstractmethod
    async def create_index(self) -> None:
        """创建索引"""
        pass

    @abstractmethod
    async def delete_index(self) -> None:
        """删除索引"""
        pass

    @abstractmethod
    async def index_post(self, post: Union[Post, dict[str, Any]]) -> bool:
        """索引帖子"""
        pass

    @abstractmethod
    async def index_posts(self, posts: list[Union[Post, dict[str, Any]]]) -> int:
        """批量索引帖子"""
        pass

    @abstractmethod
    async def index_user(self, user: Union[User, dict[str, Any]]) -> bool:
        """索引用户"""
        pass

    @abstractmethod
    async def delete_post(self, post_id: str, platform: Optional[str] = None) -> bool:
        """删除帖子索引"""
        pass

    @abstractmethod
    async def search_posts(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        sort: SortOrder = SortOrder.RELEVANCE,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """搜索帖子"""
        pass

    @abstractmethod
    async def search_users(
        self,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """搜索用户"""
        pass

    @abstractmethod
    async def search_by_hashtag(
        self,
        hashtag: str,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """按话题搜索"""
        pass

    @abstractmethod
    async def search_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> SearchResult:
        """按时间范围搜索"""
        pass

    @abstractmethod
    async def suggest(
        self,
        query: str,
        field: str = "content",
        limit: int = 10,
    ) -> list[str]:
        """搜索建议"""
        pass

    @abstractmethod
    async def aggregate_by_field(
        self,
        field: str,
        interval: Optional[str] = None,
        size: int = 10,
    ) -> dict[str, Any]:
        """聚合统计"""
        pass

    @abstractmethod
    async def get_trending_hashtags(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取热门话题"""
        pass

    @abstractmethod
    async def get_post_count_by_time(
        self,
        interval: str = "day",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """按时间统计帖子数"""
        pass


class ElasticsearchBackend(BaseSearchBackend):
    """Elasticsearch 搜索后端"""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._client: Any = None
        self._initialized = False

    async def initialize(self) -> None:
        """初始化 Elasticsearch 连接"""
        try:
            from elasticsearch import AsyncElasticsearch
            from elasticsearch.helpers import async_bulk

            self._async_bulk = async_bulk

            kwargs: dict[str, Any] = {
                "hosts": self.config.es_hosts,
                "timeout": self.config.es_timeout,
                "max_retries": self.config.es_max_retries,
                "retry_on_timeout": self.config.es_retry_on_timeout,
            }

            if self.config.es_username and self.config.es_password:
                kwargs["basic_auth"] = (self.config.es_username, self.config.es_password)

            self._client = AsyncElasticsearch(**kwargs)

            await self._client.ping()
            self._initialized = True
            logger.info("Elasticsearch backend initialized successfully")

        except ImportError:
            raise ImportError(
                "elasticsearch package is required for Elasticsearch backend. "
                "Install it with: pip install elasticsearch>=8.0.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise

    async def close(self) -> None:
        """关闭 Elasticsearch 连接"""
        if self._client:
            await self._client.close()
            self._initialized = False
            logger.info("Elasticsearch connection closed")

    async def create_index(self) -> None:
        """创建索引"""
        if not self._initialized:
            await self.initialize()

        post_index = f"{self.config.index_name}_posts"
        user_index = f"{self.config.index_name}_users"

        post_mapping = {
            "mappings": {
                "properties": {
                    "post_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "platform": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "social_analyzer",
                        "search_analyzer": "social_search_analyzer",
                    },
                    "content_raw": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "posted_at": {"type": "date"},
                    "likes_count": {"type": "integer"},
                    "shares_count": {"type": "integer"},
                    "comments_count": {"type": "integer"},
                    "hashtags": {"type": "keyword"},
                    "mentions": {"type": "keyword"},
                    "urls": {"type": "keyword"},
                    "collected_at": {"type": "date"},
                    "indexed_at": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "social_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "social_filter"],
                        },
                        "social_search_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"],
                        },
                        "chinese_analyzer": {
                            "type": "custom",
                            "tokenizer": "smartcn_tokenizer",
                            "filter": ["lowercase"],
                        },
                    },
                    "filter": {
                        "social_filter": {
                            "type": "pattern_replace",
                            "pattern": "[#@]([^\\s]+)",
                            "replacement": "$1",
                        }
                    },
                },
            },
        }

        user_mapping = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "platform": {"type": "keyword"},
                    "username": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "display_name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "bio": {"type": "text", "analyzer": "standard"},
                    "followers_count": {"type": "integer"},
                    "friends_count": {"type": "integer"},
                    "posts_count": {"type": "integer"},
                    "verified": {"type": "boolean"},
                    "created_at": {"type": "date"},
                    "indexed_at": {"type": "date"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
        }

        for index_name, mapping in [(post_index, post_mapping), (user_index, user_mapping)]:
            if not await self._client.indices.exists(index=index_name):
                await self._client.indices.create(index=index_name, body=mapping)
                logger.info(f"Created index: {index_name}")

    async def delete_index(self) -> None:
        """删除索引"""
        if not self._initialized:
            return

        for index in [f"{self.config.index_name}_posts", f"{self.config.index_name}_users"]:
            if await self._client.indices.exists(index=index):
                await self._client.indices.delete(index=index)
                logger.info(f"Deleted index: {index}")

    def _post_to_doc(self, post: Union[Post, dict[str, Any]]) -> dict[str, Any]:
        """将 Post 对象转换为文档"""
        if isinstance(post, Post):
            return {
                "post_id": post.post_id,
                "user_id": post.user_id,
                "platform": post.platform.value,
                "content": post.content or "",
                "content_raw": post.content,
                "language": post.language,
                "posted_at": post.posted_at,
                "likes_count": post.likes_count,
                "shares_count": post.shares_count,
                "comments_count": post.comments_count,
                "hashtags": post.hashtags,
                "mentions": post.mentions,
                "urls": post.urls,
                "collected_at": post.collected_at,
                "indexed_at": datetime.utcnow(),
            }
        return post

    def _user_to_doc(self, user: Union[User, dict[str, Any]]) -> dict[str, Any]:
        """将 User 对象转换为文档"""
        if isinstance(user, User):
            return {
                "user_id": user.user_id,
                "platform": user.platform.value,
                "username": user.username,
                "display_name": user.display_name,
                "bio": user.bio,
                "avatar_url": user.avatar_url,
                "followers_count": user.followers_count,
                "friends_count": user.friends_count,
                "posts_count": user.posts_count,
                "verified": user.verified,
                "created_at": user.created_at,
                "indexed_at": datetime.utcnow(),
            }
        return user

    async def index_post(self, post: Union[Post, dict[str, Any]]) -> bool:
        """索引单个帖子"""
        if not self._initialized:
            await self.initialize()

        try:
            doc = self._post_to_doc(post)
            post_id = doc.get("post_id", "")
            platform = doc.get("platform", "")

            await self._client.index(
                index=f"{self.config.index_name}_posts",
                id=f"{platform}:{post_id}",
                document=doc,
                refresh=self.config.auto_commit,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index post: {e}")
            return False

    async def index_posts(self, posts: list[Union[Post, dict[str, Any]]]) -> int:
        """批量索引帖子"""
        if not self._initialized:
            await self.initialize()

        actions = []
        for post in posts:
            doc = self._post_to_doc(post)
            post_id = doc.get("post_id", "")
            platform = doc.get("platform", "")
            actions.append({
                "_index": f"{self.config.index_name}_posts",
                "_id": f"{platform}:{post_id}",
                "_source": doc,
            })

        try:
            success, failed = await self._async_bulk(
                self._client,
                actions,
                refresh=self.config.auto_commit,
            )
            if failed:
                logger.warning(f"Failed to index {len(failed)} posts")
            return success
        except Exception as e:
            logger.error(f"Failed to bulk index posts: {e}")
            return 0

    async def index_user(self, user: Union[User, dict[str, Any]]) -> bool:
        """索引用户"""
        if not self._initialized:
            await self.initialize()

        try:
            doc = self._user_to_doc(user)
            user_id = doc.get("user_id", "")
            platform = doc.get("platform", "")

            await self._client.index(
                index=f"{self.config.index_name}_users",
                id=f"{platform}:{user_id}",
                document=doc,
                refresh=self.config.auto_commit,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to index user: {e}")
            return False

    async def delete_post(self, post_id: str, platform: Optional[str] = None) -> bool:
        """删除帖子索引"""
        if not self._initialized:
            return False

        try:
            doc_id = f"{platform}:{post_id}" if platform else post_id
            await self._client.delete(
                index=f"{self.config.index_name}_posts",
                id=doc_id,
                refresh=self.config.auto_commit,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete post: {e}")
            return False

    def _build_filter_query(self, filters: Optional[SearchFilter]) -> list[dict[str, Any]]:
        """构建过滤条件"""
        if not filters:
            return []

        filter_queries: list[dict[str, Any]] = []

        if filters.platforms:
            filter_queries.append({"terms": {"platform": filters.platforms}})

        if filters.user_ids:
            filter_queries.append({"terms": {"user_id": filters.user_ids}})

        if filters.start_time or filters.end_time:
            range_filter: dict[str, Any] = {}
            if filters.start_time:
                range_filter["gte"] = filters.start_time.isoformat()
            if filters.end_time:
                range_filter["lte"] = filters.end_time.isoformat()
            filter_queries.append({"range": {"posted_at": range_filter}})

        if filters.min_likes is not None or filters.max_likes is not None:
            range_filter: dict[str, Any] = {}
            if filters.min_likes is not None:
                range_filter["gte"] = filters.min_likes
            if filters.max_likes is not None:
                range_filter["lte"] = filters.max_likes
            filter_queries.append({"range": {"likes_count": range_filter}})

        if filters.min_comments is not None:
            filter_queries.append({"range": {"comments_count": {"gte": filters.min_comments}}})

        if filters.hashtags:
            filter_queries.append({"terms": {"hashtags": [h.lower() for h in filters.hashtags]}})

        if filters.mentions:
            filter_queries.append({"terms": {"mentions": filters.mentions}})

        if filters.language:
            filter_queries.append({"term": {"language": filters.language}})

        return filter_queries

    def _build_sort(self, sort: SortOrder) -> list[dict[str, Any]]:
        """构建排序条件"""
        sort_map = {
            SortOrder.RELEVANCE: [{"_score": "desc"}, {"posted_at": "desc"}],
            SortOrder.TIME_DESC: [{"posted_at": "desc"}],
            SortOrder.TIME_ASC: [{"posted_at": "asc"}],
            SortOrder.LIKES_DESC: [{"likes_count": "desc"}, {"posted_at": "desc"}],
            SortOrder.COMMENTS_DESC: [{"comments_count": "desc"}, {"posted_at": "desc"}],
        }
        return sort_map.get(sort, sort_map[SortOrder.RELEVANCE])

    async def search_posts(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        sort: SortOrder = SortOrder.RELEVANCE,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """搜索帖子"""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()

        must_queries: list[dict[str, Any]] = []

        if query:
            must_queries.append({
                "multi_match": {
                    "query": query,
                    "fields": ["content^2", "hashtags^3", "mentions^1.5"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            })
        else:
            must_queries.append({"match_all": {}})

        filter_queries = self._build_filter_query(filters)

        search_body: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": must_queries,
                    "filter": filter_queries,
                }
            },
            "sort": self._build_sort(sort),
            "from": offset,
            "size": limit,
            "track_total_hits": True,
        }

        try:
            response = await self._client.search(
                index=f"{self.config.index_name}_posts",
                body=search_body,
            )

            hits = [
                {
                    "score": hit.get("_score", 0),
                    **hit.get("_source", {}),
                }
                for hit in response.get("hits", {}).get("hits", [])
            ]

            took = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SearchResult(
                total=response.get("hits", {}).get("total", {}).get("value", 0),
                hits=hits,
                offset=offset,
                limit=limit,
                took=took,
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResult(total=0, hits=[], offset=offset, limit=limit)

    async def search_users(
        self,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """搜索用户"""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()

        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["username^2", "display_name^1.5", "bio"],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
            "size": limit,
        }

        try:
            response = await self._client.search(
                index=f"{self.config.index_name}_users",
                body=search_body,
            )

            hits = [
                {
                    "score": hit.get("_score", 0),
                    **hit.get("_source", {}),
                }
                for hit in response.get("hits", {}).get("hits", [])
            ]

            took = (datetime.utcnow() - start_time).total_seconds() * 1000

            return SearchResult(
                total=response.get("hits", {}).get("total", {}).get("value", 0),
                hits=hits,
                limit=limit,
                took=took,
            )
        except Exception as e:
            logger.error(f"User search failed: {e}")
            return SearchResult(total=0, hits=[], limit=limit)

    async def search_by_hashtag(
        self,
        hashtag: str,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """按话题搜索"""
        hashtag_clean = hashtag.lstrip("#").lower()

        filters = SearchFilter(hashtags=[hashtag_clean])
        return await self.search_posts(
            query="",
            filters=filters,
            sort=SortOrder.TIME_DESC,
            limit=limit,
            offset=offset,
        )

    async def search_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> SearchResult:
        """按时间范围搜索"""
        filters = SearchFilter(start_time=start_time, end_time=end_time)
        return await self.search_posts(
            query="",
            filters=filters,
            sort=SortOrder.TIME_DESC,
            limit=limit,
        )

    async def suggest(
        self,
        query: str,
        field: str = "content",
        limit: int = 10,
    ) -> list[str]:
        """搜索建议"""
        if not self._initialized:
            await self.initialize()

        search_body = {
            "query": {
                "match_phrase_prefix": {
                    field: {
                        "query": query,
                        "max_expansions": limit,
                    }
                }
            },
            "_source": [field],
            "size": limit,
        }

        try:
            response = await self._client.search(
                index=f"{self.config.index_name}_posts",
                body=search_body,
            )

            suggestions = set()
            for hit in response.get("hits", {}).get("hits", []):
                content = hit.get("_source", {}).get(field, "")
                if content:
                    suggestions.add(content[:100])

            return list(suggestions)[:limit]
        except Exception as e:
            logger.error(f"Suggest failed: {e}")
            return []

    async def aggregate_by_field(
        self,
        field: str,
        interval: Optional[str] = None,
        size: int = 10,
    ) -> dict[str, Any]:
        """聚合统计"""
        if not self._initialized:
            await self.initialize()

        agg_name = f"agg_{field}"

        if interval and field in ["posted_at", "created_at"]:
            agg_body = {
                agg_name: {
                    "date_histogram": {
                        "field": field,
                        "calendar_interval": interval,
                    }
                }
            }
        elif interval:
            agg_body = {
                agg_name: {
                    "histogram": {
                        "field": field,
                        "interval": int(interval),
                    }
                }
            }
        else:
            agg_body = {
                agg_name: {
                    "terms": {
                        "field": field,
                        "size": size,
                    }
                }
            }

        search_body = {
            "size": 0,
            "aggs": agg_body,
        }

        try:
            response = await self._client.search(
                index=f"{self.config.index_name}_posts",
                body=search_body,
            )
            return response.get("aggregations", {})
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return {}

    async def get_trending_hashtags(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取热门话题"""
        result = await self.aggregate_by_field("hashtags", size=limit)

        buckets = result.get("agg_hashtags", {}).get("buckets", [])
        return [
            {"hashtag": bucket.get("key"), "count": bucket.get("doc_count", 0)}
            for bucket in buckets
        ]

    async def get_post_count_by_time(
        self,
        interval: str = "day",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """按时间统计帖子数"""
        filters = None
        if start_time or end_time:
            filters = SearchFilter(start_time=start_time, end_time=end_time)

        filter_queries = self._build_filter_query(filters)

        search_body: dict[str, Any] = {
            "size": 0,
            "aggs": {
                "posts_over_time": {
                    "date_histogram": {
                        "field": "posted_at",
                        "calendar_interval": interval,
                    }
                }
            },
        }

        if filter_queries:
            search_body["query"] = {"bool": {"filter": filter_queries}}

        try:
            response = await self._client.search(
                index=f"{self.config.index_name}_posts",
                body=search_body,
            )

            buckets = response.get("aggregations", {}).get("posts_over_time", {}).get("buckets", [])
            return [
                {
                    "time": bucket.get("key_as_string"),
                    "count": bucket.get("doc_count", 0),
                }
                for bucket in buckets
            ]
        except Exception as e:
            logger.error(f"Time aggregation failed: {e}")
            return []


class WhooshBackend(BaseSearchBackend):
    """Whoosh 搜索后端（轻量级本地搜索）"""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._index_dir: Optional[Path] = None
        self._post_index: Any = None
        self._user_index: Any = None
        self._post_schema: Any = None
        self._user_schema: Any = None
        self._writer: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()

        self._inverted_index: dict[str, set[str]] = {}
        self._documents: dict[str, dict[str, Any]] = {}
        self._user_documents: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """初始化 Whoosh 索引"""
        try:
            from whoosh import index
            from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
            from whoosh.fields import ID, DATETIME, KEYWORD, NUMERIC, TEXT, Schema
            from whoosh.qparser import MultifieldParser, QueryParser

            self._index = index
            self._QueryParser = QueryParser
            self._MultifieldParser = MultifieldParser

            self._index_dir = Path(self.config.whoosh_index_dir)
            self._index_dir.mkdir(parents=True, exist_ok=True)

            self._post_schema = Schema(
                doc_id=ID(stored=True, unique=True),
                post_id=ID(stored=True),
                user_id=ID(stored=True),
                platform=KEYWORD(stored=True),
                content=TEXT(stored=True, analyzer=self._get_analyzer()),
                language=KEYWORD(stored=True),
                posted_at=DATETIME(stored=True),
                likes_count=NUMERIC(stored=True, sortable=True),
                shares_count=NUMERIC(stored=True),
                comments_count=NUMERIC(stored=True, sortable=True),
                hashtags=KEYWORD(stored=True, commas=True),
                mentions=KEYWORD(stored=True, commas=True),
                indexed_at=DATETIME(stored=True),
            )

            self._user_schema = Schema(
                doc_id=ID(stored=True, unique=True),
                user_id=ID(stored=True),
                platform=KEYWORD(stored=True),
                username=TEXT(stored=True),
                display_name=TEXT(stored=True),
                bio=TEXT(stored=True),
                followers_count=NUMERIC(stored=True),
                verified=KEYWORD(stored=True),
                indexed_at=DATETIME(stored=True),
            )

            if self._index.exists_in(self._index_dir, indexname="posts"):
                self._post_index = self._index.open_dir(self._index_dir, indexname="posts")
            else:
                self._post_index = self._index.create_in(
                    self._index_dir, self._post_schema, indexname="posts"
                )

            if self._index.exists_in(self._index_dir, indexname="users"):
                self._user_index = self._index.open_dir(self._index_dir, indexname="users")
            else:
                self._user_index = self._index.create_in(
                    self._index_dir, self._user_schema, indexname="users"
                )

            self._initialized = True
            logger.info("Whoosh backend initialized successfully")

        except ImportError:
            logger.warning("Whoosh not installed, falling back to memory-based search")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Whoosh: {e}")
            self._initialized = True

    def _get_analyzer(self) -> Any:
        """获取分析器"""
        try:
            from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter

            class ChineseAnalyzer:
                """中文分析器"""

                def __init__(self):
                    self.tokenizer = RegexTokenizer()
                    self.lower_filter = LowercaseFilter()

                def __call__(self, text: str):
                    tokens = list(jieba.cut(text))
                    for token in tokens:
                        if token.strip():
                            yield type(
                                "Token",
                                (),
                                {"text": token.lower(), "positions": False, "chars": False},
                            )()

            if self.config.enable_chinese_analyzer:
                return ChineseAnalyzer()
            else:
                return RegexTokenizer() | LowercaseFilter()

        except ImportError:
            return None

    async def close(self) -> None:
        """关闭索引"""
        self._initialized = False
        logger.info("Whoosh backend closed")

    async def create_index(self) -> None:
        """创建索引"""
        if not self._initialized:
            await self.initialize()

    async def delete_index(self) -> None:
        """删除索引"""
        import shutil

        if self._index_dir and self._index_dir.exists():
            shutil.rmtree(self._index_dir)
            self._documents.clear()
            self._user_documents.clear()
            self._inverted_index.clear()
            logger.info("Whoosh index deleted")

    def _post_to_doc(self, post: Union[Post, dict[str, Any]]) -> dict[str, Any]:
        """将 Post 对象转换为文档"""
        if isinstance(post, Post):
            return {
                "post_id": post.post_id,
                "user_id": post.user_id,
                "platform": post.platform.value,
                "content": post.content or "",
                "language": post.language,
                "posted_at": post.posted_at,
                "likes_count": post.likes_count,
                "shares_count": post.shares_count,
                "comments_count": post.comments_count,
                "hashtags": ",".join(post.hashtags) if post.hashtags else "",
                "mentions": ",".join(post.mentions) if post.mentions else "",
                "indexed_at": datetime.utcnow(),
            }
        return post

    def _user_to_doc(self, user: Union[User, dict[str, Any]]) -> dict[str, Any]:
        """将 User 对象转换为文档"""
        if isinstance(user, User):
            return {
                "user_id": user.user_id,
                "platform": user.platform.value,
                "username": user.username or "",
                "display_name": user.display_name or "",
                "bio": user.bio or "",
                "followers_count": user.followers_count,
                "verified": str(user.verified).lower(),
                "indexed_at": datetime.utcnow(),
            }
        return user

    def _tokenize(self, text: str) -> list[str]:
        """分词"""
        if self.config.enable_chinese_analyzer:
            tokens = list(jieba.cut(text))
        else:
            text = text.lower()
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
            tokens = text.split()

        return [t.strip() for t in tokens if t.strip()]

    async def index_post(self, post: Union[Post, dict[str, Any]]) -> bool:
        """索引帖子"""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            try:
                doc = self._post_to_doc(post)
                post_id = doc.get("post_id", "")
                platform = doc.get("platform", "")
                doc_id = f"{platform}:{post_id}"

                self._documents[doc_id] = {
                    **doc,
                    "doc_id": doc_id,
                }

                content = doc.get("content", "")
                tokens = self._tokenize(content)

                for token in tokens:
                    if token not in self._inverted_index:
                        self._inverted_index[token] = set()
                    self._inverted_index[token].add(doc_id)

                if self._post_index:
                    try:
                        writer = self._post_index.writer()
                        writer.update_document(
                            doc_id=doc_id,
                            post_id=post_id,
                            user_id=doc.get("user_id", ""),
                            platform=platform,
                            content=content,
                            language=doc.get("language", ""),
                            posted_at=doc.get("posted_at"),
                            likes_count=doc.get("likes_count", 0),
                            shares_count=doc.get("shares_count", 0),
                            comments_count=doc.get("comments_count", 0),
                            hashtags=doc.get("hashtags", ""),
                            mentions=doc.get("mentions", ""),
                            indexed_at=datetime.utcnow(),
                        )
                        writer.commit()
                    except Exception as e:
                        logger.warning(f"Whoosh index failed, using memory: {e}")

                return True
            except Exception as e:
                logger.error(f"Failed to index post: {e}")
                return False

    async def index_posts(self, posts: list[Union[Post, dict[str, Any]]]) -> int:
        """批量索引帖子"""
        count = 0
        for post in posts:
            if await self.index_post(post):
                count += 1
        return count

    async def index_user(self, user: Union[User, dict[str, Any]]) -> bool:
        """索引用户"""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            try:
                doc = self._user_to_doc(user)
                user_id = doc.get("user_id", "")
                platform = doc.get("platform", "")
                doc_id = f"{platform}:{user_id}"

                self._user_documents[doc_id] = {
                    **doc,
                    "doc_id": doc_id,
                }

                if self._user_index:
                    try:
                        writer = self._user_index.writer()
                        writer.update_document(
                            doc_id=doc_id,
                            user_id=user_id,
                            platform=platform,
                            username=doc.get("username", ""),
                            display_name=doc.get("display_name", ""),
                            bio=doc.get("bio", ""),
                            followers_count=doc.get("followers_count", 0),
                            verified=doc.get("verified", "false"),
                            indexed_at=datetime.utcnow(),
                        )
                        writer.commit()
                    except Exception as e:
                        logger.warning(f"Whoosh user index failed: {e}")

                return True
            except Exception as e:
                logger.error(f"Failed to index user: {e}")
                return False

    async def delete_post(self, post_id: str, platform: Optional[str] = None) -> bool:
        """删除帖子索引"""
        async with self._lock:
            try:
                doc_id = f"{platform}:{post_id}" if platform else post_id

                if doc_id in self._documents:
                    content = self._documents[doc_id].get("content", "")
                    tokens = self._tokenize(content)

                    for token in tokens:
                        if token in self._inverted_index:
                            self._inverted_index[token].discard(doc_id)
                            if not self._inverted_index[token]:
                                del self._inverted_index[token]

                    del self._documents[doc_id]

                if self._post_index:
                    try:
                        writer = self._post_index.writer()
                        writer.delete_by_term("doc_id", doc_id)
                        writer.commit()
                    except Exception as e:
                        logger.warning(f"Whoosh delete failed: {e}")

                return True
            except Exception as e:
                logger.error(f"Failed to delete post: {e}")
                return False

    def _apply_filters(
        self,
        doc: dict[str, Any],
        filters: Optional[SearchFilter],
    ) -> bool:
        """应用过滤条件"""
        if not filters:
            return True

        if filters.platforms and doc.get("platform") not in filters.platforms:
            return False

        if filters.user_ids and doc.get("user_id") not in filters.user_ids:
            return False

        posted_at = doc.get("posted_at")
        if posted_at:
            if isinstance(posted_at, str):
                posted_at = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))

            if filters.start_time and posted_at < filters.start_time:
                return False
            if filters.end_time and posted_at > filters.end_time:
                return False

        if filters.min_likes is not None:
            if doc.get("likes_count", 0) < filters.min_likes:
                return False

        if filters.max_likes is not None:
            if doc.get("likes_count", 0) > filters.max_likes:
                return False

        if filters.min_comments is not None:
            if doc.get("comments_count", 0) < filters.min_comments:
                return False

        if filters.hashtags:
            doc_hashtags = [h.lower() for h in doc.get("hashtags", "").split(",") if h]
            if not any(h.lower() in doc_hashtags for h in filters.hashtags):
                return False

        if filters.mentions:
            doc_mentions = doc.get("mentions", "").split(",")
            if not any(m in doc_mentions for m in filters.mentions):
                return False

        if filters.language and doc.get("language") != filters.language:
            return False

        return True

    def _sort_docs(
        self,
        docs: list[dict[str, Any]],
        sort: SortOrder,
    ) -> list[dict[str, Any]]:
        """排序文档"""
        def get_sort_key(doc: dict[str, Any]) -> tuple:
            posted_at = doc.get("posted_at")
            if posted_at:
                if isinstance(posted_at, str):
                    posted_at = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
            else:
                posted_at = datetime.min

            likes = doc.get("likes_count", 0)
            comments = doc.get("comments_count", 0)
            score = doc.get("score", 0)

            if sort == SortOrder.TIME_DESC:
                return (-posted_at.timestamp() if posted_at else 0,)
            elif sort == SortOrder.TIME_ASC:
                return (posted_at.timestamp() if posted_at else 0,)
            elif sort == SortOrder.LIKES_DESC:
                return (-likes, -posted_at.timestamp() if posted_at else 0)
            elif sort == SortOrder.COMMENTS_DESC:
                return (-comments, -posted_at.timestamp() if posted_at else 0)
            else:
                return (-score, -posted_at.timestamp() if posted_at else 0)

        return sorted(docs, key=get_sort_key)

    async def search_posts(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        sort: SortOrder = SortOrder.RELEVANCE,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """搜索帖子"""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()

        query_tokens = self._tokenize(query) if query else []

        doc_scores: dict[str, float] = {}
        for token in query_tokens:
            if token in self._inverted_index:
                for doc_id in self._inverted_index[token]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

        if query_tokens:
            for doc_id in doc_scores:
                doc_scores[doc_id] /= len(query_tokens)

        results = []
        for doc_id, score in doc_scores.items():
            doc = self._documents.get(doc_id)
            if doc and self._apply_filters(doc, filters):
                doc["score"] = score
                results.append(doc)

        if not query:
            for doc_id, doc in self._documents.items():
                if self._apply_filters(doc, filters):
                    doc["score"] = 0
                    results.append(doc)

        sorted_results = self._sort_docs(results, sort)
        paginated = sorted_results[offset : offset + limit]

        took = (datetime.utcnow() - start_time).total_seconds() * 1000

        return SearchResult(
            total=len(sorted_results),
            hits=paginated,
            offset=offset,
            limit=limit,
            took=took,
        )

    async def search_users(
        self,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """搜索用户"""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.utcnow()
        query_lower = query.lower()

        results = []
        for doc_id, doc in self._user_documents.items():
            username = (doc.get("username") or "").lower()
            display_name = (doc.get("display_name") or "").lower()
            bio = (doc.get("bio") or "").lower()

            score = 0
            if query_lower in username:
                score += 3
            if query_lower in display_name:
                score += 2
            if query_lower in bio:
                score += 1

            if score > 0:
                doc["score"] = score
                results.append(doc)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:limit]

        took = (datetime.utcnow() - start_time).total_seconds() * 1000

        return SearchResult(
            total=len(results),
            hits=results,
            limit=limit,
            took=took,
        )

    async def search_by_hashtag(
        self,
        hashtag: str,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """按话题搜索"""
        hashtag_clean = hashtag.lstrip("#").lower()

        filters = SearchFilter(hashtags=[hashtag_clean])
        return await self.search_posts(
            query="",
            filters=filters,
            sort=SortOrder.TIME_DESC,
            limit=limit,
            offset=offset,
        )

    async def search_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> SearchResult:
        """按时间范围搜索"""
        filters = SearchFilter(start_time=start_time, end_time=end_time)
        return await self.search_posts(
            query="",
            filters=filters,
            sort=SortOrder.TIME_DESC,
            limit=limit,
        )

    async def suggest(
        self,
        query: str,
        field: str = "content",
        limit: int = 10,
    ) -> list[str]:
        """搜索建议"""
        query_lower = query.lower()
        suggestions = []

        for token in self._inverted_index:
            if token.startswith(query_lower):
                suggestions.append(token)
                if len(suggestions) >= limit:
                    break

        return suggestions

    async def aggregate_by_field(
        self,
        field: str,
        interval: Optional[str] = None,
        size: int = 10,
    ) -> dict[str, Any]:
        """聚合统计"""
        counts: dict[str, int] = {}

        for doc in self._documents.values():
            value = doc.get(field)
            if value:
                if isinstance(value, list):
                    for v in value:
                        counts[str(v)] = counts.get(str(v), 0) + 1
                else:
                    counts[str(value)] = counts.get(str(value), 0) + 1

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:size]

        return {
            f"agg_{field}": {
                "buckets": [
                    {"key": k, "doc_count": v}
                    for k, v in sorted_counts
                ]
            }
        }

    async def get_trending_hashtags(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取热门话题"""
        hashtag_counts: dict[str, int] = {}

        for doc in self._documents.values():
            hashtags = doc.get("hashtags", "")
            if hashtags:
                for tag in hashtags.split(","):
                    tag = tag.strip().lower()
                    if tag:
                        hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1

        sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [
            {"hashtag": tag, "count": count}
            for tag, count in sorted_hashtags
        ]

    async def get_post_count_by_time(
        self,
        interval: str = "day",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """按时间统计帖子数"""
        counts: dict[str, int] = {}

        for doc in self._documents.values():
            posted_at = doc.get("posted_at")
            if posted_at:
                if isinstance(posted_at, str):
                    posted_at = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))

                if start_time and posted_at < start_time:
                    continue
                if end_time and posted_at > end_time:
                    continue

                if interval == "hour":
                    key = posted_at.strftime("%Y-%m-%d %H:00")
                elif interval == "day":
                    key = posted_at.strftime("%Y-%m-%d")
                elif interval == "week":
                    key = posted_at.strftime("%Y-W%W")
                elif interval == "month":
                    key = posted_at.strftime("%Y-%m")
                else:
                    key = posted_at.strftime("%Y-%m-%d")

                counts[key] = counts.get(key, 0) + 1

        return [
            {"time": k, "count": v}
            for k, v in sorted(counts.items())
        ]


class SearchEngine:
    """
    搜索引擎类

    支持 Elasticsearch 和 Whoosh 双模式的搜索引擎，提供统一的搜索接口。

    Attributes:
        config: 搜索引擎配置
        backend: 当前使用的后端
    """

    def __init__(self, config: Optional[SearchConfig] = None) -> None:
        """
        初始化搜索引擎

        Args:
            config: 搜索引擎配置，如果为 None 则使用默认配置
        """
        self.config = config or SearchConfig()
        self._backend: Optional[BaseSearchBackend] = None
        self._initialized = False

        if self.config.enable_chinese_analyzer:
            self._setup_jieba()

    def _setup_jieba(self) -> None:
        """配置 jieba 分词器"""
        if self.config.jieba_dict_path:
            jieba.set_dictionary(self.config.jieba_dict_path)
        if self.config.jieba_user_dict:
            jieba.load_userdict(self.config.jieba_user_dict)

    @property
    def backend(self) -> BaseSearchBackend:
        """获取当前后端"""
        if self._backend is None:
            raise RuntimeError("Search engine not initialized. Call initialize() first.")
        return self._backend

    @property
    def backend_type(self) -> SearchBackend:
        """获取当前后端类型"""
        if self.config.use_elasticsearch:
            return SearchBackend.ELASTICSEARCH
        elif self.config.use_whoosh:
            return SearchBackend.WHOOSH
        return SearchBackend.MEMORY

    async def initialize(self) -> None:
        """
        初始化搜索引擎

        根据配置选择合适的后端并初始化。
        """
        if self._initialized:
            return

        if self.config.use_elasticsearch:
            self._backend = ElasticsearchBackend(self.config)
        else:
            self._backend = WhooshBackend(self.config)

        await self._backend.initialize()
        self._initialized = True
        logger.info(f"Search engine initialized with {self.backend_type.value} backend")

    async def close(self) -> None:
        """关闭搜索引擎"""
        if self._backend:
            await self._backend.close()
            self._initialized = False
            logger.info("Search engine closed")

    async def __aenter__(self) -> "SearchEngine":
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """异步上下文管理器出口"""
        await self.close()

    async def create_index(self) -> None:
        """
        创建索引

        创建帖子索引和用户索引。
        """
        await self.backend.create_index()
        logger.info("Search index created")

    async def delete_index(self) -> None:
        """
        删除索引

        删除所有索引数据。
        """
        await self.backend.delete_index()
        logger.info("Search index deleted")

    async def index_post(self, post: Union[Post, dict[str, Any]]) -> bool:
        """
        索引单个帖子

        Args:
            post: Post 对象或帖子字典

        Returns:
            是否索引成功
        """
        return await self.backend.index_post(post)

    async def index_posts(self, posts: list[Union[Post, dict[str, Any]]]) -> int:
        """
        批量索引帖子

        Args:
            posts: Post 对象列表或帖子字典列表

        Returns:
            成功索引的数量
        """
        return await self.backend.index_posts(posts)

    async def index_user(self, user: Union[User, dict[str, Any]]) -> bool:
        """
        索引单个用户

        Args:
            user: User 对象或用户字典

        Returns:
            是否索引成功
        """
        return await self.backend.index_user(user)

    async def delete_post(self, post_id: str, platform: Optional[str] = None) -> bool:
        """
        删除帖子索引

        Args:
            post_id: 帖子ID
            platform: 平台名称（可选）

        Returns:
            是否删除成功
        """
        return await self.backend.delete_post(post_id, platform)

    async def search_posts(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        sort: SortOrder = SortOrder.RELEVANCE,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """
        搜索帖子

        Args:
            query: 搜索查询字符串
            filters: 过滤条件
            sort: 排序方式
            limit: 返回结果数量限制
            offset: 结果偏移量（用于分页）

        Returns:
            搜索结果
        """
        return await self.backend.search_posts(
            query=query,
            filters=filters,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    async def search_users(
        self,
        query: str,
        limit: int = 10,
    ) -> SearchResult:
        """
        搜索用户

        Args:
            query: 搜索查询字符串
            limit: 返回结果数量限制

        Returns:
            搜索结果
        """
        return await self.backend.search_users(query=query, limit=limit)

    async def search_by_hashtag(
        self,
        hashtag: str,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """
        按话题标签搜索

        Args:
            hashtag: 话题标签（可带或不带 # 前缀）
            limit: 返回结果数量限制
            offset: 结果偏移量

        Returns:
            搜索结果
        """
        return await self.backend.search_by_hashtag(
            hashtag=hashtag,
            limit=limit,
            offset=offset,
        )

    async def search_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> SearchResult:
        """
        按时间范围搜索

        Args:
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回结果数量限制

        Returns:
            搜索结果
        """
        return await self.backend.search_by_time_range(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    async def suggest(
        self,
        query: str,
        field: str = "content",
        limit: int = 10,
    ) -> list[str]:
        """
        获取搜索建议

        Args:
            query: 查询前缀
            field: 搜索字段
            limit: 返回建议数量

        Returns:
            建议列表
        """
        return await self.backend.suggest(query=query, field=field, limit=limit)

    async def aggregate_by_field(
        self,
        field: str,
        interval: Optional[str] = None,
        size: int = 10,
    ) -> dict[str, Any]:
        """
        聚合统计

        Args:
            field: 聚合字段
            interval: 时间间隔（用于日期或数值字段）
            size: 返回桶数量

        Returns:
            聚合结果
        """
        return await self.backend.aggregate_by_field(
            field=field,
            interval=interval,
            size=size,
        )

    async def get_trending_hashtags(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        获取热门话题

        Args:
            limit: 返回话题数量

        Returns:
            热门话题列表，每项包含 hashtag 和 count
        """
        return await self.backend.get_trending_hashtags(limit=limit)

    async def get_post_count_by_time(
        self,
        interval: str = "day",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """
        按时间统计帖子数

        Args:
            interval: 时间间隔（hour, day, week, month）
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            时间统计列表，每项包含 time 和 count
        """
        return await self.backend.get_post_count_by_time(
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    async def search_by_platform(
        self,
        platform: str,
        limit: int = 100,
        offset: int = 0,
    ) -> SearchResult:
        """
        按平台搜索

        Args:
            platform: 平台名称
            limit: 返回结果数量限制
            offset: 结果偏移量

        Returns:
            搜索结果
        """
        filters = SearchFilter(platforms=[platform])
        return await self.search_posts(
            query="",
            filters=filters,
            sort=SortOrder.TIME_DESC,
            limit=limit,
            offset=offset,
        )

    async def search_by_user(
        self,
        user_id: str,
        platform: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> SearchResult:
        """
        按用户搜索

        Args:
            user_id: 用户ID
            platform: 平台名称（可选）
            limit: 返回结果数量限制
            offset: 结果偏移量

        Returns:
            搜索结果
        """
        filters = SearchFilter(user_ids=[user_id])
        if platform:
            filters.platforms = [platform]
        return await self.search_posts(
            query="",
            filters=filters,
            sort=SortOrder.TIME_DESC,
            limit=limit,
            offset=offset,
        )

    async def get_statistics(self) -> dict[str, Any]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        if isinstance(self._backend, WhooshBackend):
            return {
                "backend": self.backend_type.value,
                "total_posts": len(self._backend._documents),
                "total_users": len(self._backend._user_documents),
                "total_terms": len(self._backend._inverted_index),
            }
        elif isinstance(self._backend, ElasticsearchBackend):
            try:
                post_count = await self._backend._client.count(
                    index=f"{self.config.index_name}_posts"
                )
                user_count = await self._backend._client.count(
                    index=f"{self.config.index_name}_users"
                )
                return {
                    "backend": self.backend_type.value,
                    "total_posts": post_count.get("count", 0),
                    "total_users": user_count.get("count", 0),
                }
            except Exception:
                return {"backend": self.backend_type.value}

        return {"backend": self.backend_type.value}
