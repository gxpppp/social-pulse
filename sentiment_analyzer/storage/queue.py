"""
消息队列集成模块

提供 Redis Streams 和 Kafka 的消息队列实现，支持异步消息发布和订阅。

使用示例:
    from sentiment_analyzer.storage.queue import (
        MessageQueueConfig,
        RedisStreamsQueue,
        MessageProducer,
        MessageConsumer,
    )

    # 配置消息队列
    config = MessageQueueConfig(
        use_redis=True,
        redis_url="redis://localhost:6379/0",
        topic_prefix="sentiment"
    )

    # 创建队列实例
    queue = RedisStreamsQueue(config)

    # 生产者发送消息
    producer = MessageProducer(queue)
    await producer.send_crawl_task({"platform": "twitter", "query": "python"})

    # 消费者处理消息
    consumer = MessageConsumer(queue)
    async for message in consumer.process_message(handler):
        print(f"Processed: {message.id}")
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MessageQueueConfig(BaseModel):
    """
    消息队列配置类

    Attributes:
        use_redis: 是否使用Redis Streams
        redis_url: Redis连接URL
        use_kafka: 是否使用Kafka
        kafka_servers: Kafka服务器列表
        topic_prefix: 主题前缀
        consumer_group: 消费者组名称
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        dead_letter_topic: 死信队列主题
    """

    use_redis: bool = Field(default=True, description="是否使用Redis Streams")
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis连接URL"
    )
    use_kafka: bool = Field(default=False, description="是否使用Kafka")
    kafka_servers: list[str] = Field(
        default_factory=lambda: ["localhost:9092"],
        description="Kafka服务器列表"
    )
    topic_prefix: str = Field(
        default="sentiment",
        description="主题前缀"
    )
    consumer_group: str = Field(
        default="sentiment_analyzer",
        description="消费者组名称"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="最大重试次数"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        description="重试延迟（秒）"
    )
    dead_letter_topic: str = Field(
        default="dead_letter",
        description="死信队列主题"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="批量消费消息数"
    )
    block_timeout: int = Field(
        default=5000,
        ge=100,
        description="阻塞读取超时（毫秒）"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_redis": True,
                "redis_url": "redis://localhost:6379/0",
                "use_kafka": False,
                "kafka_servers": ["localhost:9092"],
                "topic_prefix": "sentiment",
                "consumer_group": "sentiment_analyzer",
                "max_retries": 3,
                "retry_delay": 1.0,
                "dead_letter_topic": "dead_letter",
                "batch_size": 10,
                "block_timeout": 5000
            }
        }


class MessagePriority(str, Enum):
    """消息优先级枚举"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """消息状态枚举"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class MessageType(str, Enum):
    """消息类型枚举"""

    CRAWL_TASK = "crawl_task"
    ANALYSIS_TASK = "analysis_task"
    NOTIFICATION = "notification"
    SYSTEM_EVENT = "system_event"
    DATA_UPDATE = "data_update"


class Message(BaseModel):
    """
    消息数据结构

    Attributes:
        id: 消息唯一标识符
        topic: 消息主题
        payload: 消息负载
        timestamp: 消息时间戳
        headers: 消息头信息
        priority: 消息优先级
        status: 消息状态
        retry_count: 重试次数
        message_type: 消息类型
        correlation_id: 关联ID（用于追踪）
        source: 消息来源
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="消息唯一标识符"
    )
    topic: str = Field(..., description="消息主题")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="消息负载"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="消息时间戳"
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="消息头信息"
    )
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="消息优先级"
    )
    status: MessageStatus = Field(
        default=MessageStatus.PENDING,
        description="消息状态"
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="重试次数"
    )
    message_type: MessageType = Field(
        default=MessageType.SYSTEM_EVENT,
        description="消息类型"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="关联ID（用于追踪）"
    )
    source: Optional[str] = Field(
        default=None,
        description="消息来源"
    )

    def to_json(self) -> str:
        """序列化消息为JSON字符串"""
        return self.model_dump_json()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """从JSON字符串反序列化消息"""
        data = json.loads(json_str)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """从字典创建消息"""
        return cls.model_validate(data)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "topic": "crawl_tasks",
                "payload": {"platform": "twitter", "query": "python"},
                "timestamp": "2024-01-15T12:00:00",
                "headers": {"source": "api"},
                "priority": "normal",
                "status": "pending",
                "retry_count": 0,
                "message_type": "crawl_task",
                "correlation_id": "req_12345",
                "source": "scheduler"
            }
        }


class MessageQueue(ABC):
    """
    消息队列基类（抽象类）

    定义消息队列的基本接口，所有具体实现都需要继承此类。
    """

    def __init__(self, config: MessageQueueConfig) -> None:
        self.config = config
        self._connected = False

    @abstractmethod
    async def publish(self, topic: str, message: Message) -> str:
        """
        发布消息到指定主题

        Args:
            topic: 消息主题
            message: 消息对象

        Returns:
            消息ID

        Raises:
            MessageQueueError: 发布消息失败
        """
        pass

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        consumer_name: Optional[str] = None
    ) -> AsyncGenerator[Message, None]:
        """
        订阅指定主题的消息

        Args:
            topic: 消息主题
            consumer_name: 消费者名称

        Yields:
            Message: 消息对象

        Raises:
            MessageQueueError: 订阅失败
        """
        pass

    @abstractmethod
    async def acknowledge(self, topic: str, message_id: str) -> bool:
        """
        确认消息已处理

        Args:
            topic: 消息主题
            message_id: 消息ID

        Returns:
            是否确认成功

        Raises:
            MessageQueueError: 确认失败
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        关闭连接

        释放所有资源，关闭连接。
        """
        pass

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected

    def _get_full_topic(self, topic: str) -> str:
        """获取带前缀的完整主题名"""
        return f"{self.config.topic_prefix}:{topic}"


class MessageQueueError(Exception):
    """消息队列错误基类"""
    pass


class ConnectionError(MessageQueueError):
    """连接错误"""
    pass


class PublishError(MessageQueueError):
    """发布消息错误"""
    pass


class SubscribeError(MessageQueueError):
    """订阅错误"""
    pass


class AcknowledgeError(MessageQueueError):
    """确认消息错误"""
    pass


class RedisStreamsQueue(MessageQueue):
    """
    基于Redis Streams的消息队列实现

    使用Redis Streams作为消息队列后端，支持消费者组和消息确认。

    Attributes:
        config: 消息队列配置
        redis_client: Redis客户端实例
    """

    def __init__(self, config: MessageQueueConfig) -> None:
        super().__init__(config)
        self._redis: Any = None
        self._consumer_groups: set[str] = set()

    async def _ensure_connection(self) -> None:
        """确保Redis连接已建立"""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis.ping()
                self._connected = True
                logger.info(f"Redis连接已建立: {self.config.redis_url}")
            except Exception as e:
                self._connected = False
                raise ConnectionError(f"无法连接到Redis: {e}") from e

    async def publish(self, topic: str, message: Message) -> str:
        """
        使用XADD命令发布消息到Redis Streams

        Args:
            topic: 消息主题
            message: 消息对象

        Returns:
            消息ID
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)

        try:
            message_data = {
                "data": message.to_json(),
                "timestamp": message.timestamp.isoformat(),
                "priority": message.priority.value,
                "message_type": message.message_type.value
            }

            message_id = await self._redis.xadd(
                full_topic,
                message_data,
                id="*"
            )

            logger.debug(f"消息已发布到 {full_topic}: {message_id}")
            return message_id

        except Exception as e:
            raise PublishError(f"发布消息到 {full_topic} 失败: {e}") from e

    async def subscribe(
        self,
        topic: str,
        consumer_name: Optional[str] = None
    ) -> AsyncGenerator[Message, None]:
        """
        使用XREAD命令消费Redis Streams消息

        Args:
            topic: 消息主题
            consumer_name: 消费者名称

        Yields:
            Message: 消息对象
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)
        consumer = consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"

        await self.create_consumer_group(full_topic)

        while True:
            try:
                messages = await self._redis.xreadgroup(
                    groupname=self.config.consumer_group,
                    consumername=consumer,
                    streams={full_topic: ">"},
                    count=self.config.batch_size,
                    block=self.config.block_timeout
                )

                if not messages:
                    await asyncio.sleep(0.1)
                    continue

                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        try:
                            message_json = message_data.get("data", "{}")
                            message = Message.from_json(message_json)
                            message.id = message_id
                            yield message
                        except Exception as e:
                            logger.error(f"解析消息失败 {message_id}: {e}")
                            await self.acknowledge(topic, message_id)

            except Exception as e:
                logger.error(f"读取消息失败: {e}")
                await asyncio.sleep(1)

    async def create_consumer_group(
        self,
        topic: str,
        group_name: Optional[str] = None
    ) -> bool:
        """
        创建消费者组

        Args:
            topic: 消息主题
            group_name: 消费者组名称

        Returns:
            是否创建成功
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)
        group = group_name or self.config.consumer_group
        group_key = f"{full_topic}:{group}"

        if group_key in self._consumer_groups:
            return True

        try:
            await self._redis.xgroup_create(
                name=full_topic,
                groupname=group,
                id="0",
                mkstream=True
            )
            self._consumer_groups.add(group_key)
            logger.info(f"消费者组已创建: {group} @ {full_topic}")
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                self._consumer_groups.add(group_key)
                logger.debug(f"消费者组已存在: {group} @ {full_topic}")
                return True
            raise SubscribeError(f"创建消费者组失败: {e}") from e

    async def pending_messages(
        self,
        topic: str,
        consumer_name: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        获取待处理消息

        Args:
            topic: 消息主题
            consumer_name: 消费者名称

        Returns:
            待处理消息列表
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)

        try:
            pending = await self._redis.xpending_range(
                name=full_topic,
                groupname=self.config.consumer_group,
                min="-",
                max="+",
                count=self.config.batch_size
            )
            return pending or []
        except Exception as e:
            logger.error(f"获取待处理消息失败: {e}")
            return []

    async def acknowledge(self, topic: str, message_id: str) -> bool:
        """
        使用XACK确认消息

        Args:
            topic: 消息主题
            message_id: 消息ID

        Returns:
            是否确认成功
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)

        try:
            result = await self._redis.xack(
                full_topic,
                self.config.consumer_group,
                message_id
            )
            success = result > 0
            if success:
                logger.debug(f"消息已确认: {message_id} @ {full_topic}")
            return success
        except Exception as e:
            raise AcknowledgeError(f"确认消息失败: {e}") from e

    async def claim_message(
        self,
        topic: str,
        message_id: str,
        consumer_name: str,
        min_idle_time: int = 60000
    ) -> Optional[Message]:
        """
        认领超时消息

        Args:
            topic: 消息主题
            message_id: 消息ID
            consumer_name: 消费者名称
            min_idle_time: 最小空闲时间（毫秒）

        Returns:
            消息对象或None
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)

        try:
            messages = await self._redis.xclaim(
                name=full_topic,
                groupname=self.config.consumer_group,
                consumername=consumer_name,
                min_idle_time=min_idle_time,
                message_ids=[message_id]
            )

            if messages:
                _, message_data = messages[0]
                message_json = message_data.get("data", "{}")
                message = Message.from_json(message_json)
                message.id = message_id
                return message
            return None
        except Exception as e:
            logger.error(f"认领消息失败: {e}")
            return None

    async def close(self) -> None:
        """关闭Redis连接"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._connected = False
            logger.info("Redis连接已关闭")


class KafkaQueue(MessageQueue):
    """
    基于Kafka的消息队列实现

    使用Kafka作为消息队列后端，支持分区和消费者组。

    Attributes:
        config: 消息队列配置
        producer: Kafka生产者实例
        consumer: Kafka消费者实例
    """

    def __init__(self, config: MessageQueueConfig) -> None:
        super().__init__(config)
        self._producer: Any = None
        self._consumer: Any = None
        self._admin: Any = None
        self._topics: set[str] = set()

    async def _ensure_connection(self) -> None:
        """确保Kafka连接已建立"""
        if self._producer is None:
            try:
                from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
                from aiokafka.admin import AIOKafkaAdminClient

                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.config.kafka_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8")
                )
                await self._producer.start()

                self._admin = AIOKafkaAdminClient(
                    bootstrap_servers=self.config.kafka_servers
                )
                await self._admin.start()

                self._connected = True
                logger.info(f"Kafka连接已建立: {self.config.kafka_servers}")

            except ImportError:
                raise ConnectionError(
                    "aiokafka未安装，请运行: pip install aiokafka"
                )
            except Exception as e:
                self._connected = False
                raise ConnectionError(f"无法连接到Kafka: {e}") from e

    async def _ensure_topic(self, topic: str) -> None:
        """确保主题存在"""
        if topic in self._topics:
            return

        try:
            from kafka.admin import NewTopic
            from kafka.errors import TopicAlreadyExistsError

            new_topic = NewTopic(
                name=topic,
                num_partitions=3,
                replication_factor=1
            )

            try:
                await self._admin.create_topics([new_topic])
                logger.info(f"Kafka主题已创建: {topic}")
            except TopicAlreadyExistsError:
                pass

            self._topics.add(topic)

        except Exception as e:
            logger.warning(f"创建主题失败（可能已存在）: {e}")
            self._topics.add(topic)

    async def publish(self, topic: str, message: Message) -> str:
        """
        发布消息到Kafka主题

        Args:
            topic: 消息主题
            message: 消息对象

        Returns:
            消息ID
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)
        await self._ensure_topic(full_topic)

        try:
            message_data = message.to_dict()

            partition = None
            if message.correlation_id:
                partition = hash(message.correlation_id) % 3

            metadata = await self._producer.send_and_wait(
                full_topic,
                message_data,
                partition=partition
            )

            kafka_message_id = f"{metadata.topic}:{metadata.partition}:{metadata.offset}"
            logger.debug(f"消息已发布到 {full_topic}: {kafka_message_id}")
            return kafka_message_id

        except Exception as e:
            raise PublishError(f"发布消息到 {full_topic} 失败: {e}") from e

    async def subscribe(
        self,
        topic: str,
        consumer_name: Optional[str] = None
    ) -> AsyncGenerator[Message, None]:
        """
        订阅Kafka主题消息

        Args:
            topic: 消息主题
            consumer_name: 消费者名称

        Yields:
            Message: 消息对象
        """
        from aiokafka import AIOKafkaConsumer

        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)
        await self._ensure_topic(full_topic)

        consumer = AIOKafkaConsumer(
            full_topic,
            bootstrap_servers=self.config.kafka_servers,
            group_id=self.config.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False
        )

        try:
            await consumer.start()

            async for kafka_message in consumer:
                try:
                    message = Message.from_dict(kafka_message.value)
                    message.id = f"{kafka_message.topic}:{kafka_message.partition}:{kafka_message.offset}"
                    yield message
                except Exception as e:
                    logger.error(f"解析消息失败: {e}")

        finally:
            await consumer.stop()

    async def commit(self, topic: str) -> bool:
        """
        提交偏移量

        Args:
            topic: 消息主题

        Returns:
            是否提交成功
        """
        if self._consumer:
            try:
                await self._consumer.commit()
                return True
            except Exception as e:
                logger.error(f"提交偏移量失败: {e}")
                return False
        return False

    async def acknowledge(self, topic: str, message_id: str) -> bool:
        """
        确认消息（Kafka通过commit机制确认）

        Args:
            topic: 消息主题
            message_id: 消息ID

        Returns:
            是否确认成功
        """
        return await self.commit(topic)

    async def get_partition_info(self, topic: str) -> list[dict[str, Any]]:
        """
        获取分区信息

        Args:
            topic: 消息主题

        Returns:
            分区信息列表
        """
        await self._ensure_connection()
        full_topic = self._get_full_topic(topic)

        try:
            partitions = await self._admin.describe_topics([full_topic])
            return partitions
        except Exception as e:
            logger.error(f"获取分区信息失败: {e}")
            return []

    async def close(self) -> None:
        """关闭Kafka连接"""
        if self._producer:
            await self._producer.stop()
            self._producer = None

        if self._admin:
            await self._admin.close()
            self._admin = None

        self._connected = False
        logger.info("Kafka连接已关闭")


class MessageProducer:
    """
    消息生产者封装

    提供便捷的消息发送方法，封装不同类型消息的发送逻辑。

    Attributes:
        queue: 消息队列实例
    """

    def __init__(self, queue: MessageQueue) -> None:
        self.queue = queue

    async def send_crawl_task(
        self,
        task: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        发送采集任务

        Args:
            task: 任务数据
            priority: 消息优先级
            correlation_id: 关联ID

        Returns:
            消息ID
        """
        message = Message(
            topic="crawl_tasks",
            payload=task,
            message_type=MessageType.CRAWL_TASK,
            priority=priority,
            correlation_id=correlation_id,
            source="crawler"
        )

        return await self.queue.publish("crawl_tasks", message)

    async def send_analysis_task(
        self,
        task: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        发送分析任务

        Args:
            task: 任务数据
            priority: 消息优先级
            correlation_id: 关联ID

        Returns:
            消息ID
        """
        message = Message(
            topic="analysis_tasks",
            payload=task,
            message_type=MessageType.ANALYSIS_TASK,
            priority=priority,
            correlation_id=correlation_id,
            source="analyzer"
        )

        return await self.queue.publish("analysis_tasks", message)

    async def send_notification(
        self,
        notification: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        发送通知

        Args:
            notification: 通知数据
            priority: 消息优先级
            correlation_id: 关联ID

        Returns:
            消息ID
        """
        message = Message(
            topic="notifications",
            payload=notification,
            message_type=MessageType.NOTIFICATION,
            priority=priority,
            correlation_id=correlation_id,
            source="notifier"
        )

        return await self.queue.publish("notifications", message)

    async def send_system_event(
        self,
        event: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """
        发送系统事件

        Args:
            event: 事件数据
            priority: 消息优先级

        Returns:
            消息ID
        """
        message = Message(
            topic="system_events",
            payload=event,
            message_type=MessageType.SYSTEM_EVENT,
            priority=priority,
            source="system"
        )

        return await self.queue.publish("system_events", message)

    async def send_data_update(
        self,
        update: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """
        发送数据更新通知

        Args:
            update: 更新数据
            priority: 消息优先级

        Returns:
            消息ID
        """
        message = Message(
            topic="data_updates",
            payload=update,
            message_type=MessageType.DATA_UPDATE,
            priority=priority,
            source="database"
        )

        return await self.queue.publish("data_updates", message)


MessageHandler = Callable[[Message], Awaitable[bool]]


class MessageConsumer:
    """
    消息消费者封装

    提供消息处理、重试和死信队列功能。

    Attributes:
        queue: 消息队列实例
        config: 消息队列配置
    """

    def __init__(
        self,
        queue: MessageQueue,
        config: Optional[MessageQueueConfig] = None
    ) -> None:
        self.queue = queue
        self.config = config or queue.config
        self._running = False

    async def process_message(
        self,
        handler: MessageHandler,
        topic: str = "crawl_tasks",
        consumer_name: Optional[str] = None
    ) -> AsyncGenerator[Message, None]:
        """
        处理消息

        Args:
            handler: 消息处理函数
            topic: 消息主题
            consumer_name: 消费者名称

        Yields:
            Message: 已处理的消息
        """
        self._running = True

        async for message in self.queue.subscribe(topic, consumer_name):
            if not self._running:
                break

            try:
                success = await self._handle_with_retry(handler, message)

                if success:
                    await self.queue.acknowledge(topic, message.id)
                    yield message
                else:
                    await self._send_to_dead_letter(message)

            except Exception as e:
                logger.error(f"处理消息异常: {e}")
                await self._send_to_dead_letter(message)

    async def _handle_with_retry(
        self,
        handler: MessageHandler,
        message: Message
    ) -> bool:
        """
        带重试的消息处理

        Args:
            handler: 消息处理函数
            message: 消息对象

        Returns:
            是否处理成功
        """
        max_retries = self.config.max_retries

        for attempt in range(max_retries + 1):
            try:
                message.status = MessageStatus.PROCESSING
                result = await handler(message)

                if result:
                    message.status = MessageStatus.COMPLETED
                    return True

                if attempt < max_retries:
                    message.status = MessageStatus.RETRYING
                    message.retry_count = attempt + 1
                    await asyncio.sleep(
                        self.config.retry_delay * (2 ** attempt)
                    )

            except Exception as e:
                logger.warning(
                    f"消息处理失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}"
                )

                if attempt < max_retries:
                    message.status = MessageStatus.RETRYING
                    message.retry_count = attempt + 1
                    await asyncio.sleep(
                        self.config.retry_delay * (2 ** attempt)
                    )
                else:
                    message.status = MessageStatus.FAILED
                    return False

        return False

    async def _send_to_dead_letter(self, message: Message) -> None:
        """
        发送消息到死信队列

        Args:
            message: 消息对象
        """
        try:
            message.status = MessageStatus.FAILED
            message.headers["original_topic"] = message.topic
            message.headers["failed_at"] = datetime.utcnow().isoformat()

            dead_letter_message = Message(
                topic=self.config.dead_letter_topic,
                payload=message.to_dict(),
                message_type=message.message_type,
                priority=MessagePriority.LOW,
                headers={
                    "original_message_id": message.id,
                    "retry_count": str(message.retry_count),
                    "failed_at": datetime.utcnow().isoformat()
                }
            )

            await self.queue.publish(
                self.config.dead_letter_topic,
                dead_letter_message
            )

            logger.warning(
                f"消息已发送到死信队列: {message.id}"
            )

        except Exception as e:
            logger.error(f"发送到死信队列失败: {e}")

    async def retry_on_failure(
        self,
        message: Message,
        handler: MessageHandler,
        max_retries: Optional[int] = None
    ) -> bool:
        """
        失败重试

        Args:
            message: 消息对象
            handler: 消息处理函数
            max_retries: 最大重试次数

        Returns:
            是否处理成功
        """
        original_max_retries = self.config.max_retries
        if max_retries is not None:
            self.config.max_retries = max_retries

        try:
            return await self._handle_with_retry(handler, message)
        finally:
            self.config.max_retries = original_max_retries

    async def dead_letter_queue(
        self,
        consumer_name: Optional[str] = None
    ) -> AsyncGenerator[Message, None]:
        """
        消费死信队列消息

        Args:
            consumer_name: 消费者名称

        Yields:
            Message: 死信消息
        """
        async for message in self.queue.subscribe(
            self.config.dead_letter_topic,
            consumer_name
        ):
            yield message

    def stop(self) -> None:
        """停止消费"""
        self._running = False


async def create_message_queue(
    config: Optional[MessageQueueConfig] = None
) -> MessageQueue:
    """
    创建消息队列实例

    根据配置自动选择Redis Streams或Kafka实现。

    Args:
        config: 消息队列配置

    Returns:
        消息队列实例
    """
    if config is None:
        config = MessageQueueConfig()

    if config.use_redis:
        return RedisStreamsQueue(config)
    elif config.use_kafka:
        return KafkaQueue(config)
    else:
        raise ValueError("必须启用Redis或Kafka作为消息队列后端")


__all__ = [
    "MessageQueueConfig",
    "MessageQueue",
    "MessageQueueError",
    "ConnectionError",
    "PublishError",
    "SubscribeError",
    "AcknowledgeError",
    "RedisStreamsQueue",
    "KafkaQueue",
    "Message",
    "MessagePriority",
    "MessageStatus",
    "MessageType",
    "MessageProducer",
    "MessageConsumer",
    "MessageHandler",
    "create_message_queue",
]
