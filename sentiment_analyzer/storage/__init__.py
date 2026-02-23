"""
数据存储模块

提供数据库Schema定义和数据访问层接口。

使用示例:
    from sentiment_analyzer.storage import Database, UserRepository, PostRepository

    # 初始化数据库
    db = Database("sqlite+aiosqlite:///./data/sentiment.db")
    await db.init()

    # 创建Repository实例
    user_repo = UserRepository(db)
    post_repo = PostRepository(db)

    # 使用Repository操作数据
    user = await user_repo.create({
        "user_id": "twitter_12345",
        "platform": "twitter",
        "username": "example_user"
    })

    # 关闭数据库连接
    await db.close()

消息队列使用示例:
    from sentiment_analyzer.storage import (
        MessageQueueConfig,
        RedisStreamsQueue,
        MessageProducer,
        MessageConsumer,
    )

    # 配置消息队列
    config = MessageQueueConfig(
        use_redis=True,
        redis_url="redis://localhost:6379/0"
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

from .queue import (
    AcknowledgeError,
    ConnectionError,
    KafkaQueue,
    Message,
    MessageConsumer,
    MessageHandler,
    MessagePriority,
    MessageProducer,
    MessageQueue,
    MessageQueueConfig,
    MessageQueueError,
    MessageStatus,
    MessageType,
    PublishError,
    RedisStreamsQueue,
    SubscribeError,
    create_message_queue,
)
from .repository import (
    CrawlTaskRepository,
    Database,
    DatabaseError,
    DuplicateError,
    InteractionRepository,
    NotFoundError,
    PostRepository,
    SystemLogRepository,
    UserFeatureRepository,
    UserRepository,
)
from .schema import (
    Base,
    CrawlTask,
    Interaction,
    Post,
    SystemLog,
    User,
    UserFeature,
    get_all_models,
)

__all__ = [
    "Database",
    "DatabaseError",
    "DuplicateError",
    "NotFoundError",
    "Base",
    "User",
    "Post",
    "Interaction",
    "CrawlTask",
    "UserFeature",
    "SystemLog",
    "get_all_models",
    "UserRepository",
    "PostRepository",
    "InteractionRepository",
    "CrawlTaskRepository",
    "UserFeatureRepository",
    "SystemLogRepository",
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
