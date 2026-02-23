"""
生成演示数据脚本

用于生成模拟的社交媒体数据，用于测试和演示仪表盘功能。
"""

import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from sentiment_analyzer.storage.sqlite_store import SQLiteStore
from sentiment_analyzer.storage.models import Platform, Post, User


# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")


# 模拟数据模板
SAMPLE_CONTENTS = [
    "人工智能正在改变我们的生活方式，未来可期！",
    "刚刚体验了最新的AI助手，真的太强大了",
    "机器学习在医疗领域的应用越来越广泛",
    "ChatGPT的出现标志着AI进入新纪元",
    "自动驾驶技术还需要更多时间完善",
    "AI绘画工具让创作变得如此简单",
    "深度学习在图像识别上的突破令人惊叹",
    "人工智能是否会取代人类工作？",
    "神经网络的发展历史回顾",
    "大语言模型的训练成本有多高？",
    "AI伦理问题值得每个人思考",
    "智能推荐算法如何影响我们的选择",
    "机器人技术在制造业的应用",
    "自然语言处理的最新进展",
    "计算机视觉技术的商业应用",
]

SAMPLE_USERS = [
    {"username": "tech_guru", "display_name": "科技达人"},
    {"username": "ai_researcher", "display_name": "AI研究员"},
    {"username": "data_scientist", "display_name": "数据科学家"},
    {"username": "ml_engineer", "display_name": "机器学习工程师"},
    {"username": "tech_blogger", "display_name": "科技博主"},
    {"username": "startup_founder", "display_name": "创业者"},
    {"username": "product_manager", "display_name": "产品经理"},
    {"username": "code_master", "display_name": "代码大师"},
    {"username": "innovation_lead", "display_name": "创新领袖"},
    {"username": "future_tech", "display_name": "未来科技"},
]

SAMPLE_HASHTAGS = ["#人工智能", "#AI", "#机器学习", "#深度学习", "#科技", "#创新", "#未来", "#ChatGPT"]


def generate_random_user(platform: Platform, index: int) -> User:
    """生成随机用户"""
    user_data = SAMPLE_USERS[index % len(SAMPLE_USERS)]
    return User(
        user_id=f"user_{index:04d}",
        platform=platform,
        username=user_data["username"],
        display_name=user_data["display_name"],
        bio=f"热爱{random.choice(['科技', 'AI', '编程', '创新'])}",
        followers_count=random.randint(100, 10000),
        friends_count=random.randint(50, 1000),
        posts_count=random.randint(10, 500),
        created_at=datetime.now() - timedelta(days=random.randint(30, 1000)),
        verified=random.random() > 0.7,
        avatar_url=f"https://example.com/avatar_{index}.jpg",
        raw_data={},
    )


def generate_random_post(platform: Platform, author: User, index: int) -> Post:
    """生成随机帖子"""
    content = random.choice(SAMPLE_CONTENTS)
    
    # 随机添加话题标签
    if random.random() > 0.5:
        content += " " + random.choice(SAMPLE_HASHTAGS)
    
    # 随机添加提及
    if random.random() > 0.8:
        content += " @" + random.choice(SAMPLE_USERS)["username"]
    
    created_at = datetime.now() - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    
    return Post(
        post_id=f"post_{index:06d}",
        platform=platform,
        author_id=author.user_id,
        author_name=author.username,
        content=content,
        created_at=created_at,
        language="zh",
        likes=random.randint(0, 1000),
        shares=random.randint(0, 100),
        comments=random.randint(0, 50),
        hashtags=[tag for tag in SAMPLE_HASHTAGS if tag in content],
        mentions=["tech_guru"] if "@" in content else [],
        urls=[],
        media_urls=[f"https://example.com/image_{index}.jpg"] if random.random() > 0.6 else [],
        is_retweet=random.random() > 0.9,
        parent_id=None,
        sentiment_score=random.uniform(-1, 1),
        sentiment_label=random.choice(["positive", "neutral", "negative"]),
        raw_data={},
    )


async def generate_demo_data(num_users: int = 10, num_posts: int = 100):
    """
    生成演示数据
    
    Args:
        num_users: 生成用户数量
        num_posts: 生成帖子数量
    """
    # 确保数据目录存在
    Path("./data").mkdir(exist_ok=True)
    
    # 初始化存储
    store = SQLiteStore("./data/sentiment.db")
    await store.initialize()
    
    logger.info(f"开始生成演示数据: {num_users} 用户, {num_posts} 帖子")
    
    # 生成用户
    users = []
    for i in range(num_users):
        user = generate_random_user(Platform.WEIBO, i)
        await store.save_user(user)
        users.append(user)
        logger.debug(f"生成用户: {user.username}")
    
    logger.info(f"已生成 {len(users)} 个用户")
    
    # 生成帖子
    posts_count = 0
    for i in range(num_posts):
        author = random.choice(users)
        post = generate_random_post(Platform.WEIBO, author, i)
        await store.save_post(post)
        posts_count += 1
        
        if (i + 1) % 20 == 0:
            logger.info(f"已生成 {i + 1}/{num_posts} 条帖子")
    
    logger.info(f"已生成 {posts_count} 条帖子")
    
    # 关闭存储
    await store.close()
    
    logger.info("演示数据生成完成!")
    return len(users), posts_count


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成演示数据")
    parser.add_argument("--users", "-u", type=int, default=10, help="用户数量")
    parser.add_argument("--posts", "-p", type=int, default=100, help="帖子数量")
    
    args = parser.parse_args()
    
    try:
        user_count, post_count = asyncio.run(generate_demo_data(
            num_users=args.users,
            num_posts=args.posts
        ))
        print(f"\n✅ 演示数据生成完成!")
        print(f"👥 用户数: {user_count}")
        print(f"📝 帖子数: {post_count}")
        print(f"📊 数据库: ./data/sentiment.db")
        print(f"\n现在可以刷新仪表盘查看数据: http://localhost:8501")
    except Exception as e:
        logger.exception("生成数据失败")
        print(f"\n❌ 生成失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
