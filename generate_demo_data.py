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
        bio