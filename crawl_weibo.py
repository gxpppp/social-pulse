"""
微博数据采集脚本

使用方法:
    .venv\Scripts\python crawl_weibo.py --keyword "AI" --limit 100
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from sentiment_analyzer.crawlers.weibo import WeiboConfig, WeiboCrawler
from sentiment_analyzer.storage.sqlite_store import SQLiteStore


# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/crawl_weibo.log", rotation="10 MB", level="DEBUG")


async def crawl_weibo_data(
    keyword: str = "AI",
    limit: int = 100,
    cookies: str = None
):
    """
    爬取微博数据
    
    Args:
        keyword: 搜索关键词
        limit: 采集数量限制
        cookies: 微博登录Cookie（可选）
    """
    # 确保数据目录存在
    Path("./data").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    # 初始化存储
    store = SQLiteStore("./data/sentiment.db")
    await store.initialize()
    
    # 配置微博爬虫
    config = WeiboConfig(
        cookies=cookies,
        use_mobile_api=True,
        timeout=30,
        max_retries=3,
        request_delay=(2.0, 5.0),  # 随机延迟，避免被封
    )
    
    logger.info(f"开始采集微博数据: 关键词='{keyword}', 限制={limit}")
    
    crawler = WeiboCrawler(config)
    await crawler.initialize()
    count = 0
    try:
        async for post in crawler.crawl_search(keyword, limit=limit):
            try:
                # 保存用户
                if post.user:
                    await store.save_user(post.user)
                    logger.debug(f"保存用户: {post.user.username}")
                
                # 保存帖子
                await store.save_post(post)
                logger.debug(f"保存帖子: {post.content[:50]}...")
                
                count += 1
                if count % 10 == 0:
                    logger.info(f"已采集 {count}/{limit} 条数据")
                
            except Exception as e:
                logger.error(f"保存数据失败: {e}")
                continue
    finally:
        await crawler.close()
    
    logger.info(f"采集完成! 共采集 {count} 条数据")
    
    return count


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="微博数据采集工具")
    parser.add_argument("--keyword", "-k", default="AI", help="搜索关键词")
    parser.add_argument("--limit", "-l", type=int, default=100, help="采集数量限制")
    parser.add_argument("--cookies", "-c", default=None, help="微博Cookie字符串")
    
    args = parser.parse_args()
    
    # 运行采集
    try:
        count = asyncio.run(crawl_weibo_data(
            keyword=args.keyword,
            limit=args.limit,
            cookies=args.cookies
        ))
        print(f"\n✅ 采集完成! 共采集 {count} 条微博数据")
        print(f"📊 数据保存在: ./data/sentiment.db")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断采集")
    except Exception as e:
        logger.exception("采集失败")
        print(f"\n❌ 采集失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
