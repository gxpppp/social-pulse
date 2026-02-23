"""
Sentiment Analyzer - 多平台社交媒体情感分析系统
主入口文件
"""

import asyncio
from typing import Optional

import click
from loguru import logger

from sentiment_analyzer.config.settings import Settings, get_settings
from sentiment_analyzer.crawlers.base import CrawlerConfig, Platform
from sentiment_analyzer.crawlers.twitter import TwitterCrawler
from sentiment_analyzer.crawlers.weibo import WeiboCrawler
from sentiment_analyzer.crawlers.reddit import RedditCrawler
from sentiment_analyzer.crawlers.telegram import TelegramCrawler
from sentiment_analyzer.storage.sqlite_store import SQLiteStore
from sentiment_analyzer.storage.graph_store import GraphStore
from sentiment_analyzer.storage.search import SearchEngine
from sentiment_analyzer.analysis.features import FeatureExtractor
from sentiment_analyzer.analysis.anomaly import AnomalyDetector
from sentiment_analyzer.analysis.graph import GraphAnalyzer


def setup_logging(settings: Settings) -> None:
    """配置日志"""
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
    )
    logger.add(
        settings.log_file,
        rotation="10 MB",
        retention="7 days",
        level=settings.log_level,
        encoding="utf-8"
    )


@click.group()
@click.option("--env", type=str, default="development", help="运行环境")
@click.pass_context
def cli(ctx: click.Context, env: str) -> None:
    """情感分析系统命令行工具"""
    import os
    os.environ["ENVIRONMENT"] = env

    settings = get_settings()
    setup_logging(settings)
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings
    logger.info(f"Sentiment Analyzer started in {env} mode")


@cli.command()
@click.option("--platform", "-p", type=str, required=True,
              help="平台名称 (twitter, weibo, reddit, telegram)")
@click.option("--query", "-q", type=str, required=True,
              help="搜索关键词")
@click.option("--limit", "-l", type=int, default=100,
              help="爬取数量限制")
@click.pass_context
def crawl(ctx: click.Context, platform: str, query: str, limit: int) -> None:
    """执行爬虫任务"""
    settings = ctx.obj["settings"]

    async def run_crawler() -> None:
        store = SQLiteStore(settings.database_url.replace("sqlite:///", ""))
        await store.initialize()

        try:
            platform_enum = Platform(platform.lower())
            config = settings.get_crawler_config(platform_enum)

            crawler = _create_crawler(platform_enum, config, settings)

            async with crawler:
                count = 0
                async for post in crawler.search_posts(query, limit):
                    await store.save_post(post)
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"Crawled {count} posts...")

                logger.info(f"Crawling completed. Total: {count} posts")

        finally:
            await store.close()

    asyncio.run(run_crawler())


@cli.command()
@click.option("--platform", "-p", type=str, default=None,
              help="指定平台 (可选)")
@click.option("--limit", "-l", type=int, default=1000,
              help="分析数量限制")
@click.pass_context
def analyze(ctx: click.Context, platform: Optional[str], limit: int) -> None:
    """执行情感分析"""
    settings = ctx.obj["settings"]

    async def run_analysis() -> None:
        store = SQLiteStore(settings.database_url.replace("sqlite:///", ""))
        await store.initialize()

        extractor = FeatureExtractor()
        extractor.initialize()

        try:
            posts = await store.search_posts("", platform, limit)
            logger.info(f"Found {len(posts)} posts to analyze")

            for i, post in enumerate(posts):
                content = post.get("content", "")
                if not content:
                    continue

                result = extractor.analyze_sentiment(content)

                await store.update_sentiment(
                    post.get("platform", ""),
                    post.get("post_id", ""),
                    result.score,
                    result.label
                )

                if (i + 1) % 100 == 0:
                    logger.info(f"Analyzed {i + 1} posts...")

            logger.info(f"Analysis completed. Total: {len(posts)} posts")

        finally:
            await store.close()

    asyncio.run(run_analysis())


@cli.command()
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """启动仪表盘"""
    settings = ctx.obj["settings"]

    from sentiment_analyzer.dashboard.app import run_dashboard
    run_dashboard(settings.dashboard_host, settings.dashboard_port)


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """显示统计信息"""
    settings = ctx.obj["settings"]

    async def show_stats() -> None:
        store = SQLiteStore(settings.database_url.replace("sqlite:///", ""))
        await store.initialize()

        try:
            stats_data = await store.get_statistics()
            click.echo("\n=== 数据库统计 ===")
            click.echo(f"总帖子数: {stats_data['total_posts']}")
            click.echo(f"总用户数: {stats_data['total_users']}")
            click.echo("\n各平台帖子数:")
            for platform, count in stats_data['posts_by_platform'].items():
                click.echo(f"  {platform}: {count}")

        finally:
            await store.close()

    asyncio.run(show_stats())


@cli.command()
@click.option("--force", "-f", is_flag=True, help="强制重新初始化")
@click.pass_context
def init(ctx: click.Context, force: bool) -> None:
    """初始化数据库"""
    settings = ctx.obj["settings"]

    async def init_db() -> None:
        import os

        data_dir = settings.data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")

        store = SQLiteStore(settings.database_url.replace("sqlite:///", ""))
        await store.initialize()

        graph_store = GraphStore(f"{data_dir}/graph_store.json")
        graph_store.initialize()

        search_engine = SearchEngine(f"{data_dir}/search_index.json")
        search_engine.initialize()

        logger.info("Database initialized successfully")

        await store.close()
        graph_store.close()
        search_engine.close()

    asyncio.run(init_db())


@cli.command()
@click.option("--platform", "-p", type=str, required=True,
              help="平台名称")
@click.option("--user-id", "-u", type=str, required=True,
              help="用户ID")
@click.pass_context
def user(ctx: click.Context, platform: str, user_id: str) -> None:
    """获取用户信息"""
    settings = ctx.obj["settings"]

    async def get_user_info() -> None:
        platform_enum = Platform(platform.lower())
        config = settings.get_crawler_config(platform_enum)

        crawler = _create_crawler(platform_enum, config, settings)

        async with crawler:
            user_data = await crawler.crawl_user(user_id)
            if user_data:
                click.echo(f"\n=== 用户信息 ===")
                click.echo(f"平台: {user_data.platform}")
                click.echo(f"用户ID: {user_data.user_id}")
                click.echo(f"用户名: {user_data.username}")
                click.echo(f"显示名: {user_data.display_name}")
                click.echo(f"粉丝数: {user_data.followers_count}")
                click.echo(f"关注数: {user_data.following_count}")
                click.echo(f"帖子数: {user_data.posts_count}")
                click.echo(f"已认证: {user_data.verified}")
            else:
                click.echo("用户不存在或获取失败")

    asyncio.run(get_user_info())


def _create_crawler(
    platform: Platform,
    config: CrawlerConfig,
    settings: Settings
):
    """创建爬虫实例"""
    if platform == Platform.TWITTER:
        return TwitterCrawler(config, settings.twitter_bearer_token)
    elif platform == Platform.WEIBO:
        return WeiboCrawler(config)
    elif platform == Platform.REDDIT:
        return RedditCrawler(
            config,
            settings.reddit_client_id,
            settings.reddit_client_secret
        )
    elif platform == Platform.TELEGRAM:
        return TelegramCrawler(
            config,
            settings.telegram_api_id,
            settings.telegram_api_hash
        )
    else:
        raise ValueError(f"Unsupported platform: {platform}")


if __name__ == "__main__":
    cli()
