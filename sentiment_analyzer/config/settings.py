"""
配置管理模块 - 使用 Pydantic 进行配置管理
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentiment_analyzer.crawlers.base import Platform


class Settings(BaseSettings):
    """应用程序配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    database_url: str = Field(
        default="sqlite:///./data/sentiment.db",
        description="数据库连接URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis连接URL"
    )

    twitter_bearer_token: Optional[str] = Field(
        default=None,
        description="Twitter API Bearer Token"
    )
    reddit_client_id: Optional[str] = Field(
        default=None,
        description="Reddit Client ID"
    )
    reddit_client_secret: Optional[str] = Field(
        default=None,
        description="Reddit Client Secret"
    )
    telegram_api_id: Optional[str] = Field(
        default=None,
        description="Telegram API ID"
    )
    telegram_api_hash: Optional[str] = Field(
        default=None,
        description="Telegram API Hash"
    )

    proxy_enabled: bool = Field(
        default=False,
        description="是否启用代理"
    )
    proxy_url: Optional[str] = Field(
        default=None,
        description="代理URL"
    )

    crawler_delay_min: float = Field(
        default=1.0,
        ge=0.1,
        description="爬虫请求最小延迟(秒)"
    )
    crawler_delay_max: float = Field(
        default=3.0,
        ge=0.1,
        description="爬虫请求最大延迟(秒)"
    )
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        description="最大并发请求数"
    )

    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    log_file: str = Field(
        default="./logs/sentiment_analyzer.log",
        description="日志文件路径"
    )

    data_dir: str = Field(
        default="./data",
        description="数据目录"
    )

    dashboard_host: str = Field(
        default="localhost",
        description="仪表盘主机地址"
    )
    dashboard_port: int = Field(
        default=8501,
        description="仪表盘端口"
    )

    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="情感分析模型名称"
    )

    anomaly_contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="异常检测污染率"
    )

    @field_validator("crawler_delay_max")
    @classmethod
    def validate_delay_range(cls, v: float, info) -> float:
        if "crawler_delay_min" in info.data and v < info.data["crawler_delay_min"]:
            raise ValueError("crawler_delay_max must be greater than crawler_delay_min")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    def get_crawler_config(self, platform: Platform) -> dict:
        """获取爬虫配置"""
        from ..crawlers.base import CrawlerConfig

        return CrawlerConfig(
            platform=platform,
            delay_min=self.crawler_delay_min,
            delay_max=self.crawler_delay_max,
            max_concurrent=self.max_concurrent_requests,
            proxy_enabled=self.proxy_enabled,
            proxy_url=self.proxy_url
        )


class DevelopmentSettings(Settings):
    """开发环境配置"""

    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """生产环境配置"""

    log_level: str = "WARNING"


class TestingSettings(Settings):
    """测试环境配置"""

    database_url: str = "sqlite:///./data/test_sentiment.db"
    log_level: str = "DEBUG"


@lru_cache()
def get_settings() -> Settings:
    """获取配置实例 (单例模式)"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


def clear_settings_cache() -> None:
    """清除配置缓存"""
    get_settings.cache_clear()
