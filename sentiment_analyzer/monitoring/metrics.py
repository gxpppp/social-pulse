"""
监控与告警模块

提供系统指标收集、业务指标追踪、告警规则管理和健康检查功能。

Classes:
    MetricsCollector: 系统指标收集器
    BusinessMetrics: 业务指标收集器
    MetricsRegistry: 指标注册中心
    AlertRule: 告警规则
    AlertManager: 告警管理器
    Alert: 告警数据结构
    HealthChecker: 健康检查器
"""

import asyncio
import json
import os
import smtplib
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from loguru import logger


class Severity(str, Enum):
    """告警严重程度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """健康状态"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricValue:
    """
    指标值数据结构

    Attributes:
        name: 指标名称
        value: 指标值
        timestamp: 时间戳
        labels: 标签字典
        unit: 单位
        description: 描述
    """

    name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "unit": self.unit,
            "description": self.description,
        }

    def to_prometheus(self) -> str:
        """转换为 Prometheus 格式"""
        labels_str = ""
        if self.labels:
            labels_str = "{" + ", ".join(f'{k}="{v}"' for k, v in self.labels.items()) + "}"

        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        if self.unit:
            lines.append(f"# UNIT {self.name} {self.unit}")
        lines.append(f"{self.name}{labels_str} {self.value}")

        return "\n".join(lines)


@dataclass
class Alert:
    """
    告警数据结构

    Attributes:
        rule_name: 规则名称
        severity: 严重程度
        message: 告警消息
        timestamp: 时间戳
        value: 触发值
        labels: 标签
        resolved: 是否已解决
        resolved_at: 解决时间
    """

    rule_name: str
    severity: Severity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    value: Union[int, float] = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }

    def resolve(self) -> None:
        """标记告警为已解决"""
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)


@dataclass
class AlertRule:
    """
    告警规则

    Attributes:
        name: 规则名称
        condition: 条件表达式 (支持: >, <, >=, <=, ==, !=)
        threshold: 阈值
        duration: 持续时间（秒）
        severity: 严重程度
        description: 描述
        labels: 标签
        enabled: 是否启用
    """

    name: str
    condition: str
    threshold: Union[int, float]
    duration: int = 0
    severity: Severity = Severity.WARNING
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    _violation_start: Optional[datetime] = field(default=None, repr=False, compare=False)

    def evaluate(self, metrics: Dict[str, Union[int, float]]) -> Optional[Alert]:
        """
        评估规则

        Args:
            metrics: 指标字典

        Returns:
            如果触发告警则返回 Alert 对象，否则返回 None
        """
        if not self.enabled:
            return None

        metric_value = metrics.get(self.name)
        if metric_value is None:
            return None

        is_violated = self._check_condition(metric_value)

        if is_violated:
            if self._violation_start is None:
                self._violation_start = datetime.now(timezone.utc)
                if self.duration == 0:
                    return self._create_alert(metric_value)
            else:
                elapsed = (datetime.now(timezone.utc) - self._violation_start).total_seconds()
                if elapsed >= self.duration:
                    return self._create_alert(metric_value)
        else:
            self._violation_start = None

        return None

    def _check_condition(self, value: Union[int, float]) -> bool:
        """检查条件是否满足"""
        try:
            if self.condition == ">":
                return value > self.threshold
            elif self.condition == "<":
                return value < self.threshold
            elif self.condition == ">=":
                return value >= self.threshold
            elif self.condition == "<=":
                return value <= self.threshold
            elif self.condition == "==":
                return value == self.threshold
            elif self.condition == "!=":
                return value != self.threshold
            else:
                logger.warning(f"Unknown condition: {self.condition}")
                return False
        except (TypeError, ValueError) as e:
            logger.error(f"Error evaluating condition: {e}")
            return False

    def _create_alert(self, value: Union[int, float]) -> Alert:
        """创建告警"""
        return Alert(
            rule_name=self.name,
            severity=self.severity,
            message=f"{self.name} {self.condition} {self.threshold} (current: {value})",
            value=value,
            labels=self.labels.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "condition": self.condition,
            "threshold": self.threshold,
            "duration": self.duration,
            "severity": self.severity.value,
            "description": self.description,
            "labels": self.labels,
            "enabled": self.enabled,
        }


class NotificationChannel(ABC):
    """通知渠道抽象基类"""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """
        发送告警通知

        Args:
            alert: 告警对象

        Returns:
            发送是否成功
        """
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """检查渠道是否已配置"""
        pass


class EmailChannel(NotificationChannel):
    """
    邮件通知渠道

    Attributes:
        smtp_host: SMTP 服务器地址
        smtp_port: SMTP 端口
        smtp_user: SMTP 用户名
        smtp_password: SMTP 密码
        from_addr: 发件人地址
        to_addrs: 收件人地址列表
        use_tls: 是否使用 TLS
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        from_addr: str = "",
        to_addrs: List[str] = None,
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_addr = from_addr or smtp_user
        self.to_addrs = to_addrs or []
        self.use_tls = use_tls

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.smtp_host and self.to_addrs)

    async def send(self, alert: Alert) -> bool:
        """发送邮件通知"""
        if not self.is_configured():
            logger.warning("Email channel not configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] Alert: {alert.rule_name}"
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            text_content = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value}
Message: {alert.message}
Value: {alert.value}
Time: {alert.timestamp.isoformat()}
Labels: {json.dumps(alert.labels)}
            """.strip()

            html_content = f"""
<html>
<body>
<h2>Alert: {alert.rule_name}</h2>
<table>
    <tr><td><b>Severity:</b></td><td>{alert.severity.value}</td></tr>
    <tr><td><b>Message:</b></td><td>{alert.message}</td></tr>
    <tr><td><b>Value:</b></td><td>{alert.value}</td></tr>
    <tr><td><b>Time:</b></td><td>{alert.timestamp.isoformat()}</td></tr>
    <tr><td><b>Labels:</b></td><td>{json.dumps(alert.labels)}</td></tr>
</table>
</body>
</html>
            """.strip()

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            def send_email():
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    if self.use_tls:
                        server.starttls()
                    if self.smtp_user and self.smtp_password:
                        server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            await asyncio.get_event_loop().run_in_executor(None, send_email)
            logger.info(f"Email alert sent for {alert.rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """
    Webhook 通知渠道

    Attributes:
        url: Webhook URL
        headers: 请求头
        timeout: 超时时间（秒）
    """

    def __init__(
        self,
        url: str,
        headers: Dict[str, str] = None,
        timeout: int = 30,
    ):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._session = None

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.url)

    async def _get_session(self):
        """获取 aiohttp session"""
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """关闭 session"""
        if self._session:
            await self._session.close()
            self._session = None

    async def send(self, alert: Alert) -> bool:
        """发送 Webhook 通知"""
        if not self.is_configured():
            logger.warning("Webhook channel not configured")
            return False

        try:
            import aiohttp

            session = await self._get_session()
            payload = alert.to_dict()

            async with session.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status >= 200 and response.status < 300:
                    logger.info(f"Webhook alert sent for {alert.rule_name}")
                    return True
                else:
                    logger.error(f"Webhook returned status {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class SlackChannel(NotificationChannel):
    """
    Slack 通知渠道

    Attributes:
        webhook_url: Slack Webhook URL
        channel: 频道名称
        username: 机器人名称
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = "",
        username: str = "Alert Bot",
    ):
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self._webhook = None

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.webhook_url)

    async def send(self, alert: Alert) -> bool:
        """发送 Slack 通知"""
        if not self.is_configured():
            logger.warning("Slack channel not configured")
            return False

        try:
            import aiohttp

            color_map = {
                Severity.INFO: "#36a64f",
                Severity.WARNING: "#ff9900",
                Severity.ERROR: "#ff0000",
                Severity.CRITICAL: "#8b0000",
            }

            payload = {
                "username": self.username,
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": f"[{alert.severity.value.upper()}] {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Value", "value": str(alert.value), "short": True},
                            {"title": "Time", "value": alert.timestamp.isoformat(), "short": True},
                        ],
                        "footer": "Sentiment Analyzer",
                        "ts": int(alert.timestamp.timestamp()),
                    }
                ],
            }

            if self.channel:
                payload["channel"] = self.channel

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.rule_name}")
                        return True
                    else:
                        logger.error(f"Slack returned status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class MetricsCollector:
    """
    系统指标收集器

    收集 CPU、内存、磁盘、网络、进程、数据库和队列指标。

    Attributes:
        collection_interval: 收集间隔（秒）
        history_size: 历史记录大小
    """

    def __init__(
        self,
        collection_interval: int = 60,
        history_size: int = 100,
    ):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self._metrics_history: Dict[str, List[MetricValue]] = defaultdict(list)
        self._last_network_io = None
        self._last_network_time = None

    def collect_system_metrics(self) -> Dict[str, MetricValue]:
        """
        收集系统指标

        Returns:
            包含 CPU、内存、磁盘、网络指标的字典
        """
        metrics = {}

        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, returning mock metrics")
            return self._get_mock_system_metrics()

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics["cpu_usage"] = MetricValue(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                description="CPU usage percentage",
            )

            cpu_count = psutil.cpu_count()
            metrics["cpu_count"] = MetricValue(
                name="cpu_count",
                value=cpu_count,
                unit="count",
                description="Number of CPU cores",
            )

            memory = psutil.virtual_memory()
            metrics["memory_usage"] = MetricValue(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                description="Memory usage percentage",
            )
            metrics["memory_available"] = MetricValue(
                name="memory_available",
                value=memory.available / (1024**3),
                unit="GB",
                description="Available memory in GB",
            )
            metrics["memory_total"] = MetricValue(
                name="memory_total",
                value=memory.total / (1024**3),
                unit="GB",
                description="Total memory in GB",
            )

            disk = psutil.disk_usage("/")
            metrics["disk_usage"] = MetricValue(
                name="disk_usage",
                value=disk.percent,
                unit="percent",
                description="Disk usage percentage",
            )
            metrics["disk_free"] = MetricValue(
                name="disk_free",
                value=disk.free / (1024**3),
                unit="GB",
                description="Free disk space in GB",
            )

            network_io = psutil.net_io_counters()
            current_time = time.time()

            if self._last_network_io is not None and self._last_network_time is not None:
                time_delta = current_time - self._last_network_time
                if time_delta > 0:
                    bytes_sent_rate = (network_io.bytes_sent - self._last_network_io.bytes_sent) / time_delta
                    bytes_recv_rate = (network_io.bytes_recv - self._last_network_io.bytes_recv) / time_delta

                    metrics["network_send_rate"] = MetricValue(
                        name="network_send_rate",
                        value=bytes_sent_rate / 1024,
                        unit="KB/s",
                        description="Network send rate in KB/s",
                    )
                    metrics["network_recv_rate"] = MetricValue(
                        name="network_recv_rate",
                        value=bytes_recv_rate / 1024,
                        unit="KB/s",
                        description="Network receive rate in KB/s",
                    )

            self._last_network_io = network_io
            self._last_network_time = current_time

            metrics["network_bytes_sent"] = MetricValue(
                name="network_bytes_sent",
                value=network_io.bytes_sent / (1024**2),
                unit="MB",
                description="Total bytes sent in MB",
            )
            metrics["network_bytes_recv"] = MetricValue(
                name="network_bytes_recv",
                value=network_io.bytes_recv / (1024**2),
                unit="MB",
                description="Total bytes received in MB",
            )

            load_avg = os.getloadavg()
            metrics["load_avg_1"] = MetricValue(
                name="load_avg_1",
                value=load_avg[0],
                description="1-minute load average",
            )
            metrics["load_avg_5"] = MetricValue(
                name="load_avg_5",
                value=load_avg[1],
                description="5-minute load average",
            )
            metrics["load_avg_15"] = MetricValue(
                name="load_avg_15",
                value=load_avg[2],
                description="15-minute load average",
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        for metric in metrics.values():
            self._add_to_history(metric)

        return metrics

    def _get_mock_system_metrics(self) -> Dict[str, MetricValue]:
        """返回模拟的系统指标（当 psutil 不可用时）"""
        return {
            "cpu_usage": MetricValue("cpu_usage", 0.0, unit="percent", description="CPU usage (mock)"),
            "memory_usage": MetricValue("memory_usage", 0.0, unit="percent", description="Memory usage (mock)"),
            "disk_usage": MetricValue("disk_usage", 0.0, unit="percent", description="Disk usage (mock)"),
        }

    def collect_process_metrics(self, pid: Optional[int] = None) -> Dict[str, MetricValue]:
        """
        收集进程资源使用指标

        Args:
            pid: 进程 ID，默认为当前进程

        Returns:
            包含进程指标的字典
        """
        metrics = {}

        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, returning mock process metrics")
            return {
                "process_cpu": MetricValue("process_cpu", 0.0, unit="percent"),
                "process_memory": MetricValue("process_memory", 0.0, unit="MB"),
            }

        try:
            process = psutil.Process(pid)

            metrics["process_cpu"] = MetricValue(
                name="process_cpu",
                value=process.cpu_percent(),
                unit="percent",
                description="Process CPU usage percentage",
            )

            memory_info = process.memory_info()
            metrics["process_memory"] = MetricValue(
                name="process_memory",
                value=memory_info.rss / (1024**2),
                unit="MB",
                description="Process memory usage in MB",
            )
            metrics["process_memory_virtual"] = MetricValue(
                name="process_memory_virtual",
                value=memory_info.vms / (1024**2),
                unit="MB",
                description="Process virtual memory in MB",
            )

            try:
                io_counters = process.io_counters()
                metrics["process_io_read"] = MetricValue(
                    name="process_io_read",
                    value=io_counters.read_bytes / (1024**2),
                    unit="MB",
                    description="Process IO read bytes in MB",
                )
                metrics["process_io_write"] = MetricValue(
                    name="process_io_write",
                    value=io_counters.write_bytes / (1024**2),
                    unit="MB",
                    description="Process IO write bytes in MB",
                )
            except (AttributeError, psutil.AccessDenied):
                pass

            num_threads = process.num_threads()
            metrics["process_threads"] = MetricValue(
                name="process_threads",
                value=num_threads,
                unit="count",
                description="Number of threads",
            )

            try:
                num_fds = process.num_fds() if hasattr(process, "num_fds") else 0
                metrics["process_file_descriptors"] = MetricValue(
                    name="process_file_descriptors",
                    value=num_fds,
                    unit="count",
                    description="Number of file descriptors",
                )
            except (AttributeError, psutil.AccessDenied):
                pass

            metrics["process_connections"] = MetricValue(
                name="process_connections",
                value=len(process.connections()),
                unit="count",
                description="Number of network connections",
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error collecting process metrics: {e}")

        for metric in metrics.values():
            self._add_to_history(metric)

        return metrics

    def collect_database_metrics(
        self,
        connection_count: int = 0,
        query_latency: float = 0.0,
        active_queries: int = 0,
        connection_pool_size: int = 0,
    ) -> Dict[str, MetricValue]:
        """
        收集数据库指标

        Args:
            connection_count: 连接数
            query_latency: 查询延迟（毫秒）
            active_queries: 活跃查询数
            connection_pool_size: 连接池大小

        Returns:
            包含数据库指标的字典
        """
        metrics = {
            "db_connections": MetricValue(
                name="db_connections",
                value=connection_count,
                unit="count",
                description="Database connection count",
            ),
            "db_query_latency": MetricValue(
                name="db_query_latency",
                value=query_latency,
                unit="ms",
                description="Database query latency in milliseconds",
            ),
            "db_active_queries": MetricValue(
                name="db_active_queries",
                value=active_queries,
                unit="count",
                description="Number of active database queries",
            ),
            "db_pool_size": MetricValue(
                name="db_pool_size",
                value=connection_pool_size,
                unit="count",
                description="Database connection pool size",
            ),
        }

        for metric in metrics.values():
            self._add_to_history(metric)

        return metrics

    def collect_queue_metrics(
        self,
        queue_length: int = 0,
        processing_rate: float = 0.0,
        failed_tasks: int = 0,
        avg_wait_time: float = 0.0,
    ) -> Dict[str, MetricValue]:
        """
        收集队列指标

        Args:
            queue_length: 队列长度
            processing_rate: 处理速率（任务/秒）
            failed_tasks: 失败任务数
            avg_wait_time: 平均等待时间（秒）

        Returns:
            包含队列指标的字典
        """
        metrics = {
            "queue_length": MetricValue(
                name="queue_length",
                value=queue_length,
                unit="count",
                description="Queue length",
            ),
            "queue_processing_rate": MetricValue(
                name="queue_processing_rate",
                value=processing_rate,
                unit="tasks/s",
                description="Queue processing rate",
            ),
            "queue_failed_tasks": MetricValue(
                name="queue_failed_tasks",
                value=failed_tasks,
                unit="count",
                description="Number of failed queue tasks",
            ),
            "queue_avg_wait_time": MetricValue(
                name="queue_avg_wait_time",
                value=avg_wait_time,
                unit="seconds",
                description="Average queue wait time in seconds",
            ),
        }

        for metric in metrics.values():
            self._add_to_history(metric)

        return metrics

    def _add_to_history(self, metric: MetricValue) -> None:
        """添加指标到历史记录"""
        self._metrics_history[metric.name].append(metric)
        if len(self._metrics_history[metric.name]) > self.history_size:
            self._metrics_history[metric.name].pop(0)

    def get_history(self, metric_name: str) -> List[MetricValue]:
        """
        获取指标历史记录

        Args:
            metric_name: 指标名称

        Returns:
            指标历史记录列表
        """
        return self._metrics_history.get(metric_name, [])

    def get_all_metrics(self) -> Dict[str, List[MetricValue]]:
        """获取所有指标历史记录"""
        return dict(self._metrics_history)

    def clear_history(self) -> None:
        """清空历史记录"""
        self._metrics_history.clear()


class BusinessMetrics:
    """
    业务指标收集器

    收集爬虫成功率、延迟、帖子数、异常检测率等业务指标。

    Attributes:
        time_window: 时间窗口（秒）
    """

    def __init__(self, time_window: int = 300):
        self.time_window = time_window
        self._crawl_attempts: List[datetime] = []
        self._crawl_successes: List[datetime] = []
        self._crawl_latencies: List[tuple] = []
        self._posts_collected: List[datetime] = []
        self._anomalies_detected: List[datetime] = []
        self._anomalies_total: List[datetime] = []
        self._active_users: Dict[str, datetime] = {}

    def record_crawl_attempt(self, success: bool, latency: float = 0.0) -> None:
        """
        记录爬虫尝试

        Args:
            success: 是否成功
            latency: 延迟（秒）
        """
        now = datetime.now(timezone.utc)
        self._crawl_attempts.append(now)
        if success:
            self._crawl_successes.append(now)
        if latency > 0:
            self._crawl_latencies.append((now, latency))
        self._cleanup_old_records()

    def record_post_collected(self, count: int = 1) -> None:
        """
        记录帖子采集

        Args:
            count: 采集数量
        """
        now = datetime.now(timezone.utc)
        for _ in range(count):
            self._posts_collected.append(now)
        self._cleanup_old_records()

    def record_anomaly_detection(self, detected: bool) -> None:
        """
        记录异常检测

        Args:
            detected: 是否检测到异常
        """
        now = datetime.now(timezone.utc)
        self._anomalies_total.append(now)
        if detected:
            self._anomalies_detected.append(now)
        self._cleanup_old_records()

    def record_active_user(self, user_id: str) -> None:
        """
        记录活跃用户

        Args:
            user_id: 用户 ID
        """
        self._active_users[user_id] = datetime.now(timezone.utc)
        self._cleanup_old_records()

    def _cleanup_old_records(self) -> None:
        """清理过期记录"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.time_window)

        self._crawl_attempts = [t for t in self._crawl_attempts if t > cutoff]
        self._crawl_successes = [t for t in self._crawl_successes if t > cutoff]
        self._crawl_latencies = [(t, l) for t, l in self._crawl_latencies if t > cutoff]
        self._posts_collected = [t for t in self._posts_collected if t > cutoff]
        self._anomalies_detected = [t for t in self._anomalies_detected if t > cutoff]
        self._anomalies_total = [t for t in self._anomalies_total if t > cutoff]
        self._active_users = {k: v for k, v in self._active_users.items() if v > cutoff}

    def crawl_success_rate(self) -> MetricValue:
        """
        计算采集成功率

        Returns:
            成功率指标
        """
        self._cleanup_old_records()

        if not self._crawl_attempts:
            rate = 1.0
        else:
            rate = len(self._crawl_successes) / len(self._crawl_attempts)

        return MetricValue(
            name="crawl_success_rate",
            value=rate * 100,
            unit="percent",
            description="Crawl success rate percentage",
        )

    def crawl_latency(self) -> MetricValue:
        """
        计算采集延迟

        Returns:
            平均延迟指标
        """
        self._cleanup_old_records()

        if not self._crawl_latencies:
            avg_latency = 0.0
        else:
            latencies = [l for _, l in self._crawl_latencies]
            avg_latency = sum(latencies) / len(latencies)

        return MetricValue(
            name="crawl_latency",
            value=avg_latency * 1000,
            unit="ms",
            description="Average crawl latency in milliseconds",
        )

    def posts_per_minute(self) -> MetricValue:
        """
        计算每分钟帖子数

        Returns:
            每分钟帖子数指标
        """
        self._cleanup_old_records()

        minutes = self.time_window / 60
        rate = len(self._posts_collected) / minutes if minutes > 0 else 0

        return MetricValue(
            name="posts_per_minute",
            value=rate,
            unit="posts/min",
            description="Posts collected per minute",
        )

    def anomaly_detection_rate(self) -> MetricValue:
        """
        计算异常检测率

        Returns:
            异常检测率指标
        """
        self._cleanup_old_records()

        if not self._anomalies_total:
            rate = 0.0
        else:
            rate = len(self._anomalies_detected) / len(self._anomalies_total)

        return MetricValue(
            name="anomaly_detection_rate",
            value=rate * 100,
            unit="percent",
            description="Anomaly detection rate percentage",
        )

    def active_users_count(self) -> MetricValue:
        """
        计算活跃用户数

        Returns:
            活跃用户数指标
        """
        self._cleanup_old_records()

        return MetricValue(
            name="active_users_count",
            value=len(self._active_users),
            unit="count",
            description="Number of active users",
        )

    def get_all_metrics(self) -> Dict[str, MetricValue]:
        """获取所有业务指标"""
        return {
            "crawl_success_rate": self.crawl_success_rate(),
            "crawl_latency": self.crawl_latency(),
            "posts_per_minute": self.posts_per_minute(),
            "anomaly_detection_rate": self.anomaly_detection_rate(),
            "active_users_count": self.active_users_count(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取业务指标摘要"""
        metrics = self.get_all_metrics()
        return {
            "time_window_seconds": self.time_window,
            "crawl_attempts": len(self._crawl_attempts),
            "crawl_successes": len(self._crawl_successes),
            "posts_collected": len(self._posts_collected),
            "anomalies_detected": len(self._anomalies_detected),
            "active_users": len(self._active_users),
            "metrics": {k: v.value for k, v in metrics.items()},
        }


class MetricsRegistry:
    """
    指标注册中心

    管理和导出所有注册的指标。

    Attributes:
        prefix: 指标名称前缀
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._metrics: Dict[str, Callable[[], MetricValue]] = {}
        self._labels: Dict[str, Dict[str, str]] = {}

    def register(
        self,
        name: str,
        collector: Callable[[], MetricValue],
        labels: Dict[str, str] = None,
    ) -> None:
        """
        注册指标

        Args:
            name: 指标名称
            collector: 指标收集函数
            labels: 标签字典
        """
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        self._metrics[full_name] = collector
        if labels:
            self._labels[full_name] = labels
        logger.debug(f"Registered metric: {full_name}")

    def unregister(self, name: str) -> bool:
        """
        注销指标

        Args:
            name: 指标名称

        Returns:
            是否成功注销
        """
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        if full_name in self._metrics:
            del self._metrics[full_name]
            self._labels.pop(full_name, None)
            return True
        return False

    def get(self, name: str) -> Optional[MetricValue]:
        """
        获取指标

        Args:
            name: 指标名称

        Returns:
            指标值，如果不存在则返回 None
        """
        full_name = f"{self.prefix}_{name}" if self.prefix else name
        collector = self._metrics.get(full_name)
        if collector:
            metric = collector()
            if full_name in self._labels:
                metric.labels = {**self._labels[full_name], **metric.labels}
            return metric
        return None

    def get_all(self) -> Dict[str, MetricValue]:
        """
        获取所有指标

        Returns:
            指标字典
        """
        result = {}
        for name, collector in self._metrics.items():
            try:
                metric = collector()
                if name in self._labels:
                    metric.labels = {**self._labels[name], **metric.labels}
                result[name] = metric
            except Exception as e:
                logger.error(f"Error collecting metric {name}: {e}")
        return result

    def export_prometheus(self) -> str:
        """
        导出 Prometheus 格式

        Returns:
            Prometheus 格式的指标字符串
        """
        metrics = self.get_all()
        lines = []

        for name, metric in metrics.items():
            lines.append(metric.to_prometheus())

        return "\n".join(lines)

    def export_json(self) -> str:
        """
        导出 JSON 格式

        Returns:
            JSON 格式的指标字符串
        """
        metrics = self.get_all()
        return json.dumps(
            {name: metric.to_dict() for name, metric in metrics.items()},
            indent=2,
        )

    def get_metric_names(self) -> List[str]:
        """获取所有注册的指标名称"""
        return list(self._metrics.keys())


class AlertManager:
    """
    告警管理器

    管理告警规则、检查告警并发送通知。

    Attributes:
        check_interval: 检查间隔（秒）
    """

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self._rules: Dict[str, AlertRule] = {}
        self._channels: List[NotificationChannel] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._history_size = 1000

    def add_rule(self, rule: AlertRule) -> None:
        """
        添加告警规则

        Args:
            rule: 告警规则
        """
        self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """
        移除告警规则

        Args:
            rule_name: 规则名称

        Returns:
            是否成功移除
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """
        获取告警规则

        Args:
            rule_name: 规则名称

        Returns:
            告警规则，如果不存在则返回 None
        """
        return self._rules.get(rule_name)

    def get_all_rules(self) -> List[AlertRule]:
        """获取所有告警规则"""
        return list(self._rules.values())

    def add_channel(self, channel: NotificationChannel) -> None:
        """
        添加通知渠道

        Args:
            channel: 通知渠道
        """
        self._channels.append(channel)
        logger.info(f"Added notification channel: {channel.__class__.__name__}")

    def remove_channel(self, channel: NotificationChannel) -> None:
        """
        移除通知渠道

        Args:
            channel: 通知渠道
        """
        if channel in self._channels:
            self._channels.remove(channel)

    def check_alerts(self, metrics: Dict[str, Union[int, float]]) -> List[Alert]:
        """
        检查告警

        Args:
            metrics: 指标字典

        Returns:
            触发的告警列表
        """
        triggered_alerts = []

        for rule_name, rule in self._rules.items():
            alert = rule.evaluate(metrics)
            if alert:
                if rule_name not in self._active_alerts:
                    self._active_alerts[rule_name] = alert
                    self._add_to_history(alert)
                    triggered_alerts.append(alert)
                    logger.warning(f"Alert triggered: {rule_name} - {alert.message}")

        for rule_name in list(self._active_alerts.keys()):
            if rule_name in self._rules:
                rule = self._rules[rule_name]
                metric_value = metrics.get(rule_name)
                if metric_value is not None:
                    if not rule._check_condition(metric_value):
                        self._active_alerts[rule_name].resolve()
                        logger.info(f"Alert resolved: {rule_name}")
                        del self._active_alerts[rule_name]

        return triggered_alerts

    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        发送告警

        Args:
            alert: 告警对象

        Returns:
            各渠道发送结果字典
        """
        results = {}

        for channel in self._channels:
            if channel.is_configured():
                try:
                    success = await channel.send(alert)
                    results[channel.__class__.__name__] = success
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.__class__.__name__}: {e}")
                    results[channel.__class__.__name__] = False

        return results

    async def check_and_notify(self, metrics: Dict[str, Union[int, float]]) -> List[Alert]:
        """
        检查告警并发送通知

        Args:
            metrics: 指标字典

        Returns:
            触发的告警列表
        """
        alerts = self.check_alerts(metrics)

        for alert in alerts:
            await self.send_alert(alert)

        return alerts

    def _add_to_history(self, alert: Alert) -> None:
        """添加告警到历史记录"""
        self._alert_history.append(alert)
        if len(self._alert_history) > self._history_size:
            self._alert_history.pop(0)

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """
        获取告警历史

        Args:
            limit: 返回数量限制

        Returns:
            告警历史列表
        """
        return self._alert_history[-limit:]

    def clear_alert(self, rule_name: str) -> bool:
        """
        清除告警

        Args:
            rule_name: 规则名称

        Returns:
            是否成功清除
        """
        if rule_name in self._active_alerts:
            self._active_alerts[rule_name].resolve()
            del self._active_alerts[rule_name]
            logger.info(f"Alert cleared: {rule_name}")
            return True
        return False

    def get_alert_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        severity_counts = defaultdict(int)
        for alert in self._active_alerts.values():
            severity_counts[alert.severity.value] += 1

        return {
            "active_alerts_count": len(self._active_alerts),
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "channels_count": len(self._channels),
            "configured_channels": sum(1 for c in self._channels if c.is_configured()),
            "severity_breakdown": dict(severity_counts),
            "history_size": len(self._alert_history),
        }


class HealthChecker:
    """
    健康检查器

    检查系统各组件的健康状态。

    Attributes:
        timeout: 检查超时时间（秒）
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._last_checks: Dict[str, Dict[str, Any]] = {}

    async def check_database(
        self,
        db_manager=None,
        connection_string: str = "",
    ) -> Dict[str, Any]:
        """
        检查数据库健康状态

        Args:
            db_manager: 数据库管理器
            connection_string: 数据库连接字符串

        Returns:
            检查结果字典
        """
        result = {
            "component": "database",
            "status": HealthStatus.UNKNOWN.value,
            "latency_ms": 0,
            "message": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        start_time = time.time()

        try:
            if db_manager is not None:
                async with db_manager.session() as session:
                    await session.execute("SELECT 1")

                result["status"] = HealthStatus.HEALTHY.value
                result["message"] = "Database connection successful"
            elif connection_string:
                import aiohttp

                result["status"] = HealthStatus.DEGRADED.value
                result["message"] = "Connection string provided but no manager"
            else:
                result["status"] = HealthStatus.UNKNOWN.value
                result["message"] = "No database configuration provided"

        except asyncio.TimeoutError:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = "Database connection timeout"
        except Exception as e:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = f"Database error: {str(e)}"
        finally:
            result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

        self._last_checks["database"] = result
        return result

    async def check_queue(
        self,
        redis_url: str = "",
        queue_name: str = "default",
    ) -> Dict[str, Any]:
        """
        检查队列健康状态

        Args:
            redis_url: Redis 连接 URL
            queue_name: 队列名称

        Returns:
            检查结果字典
        """
        result = {
            "component": "queue",
            "status": HealthStatus.UNKNOWN.value,
            "latency_ms": 0,
            "message": "",
            "queue_length": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        start_time = time.time()

        try:
            if redis_url:
                import redis.asyncio as redis

                client = redis.from_url(redis_url)
                await client.ping()

                queue_length = await client.llen(queue_name)
                result["queue_length"] = queue_length

                await client.close()

                result["status"] = HealthStatus.HEALTHY.value
                result["message"] = f"Queue '{queue_name}' is healthy"
            else:
                result["status"] = HealthStatus.UNKNOWN.value
                result["message"] = "No Redis URL provided"

        except asyncio.TimeoutError:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = "Queue connection timeout"
        except Exception as e:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = f"Queue error: {str(e)}"
        finally:
            result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

        self._last_checks["queue"] = result
        return result

    async def check_storage(
        self,
        path: str = "./data",
        min_free_gb: float = 1.0,
    ) -> Dict[str, Any]:
        """
        检查存储健康状态

        Args:
            path: 存储路径
            min_free_gb: 最小可用空间（GB）

        Returns:
            检查结果字典
        """
        result = {
            "component": "storage",
            "status": HealthStatus.UNKNOWN.value,
            "latency_ms": 0,
            "message": "",
            "path": path,
            "free_gb": 0,
            "total_gb": 0,
            "usage_percent": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        start_time = time.time()

        try:
            if os.path.exists(path):
                if PSUTIL_AVAILABLE:
                    disk = psutil.disk_usage(path)
                    free_gb = disk.free / (1024**3)
                    total_gb = disk.total / (1024**3)
                    usage_percent = disk.percent

                    result["free_gb"] = round(free_gb, 2)
                    result["total_gb"] = round(total_gb, 2)
                    result["usage_percent"] = usage_percent

                    if free_gb < min_free_gb:
                        result["status"] = HealthStatus.UNHEALTHY.value
                        result["message"] = f"Low disk space: {free_gb:.2f}GB free (min: {min_free_gb}GB)"
                    elif usage_percent > 90:
                        result["status"] = HealthStatus.DEGRADED.value
                        result["message"] = f"Disk usage high: {usage_percent}%"
                    else:
                        result["status"] = HealthStatus.HEALTHY.value
                        result["message"] = f"Storage healthy: {free_gb:.2f}GB free"
                else:
                    result["status"] = HealthStatus.HEALTHY.value
                    result["message"] = "Storage path exists (psutil not available)"
            else:
                result["status"] = HealthStatus.UNHEALTHY.value
                result["message"] = f"Storage path does not exist: {path}"

        except Exception as e:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = f"Storage error: {str(e)}"
        finally:
            result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

        self._last_checks["storage"] = result
        return result

    async def check_external_api(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        timeout: int = 5,
    ) -> Dict[str, Any]:
        """
        检查外部 API 健康状态

        Args:
            name: API 名称
            url: API URL
            expected_status: 期望的 HTTP 状态码
            timeout: 超时时间（秒）

        Returns:
            检查结果字典
        """
        result = {
            "component": f"api_{name}",
            "status": HealthStatus.UNKNOWN.value,
            "latency_ms": 0,
            "message": "",
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        start_time = time.time()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

                    if response.status == expected_status:
                        result["status"] = HealthStatus.HEALTHY.value
                        result["message"] = f"API returned status {response.status}"
                    else:
                        result["status"] = HealthStatus.DEGRADED.value
                        result["message"] = f"API returned unexpected status {response.status}"

        except asyncio.TimeoutError:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = "API request timeout"
        except Exception as e:
            result["status"] = HealthStatus.UNHEALTHY.value
            result["message"] = f"API error: {str(e)}"
        finally:
            result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

        self._last_checks[f"api_{name}"] = result
        return result

    async def get_health_status(self) -> Dict[str, Any]:
        """
        获取整体健康状态

        Returns:
            整体健康状态字典
        """
        if not self._last_checks:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks performed yet",
                "components": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        statuses = [check.get("status") for check in self._last_checks.values()]

        if all(s == HealthStatus.HEALTHY.value for s in statuses):
            overall_status = HealthStatus.HEALTHY.value
            message = "All components are healthy"
        elif any(s == HealthStatus.UNHEALTHY.value for s in statuses):
            overall_status = HealthStatus.UNHEALTHY.value
            unhealthy = [k for k, v in self._last_checks.items() if v.get("status") == HealthStatus.UNHEALTHY.value]
            message = f"Unhealthy components: {', '.join(unhealthy)}"
        elif any(s == HealthStatus.DEGRADED.value for s in statuses):
            overall_status = HealthStatus.DEGRADED.value
            degraded = [k for k, v in self._last_checks.items() if v.get("status") == HealthStatus.DEGRADED.value]
            message = f"Degraded components: {', '.join(degraded)}"
        else:
            overall_status = HealthStatus.UNKNOWN.value
            message = "Some components have unknown status"

        return {
            "status": overall_status,
            "message": message,
            "components": self._last_checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def run_all_checks(
        self,
        db_manager=None,
        redis_url: str = "",
        storage_path: str = "./data",
    ) -> Dict[str, Any]:
        """
        运行所有健康检查

        Args:
            db_manager: 数据库管理器
            redis_url: Redis URL
            storage_path: 存储路径

        Returns:
            所有检查结果
        """
        tasks = [
            self.check_database(db_manager),
            self.check_queue(redis_url),
            self.check_storage(storage_path),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        checks = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component = ["database", "queue", "storage"][i]
                checks[component] = {
                    "component": component,
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Check failed: {str(result)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                checks[result["component"]] = result

        return await self.get_health_status()

    def get_last_checks(self) -> Dict[str, Dict[str, Any]]:
        """获取最后一次检查结果"""
        return self._last_checks.copy()


def create_default_alert_rules() -> List[AlertRule]:
    """
    创建默认告警规则

    Returns:
        默认告警规则列表
    """
    return [
        AlertRule(
            name="cpu_usage",
            condition=">",
            threshold=80,
            duration=60,
            severity=Severity.WARNING,
            description="CPU usage above 80% for 1 minute",
        ),
        AlertRule(
            name="cpu_usage",
            condition=">",
            threshold=95,
            duration=30,
            severity=Severity.CRITICAL,
            description="CPU usage above 95% for 30 seconds",
        ),
        AlertRule(
            name="memory_usage",
            condition=">",
            threshold=85,
            duration=60,
            severity=Severity.WARNING,
            description="Memory usage above 85% for 1 minute",
        ),
        AlertRule(
            name="memory_usage",
            condition=">",
            threshold=95,
            duration=30,
            severity=Severity.CRITICAL,
            description="Memory usage above 95% for 30 seconds",
        ),
        AlertRule(
            name="disk_usage",
            condition=">",
            threshold=85,
            duration=0,
            severity=Severity.WARNING,
            description="Disk usage above 85%",
        ),
        AlertRule(
            name="disk_usage",
            condition=">",
            threshold=95,
            duration=0,
            severity=Severity.CRITICAL,
            description="Disk usage above 95%",
        ),
        AlertRule(
            name="crawl_success_rate",
            condition="<",
            threshold=50,
            duration=120,
            severity=Severity.ERROR,
            description="Crawl success rate below 50% for 2 minutes",
        ),
        AlertRule(
            name="crawl_latency",
            condition=">",
            threshold=5000,
            duration=60,
            severity=Severity.WARNING,
            description="Crawl latency above 5 seconds for 1 minute",
        ),
        AlertRule(
            name="queue_length",
            condition=">",
            threshold=1000,
            duration=300,
            severity=Severity.WARNING,
            description="Queue length above 1000 for 5 minutes",
        ),
        AlertRule(
            name="db_query_latency",
            condition=">",
            threshold=1000,
            duration=60,
            severity=Severity.WARNING,
            description="Database query latency above 1 second for 1 minute",
        ),
    ]
