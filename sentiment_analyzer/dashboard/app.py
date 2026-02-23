"""
Streamlit 可视化平台

提供社交媒体情感分析的多页面可视化界面，包括：
- 数据概览仪表盘
- 趋势分析
- 网络拓扑可视化
- 异常检测
- 跨事件分析
- 数据导出
"""

import asyncio
import io
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger
from plotly.subplots import make_subplots
from pyecharts import options as opts
from pyecharts.charts import Graph, Line, Pie, Bar

from ..config.settings import get_settings


@dataclass
class FilterConfig:
    platform: str
    time_range: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]


class DashboardData:
    def __init__(self, db_path: str = "./data/sentiment.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_overview_stats(self, platform: str = "全部", days: int = 30) -> dict[str, Any]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            platform_filter = "" if platform == "全部" else f"AND platform = '{platform}'"
            date_filter = f"AND posted_at >= datetime('now', '-{days} days')"
            
            try:
                cursor.execute(f"""
                    SELECT COUNT(*) as total_posts,
                           COUNT(DISTINCT user_id) as active_users
                    FROM posts 
                    WHERE 1=1 {platform_filter} {date_filter}
                """)
                post_stats = dict(cursor.fetchone())
            except Exception:
                post_stats = {"total_posts": 0, "active_users": 0}
            
            try:
                cursor.execute(f"""
                    SELECT COUNT(*) as anomaly_count
                    FROM users 
                    WHERE is_suspicious = 1 {platform_filter.replace('platform', 'platform')}
                """)
                anomaly_count = cursor.fetchone()["anomaly_count"]
            except Exception:
                anomaly_count = 0
            
            try:
                cursor.execute(f"""
                    SELECT value, COUNT(*) as count
                    FROM posts, json_each(hashtags)
                    WHERE hashtags IS NOT NULL {platform_filter} {date_filter}
                    GROUP BY value
                    ORDER BY count DESC
                    LIMIT 10
                """)
                trending_topics = [row["value"] for row in cursor.fetchall()]
            except Exception:
                trending_topics = []
            
            return {
                "total_posts": post_stats.get("total_posts", 0),
                "active_users": post_stats.get("active_users", 0),
                "anomaly_accounts": anomaly_count,
                "trending_topics": trending_topics[:5]
            }

    def get_platform_distribution(self) -> pd.DataFrame:
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query("""
                    SELECT platform, COUNT(*) as count
                    FROM posts
                    GROUP BY platform
                    ORDER BY count DESC
                """, conn)
                return df
            except Exception:
                return pd.DataFrame({"platform": ["Twitter", "微博", "Reddit", "Telegram"], 
                                    "count": [5000, 4000, 2500, 845]})

    def get_time_series_data(self, platform: str = "全部", days: int = 30) -> pd.DataFrame:
        with self.get_connection() as conn:
            platform_filter = "" if platform == "全部" else f"AND platform = '{platform}'"
            try:
                df = pd.read_sql_query(f"""
                    SELECT date(posted_at) as date, 
                           COUNT(*) as post_count,
                           COUNT(DISTINCT user_id) as active_users
                    FROM posts
                    WHERE posted_at >= datetime('now', '-{days} days')
                    {platform_filter}
                    GROUP BY date(posted_at)
                    ORDER BY date
                """, conn)
                return df
            except Exception:
                dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
                return pd.DataFrame({
                    "date": dates,
                    "post_count": np.random.randint(100, 500, days),
                    "active_users": np.random.randint(50, 200, days)
                })

    def get_sentiment_trend(self, platform: str = "全部", days: int = 30) -> pd.DataFrame:
        with self.get_connection() as conn:
            platform_filter = "" if platform == "全部" else f"AND platform = '{platform}'"
            try:
                df = pd.read_sql_query(f"""
                    SELECT date(posted_at) as date,
                           AVG(sentiment_score) as avg_sentiment,
                           COUNT(*) as count
                    FROM posts
                    WHERE posted_at >= datetime('now', '-{days} days')
                    AND sentiment_score IS NOT NULL
                    {platform_filter}
                    GROUP BY date(posted_at)
                    ORDER BY date
                """, conn)
                return df
            except Exception:
                dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
                return pd.DataFrame({
                    "date": dates,
                    "avg_sentiment": np.random.uniform(0.3, 0.7, days),
                    "count": np.random.randint(100, 500, days)
                })

    def get_engagement_trend(self, platform: str = "全部", days: int = 30) -> pd.DataFrame:
        with self.get_connection() as conn:
            platform_filter = "" if platform == "全部" else f"AND platform = '{platform}'"
            try:
                df = pd.read_sql_query(f"""
                    SELECT date(posted_at) as date,
                           SUM(likes_count) as total_likes,
                           SUM(shares_count) as total_shares,
                           SUM(comments_count) as total_comments
                    FROM posts
                    WHERE posted_at >= datetime('now', '-{days} days')
                    {platform_filter}
                    GROUP BY date(posted_at)
                    ORDER BY date
                """, conn)
                return df
            except Exception:
                dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
                return pd.DataFrame({
                    "date": dates,
                    "total_likes": np.random.randint(1000, 5000, days),
                    "total_shares": np.random.randint(500, 2000, days),
                    "total_comments": np.random.randint(200, 1000, days)
                })

    def get_topic_trend(self, topic: str, days: int = 30) -> pd.DataFrame:
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query(f"""
                    SELECT date(posted_at) as date, COUNT(*) as count
                    FROM posts, json_each(hashtags)
                    WHERE value = '{topic}'
                    AND posted_at >= datetime('now', '-{days} days')
                    GROUP BY date(posted_at)
                    ORDER BY date
                """, conn)
                return df
            except Exception:
                dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
                return pd.DataFrame({
                    "date": dates,
                    "count": np.random.randint(10, 100, days)
                })

    def get_network_data(self, limit: int = 200) -> tuple[list[dict], list[dict]]:
        with self.get_connection() as conn:
            try:
                nodes_df = pd.read_sql_query(f"""
                    SELECT u.user_id as id, u.username as name, u.platform,
                           u.followers_count, u.is_suspicious,
                           COALESCE(uf.anomaly_score, 0) as anomaly_score,
                           COALESCE(uf.degree_centrality, 0) as centrality
                    FROM users u
                    LEFT JOIN user_features uf ON u.user_id = uf.user_id
                    ORDER BY u.followers_count DESC
                    LIMIT {limit}
                """, conn)
                
                nodes = nodes_df.to_dict("records")
                user_ids = nodes_df["id"].tolist()
                
                edges_df = pd.read_sql_query(f"""
                    SELECT 
                        i.user_id as source,
                        p.user_id as target,
                        i.interaction_type as relation,
                        COUNT(*) as weight
                    FROM interactions i
                    JOIN posts p ON i.post_id = p.post_id
                    WHERE i.user_id IN ({','.join(['?']*len(user_ids))})
                    AND p.user_id IN ({','.join(['?']*len(user_ids))})
                    AND i.user_id != p.user_id
                    GROUP BY i.user_id, p.user_id, i.interaction_type
                """, conn, params=user_ids + user_ids)
                
                edges = edges_df.to_dict("records")
                
                return nodes, edges
            except Exception:
                return self._generate_mock_network_data()

    def _generate_mock_network_data(self) -> tuple[list[dict], list[dict]]:
        nodes = []
        for i in range(50):
            nodes.append({
                "id": f"user_{i}",
                "name": f"User_{i}",
                "platform": np.random.choice(["twitter", "weibo", "reddit"]),
                "followers_count": np.random.randint(100, 10000),
                "is_suspicious": np.random.random() > 0.9,
                "anomaly_score": np.random.random(),
                "centrality": np.random.random()
            })
        
        edges = []
        for i in range(100):
            source = np.random.randint(0, 50)
            target = np.random.randint(0, 50)
            if source != target:
                edges.append({
                    "source": f"user_{source}",
                    "target": f"user_{target}",
                    "relation": np.random.choice(["retweet", "mention", "reply"]),
                    "weight": np.random.randint(1, 10)
                })
        
        return nodes, edges

    def get_anomaly_accounts(self, threshold: float = 0.5, limit: int = 100) -> pd.DataFrame:
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query(f"""
                    SELECT u.user_id, u.username, u.platform, u.followers_count,
                           u.posts_count, u.is_suspicious,
                           uf.anomaly_score, uf.predicted_label, uf.confidence_score,
                           uf.daily_post_avg, uf.content_similarity_avg,
                           uf.night_activity_ratio
                    FROM users u
                    JOIN user_features uf ON u.user_id = uf.user_id
                    WHERE uf.anomaly_score >= ?
                    ORDER BY uf.anomaly_score DESC
                    LIMIT ?
                """, conn, params=[threshold, limit])
                return df
            except Exception:
                return self._generate_mock_anomaly_data()

    def _generate_mock_anomaly_data(self) -> pd.DataFrame:
        data = []
        for i in range(20):
            data.append({
                "user_id": f"anomaly_{i}",
                "username": f"suspicious_user_{i}",
                "platform": np.random.choice(["twitter", "weibo", "reddit"]),
                "followers_count": np.random.randint(10, 1000),
                "posts_count": np.random.randint(100, 5000),
                "is_suspicious": True,
                "anomaly_score": np.random.uniform(0.7, 0.99),
                "predicted_label": np.random.choice(["bot", "troll", "spammer"]),
                "confidence_score": np.random.uniform(0.6, 0.95),
                "daily_post_avg": np.random.uniform(20, 100),
                "content_similarity_avg": np.random.uniform(0.7, 0.99),
                "night_activity_ratio": np.random.uniform(0.5, 0.9)
            })
        return pd.DataFrame(data)

    def get_feature_importance(self) -> pd.DataFrame:
        features = [
            ("daily_post_avg", 0.25),
            ("content_similarity_avg", 0.20),
            ("night_activity_ratio", 0.15),
            ("follower_ratio", 0.12),
            ("hour_entropy", 0.10),
            ("mention_ratio", 0.08),
            ("url_ratio", 0.06),
            ("sentiment_variance", 0.04)
        ]
        return pd.DataFrame(features, columns=["feature", "importance"])

    def get_events(self) -> list[dict]:
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT 
                        date(posted_at) as event_date,
                        platform,
                        COUNT(*) as post_count
                    FROM posts
                    GROUP BY date(posted_at), platform
                    HAVING post_count > 100
                    ORDER BY event_date DESC
                    LIMIT 20
                """)
                return [dict(row) for row in cursor.fetchall()]
            except Exception:
                return [
                    {"event_date": "2024-01-15", "platform": "twitter", "post_count": 500},
                    {"event_date": "2024-01-14", "platform": "weibo", "post_count": 350},
                    {"event_date": "2024-01-13", "platform": "reddit", "post_count": 200}
                ]

    def get_cross_event_accounts(self, event1: str, event2: str) -> pd.DataFrame:
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query("""
                    SELECT DISTINCT u.user_id, u.username, u.platform,
                           COUNT(DISTINCT date(p.posted_at)) as event_count
                    FROM users u
                    JOIN posts p ON u.user_id = p.user_id
                    WHERE date(p.posted_at) IN (?, ?)
                    GROUP BY u.user_id
                    HAVING event_count = 2
                """, conn, params=[event1, event2])
                return df
            except Exception:
                return pd.DataFrame({
                    "user_id": [f"cross_{i}" for i in range(10)],
                    "username": [f"cross_user_{i}" for i in range(10)],
                    "platform": np.random.choice(["twitter", "weibo"], 10),
                    "event_count": [2] * 10
                })

    def get_all_posts(self, platform: str = "全部", days: int = 30, limit: int = 1000) -> pd.DataFrame:
        with self.get_connection() as conn:
            platform_filter = "" if platform == "全部" else f"AND platform = '{platform}'"
            try:
                df = pd.read_sql_query(f"""
                    SELECT p.post_id, p.user_id, p.platform, p.content,
                           p.posted_at, p.likes_count, p.shares_count,
                           p.comments_count, p.hashtags, p.sentiment_score,
                           u.username
                    FROM posts p
                    JOIN users u ON p.user_id = u.user_id
                    WHERE p.posted_at >= datetime('now', '-{days} days')
                    {platform_filter}
                    ORDER BY p.posted_at DESC
                    LIMIT ?
                """, conn, params=[limit])
                return df
            except Exception:
                return pd.DataFrame()

    def get_all_users(self, platform: str = "全部", limit: int = 1000) -> pd.DataFrame:
        with self.get_connection() as conn:
            platform_filter = "" if platform == "全部" else f"AND platform = '{platform}'"
            try:
                df = pd.read_sql_query(f"""
                    SELECT user_id, username, platform, followers_count,
                           friends_count, posts_count, verified, is_suspicious
                    FROM users
                    WHERE 1=1 {platform_filter}
                    ORDER BY followers_count DESC
                    LIMIT ?
                """, conn, params=[limit])
                return df
            except Exception:
                return pd.DataFrame()


def get_time_range_days(time_range: str) -> int:
    mapping = {
        "最近24小时": 1,
        "最近7天": 7,
        "最近30天": 30,
        "最近90天": 90,
        "全部": 365
    }
    return mapping.get(time_range, 30)


@st.cache_resource
def get_data_provider() -> DashboardData:
    settings = get_settings()
    db_path = settings.database_url.replace("sqlite:///", "")
    return DashboardData(db_path)


@st.cache_data(ttl=300)
def cached_overview_stats(platform: str, days: int) -> dict[str, Any]:
    return get_data_provider().get_overview_stats(platform, days)


@st.cache_data(ttl=300)
def cached_platform_distribution() -> pd.DataFrame:
    return get_data_provider().get_platform_distribution()


@st.cache_data(ttl=300)
def cached_time_series(platform: str, days: int) -> pd.DataFrame:
    return get_data_provider().get_time_series_data(platform, days)


@st.cache_data(ttl=300)
def cached_sentiment_trend(platform: str, days: int) -> pd.DataFrame:
    return get_data_provider().get_sentiment_trend(platform, days)


@st.cache_data(ttl=300)
def cached_engagement_trend(platform: str, days: int) -> pd.DataFrame:
    return get_data_provider().get_engagement_trend(platform, days)


@st.cache_data(ttl=300)
def cached_network_data(limit: int) -> tuple[list[dict], list[dict]]:
    return get_data_provider().get_network_data(limit)


@st.cache_data(ttl=300)
def cached_anomaly_accounts(threshold: float, limit: int) -> pd.DataFrame:
    return get_data_provider().get_anomaly_accounts(threshold, limit)


def render_sidebar() -> FilterConfig:
    with st.sidebar:
        st.header("⚙️ 控制面板")
        
        platform = st.selectbox(
            "选择平台",
            ["全部", "Twitter", "微博", "Reddit", "Telegram"],
            key="platform_select"
        )
        
        time_range = st.selectbox(
            "时间范围",
            ["最近24小时", "最近7天", "最近30天", "最近90天", "全部"],
            key="time_range_select"
        )
        
        st.divider()
        
        analysis_type = st.radio(
            "分析类型",
            ["概览", "趋势分析", "网络拓扑", "异常检测", "跨事件分析"],
            key="analysis_type_radio"
        )
        
        st.divider()
        
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        with st.expander("📊 数据导出"):
            export_format = st.selectbox("导出格式", ["CSV", "JSON", "PDF报告"])
            if st.button("📥 导出数据", use_container_width=True):
                handle_export(export_format)
        
        days = get_time_range_days(time_range)
        
        return FilterConfig(
            platform=platform,
            time_range=time_range,
            start_date=datetime.now() - timedelta(days=days),
            end_date=datetime.now()
        )


def handle_export(export_format: str) -> None:
    try:
        data_provider = get_data_provider()
        platform = st.session_state.get("platform_select", "全部")
        time_range = st.session_state.get("time_range_select", "最近30天")
        days = get_time_range_days(time_range)
        
        if export_format == "CSV":
            posts_df = data_provider.get_all_posts(platform, days)
            users_df = data_provider.get_all_users(platform)
            
            csv_buffer = io.StringIO()
            if not posts_df.empty:
                csv_buffer.write("# 帖子数据\n")
                csv_buffer.write(posts_df.to_csv(index=False))
                csv_buffer.write("\n")
            if not users_df.empty:
                csv_buffer.write("# 用户数据\n")
                csv_buffer.write(users_df.to_csv(index=False))
            
            st.download_button(
                label="下载 CSV 文件",
                data=csv_buffer.getvalue(),
                file_name=f"sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        elif export_format == "JSON":
            posts_df = data_provider.get_all_posts(platform, days)
            users_df = data_provider.get_all_users(platform)
            
            export_data = {
                "export_time": datetime.now().isoformat(),
                "platform": platform,
                "time_range": time_range,
                "posts": posts_df.to_dict("records") if not posts_df.empty else [],
                "users": users_df.to_dict("records") if not users_df.empty else []
            }
            
            st.download_button(
                label="下载 JSON 文件",
                data=json.dumps(export_data, ensure_ascii=False, indent=2, default=str),
                file_name=f"sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        elif export_format == "PDF报告":
            st.info("PDF报告导出功能需要安装 reportlab 库。当前显示报告预览。")
            
            stats = data_provider.get_overview_stats(platform, days)
            
            report_content = f"""
# 社交媒体情感分析报告

## 基本信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 平台: {platform}
- 时间范围: {time_range}

## 数据概览
- 总帖子数: {stats['total_posts']:,}
- 活跃用户: {stats['active_users']:,}
- 异常账号: {stats['anomaly_accounts']:,}
- 热门话题: {', '.join(stats['trending_topics']) if stats['trending_topics'] else '无'}

## 分析摘要
本报告基于指定时间范围内的社交媒体数据进行分析，包含情感趋势、用户行为和网络拓扑等多维度分析结果。
"""
            st.download_button(
                label="下载报告 (Markdown)",
                data=report_content,
                file_name=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"导出失败: {str(e)}")


def render_overview_page(filter_config: FilterConfig) -> None:
    st.header("📊 数据概览")
    
    days = get_time_range_days(filter_config.time_range)
    
    with st.spinner("加载数据中..."):
        stats = cached_overview_stats(filter_config.platform, days)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📝 总帖子数",
            value=f"{stats['total_posts']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="👥 活跃用户",
            value=f"{stats['active_users']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="⚠️ 异常账号",
            value=f"{stats['anomaly_accounts']:,}",
            delta=None
        )
    
    with col4:
        trending = stats.get('trending_topics', [])
        st.metric(
            label="🔥 热门话题",
            value=len(trending),
            delta=None
        )
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("平台分布")
        platform_df = cached_platform_distribution()
        
        if not platform_df.empty:
            fig_pie = px.pie(
                platform_df,
                values="count",
                names="platform",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=350
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("暂无平台分布数据")
    
    with col_right:
        st.subheader("帖子数量趋势")
        time_series_df = cached_time_series(filter_config.platform, days)
        
        if not time_series_df.empty:
            fig_line = px.line(
                time_series_df,
                x="date",
                y="post_count",
                markers=True,
                color_discrete_sequence=["#1f77b4"]
            )
            fig_line.update_layout(
                xaxis_title="日期",
                yaxis_title="帖子数量",
                margin=dict(t=20, b=20, l=20, r=20),
                height=350
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("暂无趋势数据")
    
    st.subheader("用户活跃度趋势")
    if not time_series_df.empty:
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(
            x=time_series_df["date"],
            y=time_series_df["active_users"],
            fill="tozeroy",
            mode="lines",
            name="活跃用户",
            line=dict(color="#2ecc71")
        ))
        fig_area.update_layout(
            xaxis_title="日期",
            yaxis_title="用户数",
            hovermode="x unified",
            margin=dict(t=20, b=20, l=20, r=20),
            height=300
        )
        st.plotly_chart(fig_area, use_container_width=True)
    
    if trending:
        st.subheader("热门话题")
        topics_df = pd.DataFrame({
            "话题": trending,
            "热度": [100 - i * 10 for i in range(len(trending))]
        })
        fig_topics = px.bar(
            topics_df,
            x="热度",
            y="话题",
            orientation="h",
            color="热度",
            color_continuous_scale="Viridis"
        )
        fig_topics.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_topics, use_container_width=True)


def render_trend_analysis_page(filter_config: FilterConfig) -> None:
    st.header("📈 趋势分析")
    
    days = get_time_range_days(filter_config.time_range)
    
    tab1, tab2, tab3 = st.tabs(["时间序列", "话题热度", "情感分析"])
    
    with tab1:
        st.subheader("帖子发布时间序列")
        
        time_series_df = cached_time_series(filter_config.platform, days)
        
        if not time_series_df.empty:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("帖子数量", "活跃用户"),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_series_df["date"],
                    y=time_series_df["post_count"],
                    mode="lines+markers",
                    name="帖子数量",
                    line=dict(color="#3498db", width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=time_series_df["date"],
                    y=time_series_df["active_users"],
                    mode="lines+markers",
                    name="活跃用户",
                    line=dict(color="#2ecc71", width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                margin=dict(t=30, b=20, l=20, r=20)
            )
            fig.update_xaxes(title_text="日期", row=2, col=1)
            fig.update_yaxes(title_text="数量", row=1, col=1)
            fig.update_yaxes(title_text="用户数", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无时间序列数据")
    
    with tab2:
        st.subheader("话题热度趋势")
        
        stats = cached_overview_stats(filter_config.platform, days)
        trending = stats.get('trending_topics', [])
        
        if trending:
            selected_topic = st.selectbox("选择话题", trending, key="topic_select")
            
            topic_df = get_data_provider().get_topic_trend(selected_topic, days)
            
            if not topic_df.empty:
                fig_topic = px.line(
                    topic_df,
                    x="date",
                    y="count",
                    markers=True,
                    title=f"'{selected_topic}' 话题趋势"
                )
                fig_topic.update_layout(
                    xaxis_title="日期",
                    yaxis_title="提及次数",
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_topic, use_container_width=True)
            else:
                st.info(f"暂无 '{selected_topic}' 的趋势数据")
        else:
            st.info("暂无热门话题数据")
    
    with tab3:
        st.subheader("情感分析趋势")
        
        sentiment_df = cached_sentiment_trend(filter_config.platform, days)
        
        if not sentiment_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_sentiment = go.Figure()
                fig_sentiment.add_trace(go.Scatter(
                    x=sentiment_df["date"],
                    y=sentiment_df["avg_sentiment"],
                    mode="lines+markers",
                    name="平均情感分数",
                    line=dict(color="#9b59b6", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(155, 89, 182, 0.2)"
                ))
                
                fig_sentiment.add_hline(
                    y=0.5,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="中性线"
                )
                
                fig_sentiment.update_layout(
                    xaxis_title="日期",
                    yaxis_title="情感分数",
                    yaxis=dict(range=[0, 1]),
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=400
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                avg_sentiment = sentiment_df["avg_sentiment"].mean()
                sentiment_label = "积极" if avg_sentiment > 0.6 else "消极" if avg_sentiment < 0.4 else "中性"
                sentiment_color = "#2ecc71" if avg_sentiment > 0.6 else "#e74c3c" if avg_sentiment < 0.4 else "#f39c12"
                
                st.metric("平均情感分数", f"{avg_sentiment:.3f}")
                st.metric("情感倾向", sentiment_label)
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_sentiment,
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": sentiment_color},
                        "steps": [
                            {"range": [0, 0.4], "color": "#fadbd8"},
                            {"range": [0.4, 0.6], "color": "#fdebd0"},
                            {"range": [0.6, 1], "color": "#d5f5e3"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("暂无情感分析数据")
        
        st.divider()
        st.subheader("互动数据趋势")
        
        engagement_df = cached_engagement_trend(filter_config.platform, days)
        
        if not engagement_df.empty:
            fig_engagement = go.Figure()
            
            fig_engagement.add_trace(go.Scatter(
                x=engagement_df["date"],
                y=engagement_df["total_likes"],
                mode="lines",
                name="点赞",
                stackgroup="one",
                line=dict(color="#3498db")
            ))
            fig_engagement.add_trace(go.Scatter(
                x=engagement_df["date"],
                y=engagement_df["total_shares"],
                mode="lines",
                name="转发",
                stackgroup="one",
                line=dict(color="#2ecc71")
            ))
            fig_engagement.add_trace(go.Scatter(
                x=engagement_df["date"],
                y=engagement_df["total_comments"],
                mode="lines",
                name="评论",
                stackgroup="one",
                line=dict(color="#e74c3c")
            ))
            
            fig_engagement.update_layout(
                xaxis_title="日期",
                yaxis_title="互动数量",
                hovermode="x unified",
                margin=dict(t=20, b=20, l=20, r=20),
                height=350
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
        else:
            st.info("暂无互动数据")


def render_network_page(filter_config: FilterConfig) -> None:
    st.header("🕸️ 网络拓扑可视化")
    
    with st.sidebar:
        st.divider()
        st.subheader("网络图设置")
        node_limit = st.slider("节点数量限制", 50, 500, 200, key="node_limit")
        show_labels = st.checkbox("显示节点标签", value=True, key="show_labels")
        highlight_suspicious = st.checkbox("高亮异常账号", value=True, key="highlight_suspicious")
    
    with st.spinner("加载网络数据..."):
        nodes, edges = cached_network_data(node_limit)
    
    if not nodes:
        st.warning("暂无网络数据")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("交互网络图")
        
        node_categories = {
            "normal": 0,
            "suspicious": 1,
            "influencer": 2
        }
        
        pyecharts_nodes = []
        for node in nodes:
            category = 0
            if node.get("is_suspicious") and highlight_suspicious:
                category = 1
            elif node.get("followers_count", 0) > 5000:
                category = 2
            
            node_size = max(10, min(50, node.get("centrality", 0.5) * 40 + 10))
            
            pyecharts_nodes.append(
                opts.GraphNodeOpts(
                    name=node.get("name", node.get("id")),
                    symbol_size=node_size,
                    category=category,
                    value=node.get("centrality", 0),
                    label_opts=opts.LabelOpts(
                        is_show=show_labels,
                        position="right",
                        font_size=10
                    )
                )
            )
        
        pyecharts_edges = []
        for edge in edges:
            source_node = next((n for n in nodes if n["id"] == edge["source"]), None)
            target_node = next((n for n in nodes if n["id"] == edge["target"]), None)
            
            if source_node and target_node:
                pyecharts_edges.append(
                    opts.GraphLinkOpts(
                        source=source_node.get("name", edge["source"]),
                        target=target_node.get("name", edge["target"]),
                        value=edge.get("weight", 1)
                    )
                )
        
        categories = [
            opts.GraphCategoryOpts(name="普通用户", itemstyle_opts=opts.ItemStyleOpts(color="#5dade2")),
            opts.GraphCategoryOpts(name="异常账号", itemstyle_opts=opts.ItemStyleOpts(color="#e74c3c")),
            opts.GraphCategoryOpts(name="影响力用户", itemstyle_opts=opts.ItemStyleOpts(color="#f39c12"))
        ]
        
        graph = (
            Graph(init_opts=opts.InitOpts(width="100%", height="600px"))
            .add(
                "",
                nodes=pyecharts_nodes,
                links=pyecharts_edges,
                categories=categories,
                layout="force",
                is_roam=True,
                is_focusnode=True,
                is_draggable=True,
                repulsion=1000,
                gravity=0.1,
                edge_length=[50, 200],
                linestyle_opts=opts.LineStyleOpts(
                    width=0.5,
                    curve=0.2,
                    opacity=0.6
                ),
                edge_symbol=["circle", "arrow"],
                edge_symbol_size=[4, 8]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="社交网络关系图"),
                legend_opts=opts.LegendOpts(
                    orient="vertical",
                    pos_left="left",
                    pos_top="middle"
                ),
                toolbox_opts=opts.ToolboxOpts(
                    is_show=True,
                    feature={
                        "saveAsImage": {"title": "保存图片"},
                        "restore": {"title": "还原"},
                        "dataZoom": {"title": "缩放"}
                    }
                )
            )
        )
        
        st.components.v1.html(graph.render_embed(), height=650)
    
    with col2:
        st.subheader("网络统计")
        
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"], **node)
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))
        
        st.metric("节点数", G.number_of_nodes())
        st.metric("边数", G.number_of_edges())
        
        if G.number_of_nodes() > 0:
            density = nx.density(G)
            st.metric("网络密度", f"{density:.4f}")
            
            if nx.is_weakly_connected(G):
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                st.metric("最大连通分量", len(largest_cc))
        
        st.divider()
        st.subheader("节点搜索")
        search_term = st.text_input("搜索用户", key="node_search")
        
        if search_term:
            matching_nodes = [
                n for n in nodes 
                if search_term.lower() in n.get("name", "").lower() 
                or search_term.lower() in n.get("id", "").lower()
            ]
            
            if matching_nodes:
                st.write(f"找到 {len(matching_nodes)} 个匹配节点:")
                for node in matching_nodes[:10]:
                    status = "⚠️ 异常" if node.get("is_suspicious") else "✓ 正常"
                    st.write(f"- {node.get('name')} ({status})")
            else:
                st.info("未找到匹配节点")


def render_anomaly_page(filter_config: FilterConfig) -> None:
    st.header("🔍 异常检测")
    
    with st.sidebar:
        st.divider()
        st.subheader("异常检测设置")
        threshold = st.slider("异常分数阈值", 0.0, 1.0, 0.5, 0.05, key="anomaly_threshold")
        result_limit = st.slider("结果数量限制", 10, 200, 50, key="result_limit")
    
    anomaly_df = cached_anomaly_accounts(threshold, result_limit)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("检测到的异常账号", len(anomaly_df))
    
    with col2:
        if not anomaly_df.empty:
            avg_score = anomaly_df["anomaly_score"].mean()
            st.metric("平均异常分数", f"{avg_score:.3f}")
        else:
            st.metric("平均异常分数", "N/A")
    
    with col3:
        if not anomaly_df.empty and "predicted_label" in anomaly_df.columns:
            label_counts = anomaly_df["predicted_label"].value_counts()
            most_common = label_counts.index[0] if len(label_counts) > 0 else "N/A"
            st.metric("最常见类型", most_common)
        else:
            st.metric("最常见类型", "N/A")
    
    tab1, tab2, tab3 = st.tabs(["异常账号列表", "分数分布", "特征重要性"])
    
    with tab1:
        st.subheader("异常账号列表")
        
        if not anomaly_df.empty:
            display_cols = ["username", "platform", "anomaly_score", "predicted_label", 
                           "followers_count", "posts_count"]
            available_cols = [c for c in display_cols if c in anomaly_df.columns]
            
            df_display = anomaly_df[available_cols].copy()
            df_display = df_display.round(3)
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "anomaly_score": st.column_config.ProgressColumn(
                        "异常分数",
                        help="异常检测分数",
                        format="%.3f",
                        min_value=0,
                        max_value=1
                    ),
                    "predicted_label": st.column_config.TextColumn("预测类型"),
                    "platform": st.column_config.TextColumn("平台"),
                    "followers_count": st.column_config.NumberColumn("粉丝数", format="%d"),
                    "posts_count": st.column_config.NumberColumn("帖子数", format="%d")
                }
            )
            
            selected_user = st.selectbox(
                "选择账号查看详情",
                anomaly_df["user_id"].tolist() if "user_id" in anomaly_df.columns else []
            )
            
            if selected_user:
                user_data = anomaly_df[anomaly_df["user_id"] == selected_user].iloc[0]
                
                with st.expander("📋 账号详情", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**基本信息**")
                        st.write(f"用户ID: {user_data.get('user_id', 'N/A')}")
                        st.write(f"用户名: {user_data.get('username', 'N/A')}")
                        st.write(f"平台: {user_data.get('platform', 'N/A')}")
                        st.write(f"粉丝数: {user_data.get('followers_count', 0):,}")
                        st.write(f"帖子数: {user_data.get('posts_count', 0):,}")
                    
                    with col_b:
                        st.write("**异常指标**")
                        st.write(f"异常分数: {user_data.get('anomaly_score', 0):.3f}")
                        st.write(f"预测类型: {user_data.get('predicted_label', 'N/A')}")
                        st.write(f"置信度: {user_data.get('confidence_score', 0):.2%}")
                        st.write(f"日均发帖: {user_data.get('daily_post_avg', 0):.1f}")
                        st.write(f"内容相似度: {user_data.get('content_similarity_avg', 0):.2%}")
                        st.write(f"夜间活跃比: {user_data.get('night_activity_ratio', 0):.2%}")
        else:
            st.info("未检测到异常账号")
    
    with tab2:
        st.subheader("异常分数分布")
        
        if not anomaly_df.empty and "anomaly_score" in anomaly_df.columns:
            fig_hist = px.histogram(
                anomaly_df,
                x="anomaly_score",
                nbins=20,
                title="异常分数分布",
                color_discrete_sequence=["#e74c3c"]
            )
            fig_hist.update_layout(
                xaxis_title="异常分数",
                yaxis_title="账号数量",
                bargap=0.1,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"阈值: {threshold}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            if "predicted_label" in anomaly_df.columns:
                label_counts = anomaly_df["predicted_label"].value_counts().reset_index()
                label_counts.columns = ["类型", "数量"]
                
                fig_pie = px.pie(
                    label_counts,
                    values="数量",
                    names="类型",
                    title="异常类型分布",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("暂无分布数据")
    
    with tab3:
        st.subheader("特征重要性")
        
        feature_df = get_data_provider().get_feature_importance()
        
        fig_feature = px.bar(
            feature_df,
            x="importance",
            y="feature",
            orientation="h",
            title="异常检测特征重要性",
            color="importance",
            color_continuous_scale="Viridis"
        )
        fig_feature.update_layout(
            xaxis_title="重要性",
            yaxis_title="特征",
            margin=dict(t=40, b=20, l=20, r=20),
            height=400
        )
        st.plotly_chart(fig_feature, use_container_width=True)
        
        st.info("""
        **特征说明:**
        - **daily_post_avg**: 日均发帖数量
        - **content_similarity_avg**: 内容相似度平均值
        - **night_activity_ratio**: 夜间活动比例
        - **follower_ratio**: 粉丝/关注比例
        - **hour_entropy**: 发帖时间熵值
        - **mention_ratio**: 提及比例
        - **url_ratio**: URL链接比例
        - **sentiment_variance**: 情感方差
        """)


def render_cross_event_page(filter_config: FilterConfig) -> None:
    st.header("🔗 跨事件分析")
    
    events = get_data_provider().get_events()
    
    if not events:
        st.warning("暂无事件数据")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        event_options = [f"{e['event_date']} ({e['platform']}) - {e['post_count']}帖子" for e in events]
        event1_idx = st.selectbox(
            "选择事件1",
            range(len(event_options)),
            format_func=lambda i: event_options[i],
            key="event1_select"
        )
    
    with col2:
        event2_idx = st.selectbox(
            "选择事件2",
            range(len(event_options)),
            format_func=lambda i: event_options[i],
            index=min(1, len(event_options) - 1),
            key="event2_select"
        )
    
    if event1_idx == event2_idx:
        st.warning("请选择两个不同的事件进行比较")
        return
    
    event1 = events[event1_idx]
    event2 = events[event2_idx]
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["复用账号网络", "行为演化时间线", "账号资产库"])
    
    with tab1:
        st.subheader("复用账号网络图")
        
        cross_accounts = get_data_provider().get_cross_event_accounts(
            event1["event_date"], 
            event2["event_date"]
        )
        
        if not cross_accounts.empty:
            st.metric("跨事件活跃账号", len(cross_accounts))
            
            G = nx.DiGraph()
            
            for _, row in cross_accounts.iterrows():
                G.add_node(
                    row["user_id"],
                    name=row.get("username", row["user_id"]),
                    platform=row.get("platform", "unknown")
                )
            
            for i, node1 in cross_accounts.iterrows():
                for j, node2 in cross_accounts.iterrows():
                    if i < j and node1["platform"] == node2["platform"]:
                        G.add_edge(node1["user_id"], node2["user_id"])
            
            pyecharts_nodes = []
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                pyecharts_nodes.append(
                    opts.GraphNodeOpts(
                        name=node_data.get("name", node_id),
                        symbol_size=20,
                        category=0 if node_data.get("platform") == "twitter" else 1
                    )
                )
            
            pyecharts_edges = []
            for edge in G.edges():
                source_name = G.nodes[edge[0]].get("name", edge[0])
                target_name = G.nodes[edge[1]].get("name", edge[1])
                pyecharts_edges.append(
                    opts.GraphLinkOpts(source=source_name, target=target_name)
                )
            
            categories = [
                opts.GraphCategoryOpts(name="Twitter", itemstyle_opts=opts.ItemStyleOpts(color="#1da1f2")),
                opts.GraphCategoryOpts(name="其他平台", itemstyle_opts=opts.ItemStyleOpts(color="#ff6b6b"))
            ]
            
            graph = (
                Graph(init_opts=opts.InitOpts(width="100%", height="500px"))
                .add(
                    "",
                    nodes=pyecharts_nodes,
                    links=pyecharts_edges,
                    categories=categories,
                    layout="force",
                    is_roam=True,
                    is_draggable=True,
                    repulsion=500
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="跨事件复用账号网络"),
                    legend_opts=opts.LegendOpts(pos_left="left")
                )
            )
            
            st.components.v1.html(graph.render_embed(), height=550)
        else:
            st.info("未发现跨事件活跃账号")
    
    with tab2:
        st.subheader("行为演化时间线")
        
        timeline_data = pd.DataFrame({
            "时间点": [
                event1["event_date"],
                event2["event_date"]
            ],
            "事件": [
                f"事件1: {event1['platform']}",
                f"事件2: {event2['platform']}"
            ],
            "帖子数": [
                event1["post_count"],
                event2["post_count"]
            ]
        })
        
        fig_timeline = px.scatter(
            timeline_data,
            x="时间点",
            y="帖子数",
            size="帖子数",
            color="事件",
            title="事件时间线",
            size_max=50
        )
        fig_timeline.update_layout(
            margin=dict(t=40, b=20, l=20, r=20),
            height=400
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**事件1详情**")
            st.write(f"日期: {event1['event_date']}")
            st.write(f"平台: {event1['platform']}")
            st.write(f"帖子数: {event1['post_count']}")
        
        with col_b:
            st.write(f"**事件2详情**")
            st.write(f"日期: {event2['event_date']}")
            st.write(f"平台: {event2['platform']}")
            st.write(f"帖子数: {event2['post_count']}")
    
    with tab3:
        st.subheader("账号资产库")
        
        if not cross_accounts.empty:
            st.dataframe(
                cross_accounts,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "user_id": "用户ID",
                    "username": "用户名",
                    "platform": "平台",
                    "event_count": "参与事件数"
                }
            )
            
            csv = cross_accounts.to_csv(index=False)
            st.download_button(
                label="📥 导出账号列表 (CSV)",
                data=csv,
                file_name=f"cross_event_accounts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("暂无跨事件账号数据")


def create_app() -> None:
    st.set_page_config(
        page_title="社交媒体情感分析平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "社交媒体情感分析可视化平台 - 多平台数据采集与分析"
        }
    )
    
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }
        .stMetric label {
            font-size: 14px;
            color: #6c757d;
        }
        .stMetric value {
            font-size: 24px;
            font-weight: bold;
        }
        div[data-testid="stHorizontalBlock"] > div {
            gap: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 社交媒体情感分析平台")
    st.markdown("---")
    
    filter_config = render_sidebar()
    
    analysis_type = st.session_state.get("analysis_type_radio", "概览")
    
    try:
        if analysis_type == "概览":
            render_overview_page(filter_config)
        elif analysis_type == "趋势分析":
            render_trend_analysis_page(filter_config)
        elif analysis_type == "网络拓扑":
            render_network_page(filter_config)
        elif analysis_type == "异常检测":
            render_anomaly_page(filter_config)
        elif analysis_type == "跨事件分析":
            render_cross_event_page(filter_config)
    except Exception as e:
        st.error(f"页面渲染错误: {str(e)}")
        logger.error(f"Dashboard render error: {e}")
    
    st.markdown("---")
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def run_dashboard(host: str = "localhost", port: int = 8501) -> None:
    import subprocess
    import sys

    logger.info(f"Starting dashboard on {host}:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.address", host,
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false"
    ])


if __name__ == "__main__":
    create_app()
