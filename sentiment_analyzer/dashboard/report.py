"""
报告自动化模块

提供报告配置、模板渲染、PDF/HTML报告生成、数据导出和报告调度功能。
"""

import csv
import io
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

try:
    from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Template = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False


class ReportFormat(str, Enum):
    """报告格式枚举"""
    PDF = "pdf"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PARQUET = "parquet"


class ScheduleType(str, Enum):
    """调度类型枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """
    报告配置类

    用于定义报告的基本配置信息。

    Attributes:
        title: 报告标题
        author: 报告作者
        date_range: 报告数据的时间范围 (开始日期, 结束日期)
        platforms: 报告涵盖的平台列表
        template_path: 模板文件路径
        output_format: 输出格式
        output_dir: 输出目录
        include_charts: 是否包含图表
        include_tables: 是否包含表格
        custom_settings: 自定义设置

    Example:
        >>> config = ReportConfig(
        ...     title="每周情感分析报告",
        ...     author="数据分析团队",
        ...     date_range=(datetime(2024, 1, 1), datetime(2024, 1, 7)),
        ...     platforms=["twitter", "weibo"]
        ... )
    """

    title: str
    author: str
    date_range: tuple[datetime, datetime]
    platforms: list[str] = field(default_factory=list)
    template_path: Optional[str] = None
    output_format: ReportFormat = ReportFormat.PDF
    output_dir: str = "./reports"
    include_charts: bool = True
    include_tables: bool = True
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("报告标题不能为空")
        if not self.author:
            raise ValueError("报告作者不能为空")
        if len(self.date_range) != 2:
            raise ValueError("date_range 必须包含开始和结束日期")
        if self.date_range[0] > self.date_range[1]:
            raise ValueError("开始日期不能晚于结束日期")

    @property
    def start_date(self) -> datetime:
        """获取开始日期"""
        return self.date_range[0]

    @property
    def end_date(self) -> datetime:
        """获取结束日期"""
        return self.date_range[1]

    @property
    def duration_days(self) -> int:
        """获取时间跨度（天数）"""
        return (self.end_date - self.start_date).days

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "author": self.author,
            "date_range": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
            },
            "platforms": self.platforms,
            "template_path": self.template_path,
            "output_format": self.output_format.value,
            "output_dir": self.output_dir,
            "include_charts": self.include_charts,
            "include_tables": self.include_tables,
            "custom_settings": self.custom_settings,
        }


class ReportTemplate:
    """
    报告模板引擎

    支持使用 Jinja2 模板语法渲染报告内容。

    Example:
        >>> template = ReportTemplate()
        >>> template.load_template("report_template.html")
        >>> content = template.render({"title": "报告标题", "data": [...]})
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        初始化模板引擎

        Args:
            template_dir: 模板文件目录
        """
        if not JINJA2_AVAILABLE:
            raise ImportError("Jinja2 未安装，请运行: pip install jinja2")

        self.template_dir = template_dir
        self._template: Optional[Template] = None
        self._env: Optional[Environment] = None

        if template_dir:
            self._env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )

    def load_template(self, path: str) -> None:
        """
        加载模板文件

        Args:
            path: 模板文件路径

        Raises:
            FileNotFoundError: 模板文件不存在
            TemplateNotFound: Jinja2 模板未找到
        """
        if self._env and not os.path.isabs(path):
            try:
                self._template = self._env.get_template(path)
                logger.info(f"已加载模板: {path}")
                return
            except TemplateNotFound:
                pass

        template_path = Path(path)
        if not template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {path}")

        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        self._template = Template(template_content)
        logger.info(f"已加载模板: {path}")

    def load_template_string(self, template_string: str) -> None:
        """
        从字符串加载模板

        Args:
            template_string: 模板字符串
        """
        self._template = Template(template_string)
        logger.info("已从字符串加载模板")

    def render(self, context: dict[str, Any]) -> str:
        """
        渲染模板

        Args:
            context: 模板上下文数据

        Returns:
            渲染后的内容字符串

        Raises:
            ValueError: 未加载模板
        """
        if self._template is None:
            raise ValueError("未加载模板，请先调用 load_template()")

        default_context = {
            "current_time": datetime.now(),
            "format_date": lambda d, fmt="%Y-%m-%d": d.strftime(fmt) if d else "",
            "format_number": lambda n: f"{n:,}" if isinstance(n, (int, float)) else n,
        }
        context = {**default_context, **context}

        try:
            rendered = self._template.render(**context)
            logger.debug("模板渲染成功")
            return rendered
        except Exception as e:
            logger.error(f"模板渲染失败: {e}")
            raise

    def render_to_file(self, context: dict[str, Any], output_path: str) -> None:
        """
        渲染模板并保存到文件

        Args:
            context: 模板上下文数据
            output_path: 输出文件路径
        """
        content = self.render(context)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"报告已保存到: {output_path}")


@dataclass
class ChartData:
    """图表数据结构"""
    chart_type: str
    title: str
    data: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    width: int = 800
    height: int = 400


@dataclass
class TableData:
    """表格数据结构"""
    headers: list[str]
    rows: list[list[Any]]
    title: Optional[str] = None
    caption: Optional[str] = None


class ReportSection:
    """
    报告章节类

    用于组织报告的各个章节内容。

    Example:
        >>> section = ReportSection(title="情感分析概览")
        >>> section.add_paragraph("本节展示整体情感分布情况")
        >>> section.add_chart(ChartData(...))
        >>> section.add_table(TableData(...))
    """

    def __init__(
        self,
        title: str,
        content: Optional[str] = None,
        level: int = 1,
    ):
        """
        初始化报告章节

        Args:
            title: 章节标题
            content: 章节内容
            level: 标题级别 (1-6)
        """
        self.title = title
        self.content = content or ""
        self.level = max(1, min(6, level))
        self.charts: list[ChartData] = []
        self.tables: list[TableData] = []
        self.paragraphs: list[str] = []
        self.subsections: list["ReportSection"] = []

        if content:
            self.paragraphs.append(content)

    def add_paragraph(self, text: str) -> None:
        """
        添加段落文本

        Args:
            text: 段落内容
        """
        self.paragraphs.append(text)

    def add_chart(
        self,
        chart_type: str,
        title: str,
        data: dict[str, Any],
        options: Optional[dict[str, Any]] = None,
        width: int = 800,
        height: int = 400,
    ) -> None:
        """
        添加图表

        Args:
            chart_type: 图表类型 (pie, bar, line, scatter, etc.)
            title: 图表标题
            data: 图表数据
            options: 图表配置选项
            width: 图表宽度
            height: 图表高度
        """
        chart = ChartData(
            chart_type=chart_type,
            title=title,
            data=data,
            options=options or {},
            width=width,
            height=height,
        )
        self.charts.append(chart)
        logger.debug(f"添加图表: {title}")

    def add_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        title: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        添加表格

        Args:
            headers: 表头
            rows: 表格行数据
            title: 表格标题
            caption: 表格说明
        """
        table = TableData(
            headers=headers,
            rows=rows,
            title=title,
            caption=caption,
        )
        self.tables.append(table)
        logger.debug(f"添加表格: {title or '未命名表格'}")

    def add_subsection(self, subsection: "ReportSection") -> None:
        """
        添加子章节

        Args:
            subsection: 子章节对象
        """
        subsection.level = self.level + 1
        self.subsections.append(subsection)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "paragraphs": self.paragraphs,
            "charts": [
                {
                    "type": c.chart_type,
                    "title": c.title,
                    "data": c.data,
                    "options": c.options,
                    "width": c.width,
                    "height": c.height,
                }
                for c in self.charts
            ],
            "tables": [
                {
                    "headers": t.headers,
                    "rows": t.rows,
                    "title": t.title,
                    "caption": t.caption,
                }
                for t in self.tables
            ],
            "subsections": [s.to_dict() for s in self.subsections],
        }


class BaseReportGenerator(ABC):
    """报告生成器基类"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.sections: list[ReportSection] = []
        self._cover_info: Optional[dict[str, Any]] = None

    @abstractmethod
    def generate(self, report_data: dict[str, Any], output_path: str) -> str:
        """
        生成报告

        Args:
            report_data: 报告数据
            output_path: 输出路径

        Returns:
            生成的报告文件路径
        """
        pass

    def add_section(self, section: ReportSection) -> None:
        """添加章节"""
        self.sections.append(section)

    def set_cover(self, title: str, author: str, date: Optional[datetime] = None) -> None:
        """设置封面信息"""
        self._cover_info = {
            "title": title,
            "author": author,
            "date": date or datetime.now(),
        }


class PDFReportGenerator(BaseReportGenerator):
    """
    PDF报告生成器

    使用 reportlab 生成 PDF 格式的报告。

    Example:
        >>> config = ReportConfig(...)
        >>> generator = PDFReportGenerator(config)
        >>> generator.add_cover_page("报告标题", "作者", datetime.now())
        >>> generator.add_section(section)
        >>> generator.generate(report_data, "output.pdf")
    """

    def __init__(self, config: ReportConfig):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab 未安装，请运行: pip install reportlab")
        super().__init__(config)
        self._styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self._elements: list[Any] = []
        self._toc_entries: list[dict[str, Any]] = []

    def _setup_custom_styles(self) -> None:
        """设置自定义样式"""
        self._styles.add(ParagraphStyle(
            name="CustomTitle",
            parent=self._styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor("#2c3e50"),
        ))
        self._styles.add(ParagraphStyle(
            name="CustomHeading1",
            parent=self._styles["Heading1"],
            fontSize=18,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor("#34495e"),
        ))
        self._styles.add(ParagraphStyle(
            name="CustomHeading2",
            parent=self._styles["Heading2"],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor("#34495e"),
        ))
        self._styles.add(ParagraphStyle(
            name="CustomBody",
            parent=self._styles["Normal"],
            fontSize=10,
            spaceBefore=6,
            spaceAfter=6,
            leading=14,
        ))
        self._styles.add(ParagraphStyle(
            name="CoverTitle",
            parent=self._styles["Title"],
            fontSize=28,
            spaceAfter=20,
            alignment=1,
        ))
        self._styles.add(ParagraphStyle(
            name="CoverSubtitle",
            parent=self._styles["Normal"],
            fontSize=14,
            spaceAfter=10,
            alignment=1,
            textColor=colors.HexColor("#7f8c8d"),
        ))

    def add_cover_page(
        self,
        title: str,
        author: str,
        date: Optional[datetime] = None,
    ) -> None:
        """
        添加封面页

        Args:
            title: 报告标题
            author: 作者
            date: 日期
        """
        self.set_cover(title, author, date)

    def add_toc(self, sections: Optional[list[ReportSection]] = None) -> None:
        """
        添加目录

        Args:
            sections: 章节列表，如果为None则使用已添加的章节
        """
        sections = sections or self.sections
        self._toc_entries = []

        def collect_entries(section_list: list[ReportSection], level: int = 1) -> None:
            for section in section_list:
                self._toc_entries.append({
                    "title": section.title,
                    "level": level,
                })
                if section.subsections:
                    collect_entries(section.subsections, level + 1)

        collect_entries(sections)

    def add_section(self, section: ReportSection) -> None:
        """添加章节"""
        super().add_section(section)

    def _build_section(self, section: ReportSection) -> list[Any]:
        """构建章节内容"""
        elements = []

        heading_style = f"CustomHeading{min(section.level, 2)}"
        elements.append(Paragraph(section.title, self._styles[heading_style]))
        elements.append(Spacer(1, 6))

        for para in section.paragraphs:
            elements.append(Paragraph(para, self._styles["CustomBody"]))
            elements.append(Spacer(1, 6))

        for table_data in section.tables:
            table_elements = self._build_table(table_data)
            elements.extend(table_elements)

        for chart_data in section.charts:
            chart_elements = self._build_chart(chart_data)
            elements.extend(chart_elements)

        for subsection in section.subsections:
            elements.extend(self._build_section(subsection))

        return elements

    def _build_table(self, table_data: TableData) -> list[Any]:
        """构建表格"""
        elements = []

        if table_data.title:
            elements.append(Paragraph(table_data.title, self._styles["CustomHeading2"]))

        data = [table_data.headers] + table_data.rows
        table = Table(data, repeatRows=1)

        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ])

        table.setStyle(style)
        elements.append(table)
        elements.append(Spacer(1, 12))

        if table_data.caption:
            elements.append(Paragraph(table_data.caption, self._styles["CustomBody"]))

        return elements

    def _build_chart(self, chart_data: ChartData) -> list[Any]:
        """构建图表（PDF中使用占位符）"""
        elements = []

        elements.append(Paragraph(f"图表: {chart_data.title}", self._styles["CustomHeading2"]))
        elements.append(Spacer(1, 6))

        chart_placeholder = Table(
            [[f"[{chart_data.chart_type.upper()} 图表: {chart_data.title}]"]],
            colWidths=[chart_data.width * 0.5],
        )
        chart_placeholder.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecf0f1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 12),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#7f8c8d")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
        ]))
        elements.append(chart_placeholder)
        elements.append(Spacer(1, 12))

        return elements

    def generate(
        self,
        report_data: dict[str, Any],
        output_path: str,
    ) -> str:
        """
        生成PDF报告

        Args:
            report_data: 报告数据
            output_path: 输出文件路径

        Returns:
            生成的PDF文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_file),
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )

        self._elements = []

        if self._cover_info:
            self._elements.extend(self._build_cover())

        if self._toc_entries:
            self._elements.extend(self._build_toc_page())

        for section in self.sections:
            self._elements.extend(self._build_section(section))
            self._elements.append(PageBreak())

        if self._elements and self._elements[-1] == PageBreak():
            self._elements.pop()

        doc.build(self._elements)
        logger.info(f"PDF报告已生成: {output_path}")

        return str(output_file)

    def _build_cover(self) -> list[Any]:
        """构建封面"""
        elements = []

        elements.append(Spacer(1, 100))
        elements.append(Paragraph(self._cover_info["title"], self._styles["CoverTitle"]))
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(f"作者: {self._cover_info['author']}", self._styles["CoverSubtitle"]))
        elements.append(Spacer(1, 10))

        date_str = self._cover_info["date"].strftime("%Y年%m月%d日")
        elements.append(Paragraph(f"日期: {date_str}", self._styles["CoverSubtitle"]))

        if self.config.platforms:
            platforms_str = "、".join(self.config.platforms)
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(f"平台: {platforms_str}", self._styles["CoverSubtitle"]))

        elements.append(PageBreak())

        return elements

    def _build_toc_page(self) -> list[Any]:
        """构建目录页"""
        elements = []

        elements.append(Paragraph("目录", self._styles["CustomTitle"]))
        elements.append(Spacer(1, 20))

        for entry in self._toc_entries:
            indent = "    " * (entry["level"] - 1)
            toc_line = f"{indent}{entry['title']}"
            elements.append(Paragraph(toc_line, self._styles["CustomBody"]))

        elements.append(PageBreak())

        return elements


class HTMLReportGenerator(BaseReportGenerator):
    """
    HTML交互报告生成器

    生成支持动态图表（ECharts/Plotly）和交互式表格的HTML报告。

    Example:
        >>> config = ReportConfig(...)
        >>> generator = HTMLReportGenerator(config)
        >>> generator.add_section(section)
        >>> generator.generate(report_data, "output.html")
    """

    def __init__(self, config: ReportConfig, chart_library: str = "echarts"):
        """
        初始化HTML报告生成器

        Args:
            config: 报告配置
            chart_library: 图表库 (echarts/plotly)
        """
        super().__init__(config)
        self.chart_library = chart_library.lower()
        self._template: Optional[ReportTemplate] = None

    def _get_echarts_chart(self, chart_data: ChartData) -> str:
        """生成ECharts图表HTML"""
        chart_id = f"chart_{id(chart_data)}"

        option = {
            "title": {"text": chart_data.title},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": []},
            "series": [],
        }

        if chart_data.chart_type == "pie":
            series_data = []
            for name, value in chart_data.data.items():
                series_data.append({"name": name, "value": value})
                option["legend"]["data"].append(name)

            option["series"] = [{
                "type": "pie",
                "radius": "50%",
                "data": series_data,
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)",
                    }
                },
            }]
        elif chart_data.chart_type == "bar":
            x_data = chart_data.data.get("categories", [])
            series_data = chart_data.data.get("values", [])

            option["xAxis"] = {"type": "category", "data": x_data}
            option["yAxis"] = {"type": "value"}
            option["series"] = [{
                "type": "bar",
                "data": series_data,
            }]
        elif chart_data.chart_type == "line":
            x_data = chart_data.data.get("categories", [])
            series_list = chart_data.data.get("series", [])

            option["xAxis"] = {"type": "category", "data": x_data}
            option["yAxis"] = {"type": "value"}

            for series in series_list:
                option["legend"]["data"].append(series.get("name", ""))
                option["series"].append({
                    "type": "line",
                    "name": series.get("name", ""),
                    "data": series.get("data", []),
                })

        option.update(chart_data.options)

        return f'''
        <div id="{chart_id}" style="width: {chart_data.width}px; height: {chart_data.height}px;"></div>
        <script>
            var chart_{chart_id} = echarts.init(document.getElementById('{chart_id}'));
            chart_{chart_id}.setOption({json.dumps(option, ensure_ascii=False)});
        </script>
        '''

    def _get_plotly_chart(self, chart_data: ChartData) -> str:
        """生成Plotly图表HTML"""
        import plotly.express as px
        import plotly.graph_objects as go

        if chart_data.chart_type == "pie":
            labels = list(chart_data.data.keys())
            values = list(chart_data.data.values())
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, title=chart_data.title)])
        elif chart_data.chart_type == "bar":
            categories = chart_data.data.get("categories", [])
            values = chart_data.data.get("values", [])
            fig = go.Figure(data=[go.Bar(x=categories, y=values)])
        elif chart_data.chart_type == "line":
            categories = chart_data.data.get("categories", [])
            series_list = chart_data.data.get("series", [])
            fig = go.Figure()
            for series in series_list:
                fig.add_trace(go.Scatter(
                    x=categories,
                    y=series.get("data", []),
                    name=series.get("name", ""),
                    mode="lines+markers",
                ))
        else:
            fig = go.Figure()

        fig.update_layout(
            title=chart_data.title,
            width=chart_data.width,
            height=chart_data.height,
        )

        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _get_interactive_table(self, table_data: TableData) -> str:
        """生成交互式表格HTML"""
        table_id = f"table_{id(table_data)}"

        thead = "".join(f"<th>{h}</th>" for h in table_data.headers)
        tbody_rows = ""
        for row in table_data.rows:
            cells = "".join(f"<td>{cell}</td>" for cell in row)
            tbody_rows += f"<tr>{cells}</tr>"

        html = f'''
        <div class="table-container">
            {f'<h3 class="table-title">{table_data.title}</h3>' if table_data.title else ''}
            <table id="{table_id}" class="display responsive nowrap" style="width:100%">
                <thead><tr>{thead}</tr></thead>
                <tbody>{tbody_rows}</tbody>
            </table>
            {f'<p class="table-caption">{table_data.caption}</p>' if table_data.caption else ''}
        </div>
        '''

        return html

    def _build_section_html(self, section: ReportSection) -> str:
        """构建章节HTML"""
        heading_tag = f"h{min(section.level + 1, 6)}"

        paragraphs_html = ""
        for para in section.paragraphs:
            paragraphs_html += f"<p>{para}</p>"

        charts_html = ""
        for chart in section.charts:
            if self.chart_library == "echarts":
                charts_html += self._get_echarts_chart(chart)
            else:
                charts_html += self._get_plotly_chart(chart)

        tables_html = ""
        for table in section.tables:
            tables_html += self._get_interactive_table(table)

        subsections_html = ""
        for subsection in section.subsections:
            subsections_html += self._build_section_html(subsection)

        return f'''
        <section class="report-section">
            <{heading_tag} class="section-title">{section.title}</{heading_tag}>
            <div class="section-content">
                {paragraphs_html}
                <div class="charts-container">
                    {charts_html}
                </div>
                <div class="tables-container">
                    {tables_html}
                </div>
                {subsections_html}
            </div>
        </section>
        '''

    def generate(
        self,
        report_data: dict[str, Any],
        output_path: str,
    ) -> str:
        """
        生成HTML报告

        Args:
            report_data: 报告数据
            output_path: 输出文件路径

        Returns:
            生成的HTML文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        sections_html = ""
        for section in self.sections:
            sections_html += self._build_section_html(section)

        cover_html = ""
        if self._cover_info:
            date_str = self._cover_info["date"].strftime("%Y年%m月%d日")
            cover_html = f'''
            <header class="report-cover">
                <h1 class="cover-title">{self._cover_info["title"]}</h1>
                <div class="cover-meta">
                    <p>作者: {self._cover_info["author"]}</p>
                    <p>日期: {date_str}</p>
                    {f'<p>平台: {", ".join(self.config.platforms)}</p>' if self.config.platforms else ''}
                </div>
            </header>
            '''

        chart_library_js = ""
        if self.chart_library == "echarts":
            chart_library_js = '<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>'
        else:
            chart_library_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

        html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css">
    <style>
        :root {{
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color: #e74c3c;
            --background-color: #f8f9fa;
            --text-color: #333;
            --border-color: #ddd;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .report-cover {{
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            margin-bottom: 40px;
        }}

        .cover-title {{
            font-size: 2.5rem;
            margin-bottom: 20px;
        }}

        .cover-meta {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .cover-meta p {{
            margin: 5px 0;
        }}

        .report-section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .section-title {{
            color: var(--secondary-color);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .section-content p {{
            margin-bottom: 15px;
            text-align: justify;
        }}

        .charts-container {{
            margin: 30px 0;
        }}

        .charts-container > div {{
            margin: 20px auto;
        }}

        .tables-container {{
            margin: 30px 0;
            overflow-x: auto;
        }}

        .table-container {{
            margin: 20px 0;
        }}

        .table-title {{
            color: var(--secondary-color);
            margin-bottom: 10px;
        }}

        .table-caption {{
            font-size: 0.9rem;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }}

        table.dataTable {{
            border-collapse: collapse;
            width: 100%;
        }}

        table.dataTable thead th {{
            background-color: var(--primary-color);
            color: white;
            padding: 12px;
            text-align: left;
        }}

        table.dataTable tbody td {{
            padding: 10px;
            border-bottom: 1px solid var(--border-color);
        }}

        table.dataTable tbody tr:hover {{
            background-color: #f5f5f5;
        }}

        .report-footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}

            .report-section {{
                padding: 15px;
            }}

            .cover-title {{
                font-size: 1.8rem;
            }}
        }}

        @media print {{
            .report-section {{
                break-inside: avoid;
                box-shadow: none;
            }}
        }}
    </style>
    {chart_library_js}
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>
</head>
<body>
    <div class="container">
        {cover_html}
        <main class="report-content">
            {sections_html}
        </main>
        <footer class="report-footer">
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>© {datetime.now().year} 情感分析系统</p>
        </footer>
    </div>
    <script>
        $(document).ready(function() {{
            $('table.display').DataTable({{
                responsive: true,
                pageLength: 10,
                language: {{
                    search: "搜索:",
                    lengthMenu: "显示 _MENU_ 条记录",
                    info: "显示第 _START_ 至 _END_ 条记录，共 _TOTAL_ 条",
                    infoEmpty: "没有数据",
                    infoFiltered: "(从 _MAX_ 条记录中筛选)",
                    paginate: {{
                        first: "首页",
                        last: "末页",
                        next: "下一页",
                        previous: "上一页"
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
        '''

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML报告已生成: {output_path}")
        return str(output_file)


class DataExporter:
    """
    数据导出器

    支持多种数据格式的导出功能。

    Example:
        >>> exporter = DataExporter()
        >>> exporter.export_csv(data, "output.csv")
        >>> exporter.export_json(data, "output.json")
        >>> exporter.export_excel(data, "output.xlsx")
    """

    def __init__(self, output_dir: str = "./exports"):
        """
        初始化数据导出器

        Args:
            output_dir: 默认输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_path(self, path: str) -> Path:
        """确保路径存在"""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.output_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path

    def export_csv(
        self,
        data: Union[list[dict], "pd.DataFrame"],
        path: str,
        encoding: str = "utf-8-sig",
        **kwargs,
    ) -> str:
        """
        导出CSV文件

        Args:
            data: 数据（字典列表或DataFrame）
            path: 输出路径
            encoding: 文件编码
            **kwargs: 传递给pandas.to_csv的其他参数

        Returns:
            导出的文件路径
        """
        file_path = self._ensure_path(path)

        if PANDAS_AVAILABLE:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            df.to_csv(file_path, index=False, encoding=encoding, **kwargs)
        else:
            if not isinstance(data, list):
                raise ValueError("无pandas时，data必须是字典列表")

            if not data:
                with open(file_path, "w", encoding=encoding) as f:
                    f.write("")
            else:
                with open(file_path, "w", encoding=encoding, newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)

        logger.info(f"CSV文件已导出: {file_path}")
        return str(file_path)

    def export_json(
        self,
        data: Union[list, dict],
        path: str,
        indent: int = 2,
        ensure_ascii: bool = False,
        **kwargs,
    ) -> str:
        """
        导出JSON文件

        Args:
            data: 数据
            path: 输出路径
            indent: 缩进空格数
            ensure_ascii: 是否转义非ASCII字符
            **kwargs: 传递给json.dump的其他参数

        Returns:
            导出的文件路径
        """
        file_path = self._ensure_path(path)

        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=json_serial, **kwargs)

        logger.info(f"JSON文件已导出: {file_path}")
        return str(file_path)

    def export_excel(
        self,
        data: Union[list[dict], "pd.DataFrame"],
        path: str,
        sheet_name: str = "Sheet1",
        **kwargs,
    ) -> str:
        """
        导出Excel文件

        Args:
            data: 数据（字典列表或DataFrame）
            path: 输出路径
            sheet_name: 工作表名称
            **kwargs: 传递给pandas.to_excel的其他参数

        Returns:
            导出的文件路径
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas 未安装，请运行: pip install pandas openpyxl")

        file_path = self._ensure_path(path)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        df.to_excel(file_path, sheet_name=sheet_name, index=False, **kwargs)

        logger.info(f"Excel文件已导出: {file_path}")
        return str(file_path)

    def export_parquet(
        self,
        data: Union[list[dict], "pd.DataFrame"],
        path: str,
        **kwargs,
    ) -> str:
        """
        导出Parquet文件

        Args:
            data: 数据（字典列表或DataFrame）
            path: 输出路径
            **kwargs: 传递给pyarrow的其他参数

        Returns:
            导出的文件路径
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("pyarrow 未安装，请运行: pip install pyarrow")

        file_path = self._ensure_path(path)

        if PANDAS_AVAILABLE:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            df.to_parquet(file_path, index=False, **kwargs)
        else:
            if isinstance(data, list):
                table = pa.Table.from_pylist(data)
            else:
                raise ValueError("无pandas时，data必须是字典列表")
            pq.write_table(table, file_path, **kwargs)

        logger.info(f"Parquet文件已导出: {file_path}")
        return str(file_path)

    def export(
        self,
        data: Any,
        path: str,
        format: Optional[ReportFormat] = None,
    ) -> str:
        """
        通用导出方法

        Args:
            data: 数据
            path: 输出路径
            format: 导出格式（自动检测文件扩展名）

        Returns:
            导出的文件路径
        """
        if format is None:
            ext = Path(path).suffix.lower()
            format_map = {
                ".csv": ReportFormat.CSV,
                ".json": ReportFormat.JSON,
                ".xlsx": ReportFormat.EXCEL,
                ".xls": ReportFormat.EXCEL,
                ".parquet": ReportFormat.PARQUET,
            }
            format = format_map.get(ext)
            if format is None:
                raise ValueError(f"不支持的文件格式: {ext}")

        exporters = {
            ReportFormat.CSV: self.export_csv,
            ReportFormat.JSON: self.export_json,
            ReportFormat.EXCEL: self.export_excel,
            ReportFormat.PARQUET: self.export_parquet,
        }

        exporter = exporters.get(format)
        if exporter is None:
            raise ValueError(f"不支持的导出格式: {format}")

        return exporter(data, path)


@dataclass
class ScheduledReport:
    """调度报告配置"""
    report_id: str
    report_type: str
    schedule_type: ScheduleType
    schedule_config: dict[str, Any]
    config: ReportConfig
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class ReportScheduler:
    """
    报告调度器

    支持定时任务的报告生成调度。

    Example:
        >>> scheduler = ReportScheduler()
        >>> scheduler.schedule_report(
        ...     "daily_sentiment",
        ...     ScheduleType.DAILY,
        ...     config,
        ...     callback=generate_report
        ... )
        >>> scheduler.start()
    """

    def __init__(self):
        """初始化报告调度器"""
        if not APSCHEDULER_AVAILABLE:
            raise ImportError("apscheduler 未安装，请运行: pip install apscheduler")

        self._scheduler = BackgroundScheduler()
        self._scheduled_reports: dict[str, ScheduledReport] = {}
        self._callbacks: dict[str, Callable] = {}
        self._running = False

    def schedule_report(
        self,
        report_id: str,
        report_type: str,
        schedule_type: ScheduleType,
        config: ReportConfig,
        callback: Callable[[ReportConfig], None],
        schedule_config: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        调度报告

        Args:
            report_id: 报告唯一标识
            report_type: 报告类型
            schedule_type: 调度类型
            config: 报告配置
            callback: 报告生成回调函数
            schedule_config: 调度配置（如小时、分钟等）

        Returns:
            调度任务ID
        """
        schedule_config = schedule_config or {}

        if schedule_type == ScheduleType.DAILY:
            hour = schedule_config.get("hour", 8)
            minute = schedule_config.get("minute", 0)
            trigger = CronTrigger(hour=hour, minute=minute)
            next_run = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= datetime.now():
                next_run += timedelta(days=1)

        elif schedule_type == ScheduleType.WEEKLY:
            day_of_week = schedule_config.get("day_of_week", "mon")
            hour = schedule_config.get("hour", 8)
            minute = schedule_config.get("minute", 0)
            trigger = CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute)
            next_run = datetime.now()

        elif schedule_type == ScheduleType.MONTHLY:
            day = schedule_config.get("day", 1)
            hour = schedule_config.get("hour", 8)
            minute = schedule_config.get("minute", 0)
            trigger = CronTrigger(day=day, hour=hour, minute=minute)
            next_run = datetime.now()

        elif schedule_type == ScheduleType.CUSTOM:
            trigger_type = schedule_config.get("trigger_type", "cron")
            if trigger_type == "cron":
                trigger = CronTrigger(**schedule_config.get("cron_config", {}))
            elif trigger_type == "interval":
                trigger = IntervalTrigger(**schedule_config.get("interval_config", {"hours": 24}))
            else:
                raise ValueError(f"不支持的触发器类型: {trigger_type}")
            next_run = datetime.now()

        else:
            raise ValueError(f"不支持的调度类型: {schedule_type}")

        scheduled_report = ScheduledReport(
            report_id=report_id,
            report_type=report_type,
            schedule_type=schedule_type,
            schedule_config=schedule_config,
            config=config,
            next_run=next_run if schedule_type != ScheduleType.CUSTOM else None,
        )

        self._scheduled_reports[report_id] = scheduled_report
        self._callbacks[report_id] = callback

        self._scheduler.add_job(
            func=self._execute_report,
            trigger=trigger,
            id=report_id,
            args=[report_id],
            name=f"{report_type}_{report_id}",
            replace_existing=True,
        )

        logger.info(f"已调度报告: {report_id}, 类型: {schedule_type.value}")
        return report_id

    def _execute_report(self, report_id: str) -> None:
        """执行报告生成"""
        scheduled_report = self._scheduled_reports.get(report_id)
        if not scheduled_report or not scheduled_report.enabled:
            return

        callback = self._callbacks.get(report_id)
        if not callback:
            return

        try:
            logger.info(f"开始执行报告: {report_id}")
            callback(scheduled_report.config)

            scheduled_report.last_run = datetime.now()

            if scheduled_report.schedule_type == ScheduleType.DAILY:
                scheduled_report.next_run = scheduled_report.last_run + timedelta(days=1)
            elif scheduled_report.schedule_type == ScheduleType.WEEKLY:
                scheduled_report.next_run = scheduled_report.last_run + timedelta(weeks=1)
            elif scheduled_report.schedule_type == ScheduleType.MONTHLY:
                scheduled_report.next_run = scheduled_report.last_run + timedelta(days=30)

            logger.info(f"报告执行完成: {report_id}")

        except Exception as e:
            logger.error(f"报告执行失败: {report_id}, 错误: {e}")

    def unschedule_report(self, report_id: str) -> bool:
        """
        取消报告调度

        Args:
            report_id: 报告ID

        Returns:
            是否成功取消
        """
        if report_id in self._scheduled_reports:
            self._scheduler.remove_job(report_id)
            del self._scheduled_reports[report_id]
            del self._callbacks[report_id]
            logger.info(f"已取消报告调度: {report_id}")
            return True
        return False

    def enable_report(self, report_id: str) -> bool:
        """启用报告"""
        if report_id in self._scheduled_reports:
            self._scheduled_reports[report_id].enabled = True
            self._scheduler.resume_job(report_id)
            logger.info(f"已启用报告: {report_id}")
            return True
        return False

    def disable_report(self, report_id: str) -> bool:
        """禁用报告"""
        if report_id in self._scheduled_reports:
            self._scheduled_reports[report_id].enabled = False
            self._scheduler.pause_job(report_id)
            logger.info(f"已禁用报告: {report_id}")
            return True
        return False

    def get_scheduled_reports(self) -> list[dict[str, Any]]:
        """获取所有调度报告"""
        return [
            {
                "report_id": r.report_id,
                "report_type": r.report_type,
                "schedule_type": r.schedule_type.value,
                "enabled": r.enabled,
                "last_run": r.last_run.isoformat() if r.last_run else None,
                "next_run": r.next_run.isoformat() if r.next_run else None,
            }
            for r in self._scheduled_reports.values()
        ]

    def run_scheduled_reports(self) -> None:
        """手动运行所有调度报告"""
        for report_id, scheduled_report in self._scheduled_reports.items():
            if scheduled_report.enabled:
                self._execute_report(report_id)

    def start(self) -> None:
        """启动调度器"""
        if not self._running:
            self._scheduler.start()
            self._running = True
            logger.info("报告调度器已启动")

    def stop(self, wait: bool = True) -> None:
        """
        停止调度器

        Args:
            wait: 是否等待正在执行的任务完成
        """
        if self._running:
            self._scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("报告调度器已停止")

    def __enter__(self) -> "ReportScheduler":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.stop()


def create_report(
    config: ReportConfig,
    data: dict[str, Any],
    sections: Optional[list[ReportSection]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    便捷函数：创建报告

    Args:
        config: 报告配置
        data: 报告数据
        sections: 章节列表
        output_path: 输出路径

    Returns:
        生成的报告路径
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{config.output_dir}/{config.title}_{timestamp}.{config.output_format.value}"

    if config.output_format == ReportFormat.PDF:
        generator = PDFReportGenerator(config)
    elif config.output_format == ReportFormat.HTML:
        generator = HTMLReportGenerator(config)
    else:
        raise ValueError(f"不支持的报告格式: {config.output_format}")

    generator.set_cover(config.title, config.author, datetime.now())

    if sections:
        for section in sections:
            generator.add_section(section)

    return generator.generate(data, output_path)
