"""
证据链生成模块 - 异常检测证据追踪与可视化

该模块提供完整的证据链生成、分析和可视化功能，包括：
- EvidenceItem: 单个证据项
- EvidenceChain: 证据链管理
- EvidenceGenerator: 证据生成器
- SHAPExplainer: SHAP特征归因
- EvidenceReportGenerator: 证据报告生成器
- EvidenceVisualizer: 证据可视化
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from numpy.typing import NDArray

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class EvidenceType(Enum):
    """证据类型枚举"""
    TEMPORAL = "temporal"
    CONTENT = "content"
    NETWORK = "network"
    METADATA = "metadata"
    BEHAVIORAL = "behavioral"
    ENGAGEMENT = "engagement"
    SENTIMENT = "sentiment"
    UNKNOWN = "unknown"


class EvidenceSeverity(Enum):
    """证据严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EvidenceItem:
    """
    单个证据项
    
    存储单个异常证据的详细信息，包括证据类型、特征值、偏离度等。
    
    Attributes:
        evidence_type: 证据类型
        feature_name: 特征名称
        value: 实际特征值
        deviation: 偏离度（Z-score或百分比）
        normal_range: 正常范围（均值和标准差或最小最大值）
        weight: 证据权重
        severity: 严重程度
        description: 证据描述
        timestamp: 证据产生时间
        source: 证据来源
    """
    evidence_type: EvidenceType
    feature_name: str
    value: float
    deviation: float
    normal_range: tuple[float, float]
    weight: float = 1.0
    severity: EvidenceSeverity = EvidenceSeverity.MEDIUM
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    
    def __post_init__(self) -> None:
        if not self.description:
            self.description = self._generate_description()
    
    def _generate_description(self) -> str:
        """生成证据描述"""
        deviation_abs = abs(self.deviation)
        direction = "高于" if self.deviation > 0 else "低于"
        
        severity_desc = {
            EvidenceSeverity.LOW: "轻微",
            EvidenceSeverity.MEDIUM: "中等",
            EvidenceSeverity.HIGH: "显著",
            EvidenceSeverity.CRITICAL: "严重"
        }
        
        return (
            f"{severity_desc[self.severity]}异常: {self.feature_name} "
            f"当前值 {self.value:.4f}，{direction}正常范围 "
            f"[{self.normal_range[0]:.4f}, {self.normal_range[1]:.4f}]，"
            f"偏离度 {deviation_abs:.2f}"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含所有证据信息的字典
        """
        return {
            "evidence_type": self.evidence_type.value,
            "feature_name": self.feature_name,
            "value": self.value,
            "deviation": self.deviation,
            "normal_range": list(self.normal_range),
            "weight": self.weight,
            "severity": self.severity.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }
    
    def get_impact_score(self) -> float:
        """
        计算证据影响分数
        
        Returns:
            影响分数（权重 * 偏离度 * 严重程度系数）
        """
        severity_multiplier = {
            EvidenceSeverity.LOW: 1.0,
            EvidenceSeverity.MEDIUM: 2.0,
            EvidenceSeverity.HIGH: 3.0,
            EvidenceSeverity.CRITICAL: 5.0
        }
        return self.weight * abs(self.deviation) * severity_multiplier[self.severity]
    
    def is_significant(self, threshold: float = 2.0) -> bool:
        """
        判断证据是否显著
        
        Args:
            threshold: 偏离度阈值
            
        Returns:
            是否为显著证据
        """
        return abs(self.deviation) >= threshold


@dataclass
class EvidenceChain:
    """
    证据链
    
    管理单个用户的完整证据链，支持证据添加、置信度计算和排序。
    
    Attributes:
        user_id: 用户唯一标识符
        platform: 平台名称
        evidence_items: 证据项列表
        confidence_score: 置信度分数
        created_at: 创建时间
        updated_at: 更新时间
        metadata: 元数据
    """
    user_id: str
    platform: str
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_evidence(self, item: EvidenceItem) -> None:
        """
        添加证据项
        
        Args:
            item: 证据项
        """
        self.evidence_items.append(item)
        self.updated_at = datetime.now()
        self._update_confidence()
    
    def add_evidences(self, items: list[EvidenceItem]) -> None:
        """
        批量添加证据项
        
        Args:
            items: 证据项列表
        """
        self.evidence_items.extend(items)
        self.updated_at = datetime.now()
        self._update_confidence()
    
    def _update_confidence(self) -> None:
        """更新置信度分数"""
        if not self.evidence_items:
            self.confidence_score = 0.0
            return
        
        total_weight = sum(item.weight for item in self.evidence_items)
        if total_weight == 0:
            self.confidence_score = 0.0
            return
        
        weighted_deviations = sum(
            item.weight * min(abs(item.deviation) / 3.0, 1.0)
            for item in self.evidence_items
        )
        
        self.confidence_score = min(weighted_deviations / total_weight, 1.0)
    
    def calculate_confidence(self) -> float:
        """
        计算置信度
        
        Returns:
            置信度分数（0-1）
        """
        self._update_confidence()
        return self.confidence_score
    
    def get_top_evidence(self, k: int = 5) -> list[EvidenceItem]:
        """
        获取Top-K证据
        
        Args:
            k: 返回的证据数量
            
        Returns:
            按影响分数排序的前K个证据
        """
        sorted_items = sorted(
            self.evidence_items,
            key=lambda x: x.get_impact_score(),
            reverse=True
        )
        return sorted_items[:k]
    
    def get_evidence_by_type(self, evidence_type: EvidenceType) -> list[EvidenceItem]:
        """
        按类型获取证据
        
        Args:
            evidence_type: 证据类型
            
        Returns:
            指定类型的证据列表
        """
        return [
            item for item in self.evidence_items
            if item.evidence_type == evidence_type
        ]
    
    def get_evidence_by_severity(self, severity: EvidenceSeverity) -> list[EvidenceItem]:
        """
        按严重程度获取证据
        
        Args:
            severity: 严重程度
            
        Returns:
            指定严重程度的证据列表
        """
        return [
            item for item in self.evidence_items
            if item.severity == severity
        ]
    
    def get_significant_evidence(self, threshold: float = 2.0) -> list[EvidenceItem]:
        """
        获取显著证据
        
        Args:
            threshold: 偏离度阈值
            
        Returns:
            显著证据列表
        """
        return [
            item for item in self.evidence_items
            if item.is_significant(threshold)
        ]
    
    def to_dict(self) -> dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            包含完整证据链信息的字典
        """
        return {
            "user_id": self.user_id,
            "platform": self.platform,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "confidence_score": self.confidence_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "summary": {
                "total_evidence_count": len(self.evidence_items),
                "significant_evidence_count": len(self.get_significant_evidence()),
                "evidence_types": {
                    et.value: len(self.get_evidence_by_type(et))
                    for et in EvidenceType
                },
                "severity_distribution": {
                    es.value: len(self.get_evidence_by_severity(es))
                    for es in EvidenceSeverity
                }
            }
        }
    
    def get_summary(self) -> str:
        """
        获取证据链摘要
        
        Returns:
            证据链摘要文本
        """
        lines = [
            f"用户 {self.user_id} ({self.platform}) 证据链摘要",
            f"=" * 50,
            f"总证据数: {len(self.evidence_items)}",
            f"置信度: {self.confidence_score:.2%}",
            f"显著证据数: {len(self.get_significant_evidence())}",
            "",
            "证据类型分布:"
        ]
        
        for et in EvidenceType:
            count = len(self.get_evidence_by_type(et))
            if count > 0:
                lines.append(f"  - {et.value}: {count}")
        
        lines.append("")
        lines.append("Top-5 证据:")
        
        for i, item in enumerate(self.get_top_evidence(5), 1):
            lines.append(f"  {i}. {item.feature_name}: 偏离度 {item.deviation:.2f}")
        
        return "\n".join(lines)


class EvidenceGenerator:
    """
    证据生成器
    
    根据用户特征和异常检测结果生成证据链。
    
    Attributes:
        feature_stats: 特征统计信息
        severity_thresholds: 严重程度阈值配置
    """
    
    DEFAULT_SEVERITY_THRESHOLDS = {
        EvidenceSeverity.LOW: 1.5,
        EvidenceSeverity.MEDIUM: 2.0,
        EvidenceSeverity.HIGH: 3.0,
        EvidenceSeverity.CRITICAL: 4.0
    }
    
    def __init__(
        self,
        feature_stats: Optional[dict[str, dict[str, float]]] = None,
        severity_thresholds: Optional[dict[EvidenceSeverity, float]] = None
    ) -> None:
        """
        初始化证据生成器
        
        Args:
            feature_stats: 特征统计信息，格式为 {feature_name: {mean, std, min, max}}
            severity_thresholds: 严重程度阈值配置
        """
        self.feature_stats = feature_stats or {}
        self.severity_thresholds = severity_thresholds or self.DEFAULT_SEVERITY_THRESHOLDS
    
    def set_feature_stats(self, stats: dict[str, dict[str, float]]) -> None:
        """
        设置特征统计信息
        
        Args:
            stats: 特征统计信息
        """
        self.feature_stats = stats
    
    def generate(
        self,
        user_id: str,
        features: dict[str, float],
        anomaly_result: dict[str, Any],
        platform: str = "unknown"
    ) -> EvidenceChain:
        """
        生成证据链
        
        Args:
            user_id: 用户ID
            features: 用户特征字典
            anomaly_result: 异常检测结果
            platform: 平台名称
            
        Returns:
            完整的证据链
            
        Raises:
            ValueError: 当输入数据无效时
        """
        if not user_id:
            raise ValueError("用户ID不能为空")
        
        if not features:
            raise ValueError("特征数据不能为空")
        
        chain = EvidenceChain(
            user_id=user_id,
            platform=platform,
            metadata={
                "anomaly_score": anomaly_result.get("anomaly_score", 0.0),
                "anomaly_type": anomaly_result.get("anomaly_type", "unknown")
            }
        )
        
        feature_contributions = anomaly_result.get("feature_contributions", {})
        
        for feature_name, value in features.items():
            if feature_name not in self.feature_stats:
                continue
            
            stats = self.feature_stats[feature_name]
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            
            if std == 0:
                continue
            
            deviation = self.calculate_feature_deviation(value, mean, std)
            
            if abs(deviation) < 1.0:
                continue
            
            contribution = feature_contributions.get(feature_name, 0.0)
            weight = min(abs(contribution) + 0.5, 2.0)
            
            normal_range = (mean - 2 * std, mean + 2 * std)
            
            severity = self._determine_severity(abs(deviation))
            
            evidence_type = self._infer_evidence_type(feature_name)
            
            item = EvidenceItem(
                evidence_type=evidence_type,
                feature_name=feature_name,
                value=value,
                deviation=deviation,
                normal_range=normal_range,
                weight=weight,
                severity=severity,
                source="anomaly_detection"
            )
            
            chain.add_evidence(item)
        
        ranked_items = self.rank_evidence(chain.evidence_items)
        chain.evidence_items = ranked_items
        
        logger.info(f"为用户 {user_id} 生成了 {len(chain.evidence_items)} 条证据")
        return chain
    
    def calculate_feature_deviation(
        self,
        value: float,
        mean: float,
        std: float
    ) -> float:
        """
        计算特征偏离度
        
        Args:
            value: 特征值
            mean: 均值
            std: 标准差
            
        Returns:
            Z-score偏离度
        """
        if std == 0:
            return 0.0
        return (value - mean) / std
    
    def _determine_severity(self, deviation_abs: float) -> EvidenceSeverity:
        """
        确定证据严重程度
        
        Args:
            deviation_abs: 偏离度绝对值
            
        Returns:
            严重程度枚举值
        """
        if deviation_abs >= self.severity_thresholds[EvidenceSeverity.CRITICAL]:
            return EvidenceSeverity.CRITICAL
        elif deviation_abs >= self.severity_thresholds[EvidenceSeverity.HIGH]:
            return EvidenceSeverity.HIGH
        elif deviation_abs >= self.severity_thresholds[EvidenceSeverity.MEDIUM]:
            return EvidenceSeverity.MEDIUM
        else:
            return EvidenceSeverity.LOW
    
    def _infer_evidence_type(self, feature_name: str) -> EvidenceType:
        """
        推断证据类型
        
        Args:
            feature_name: 特征名称
            
        Returns:
            证据类型枚举值
        """
        feature_lower = feature_name.lower()
        
        temporal_keywords = ["daily", "hour", "time", "frequency", "burst", "delay", "autocorr"]
        content_keywords = ["text", "similarity", "topic", "sentiment", "template", "content"]
        network_keywords = ["centrality", "pagerank", "cluster", "community", "network"]
        metadata_keywords = ["username", "avatar", "profile", "registration"]
        engagement_keywords = ["engagement", "like", "share", "comment", "retweet"]
        
        if any(kw in feature_lower for kw in temporal_keywords):
            return EvidenceType.TEMPORAL
        elif any(kw in feature_lower for kw in content_keywords):
            return EvidenceType.CONTENT
        elif any(kw in feature_lower for kw in network_keywords):
            return EvidenceType.NETWORK
        elif any(kw in feature_lower for kw in metadata_keywords):
            return EvidenceType.METADATA
        elif any(kw in feature_lower for kw in engagement_keywords):
            return EvidenceType.ENGAGEMENT
        else:
            return EvidenceType.BEHAVIORAL
    
    def rank_evidence(
        self,
        evidence_items: list[EvidenceItem]
    ) -> list[EvidenceItem]:
        """
        证据排序
        
        按影响分数从高到低排序。
        
        Args:
            evidence_items: 证据项列表
            
        Returns:
            排序后的证据列表
        """
        return sorted(
            evidence_items,
            key=lambda x: x.get_impact_score(),
            reverse=True
        )
    
    def filter_significant_evidence(
        self,
        items: list[EvidenceItem],
        threshold: float = 2.0
    ) -> list[EvidenceItem]:
        """
        过滤显著证据
        
        Args:
            items: 证据项列表
            threshold: 偏离度阈值
            
        Returns:
            显著证据列表
        """
        return [item for item in items if item.is_significant(threshold)]
    
    def compute_feature_stats_from_data(
        self,
        data: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """
        从数据计算特征统计信息
        
        Args:
            data: 特征数据字典，格式为 {feature_name: [values]}
            
        Returns:
            特征统计信息字典
        """
        stats = {}
        
        for feature_name, values in data.items():
            if not values:
                continue
            
            values_array = np.array(values)
            stats[feature_name] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "median": float(np.median(values_array)),
                "q1": float(np.percentile(values_array, 25)),
                "q3": float(np.percentile(values_array, 75))
            }
        
        self.feature_stats = stats
        return stats


class SHAPExplainer:
    """
    SHAP特征归因
    
    使用SHAP值解释模型预测，分析各特征对异常检测的贡献。
    
    Attributes:
        model: 待解释的模型
        explainer: SHAP解释器
        shap_values: 计算得到的SHAP值
        feature_names: 特征名称列表
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        background_data: Optional[NDArray[np.floating]] = None
    ) -> None:
        """
        初始化SHAP解释器
        
        Args:
            model: 待解释的模型
            background_data: 背景数据（用于创建解释器）
        """
        self.model = model
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[NDArray[np.floating]] = None
        self.feature_names: list[str] = []
        self._background_data = background_data
        self._is_fitted: bool = False
    
    def explain(
        self,
        model: Optional[Any] = None,
        data: Optional[NDArray[np.floating]] = None,
        feature_names: Optional[list[str]] = None
    ) -> NDArray[np.floating]:
        """
        解释模型预测
        
        Args:
            model: 待解释的模型（可选，如果初始化时已提供）
            data: 待解释的数据
            feature_names: 特征名称列表
            
        Returns:
            SHAP值数组
            
        Raises:
            ImportError: 当SHAP库未安装时
            ValueError: 当输入数据无效时
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP库未安装，请使用 'pip install shap' 安装")
        
        if model is not None:
            self.model = model
        
        if self.model is None:
            raise ValueError("未提供模型")
        
        if data is None:
            raise ValueError("未提供数据")
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        try:
            model_type = type(self.model).__name__
            
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'estimators_') else shap.KernelExplainer(self.model.predict_proba, self._background_data if self._background_data is not None else shap.kmeans(data, 10))
            elif hasattr(self.model, 'decision_function'):
                self.explainer = shap.KernelExplainer(
                    self.model.decision_function,
                    self._background_data if self._background_data is not None else shap.kmeans(data, 10)
                )
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict,
                    self._background_data if self._background_data is not None else shap.kmeans(data, 10)
                )
            
            self.shap_values = self.explainer.shap_values(data)
            self._is_fitted = True
            
            logger.info(f"SHAP解释完成，形状: {self.shap_values.shape if self.shap_values is not None else 'N/A'}")
            
            return self.shap_values
            
        except Exception as e:
            logger.error(f"SHAP解释失败: {e}")
            raise
    
    def get_feature_importance(
        self,
        aggregate: str = "mean"
    ) -> dict[str, float]:
        """
        获取特征重要性
        
        Args:
            aggregate: 聚合方法 ('mean', 'max', 'sum')
            
        Returns:
            特征名称和重要性分数的字典
            
        Raises:
            RuntimeError: 当模型未解释时
        """
        if not self._is_fitted or self.shap_values is None:
            raise RuntimeError("模型尚未解释，请先调用 explain() 方法")
        
        if isinstance(self.shap_values, list):
            shap_array = np.abs(self.shap_values[0])
        else:
            shap_array = np.abs(self.shap_values)
        
        if aggregate == "mean":
            importance = np.mean(shap_array, axis=0)
        elif aggregate == "max":
            importance = np.max(shap_array, axis=0)
        elif aggregate == "sum":
            importance = np.sum(shap_array, axis=0)
        else:
            raise ValueError(f"未知的聚合方法: {aggregate}")
        
        if self.feature_names:
            return {
                name: float(imp)
                for name, imp in zip(self.feature_names, importance)
            }
        else:
            return {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(importance)
            }
    
    def get_sample_explanation(
        self,
        sample_index: int
    ) -> dict[str, float]:
        """
        获取单个样本的解释
        
        Args:
            sample_index: 样本索引
            
        Returns:
            特征名称和SHAP值的字典
        """
        if not self._is_fitted or self.shap_values is None:
            raise RuntimeError("模型尚未解释，请先调用 explain() 方法")
        
        if isinstance(self.shap_values, list):
            sample_shap = self.shap_values[0][sample_index]
        else:
            sample_shap = self.shap_values[sample_index]
        
        if self.feature_names:
            return {
                name: float(val)
                for name, val in zip(self.feature_names, sample_shap)
            }
        else:
            return {
                f"feature_{i}": float(val)
                for i, val in enumerate(sample_shap)
            }
    
    def plot_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        绘制特征重要性图表
        
        Args:
            top_n: 显示前N个特征
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象（如果MATPLOTLIB_AVAILABLE）
            
        Raises:
            ImportError: 当matplotlib未安装时
            RuntimeError: 当模型未解释时
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装，请使用 'pip install matplotlib' 安装")
        
        if not self._is_fitted or self.shap_values is None:
            raise RuntimeError("模型尚未解释，请先调用 explain() 方法")
        
        importance = self.get_feature_importance()
        
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        features = [x[0] for x in sorted_importance]
        values = [x[1] for x in sorted_importance]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Value (mean absolute)')
        ax.set_title('Feature Importance (SHAP)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存至 {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_waterfall(
        self,
        sample_index: int,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        绘制瀑布图（单样本解释）
        
        Args:
            sample_index: 样本索引
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装")
        
        if not self._is_fitted or self.shap_values is None:
            raise RuntimeError("模型尚未解释")
        
        sample_explanation = self.get_sample_explanation(sample_index)
        
        sorted_items = sorted(
            sample_explanation.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15]
        
        features = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cumulative = 0
        for i, (feature, value) in enumerate(zip(features, values)):
            color = '#ff6b6b' if value > 0 else '#4ecdc4'
            ax.barh(i, value, left=cumulative, color=color, alpha=0.8)
            cumulative += value
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Cumulative SHAP Value')
        ax.set_title(f'SHAP Waterfall Plot - Sample {sample_index}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


class EvidenceReportGenerator:
    """
    证据报告生成器
    
    生成各种格式的证据报告，包括单用户报告、对比报告和时间线报告。
    
    Attributes:
        output_format: 输出格式
        template: 报告模板
    """
    
    def __init__(
        self,
        output_format: str = "dict"
    ) -> None:
        """
        初始化报告生成器
        
        Args:
            output_format: 输出格式 ('dict', 'json', 'markdown', 'html')
        """
        self.output_format = output_format
    
    def generate_report(
        self,
        user_id: str,
        evidence_chain: EvidenceChain
    ) -> dict[str, Any]:
        """
        生成单用户证据报告
        
        Args:
            user_id: 用户ID
            evidence_chain: 证据链
            
        Returns:
            报告字典
            
        Raises:
            ValueError: 当输入数据无效时
        """
        if not user_id:
            raise ValueError("用户ID不能为空")
        
        if not evidence_chain or not evidence_chain.evidence_items:
            return {
                "user_id": user_id,
                "status": "no_evidence",
                "message": "未发现显著证据"
            }
        
        top_evidence = evidence_chain.get_top_evidence(10)
        significant_evidence = evidence_chain.get_significant_evidence()
        
        evidence_by_type: dict[EvidenceType, list[EvidenceItem]] = {}
        for item in evidence_chain.evidence_items:
            if item.evidence_type not in evidence_by_type:
                evidence_by_type[item.evidence_type] = []
            evidence_by_type[item.evidence_type].append(item)
        
        severity_counts = {
            EvidenceSeverity.LOW: 0,
            EvidenceSeverity.MEDIUM: 0,
            EvidenceSeverity.HIGH: 0,
            EvidenceSeverity.CRITICAL: 0
        }
        for item in evidence_chain.evidence_items:
            severity_counts[item.severity] += 1
        
        risk_level = self._calculate_risk_level(evidence_chain)
        
        report = {
            "report_type": "single_user",
            "user_id": user_id,
            "platform": evidence_chain.platform,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_evidence_count": len(evidence_chain.evidence_items),
                "significant_evidence_count": len(significant_evidence),
                "confidence_score": evidence_chain.confidence_score,
                "risk_level": risk_level
            },
            "severity_distribution": {
                level.value: count
                for level, count in severity_counts.items()
            },
            "evidence_type_distribution": {
                etype.value: len(items)
                for etype, items in evidence_by_type.items()
            },
            "top_evidence": [
                {
                    "rank": i + 1,
                    "feature_name": item.feature_name,
                    "value": item.value,
                    "deviation": item.deviation,
                    "severity": item.severity.value,
                    "description": item.description
                }
                for i, item in enumerate(top_evidence)
            ],
            "recommendations": self._generate_recommendations(evidence_chain),
            "detailed_evidence": evidence_chain.to_dict()
        }
        
        logger.info(f"为用户 {user_id} 生成了证据报告")
        return report
    
    def generate_comparison_report(
        self,
        user_ids: list[str],
        evidence_chains: dict[str, EvidenceChain]
    ) -> dict[str, Any]:
        """
        生成对比报告
        
        Args:
            user_ids: 用户ID列表
            evidence_chains: 用户ID到证据链的映射
            
        Returns:
            对比报告字典
        """
        if not user_ids or not evidence_chains:
            raise ValueError("用户ID列表和证据链不能为空")
        
        comparison_data = []
        
        for user_id in user_ids:
            if user_id not in evidence_chains:
                continue
            
            chain = evidence_chains[user_id]
            comparison_data.append({
                "user_id": user_id,
                "platform": chain.platform,
                "total_evidence": len(chain.evidence_items),
                "significant_evidence": len(chain.get_significant_evidence()),
                "confidence_score": chain.confidence_score,
                "top_features": [
                    item.feature_name for item in chain.get_top_evidence(3)
                ]
            })
        
        comparison_data.sort(
            key=lambda x: x["confidence_score"],
            reverse=True
        )
        
        common_features = self._find_common_features(evidence_chains)
        
        report = {
            "report_type": "comparison",
            "generated_at": datetime.now().isoformat(),
            "user_count": len(comparison_data),
            "comparison_table": comparison_data,
            "common_anomalous_features": common_features,
            "statistics": {
                "avg_confidence": np.mean([d["confidence_score"] for d in comparison_data]) if comparison_data else 0,
                "max_confidence": max(d["confidence_score"] for d in comparison_data) if comparison_data else 0,
                "avg_evidence_count": np.mean([d["total_evidence"] for d in comparison_data]) if comparison_data else 0
            }
        }
        
        return report
    
    def generate_timeline_report(
        self,
        user_id: str,
        events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        生成时间线报告
        
        Args:
            user_id: 用户ID
            events: 事件列表，每个事件应包含 'timestamp', 'event_type', 'description'
            
        Returns:
            时间线报告字典
        """
        if not events:
            return {
                "user_id": user_id,
                "status": "no_events",
                "message": "未提供事件数据"
            }
        
        sorted_events = sorted(
            events,
            key=lambda x: x.get("timestamp", datetime.min)
        )
        
        event_types: dict[str, list[dict[str, Any]]] = {}
        for event in sorted_events:
            event_type = event.get("event_type", "unknown")
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        if sorted_events:
            start_time = sorted_events[0].get("timestamp", datetime.min)
            end_time = sorted_events[-1].get("timestamp", datetime.min)
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            time_span = (end_time - start_time).total_seconds()
        else:
            time_span = 0
        
        report = {
            "report_type": "timeline",
            "user_id": user_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_events": len(events),
                "time_span_seconds": time_span,
                "event_types": list(event_types.keys())
            },
            "event_type_distribution": {
                etype: len(events_list)
                for etype, events_list in event_types.items()
            },
            "timeline": [
                {
                    "timestamp": event.get("timestamp", "").isoformat() if isinstance(event.get("timestamp"), datetime) else event.get("timestamp", ""),
                    "event_type": event.get("event_type", "unknown"),
                    "description": event.get("description", ""),
                    "severity": event.get("severity", "medium"),
                    "metadata": event.get("metadata", {})
                }
                for event in sorted_events
            ]
        }
        
        return report
    
    def _calculate_risk_level(self, evidence_chain: EvidenceChain) -> str:
        """计算风险等级"""
        confidence = evidence_chain.confidence_score
        critical_count = len(evidence_chain.get_evidence_by_severity(EvidenceSeverity.CRITICAL))
        high_count = len(evidence_chain.get_evidence_by_severity(EvidenceSeverity.HIGH))
        
        if critical_count >= 2 or (critical_count >= 1 and high_count >= 3):
            return "critical"
        elif critical_count >= 1 or high_count >= 3 or confidence > 0.8:
            return "high"
        elif high_count >= 1 or confidence > 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, evidence_chain: EvidenceChain) -> list[str]:
        """生成建议"""
        recommendations = []
        
        if evidence_chain.confidence_score > 0.8:
            recommendations.append("建议立即进行人工审核")
        
        critical_evidence = evidence_chain.get_evidence_by_severity(EvidenceSeverity.CRITICAL)
        if critical_evidence:
            recommendations.append(f"发现 {len(critical_evidence)} 条关键证据，需要优先处理")
        
        temporal_evidence = evidence_chain.get_evidence_by_type(EvidenceType.TEMPORAL)
        if len(temporal_evidence) > 3:
            recommendations.append("时序行为异常明显，建议分析用户活动模式")
        
        content_evidence = evidence_chain.get_evidence_by_type(EvidenceType.CONTENT)
        if len(content_evidence) > 3:
            recommendations.append("内容特征异常，建议检查内容原创性和质量")
        
        network_evidence = evidence_chain.get_evidence_by_type(EvidenceType.NETWORK)
        if len(network_evidence) > 2:
            recommendations.append("网络特征异常，建议分析用户社交关系")
        
        if not recommendations:
            recommendations.append("建议持续监控用户行为")
        
        return recommendations
    
    def _find_common_features(
        self,
        evidence_chains: dict[str, EvidenceChain]
    ) -> list[dict[str, Any]]:
        """找出共同的异常特征"""
        feature_counts: dict[str, int] = {}
        
        for chain in evidence_chains.values():
            seen_features = set()
            for item in chain.evidence_items:
                if item.feature_name not in seen_features:
                    feature_counts[item.feature_name] = feature_counts.get(item.feature_name, 0) + 1
                    seen_features.add(item.feature_name)
        
        common = [
            {"feature_name": name, "occurrence_count": count}
            for name, count in feature_counts.items()
            if count >= max(2, len(evidence_chains) // 2)
        ]
        
        return sorted(common, key=lambda x: x["occurrence_count"], reverse=True)


class EvidenceVisualizer:
    """
    证据可视化
    
    提供证据链、特征对比和时间线的可视化功能。
    
    Attributes:
        style: 图表样式
        figsize: 默认图形大小
    """
    
    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: tuple[int, int] = (12, 8)
    ) -> None:
        """
        初始化可视化器
        
        Args:
            style: matplotlib样式
            figsize: 默认图形大小
        """
        self.style = style
        self.figsize = figsize
        
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except Exception:
                logger.warning(f"无法使用样式 {style}，使用默认样式")
    
    def plot_evidence_chain(
        self,
        chain: EvidenceChain,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        绘制证据链可视化图
        
        Args:
            chain: 证据链
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装，请使用 'pip install matplotlib' 安装")
        
        if not chain.evidence_items:
            logger.warning("证据链为空，无法绘制")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f"证据链分析 - 用户 {chain.user_id}", fontsize=14, fontweight='bold')
        
        ax1 = axes[0, 0]
        evidence_types = {}
        for item in chain.evidence_items:
            etype = item.evidence_type.value
            evidence_types[etype] = evidence_types.get(etype, 0) + 1
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(evidence_types)))
        ax1.pie(
            evidence_types.values(),
            labels=evidence_types.keys(),
            autopct='%1.1f%%',
            colors=colors
        )
        ax1.set_title('证据类型分布')
        
        ax2 = axes[0, 1]
        top_evidence = chain.get_top_evidence(10)
        features = [item.feature_name for item in top_evidence]
        deviations = [item.deviation for item in top_evidence]
        
        colors = ['#ff6b6b' if d > 0 else '#4ecdc4' for d in deviations]
        y_pos = np.arange(len(features))
        ax2.barh(y_pos, deviations, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features)
        ax2.invert_yaxis()
        ax2.set_xlabel('偏离度 (Z-score)')
        ax2.set_title('Top-10 证据偏离度')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax3 = axes[1, 0]
        severity_counts = {
            EvidenceSeverity.LOW: 0,
            EvidenceSeverity.MEDIUM: 0,
            EvidenceSeverity.HIGH: 0,
            EvidenceSeverity.CRITICAL: 0
        }
        for item in chain.evidence_items:
            severity_counts[item.severity] += 1
        
        severity_labels = ['Low', 'Medium', 'High', 'Critical']
        severity_values = [severity_counts[EvidenceSeverity.LOW], 
                          severity_counts[EvidenceSeverity.MEDIUM],
                          severity_counts[EvidenceSeverity.HIGH],
                          severity_counts[EvidenceSeverity.CRITICAL]]
        severity_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        
        bars = ax3.bar(severity_labels, severity_values, color=severity_colors)
        ax3.set_xlabel('严重程度')
        ax3.set_ylabel('证据数量')
        ax3.set_title('严重程度分布')
        
        for bar, val in zip(bars, severity_values):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(val), ha='center', va='bottom')
        
        ax4 = axes[1, 1]
        impact_scores = [item.get_impact_score() for item in chain.evidence_items]
        ax4.hist(impact_scores, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
        ax4.set_xlabel('影响分数')
        ax4.set_ylabel('证据数量')
        ax4.set_title('影响分数分布')
        ax4.axvline(x=np.mean(impact_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(impact_scores):.2f}')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存至 {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_comparison(
        self,
        user_id: str,
        normal_users: list[dict[str, float]],
        anomaly_user: dict[str, float],
        top_features: Optional[list[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        绘制特征对比图
        
        Args:
            user_id: 目标用户ID
            normal_users: 正常用户特征列表
            anomaly_user: 异常用户特征
            top_features: 要对比的特征列表（可选）
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装")
        
        if not normal_users or not anomaly_user:
            raise ValueError("正常用户和异常用户数据不能为空")
        
        if top_features is None:
            top_features = list(anomaly_user.keys())[:10]
        
        if PANDAS_AVAILABLE:
            normal_df = pd.DataFrame(normal_users)
            normal_means = normal_df.mean()
            normal_stds = normal_df.std()
        else:
            normal_means = {
                k: np.mean([u.get(k, 0) for u in normal_users])
                for k in top_features
            }
            normal_stds = {
                k: np.std([u.get(k, 0) for u in normal_users])
                for k in top_features
            }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(top_features))
        width = 0.35
        
        normal_values = [normal_means.get(f, 0) for f in top_features]
        anomaly_values = [anomaly_user.get(f, 0) for f in top_features]
        
        bars1 = ax.bar(x - width/2, normal_values, width, label='正常用户均值', 
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, anomaly_values, width, label='目标用户', 
                       color='#e74c3c', alpha=0.8)
        
        errors = [normal_stds.get(f, 0) for f in top_features]
        ax.errorbar(x - width/2, normal_values, yerr=errors, fmt='none', 
                   color='black', capsize=3, alpha=0.5)
        
        ax.set_xlabel('特征')
        ax.set_ylabel('特征值')
        ax.set_title(f'特征对比 - 用户 {user_id}')
        ax.set_xticks(x)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_timeline(
        self,
        user_id: str,
        events: list[dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        绘制时间线图
        
        Args:
            user_id: 用户ID
            events: 事件列表
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装")
        
        if not events:
            logger.warning("事件列表为空，无法绘制时间线")
            return None
        
        sorted_events = sorted(
            events,
            key=lambda x: x.get("timestamp", datetime.min)
        )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        timestamps = []
        for event in sorted_events:
            ts = event.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            timestamps.append(ts)
        
        event_types = [event.get("event_type", "unknown") for event in sorted_events]
        unique_types = list(set(event_types))
        type_to_y = {t: i for i, t in enumerate(unique_types)}
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, event_type in enumerate(unique_types):
            type_events = [
                (ts, event) for ts, event in zip(timestamps, sorted_events)
                if event.get("event_type", "unknown") == event_type
            ]
            type_timestamps = [te[0] for te in type_events]
            y_values = [type_to_y[event_type]] * len(type_timestamps)
            
            ax.scatter(type_timestamps, y_values, c=[colors[i]], s=100, 
                      label=event_type, alpha=0.7)
        
        ax.set_yticks(range(len(unique_types)))
        ax.set_yticklabels(unique_types)
        ax.set_xlabel('时间')
        ax.set_ylabel('事件类型')
        ax.set_title(f'事件时间线 - 用户 {user_id}')
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_radar_chart(
        self,
        user_id: str,
        feature_scores: dict[str, float],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        绘制雷达图
        
        Args:
            user_id: 用户ID
            feature_scores: 特征分数字典
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装")
        
        if not feature_scores:
            logger.warning("特征分数为空，无法绘制雷达图")
            return None
        
        categories = list(feature_scores.keys())
        values = list(feature_scores.values())
        
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        values += values[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.fill(angles, values, color='#3498db', alpha=0.25)
        ax.plot(angles, values, color='#3498db', linewidth=2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        ax.set_title(f'特征雷达图 - 用户 {user_id}', size=14, y=1.08)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_dashboard(
        self,
        evidence_chain: EvidenceChain,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        创建综合仪表盘
        
        Args:
            evidence_chain: 证据链
            save_path: 保存路径（可选）
            show: 是否显示图表
            
        Returns:
            matplotlib图形对象
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib库未安装")
        
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        evidence_types = {}
        for item in evidence_chain.evidence_items:
            etype = item.evidence_type.value
            evidence_types[etype] = evidence_types.get(etype, 0) + 1
        
        if evidence_types:
            colors = plt.cm.Set3(np.linspace(0, 1, len(evidence_types)))
            ax1.pie(evidence_types.values(), labels=evidence_types.keys(), 
                   autopct='%1.1f%%', colors=colors)
        ax1.set_title('证据类型分布')
        
        ax2 = fig.add_subplot(gs[0, 1])
        top_evidence = evidence_chain.get_top_evidence(5)
        if top_evidence:
            features = [item.feature_name[:15] for item in top_evidence]
            deviations = [item.deviation for item in top_evidence]
            colors = ['#ff6b6b' if d > 0 else '#4ecdc4' for d in deviations]
            ax2.barh(range(len(features)), deviations, color=colors)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_xlabel('偏离度')
        ax2.set_title('Top-5 证据')
        
        ax3 = fig.add_subplot(gs[0, 2])
        severity_counts = {}
        for item in evidence_chain.evidence_items:
            sev = item.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        if severity_counts:
            severity_colors = {'low': '#2ecc71', 'medium': '#f1c40f', 
                              'high': '#e67e22', 'critical': '#e74c3c'}
            colors = [severity_colors.get(k, '#95a5a6') for k in severity_counts.keys()]
            ax3.bar(severity_counts.keys(), severity_counts.values(), color=colors)
        ax3.set_title('严重程度分布')
        ax3.set_xlabel('严重程度')
        ax3.set_ylabel('数量')
        
        ax4 = fig.add_subplot(gs[1, :])
        impact_scores = [item.get_impact_score() for item in evidence_chain.evidence_items]
        if impact_scores:
            ax4.hist(impact_scores, bins=15, color='#3498db', edgecolor='white', alpha=0.7)
            ax4.axvline(x=np.mean(impact_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(impact_scores):.2f}')
            ax4.legend()
        ax4.set_title('影响分数分布')
        ax4.set_xlabel('影响分数')
        ax4.set_ylabel('证据数量')
        
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = evidence_chain.get_summary()
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(f'证据链仪表盘 - 用户 {evidence_chain.user_id}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
