"""
异常检测模块 - 检测异常行为和趋势

该模块提供多种异常检测算法，包括：
- IsolationForest: 基于隔离森林的异常检测
- LOF: 局部异常因子检测
- Z-Score: 基于统计的异常检测
- Autoencoder: 基于自编码器的异常检测
- Ensemble: 集成多种检测器的异常检测
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from numpy.typing import NDArray


class AnomalyType(Enum):
    """异常类型枚举"""
    NORMAL = "normal"
    VIRAL_CONTENT = "viral_content"
    RAPID_SPREAD = "rapid_spread"
    UNUSUAL_ENGAGEMENT = "unusual_engagement"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    COORDINATED_BEHAVIOR = "coordinated_behavior"
    BOT_LIKE = "bot_like"
    UNKNOWN = "unknown"


@dataclass
class AnomalyReport:
    """
    异常检测报告类
    
    存储检测结果，包含异常分数、特征贡献度等
    
    Attributes:
        is_anomaly: 是否为异常
        anomaly_score: 异常分数（越高越异常）
        anomaly_type: 异常类型
        confidence: 检测置信度
        feature_contributions: 各特征对异常的贡献度
        details: 其他详细信息
        timestamp: 检测时间戳
    """
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    confidence: float = 0.0
    feature_contributions: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "anomaly_type": self.anomaly_type.value,
            "confidence": self.confidence,
            "feature_contributions": self.feature_contributions,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

    def get_top_contributing_features(self, top_n: int = 5) -> list[tuple[str, float]]:
        """
        获取贡献度最高的特征
        
        Args:
            top_n: 返回的特征数量
            
        Returns:
            按贡献度排序的特征列表
        """
        sorted_features = sorted(
            self.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_n]

    def get_explanation(self) -> str:
        """
        生成可读的异常解释
        
        Returns:
            异常解释文本
        """
        if not self.is_anomaly:
            return "数据点正常，未检测到异常。"

        parts = [f"检测到异常类型: {self.anomaly_type.value}"]
        parts.append(f"异常分数: {self.anomaly_score:.4f}")
        parts.append(f"置信度: {self.confidence:.2%}")

        if self.feature_contributions:
            top_features = self.get_top_contributing_features(3)
            parts.append("主要贡献特征:")
            for name, contribution in top_features:
                parts.append(f"  - {name}: {contribution:.4f}")

        return "\n".join(parts)


@dataclass
class TrendPoint:
    """趋势数据点"""
    timestamp: datetime
    value: float
    is_anomaly: bool = False


@dataclass
class AnomalyResult:
    """异常检测结果（兼容旧版本）"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    details: dict[str, Any]


class BaseAnomalyDetector(ABC):
    """
    异常检测器抽象基类
    
    所有异常检测算法都应继承此类并实现相应方法。
    
    Attributes:
        is_fitted: 模型是否已训练
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练"""
        return self._is_fitted

    @abstractmethod
    def fit(self, X: NDArray[np.floating]) -> "BaseAnomalyDetector":
        """
        训练异常检测模型
        
        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            
        Returns:
            self: 训练后的检测器实例
            
        Raises:
            ValueError: 当输入数据无效时
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        预测数据点是否为异常
        
        Args:
            X: 待预测数据，形状为 (n_samples, n_features)
            
        Returns:
            布尔数组，True 表示异常
            
        Raises:
            RuntimeError: 当模型未训练时
        """
        pass

    def fit_predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        训练模型并预测异常
        
        Args:
            X: 输入数据
            
        Returns:
            布尔数组，True 表示异常
        """
        self.fit(X)
        return self.predict(X)

    @abstractmethod
    def get_anomaly_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组，分数越高越异常
            
        Raises:
            RuntimeError: 当模型未训练时
        """
        pass

    def detect(self, X: NDArray[np.floating]) -> list[AnomalyReport]:
        """
        检测异常并生成报告
        
        Args:
            X: 输入数据
            
        Returns:
            异常报告列表
        """
        predictions = self.predict(X)
        scores = self.get_anomaly_scores(X)

        reports = []
        for i in range(len(X)):
            report = AnomalyReport(
                is_anomaly=bool(predictions[i]),
                anomaly_score=float(scores[i]),
                anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY if predictions[i] else AnomalyType.NORMAL,
                confidence=min(abs(scores[i]) / 2.0, 1.0),
                details={"sample_index": i}
            )
            reports.append(report)

        return reports

    def _validate_input(self, X: NDArray[np.floating]) -> None:
        """
        验证输入数据
        
        Args:
            X: 输入数据
            
        Raises:
            ValueError: 当输入数据无效时
        """
        if X is None or len(X) == 0:
            raise ValueError("输入数据不能为空")
        if not isinstance(X, np.ndarray):
            raise ValueError("输入必须是 numpy 数组")
        if X.ndim != 2:
            raise ValueError("输入数据必须是二维数组 (n_samples, n_features)")

    def _check_fitted(self) -> None:
        """
        检查模型是否已训练
        
        Raises:
            RuntimeError: 当模型未训练时
        """
        if not self._is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit() 方法")


class IsolationForestDetector(BaseAnomalyDetector):
    """
    基于隔离森林的异常检测器
    
    使用 sklearn 的 IsolationForest 实现，适合高维数据和大规模数据集。
    
    Attributes:
        contamination: 预期异常比例
        n_estimators: 决策树数量
        max_samples: 每棵树使用的样本数
        random_state: 随机种子
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        random_state: int = 42,
        max_features: float = 1.0
    ) -> None:
        super().__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_features = max_features
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._feature_importances: Optional[NDArray[np.floating]] = None
        self._n_features: int = 0

    def fit(self, X: NDArray[np.floating]) -> "IsolationForestDetector":
        """
        训练隔离森林模型
        
        Args:
            X: 训练数据
            
        Returns:
            训练后的检测器
        """
        self._validate_input(X)

        if len(X) < 10:
            logger.warning("训练样本数量较少，可能导致模型不稳定")

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        self._n_features = X.shape[1]
        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            max_features=self.max_features
        )
        self._model.fit(scaled_data)
        self._is_fitted = True

        self._compute_feature_importance(X)

        logger.info(f"IsolationForest 模型训练完成，特征数: {self._n_features}")
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        预测异常
        
        Args:
            X: 待预测数据
            
        Returns:
            布尔数组，True 表示异常
        """
        self._check_fitted()
        self._validate_input(X)

        if X.shape[1] != self._n_features:
            raise ValueError(f"特征数不匹配: 期望 {self._n_features}，实际 {X.shape[1]}")

        scaled_data = self._scaler.transform(X)
        predictions = self._model.predict(scaled_data)
        return predictions == -1

    def get_anomaly_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组
        """
        self._check_fitted()
        self._validate_input(X)

        scaled_data = self._scaler.transform(X)
        scores = self._model.score_samples(scaled_data)
        return -scores

    def _compute_feature_importance(self, X: NDArray[np.floating]) -> None:
        """
        计算特征重要性
        
        通过计算每个特征对异常分数的贡献来估计重要性
        
        Args:
            X: 训练数据
        """
        scaled_data = self._scaler.transform(X)
        base_scores = self._model.score_samples(scaled_data)

        importances = []
        for i in range(self._n_features):
            perturbed = scaled_data.copy()
            perturbed[:, i] = 0
            perturbed_scores = self._model.score_samples(perturbed)
            importance = np.mean(np.abs(base_scores - perturbed_scores))
            importances.append(importance)

        total = sum(importances)
        if total > 0:
            self._feature_importances = np.array(importances) / total
        else:
            self._feature_importances = np.ones(self._n_features) / self._n_features

    def get_feature_importance(self) -> dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征名称和重要性分数的字典
        """
        if self._feature_importances is None:
            return {}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(self._feature_importances)}

    def detect_with_contributions(
        self,
        X: NDArray[np.floating],
        feature_names: Optional[list[str]] = None
    ) -> list[AnomalyReport]:
        """
        检测异常并计算特征贡献度
        
        Args:
            X: 输入数据
            feature_names: 特征名称列表
            
        Returns:
            包含特征贡献度的异常报告
        """
        predictions = self.predict(X)
        scores = self.get_anomaly_scores(X)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self._n_features)]

        reports = []
        for i in range(len(X)):
            contributions = self._compute_sample_contributions(X[i:i+1])
            named_contributions = {
                feature_names[j]: contributions[j]
                for j in range(min(len(feature_names), len(contributions)))
            }

            report = AnomalyReport(
                is_anomaly=bool(predictions[i]),
                anomaly_score=float(scores[i]),
                anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY if predictions[i] else AnomalyType.NORMAL,
                confidence=min(abs(scores[i]) / 2.0, 1.0),
                feature_contributions=named_contributions,
                details={"sample_index": i}
            )
            reports.append(report)

        return reports

    def _compute_sample_contributions(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        计算单个样本的特征贡献度
        
        Args:
            X: 单个样本（形状为 (1, n_features)）
            
        Returns:
            特征贡献度数组
        """
        scaled = self._scaler.transform(X)
        base_score = self._model.score_samples(scaled)[0]

        contributions = np.zeros(self._n_features)
        for i in range(self._n_features):
            perturbed = scaled.copy()
            perturbed[0, i] = 0
            perturbed_score = self._model.score_samples(perturbed)[0]
            contributions[i] = abs(base_score - perturbed_score)

        return contributions

    def get_params(self) -> dict[str, Any]:
        """获取模型参数"""
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "max_features": self.max_features,
            "is_fitted": self._is_fitted,
            "n_features": self._n_features
        }


class LOFDetector(BaseAnomalyDetector):
    """
    局部异常因子检测器
    
    基于局部密度的异常检测方法，适合密度不均匀的数据集。
    
    Attributes:
        n_neighbors: 近邻数量
        contamination: 预期异常比例
        algorithm: 近邻搜索算法
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._n_features: int = 0
        self._threshold: float = 0.0

    def fit(self, X: NDArray[np.floating]) -> "LOFDetector":
        """
        训练 LOF 模型
        
        Args:
            X: 训练数据
            
        Returns:
            训练后的检测器
        """
        self._validate_input(X)

        if len(X) <= self.n_neighbors:
            raise ValueError(f"样本数量 ({len(X)}) 必须大于 n_neighbors ({self.n_neighbors})")

        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler

        self._n_features = X.shape[1]
        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(X)

        self._model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            novelty=True
        )
        self._model.fit(scaled_data)
        self._is_fitted = True

        training_scores = -self._model.negative_outlier_factor_
        self._threshold = np.percentile(training_scores, (1 - self.contamination) * 100)

        logger.info(f"LOF 模型训练完成，阈值: {self._threshold:.4f}")
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        预测异常
        
        Args:
            X: 待预测数据
            
        Returns:
            布尔数组，True 表示异常
        """
        self._check_fitted()
        self._validate_input(X)

        if X.shape[1] != self._n_features:
            raise ValueError(f"特征数不匹配: 期望 {self._n_features}，实际 {X.shape[1]}")

        scaled_data = self._scaler.transform(X)
        predictions = self._model.predict(scaled_data)
        return predictions == -1

    def get_anomaly_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组（LOF 值，越大越异常）
        """
        self._check_fitted()
        self._validate_input(X)

        scaled_data = self._scaler.transform(X)
        negative_scores = self._model.score_samples(scaled_data)
        return -negative_scores

    def get_local_reachability_density(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取局部可达密度
        
        Args:
            X: 输入数据
            
        Returns:
            局部可达密度数组
        """
        self._check_fitted()
        scores = self.get_anomaly_scores(X)
        return 1.0 / (scores + 1e-10)

    def get_params(self) -> dict[str, Any]:
        """获取模型参数"""
        return {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "metric": self.metric,
            "p": self.p,
            "is_fitted": self._is_fitted,
            "n_features": self._n_features,
            "threshold": self._threshold
        }


class ZScoreDetector(BaseAnomalyDetector):
    """
    基于 Z-Score 的异常检测器
    
    简单高效的统计方法，适合单变量或多变量数据。
    
    Attributes:
        threshold: Z-score 阈值
        method: 多特征联合判定方法 ('any', 'all', 'majority', 'mahalanobis')
    """

    def __init__(
        self,
        threshold: float = 3.0,
        method: str = "any"
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.method = method
        self._mean: Optional[NDArray[np.floating]] = None
        self._std: Optional[NDArray[np.floating]] = None
        self._covariance: Optional[NDArray[np.floating]] = None
        self._inv_covariance: Optional[NDArray[np.floating]] = None
        self._n_features: int = 0

    def fit(self, X: NDArray[np.floating]) -> "ZScoreDetector":
        """
        计算训练数据的统计量
        
        Args:
            X: 训练数据
            
        Returns:
            训练后的检测器
        """
        self._validate_input(X)

        self._n_features = X.shape[1]
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

        self._std = np.where(self._std == 0, 1e-10, self._std)

        if self.method == "mahalanobis":
            self._covariance = np.cov(X.T)
            if self._covariance.ndim == 0:
                self._covariance = np.array([[self._covariance]])
            try:
                self._inv_covariance = np.linalg.inv(self._covariance)
            except np.linalg.LinAlgError:
                logger.warning("协方差矩阵不可逆，使用伪逆")
                self._inv_covariance = np.linalg.pinv(self._covariance)

        self._is_fitted = True
        logger.info(f"Z-Score 检测器初始化完成，阈值: {self.threshold}")
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        预测异常
        
        Args:
            X: 待预测数据
            
        Returns:
            布尔数组，True 表示异常
        """
        self._check_fitted()
        self._validate_input(X)

        z_scores = self._compute_z_scores(X)

        if self.method == "any":
            return np.any(np.abs(z_scores) > self.threshold, axis=1)
        elif self.method == "all":
            return np.all(np.abs(z_scores) > self.threshold, axis=1)
        elif self.method == "majority":
            majority_threshold = self._n_features // 2 + 1
            return np.sum(np.abs(z_scores) > self.threshold, axis=1) >= majority_threshold
        elif self.method == "mahalanobis":
            mahal_scores = self._compute_mahalanobis(X)
            return mahal_scores > (self.threshold ** 2)
        else:
            raise ValueError(f"未知的方法: {self.method}")

    def get_anomaly_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组
        """
        self._check_fitted()
        self._validate_input(X)

        z_scores = self._compute_z_scores(X)

        if self.method == "mahalanobis":
            return np.sqrt(self._compute_mahalanobis(X))
        else:
            return np.max(np.abs(z_scores), axis=1)

    def _compute_z_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        计算 Z-scores
        
        Args:
            X: 输入数据
            
        Returns:
            Z-scores 数组
        """
        return (X - self._mean) / self._std

    def _compute_mahalanobis(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        计算马氏距离
        
        Args:
            X: 输入数据
            
        Returns:
            马氏距离平方数组
        """
        diff = X - self._mean
        return np.sum(diff @ self._inv_covariance * diff, axis=1)

    def get_z_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取各特征的 Z-scores
        
        Args:
            X: 输入数据
            
        Returns:
            Z-scores 数组
        """
        self._check_fitted()
        return self._compute_z_scores(X)

    def get_params(self) -> dict[str, Any]:
        """获取模型参数"""
        return {
            "threshold": self.threshold,
            "method": self.method,
            "is_fitted": self._is_fitted,
            "n_features": self._n_features
        }


class AutoencoderDetector(BaseAnomalyDetector):
    """
    基于自编码器的异常检测器
    
    使用重构误差作为异常分数，支持普通自编码器和变分自编码器(VAE)。
    
    Attributes:
        latent_dim: 潜在空间维度
        hidden_dims: 隐藏层维度列表
        learning_rate: 学习率
        epochs: 训练轮数
        batch_size: 批次大小
        use_vae: 是否使用变分自编码器
        device: 计算设备 ('cpu', 'cuda', 'auto')
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: Optional[list[int]] = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_vae: bool = False,
        device: str = "auto",
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_vae = use_vae
        self.device = device
        self.dropout = dropout
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._n_features: int = 0
        self._threshold: float = 0.0
        self._device_resolved: Optional[str] = None

    def _resolve_device(self) -> str:
        """
        解析计算设备
        
        Returns:
            设备字符串
        """
        if self._device_resolved is not None:
            return self._device_resolved

        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self._device_resolved = "cuda"
                    logger.info("使用 GPU 加速")
                else:
                    self._device_resolved = "cpu"
                    logger.info("使用 CPU")
            except ImportError:
                self._device_resolved = "cpu"
        else:
            self._device_resolved = self.device

        return self._device_resolved

    def fit(self, X: NDArray[np.floating]) -> "AutoencoderDetector":
        """
        训练自编码器模型
        
        Args:
            X: 训练数据
            
        Returns:
            训练后的检测器
        """
        self._validate_input(X)

        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import MinMaxScaler
        except ImportError as e:
            raise ImportError("需要安装 PyTorch 才能使用 AutoencoderDetector") from e

        self._n_features = X.shape[1]
        device = self._resolve_device()

        self._scaler = MinMaxScaler()
        scaled_data = self._scaler.fit_transform(X)

        self._model = self._build_model()
        self._model = self._model.to(device)

        X_tensor = torch.FloatTensor(scaled_data).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss(reduction="none")

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                batch_x = batch[0]
                optimizer.zero_grad()

                if self.use_vae:
                    recon, mu, log_var = self._model(batch_x)
                    recon_loss = criterion(recon, batch_x).mean(dim=1)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                    loss = (recon_loss + 0.1 * kl_loss).mean()
                else:
                    recon = self._model(batch_x)
                    loss = criterion(recon, batch_x).mean()

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.debug(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        self._is_fitted = True

        self._model.eval()
        with torch.no_grad():
            recon_errors = self._compute_reconstruction_error(X_tensor)
            self._threshold = np.percentile(recon_errors, 95)

        logger.info(f"Autoencoder 训练完成，重构误差阈值: {self._threshold:.6f}")
        return self

    def _build_model(self) -> Any:
        """
        构建自编码器模型
        
        Returns:
            PyTorch 模型
        """
        import torch
        import torch.nn as nn

        class Encoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, latent_dim))
                self.encoder = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.encoder(x)

        class Decoder(nn.Module):
            def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int, dropout: float):
                super().__init__()
                layers = []
                prev_dim = latent_dim
                for hidden_dim in reversed(hidden_dims):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, output_dim))
                layers.append(nn.Sigmoid())
                self.decoder = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.decoder(x)

        class Autoencoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
                super().__init__()
                self.encoder = Encoder(input_dim, hidden_dims, latent_dim, dropout)
                self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.encoder(x)
                return self.decoder(z)

        class VAE(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
                super().__init__()
                self.encoder = Encoder(input_dim, hidden_dims, latent_dim * 2, dropout)
                self.decoder = Decoder(latent_dim, hidden_dims, input_dim, dropout)
                self.latent_dim = latent_dim

            def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                h = self.encoder(x)
                mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
                z = self.reparameterize(mu, log_var)
                return self.decoder(z), mu, log_var

        if self.use_vae:
            return VAE(self._n_features, self.hidden_dims, self.latent_dim, self.dropout)
        else:
            return Autoencoder(self._n_features, self.hidden_dims, self.latent_dim, self.dropout)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        预测异常
        
        Args:
            X: 待预测数据
            
        Returns:
            布尔数组，True 表示异常
        """
        scores = self.get_anomaly_scores(X)
        return scores > self._threshold

    def get_anomaly_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取异常分数（重构误差）
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组
        """
        self._check_fitted()
        self._validate_input(X)

        import torch

        device = self._resolve_device()
        scaled_data = self._scaler.transform(X)
        X_tensor = torch.FloatTensor(scaled_data).to(device)

        return self._compute_reconstruction_error(X_tensor)

    def _compute_reconstruction_error(self, X_tensor: Any) -> NDArray[np.floating]:
        """
        计算重构误差
        
        Args:
            X_tensor: 输入张量
            
        Returns:
            重构误差数组
        """
        import torch

        self._model.eval()
        with torch.no_grad():
            if self.use_vae:
                recon, _, _ = self._model(X_tensor)
            else:
                recon = self._model(X_tensor)

            errors = torch.mean((X_tensor - recon) ** 2, dim=1)
            return errors.cpu().numpy()

    def get_latent_representation(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取潜在空间表示
        
        Args:
            X: 输入数据
            
        Returns:
            潜在空间表示
        """
        self._check_fitted()
        self._validate_input(X)

        import torch

        device = self._resolve_device()
        scaled_data = self._scaler.transform(X)
        X_tensor = torch.FloatTensor(scaled_data).to(device)

        self._model.eval()
        with torch.no_grad():
            if self.use_vae:
                h = self._model.encoder(X_tensor)
                mu = h[:, :self.latent_dim]
                return mu.cpu().numpy()
            else:
                z = self._model.encoder(X_tensor)
                return z.cpu().numpy()

    def get_params(self) -> dict[str, Any]:
        """获取模型参数"""
        return {
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "use_vae": self.use_vae,
            "device": self._resolve_device(),
            "dropout": self.dropout,
            "is_fitted": self._is_fitted,
            "n_features": self._n_features,
            "threshold": self._threshold
        }


class EnsembleDetector(BaseAnomalyDetector):
    """
    集成异常检测器
    
    组合多种检测器，通过投票或加权平均提高检测稳定性。
    
    Attributes:
        detectors: 检测器列表
        weights: 各检测器权重
        method: 集成方法 ('voting', 'weighted_average', 'max', 'stacking')
        voting_threshold: 投票阈值（用于 voting 方法）
    """

    def __init__(
        self,
        detectors: Optional[list[BaseAnomalyDetector]] = None,
        weights: Optional[list[float]] = None,
        method: str = "weighted_average",
        voting_threshold: float = 0.5
    ) -> None:
        super().__init__()
        self.detectors: list[BaseAnomalyDetector] = detectors or []
        self.weights: list[float] = weights or []
        self.method = method
        self.voting_threshold = voting_threshold
        self._meta_model: Optional[Any] = None
        self._is_stacking_fitted: bool = False

        if self.detectors and not self.weights:
            self.weights = [1.0 / len(self.detectors)] * len(self.detectors)

    def add_detector(
        self,
        detector: BaseAnomalyDetector,
        weight: Optional[float] = None
    ) -> "EnsembleDetector":
        """
        添加检测器
        
        Args:
            detector: 异常检测器
            weight: 权重（可选）
            
        Returns:
            self
        """
        self.detectors.append(detector)
        if weight is not None:
            self.weights.append(weight)
        else:
            n = len(self.detectors)
            self.weights = [1.0 / n] * n
        return self

    def fit(self, X: NDArray[np.floating]) -> "EnsembleDetector":
        """
        训练所有检测器
        
        Args:
            X: 训练数据
            
        Returns:
            训练后的集成检测器
        """
        self._validate_input(X)

        if not self.detectors:
            raise ValueError("没有添加任何检测器")

        for i, detector in enumerate(self.detectors):
            try:
                detector.fit(X)
                logger.info(f"检测器 {i} ({type(detector).__name__}) 训练完成")
            except Exception as e:
                logger.error(f"检测器 {i} 训练失败: {e}")
                raise

        if self.method == "stacking":
            self._fit_stacking(X)

        self._is_fitted = True
        logger.info(f"集成检测器训练完成，包含 {len(self.detectors)} 个检测器")
        return self

    def _fit_stacking(self, X: NDArray[np.floating]) -> None:
        """
        训练 Stacking 元模型
        
        Args:
            X: 训练数据
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict

        meta_features = self._get_meta_features(X)

        y_pseudo = np.zeros(len(X))
        for i, detector in enumerate(self.detectors):
            predictions = detector.predict(X)
            y_pseudo |= predictions.astype(int)

        self._meta_model = LogisticRegression(random_state=42)
        self._meta_model.fit(meta_features, y_pseudo)
        self._is_stacking_fitted = True

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        集成预测异常
        
        Args:
            X: 待预测数据
            
        Returns:
            布尔数组，True 表示异常
        """
        scores = self.get_anomaly_scores(X)

        if self.method == "voting":
            votes = self._get_votes(X)
            vote_ratio = np.mean(votes, axis=1)
            return vote_ratio > self.voting_threshold
        elif self.method == "stacking":
            return self._stacking_predict(X)
        else:
            threshold = self._get_ensemble_threshold()
            return scores > threshold

    def get_anomaly_scores(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取集成异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组
        """
        self._check_fitted()
        self._validate_input(X)

        all_scores = []
        for detector in self.detectors:
            scores = detector.get_anomaly_scores(X)
            all_scores.append(scores)

        all_scores = np.array(all_scores)

        if self.method == "max":
            return np.max(all_scores, axis=0)
        elif self.method == "weighted_average":
            weights = np.array(self.weights).reshape(-1, 1)
            return np.sum(all_scores * weights, axis=0)
        elif self.method == "voting":
            votes = self._get_votes(X)
            return np.mean(votes, axis=1)
        elif self.method == "stacking":
            meta_features = self._get_meta_features(X)
            proba = self._meta_model.predict_proba(meta_features)
            return proba[:, 1]
        else:
            raise ValueError(f"未知的集成方法: {self.method}")

    def _get_votes(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        获取各检测器的投票结果
        
        Args:
            X: 输入数据
            
        Returns:
            投票结果数组
        """
        votes = []
        for detector in self.detectors:
            predictions = detector.predict(X)
            votes.append(predictions)
        return np.array(votes).T

    def _get_meta_features(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        获取元特征（用于 Stacking）
        
        Args:
            X: 输入数据
            
        Returns:
            元特征数组
        """
        meta_features = []
        for detector in self.detectors:
            scores = detector.get_anomaly_scores(X)
            predictions = detector.predict(X).astype(float)
            meta_features.append(scores)
            meta_features.append(predictions)
        return np.array(meta_features).T

    def _stacking_predict(self, X: NDArray[np.floating]) -> NDArray[np.bool_]:
        """
        Stacking 预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        meta_features = self._get_meta_features(X)
        return self._meta_model.predict(meta_features).astype(bool)

    def _get_ensemble_threshold(self) -> float:
        """
        获取集成阈值
        
        Returns:
            阈值
        """
        return 0.5

    def get_detector_scores(
        self,
        X: NDArray[np.floating]
    ) -> dict[str, NDArray[np.floating]]:
        """
        获取各检测器的异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            检测器名称和分数的字典
        """
        self._check_fitted()

        scores_dict = {}
        for i, detector in enumerate(self.detectors):
            name = f"{type(detector).__name__}_{i}"
            scores_dict[name] = detector.get_anomaly_scores(X)

        return scores_dict

    def get_params(self) -> dict[str, Any]:
        """获取模型参数"""
        return {
            "n_detectors": len(self.detectors),
            "weights": self.weights,
            "method": self.method,
            "voting_threshold": self.voting_threshold,
            "is_fitted": self._is_fitted,
            "detector_types": [type(d).__name__ for d in self.detectors]
        }


class AnomalyDetector:
    """
    异常检测器（兼容旧版本接口）
    
    提供统一的异常检测接口，包含多种检测方法。
    """

    def __init__(self, contamination: float = 0.1) -> None:
        self.contamination = contamination
        self._model: Optional[IsolationForestDetector] = None
        self._is_fitted: bool = False

    def fit(self, data: NDArray[np.floating]) -> None:
        """训练异常检测模型"""
        if len(data) < 10:
            logger.warning("训练数据不足")
            return

        self._model = IsolationForestDetector(contamination=self.contamination)
        self._model.fit(data)
        self._is_fitted = True
        logger.info("异常检测器训练完成")

    def detect(self, data: NDArray[np.floating]) -> list[AnomalyResult]:
        """检测异常"""
        if not self._is_fitted or self._model is None:
            return [AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomaly_type="unknown",
                details={"error": "模型未训练"}
            ) for _ in range(len(data))]

        predictions = self._model.predict(data)
        scores = self._model.get_anomaly_scores(data)

        results = []
        for i, (is_anomaly, score) in enumerate(zip(predictions, scores)):
            anomaly_type = self._classify_anomaly(data[i]) if is_anomaly else "normal"

            results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(score),
                anomaly_type=anomaly_type,
                details={"index": i}
            ))

        return results

    def _classify_anomaly(self, features: NDArray[np.floating]) -> str:
        """分类异常类型"""
        if len(features) < 5:
            return "unknown"

        if features[0] > 3:
            return "viral_content"
        elif features[1] > 3:
            return "rapid_spread"
        elif features[2] > 3:
            return "unusual_engagement"
        else:
            return "behavioral_anomaly"

    def detect_volume_anomaly(
        self,
        time_series: list[TrendPoint],
        window_size: int = 7,
        threshold: float = 3.0
    ) -> list[TrendPoint]:
        """检测流量异常"""
        if len(time_series) < window_size:
            return time_series

        values = np.array([p.value for p in time_series])
        results = []

        for i in range(len(values)):
            start_idx = max(0, i - window_size)
            window = values[start_idx:i]

            if len(window) < 2:
                results.append(TrendPoint(
                    timestamp=time_series[i].timestamp,
                    value=time_series[i].value,
                    is_anomaly=False
                ))
                continue

            mean = np.mean(window)
            std = np.std(window)

            if std == 0:
                z_score = 0
            else:
                z_score = abs(values[i] - mean) / std

            results.append(TrendPoint(
                timestamp=time_series[i].timestamp,
                value=time_series[i].value,
                is_anomaly=z_score > threshold
            ))

        return results

    def detect_sentiment_shift(
        self,
        sentiment_series: list[TrendPoint],
        window_size: int = 7,
        threshold: float = 0.3
    ) -> list[dict[str, Any]]:
        """检测情感突变"""
        if len(sentiment_series) < window_size * 2:
            return []

        shifts = []
        values = [p.value for p in sentiment_series]

        for i in range(window_size, len(values) - window_size):
            before = np.mean(values[i - window_size:i])
            after = np.mean(values[i:i + window_size])
            change = after - before

            if abs(change) > threshold:
                shifts.append({
                    "timestamp": sentiment_series[i].timestamp,
                    "before_avg": before,
                    "after_avg": after,
                    "change": change,
                    "direction": "positive" if change > 0 else "negative"
                })

        return shifts

    def detect_coordinated_behavior(
        self,
        posts: list[dict[str, Any]],
        time_window: int = 3600,
        similarity_threshold: float = 0.8
    ) -> list[dict[str, Any]]:
        """检测协同行为"""
        if len(posts) < 2:
            return []

        posts_sorted = sorted(posts, key=lambda x: x.get("created_at", datetime.min))

        coordinated_groups = []
        current_group: list[dict[str, Any]] = []

        for i, post in enumerate(posts_sorted):
            if not current_group:
                current_group.append(post)
                continue

            time_diff = (post.get("created_at", datetime.min) -
                        current_group[-1].get("created_at", datetime.min)).total_seconds()

            if time_diff <= time_window:
                current_group.append(post)
            else:
                if len(current_group) >= 3:
                    coordinated_groups.append({
                        "posts": current_group,
                        "size": len(current_group),
                        "time_span": (
                            current_group[-1].get("created_at", datetime.min) -
                            current_group[0].get("created_at", datetime.min)
                        ).total_seconds()
                    })
                current_group = [post]

        if len(current_group) >= 3:
            coordinated_groups.append({
                "posts": current_group,
                "size": len(current_group),
                "time_span": (
                    current_group[-1].get("created_at", datetime.min) -
                    current_group[0].get("created_at", datetime.min)
                ).total_seconds()
            })

        return coordinated_groups

    def detect_bot_behavior(
        self,
        user_data: dict[str, Any],
        posts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """检测机器人行为"""
        bot_score = 0.0
        indicators = []

        if len(posts) > 0:
            intervals = []
            posts_sorted = sorted(posts, key=lambda x: x.get("created_at", datetime.min))
            for i in range(1, len(posts_sorted)):
                interval = (posts_sorted[i].get("created_at", datetime.min) -
                           posts_sorted[i-1].get("created_at", datetime.min)).total_seconds()
                intervals.append(interval)

            if intervals:
                interval_std = np.std(intervals)
                if interval_std < 60:
                    bot_score += 0.3
                    indicators.append("regular_posting_interval")

        contents = [p.get("content", "") for p in posts]
        if contents:
            unique_ratio = len(set(contents)) / len(contents)
            if unique_ratio < 0.3:
                bot_score += 0.3
                indicators.append("low_content_diversity")

        followers = user_data.get("followers_count", 0)
        following = user_data.get("following_count", 0)
        if followers > 0 and following > 0:
            ratio = following / followers
            if ratio > 10:
                bot_score += 0.2
                indicators.append("high_following_ratio")

        return {
            "bot_probability": min(bot_score, 1.0),
            "is_likely_bot": bot_score > 0.5,
            "indicators": indicators
        }

    def get_statistics(self) -> dict[str, Any]:
        """获取检测器统计信息"""
        return {
            "is_fitted": self._is_fitted,
            "contamination": self.contamination
        }
