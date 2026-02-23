"""
异常检测模块测试

测试孤立森林、Z-Score检测器、LOF检测器等功能。
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from analysis.anomaly import (
    AnomalyType,
    AnomalyReport,
    TrendPoint,
    AnomalyResult,
    BaseAnomalyDetector,
    IsolationForestDetector,
    LOFDetector,
    ZScoreDetector,
    AutoencoderDetector,
    EnsembleDetector,
    AnomalyDetector,
)


class TestAnomalyType:
    """异常类型枚举测试"""

    def test_anomaly_type_values(self):
        """测试异常类型值"""
        assert AnomalyType.NORMAL.value == "normal"
        assert AnomalyType.VIRAL_CONTENT.value == "viral_content"
        assert AnomalyType.RAPID_SPREAD.value == "rapid_spread"
        assert AnomalyType.BEHAVIORAL_ANOMALY.value == "behavioral_anomaly"
        assert AnomalyType.COORDINATED_BEHAVIOR.value == "coordinated_behavior"
        assert AnomalyType.BOT_LIKE.value == "bot_like"


class TestAnomalyReport:
    """异常报告测试"""

    def test_anomaly_report_creation(self):
        """测试创建异常报告"""
        report = AnomalyReport(
            is_anomaly=True,
            anomaly_score=0.85,
            anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
            confidence=0.9,
            feature_contributions={"f1": 0.5, "f2": 0.3},
        )
        
        assert report.is_anomaly is True
        assert report.anomaly_score == 0.85
        assert report.anomaly_type == AnomalyType.BEHAVIORAL_ANOMALY

    def test_anomaly_report_to_dict(self):
        """测试转换为字典"""
        report = AnomalyReport(
            is_anomaly=True,
            anomaly_score=0.85,
            anomaly_type=AnomalyType.BOT_LIKE,
            confidence=0.9,
        )
        data = report.to_dict()
        
        assert data["is_anomaly"] is True
        assert data["anomaly_score"] == 0.85
        assert data["anomaly_type"] == "bot_like"

    def test_get_top_contributing_features(self):
        """测试获取主要贡献特征"""
        report = AnomalyReport(
            is_anomaly=True,
            anomaly_score=0.85,
            feature_contributions={
                "f1": 0.1,
                "f2": 0.5,
                "f3": 0.3,
                "f4": 0.05,
            },
        )
        top_features = report.get_top_contributing_features(top_n=2)
        
        assert len(top_features) == 2
        assert top_features[0][0] == "f2"
        assert top_features[1][0] == "f3"

    def test_get_explanation_anomaly(self):
        """测试获取异常解释"""
        report = AnomalyReport(
            is_anomaly=True,
            anomaly_score=0.85,
            anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
            confidence=0.9,
            feature_contributions={"f1": 0.5},
        )
        explanation = report.get_explanation()
        
        assert "behavioral_anomaly" in explanation
        assert "0.85" in explanation

    def test_get_explanation_normal(self):
        """测试获取正常解释"""
        report = AnomalyReport(
            is_anomaly=False,
            anomaly_score=0.1,
        )
        explanation = report.get_explanation()
        
        assert "正常" in explanation


class TestTrendPoint:
    """趋势数据点测试"""

    def test_trend_point_creation(self):
        """测试创建趋势数据点"""
        point = TrendPoint(
            timestamp=datetime.utcnow(),
            value=100.0,
            is_anomaly=False,
        )
        
        assert point.value == 100.0
        assert point.is_anomaly is False


class TestAnomalyResult:
    """异常检测结果测试"""

    def test_anomaly_result_creation(self):
        """测试创建异常检测结果"""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_score=0.85,
            anomaly_type="behavioral_anomaly",
            details={"index": 0},
        )
        
        assert result.is_anomaly is True
        assert result.anomaly_type == "behavioral_anomaly"


class TestIsolationForestDetector:
    """孤立森林检测器测试"""

    @pytest.fixture
    def detector(self):
        """创建检测器"""
        return IsolationForestDetector(
            contamination=0.1,
            n_estimators=50,
            random_state=42,
        )

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)
        normal_data = np.random.randn(100, 5)
        return normal_data

    @pytest.fixture
    def data_with_anomalies(self):
        """生成包含异常的数据"""
        np.random.seed(42)
        normal_data = np.random.randn(100, 5)
        anomaly_data = np.random.randn(10, 5) * 3 + 5
        return np.vstack([normal_data, anomaly_data])

    def test_initialization(self):
        """测试初始化"""
        detector = IsolationForestDetector(
            contamination=0.05,
            n_estimators=100,
        )
        
        assert detector.contamination == 0.05
        assert detector.n_estimators == 100
        assert detector._is_fitted is False

    def test_fit(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试训练"""
        result = detector.fit(sample_data)
        
        assert result is detector
        assert detector._is_fitted is True
        assert detector._n_features == 5

    def test_fit_invalid_data(self, detector: IsolationForestDetector):
        """测试无效数据训练"""
        with pytest.raises(ValueError):
            detector.fit(None)
        
        with pytest.raises(ValueError):
            detector.fit(np.array([]))

    def test_predict(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试预测"""
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert all(isinstance(p, (bool, np.bool_)) for p in predictions)

    def test_predict_before_fit(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试训练前预测"""
        with pytest.raises(RuntimeError):
            detector.predict(sample_data)

    def test_predict_wrong_features(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试特征数不匹配"""
        detector.fit(sample_data)
        
        wrong_data = np.random.randn(10, 3)
        with pytest.raises(ValueError):
            detector.predict(wrong_data)

    def test_get_anomaly_scores(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试获取异常分数"""
        detector.fit(sample_data)
        scores = detector.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)
        assert all(isinstance(s, (float, np.floating)) for s in scores)

    def test_fit_predict(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试训练并预测"""
        predictions = detector.fit_predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_detect(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试检测"""
        detector.fit(sample_data)
        reports = detector.detect(sample_data)
        
        assert len(reports) == len(sample_data)
        assert all(isinstance(r, AnomalyReport) for r in reports)

    def test_get_feature_importance(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试获取特征重要性"""
        detector.fit(sample_data)
        importance = detector.get_feature_importance()
        
        assert len(importance) == 5

    def test_detect_with_contributions(self, detector: IsolationForestDetector, sample_data: NDArray):
        """测试带贡献度的检测"""
        detector.fit(sample_data)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        reports = detector.detect_with_contributions(sample_data[:5], feature_names)
        
        assert len(reports) == 5
        for report in reports:
            assert len(report.feature_contributions) == 5

    def test_get_params(self, detector: IsolationForestDetector):
        """测试获取参数"""
        params = detector.get_params()
        
        assert "contamination" in params
        assert "n_estimators" in params
        assert "is_fitted" in params


class TestLOFDetector:
    """LOF检测器测试"""

    @pytest.fixture
    def detector(self):
        """创建检测器"""
        return LOFDetector(
            n_neighbors=10,
            contamination=0.1,
        )

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)
        return np.random.randn(50, 5)

    def test_initialization(self):
        """测试初始化"""
        detector = LOFDetector(n_neighbors=15, contamination=0.05)
        
        assert detector.n_neighbors == 15
        assert detector.contamination == 0.05

    def test_fit(self, detector: LOFDetector, sample_data: NDArray):
        """测试训练"""
        result = detector.fit(sample_data)
        
        assert result is detector
        assert detector._is_fitted is True

    def test_fit_insufficient_samples(self):
        """测试样本不足"""
        detector = LOFDetector(n_neighbors=20)
        small_data = np.random.randn(10, 5)
        
        with pytest.raises(ValueError):
            detector.fit(small_data)

    def test_predict(self, detector: LOFDetector, sample_data: NDArray):
        """测试预测"""
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_get_anomaly_scores(self, detector: LOFDetector, sample_data: NDArray):
        """测试获取异常分数"""
        detector.fit(sample_data)
        scores = detector.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)

    def test_get_local_reachability_density(self, detector: LOFDetector, sample_data: NDArray):
        """测试获取局部可达密度"""
        detector.fit(sample_data)
        density = detector.get_local_reachability_density(sample_data)
        
        assert len(density) == len(sample_data)

    def test_get_params(self, detector: LOFDetector):
        """测试获取参数"""
        params = detector.get_params()
        
        assert "n_neighbors" in params
        assert "contamination" in params


class TestZScoreDetector:
    """Z-Score检测器测试"""

    @pytest.fixture
    def detector(self):
        """创建检测器"""
        return ZScoreDetector(threshold=3.0, method="any")

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)
        return np.random.randn(100, 5)

    @pytest.fixture
    def data_with_outliers(self):
        """生成包含离群值的数据"""
        np.random.seed(42)
        normal_data = np.random.randn(100, 5)
        outlier_data = np.array([[10, 10, 10, 10, 10]])
        return np.vstack([normal_data, outlier_data])

    def test_initialization(self):
        """测试初始化"""
        detector = ZScoreDetector(threshold=2.5, method="all")
        
        assert detector.threshold == 2.5
        assert detector.method == "all"

    def test_fit(self, detector: ZScoreDetector, sample_data: NDArray):
        """测试训练"""
        result = detector.fit(sample_data)
        
        assert result is detector
        assert detector._is_fitted is True
        assert detector._mean is not None
        assert detector._std is not None

    def test_predict_any_method(self, detector: ZScoreDetector, data_with_outliers: NDArray):
        """测试any方法预测"""
        detector.fit(data_with_outliers)
        predictions = detector.predict(data_with_outliers)
        
        assert predictions[-1] is True

    def test_predict_all_method(self, sample_data: NDArray):
        """测试all方法预测"""
        detector = ZScoreDetector(threshold=3.0, method="all")
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_predict_majority_method(self, sample_data: NDArray):
        """测试majority方法预测"""
        detector = ZScoreDetector(threshold=3.0, method="majority")
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_predict_mahalanobis_method(self, sample_data: NDArray):
        """测试mahalanobis方法预测"""
        detector = ZScoreDetector(threshold=3.0, method="mahalanobis")
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_get_anomaly_scores(self, detector: ZScoreDetector, sample_data: NDArray):
        """测试获取异常分数"""
        detector.fit(sample_data)
        scores = detector.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)

    def test_get_z_scores(self, detector: ZScoreDetector, sample_data: NDArray):
        """测试获取Z分数"""
        detector.fit(sample_data)
        z_scores = detector.get_z_scores(sample_data)
        
        assert z_scores.shape == sample_data.shape

    def test_zero_std_handling(self):
        """测试零标准差处理"""
        detector = ZScoreDetector()
        constant_data = np.ones((100, 5))
        detector.fit(constant_data)
        
        assert detector._is_fitted is True
        assert all(detector._std > 0)

    def test_get_params(self, detector: ZScoreDetector):
        """测试获取参数"""
        params = detector.get_params()
        
        assert "threshold" in params
        assert "method" in params


class TestAutoencoderDetector:
    """自编码器检测器测试"""

    @pytest.fixture
    def detector(self):
        """创建检测器"""
        return AutoencoderDetector(
            latent_dim=4,
            hidden_dims=[16, 8],
            epochs=10,
            batch_size=16,
            device="cpu",
        )

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)
        return np.random.randn(100, 10).astype(np.float32)

    def test_initialization(self):
        """测试初始化"""
        detector = AutoencoderDetector(
            latent_dim=8,
            hidden_dims=[32, 16],
            epochs=50,
        )
        
        assert detector.latent_dim == 8
        assert detector.hidden_dims == [32, 16]
        assert detector.epochs == 50

    def test_fit(self, detector: AutoencoderDetector, sample_data: NDArray):
        """测试训练"""
        result = detector.fit(sample_data)
        
        assert result is detector
        assert detector._is_fitted is True

    def test_predict(self, detector: AutoencoderDetector, sample_data: NDArray):
        """测试预测"""
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_get_anomaly_scores(self, detector: AutoencoderDetector, sample_data: NDArray):
        """测试获取异常分数"""
        detector.fit(sample_data)
        scores = detector.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)

    def test_get_latent_representation(self, detector: AutoencoderDetector, sample_data: NDArray):
        """测试获取潜在表示"""
        detector.fit(sample_data)
        latent = detector.get_latent_representation(sample_data)
        
        assert latent.shape == (len(sample_data), detector.latent_dim)

    def test_get_params(self, detector: AutoencoderDetector):
        """测试获取参数"""
        params = detector.get_params()
        
        assert "latent_dim" in params
        assert "hidden_dims" in params
        assert "epochs" in params


class TestEnsembleDetector:
    """集成检测器测试"""

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)
        return np.random.randn(100, 5)

    @pytest.fixture
    def detectors(self, sample_data: NDArray):
        """创建多个检测器"""
        if_detector = IsolationForestDetector(contamination=0.1, n_estimators=20)
        zscore_detector = ZScoreDetector(threshold=2.5)
        
        return [if_detector, zscore_detector]

    def test_initialization(self):
        """测试初始化"""
        detector = EnsembleDetector(
            method="weighted_average",
            weights=[0.6, 0.4],
        )
        
        assert detector.method == "weighted_average"
        assert detector.weights == [0.6, 0.4]

    def test_add_detector(self):
        """测试添加检测器"""
        ensemble = EnsembleDetector()
        if_detector = IsolationForestDetector()
        
        result = ensemble.add_detector(if_detector, weight=0.5)
        
        assert result is ensemble
        assert len(ensemble.detectors) == 1

    def test_fit(self, detectors: list, sample_data: NDArray):
        """测试训练"""
        ensemble = EnsembleDetector(detectors=detectors)
        result = ensemble.fit(sample_data)
        
        assert result is ensemble
        assert ensemble._is_fitted is True

    def test_fit_no_detectors(self, sample_data: NDArray):
        """测试无检测器训练"""
        ensemble = EnsembleDetector()
        
        with pytest.raises(ValueError):
            ensemble.fit(sample_data)

    def test_predict_weighted_average(self, detectors: list, sample_data: NDArray):
        """测试加权平均预测"""
        ensemble = EnsembleDetector(detectors=detectors, method="weighted_average")
        ensemble.fit(sample_data)
        predictions = ensemble.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_predict_voting(self, detectors: list, sample_data: NDArray):
        """测试投票预测"""
        ensemble = EnsembleDetector(detectors=detectors, method="voting")
        ensemble.fit(sample_data)
        predictions = ensemble.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_predict_max(self, detectors: list, sample_data: NDArray):
        """测试最大值预测"""
        ensemble = EnsembleDetector(detectors=detectors, method="max")
        ensemble.fit(sample_data)
        predictions = ensemble.predict(sample_data)
        
        assert len(predictions) == len(sample_data)

    def test_get_anomaly_scores(self, detectors: list, sample_data: NDArray):
        """测试获取异常分数"""
        ensemble = EnsembleDetector(detectors=detectors)
        ensemble.fit(sample_data)
        scores = ensemble.get_anomaly_scores(sample_data)
        
        assert len(scores) == len(sample_data)

    def test_get_detector_scores(self, detectors: list, sample_data: NDArray):
        """测试获取各检测器分数"""
        ensemble = EnsembleDetector(detectors=detectors)
        ensemble.fit(sample_data)
        scores = ensemble.get_detector_scores(sample_data)
        
        assert len(scores) == 2

    def test_get_params(self, detectors: list):
        """测试获取参数"""
        ensemble = EnsembleDetector(detectors=detectors)
        params = ensemble.get_params()
        
        assert "n_detectors" in params
        assert "method" in params


class TestAnomalyDetector:
    """异常检测器（兼容接口）测试"""

    @pytest.fixture
    def detector(self):
        """创建检测器"""
        return AnomalyDetector(contamination=0.1)

    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_initialization(self):
        """测试初始化"""
        detector = AnomalyDetector(contamination=0.05)
        
        assert detector.contamination == 0.05

    def test_fit(self, detector: AnomalyDetector, sample_data: NDArray):
        """测试训练"""
        detector.fit(sample_data)
        
        assert detector._is_fitted is True

    def test_detect(self, detector: AnomalyDetector, sample_data: NDArray):
        """测试检测"""
        detector.fit(sample_data)
        results = detector.detect(sample_data)
        
        assert len(results) == len(sample_data)
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_detect_before_fit(self, detector: AnomalyDetector, sample_data: NDArray):
        """测试训练前检测"""
        results = detector.detect(sample_data)
        
        assert all(not r.is_anomaly for r in results)

    def test_detect_volume_anomaly(self, detector: AnomalyDetector):
        """测试流量异常检测"""
        time_series = []
        base_time = datetime.utcnow()
        
        for i in range(30):
            value = 100 + np.random.randn() * 10
            if i == 20:
                value = 500
            time_series.append(TrendPoint(
                timestamp=base_time + timedelta(hours=i),
                value=value,
            ))
        
        results = detector.detect_volume_anomaly(time_series, window_size=7, threshold=3.0)
        
        assert len(results) == len(time_series)
        assert results[20].is_anomaly is True

    def test_detect_sentiment_shift(self, detector: AnomalyDetector):
        """测试情感突变检测"""
        sentiment_series = []
        base_time = datetime.utcnow()
        
        for i in range(30):
            if i < 15:
                value = 0.1
            else:
                value = 0.8
            sentiment_series.append(TrendPoint(
                timestamp=base_time + timedelta(hours=i),
                value=value,
            ))
        
        shifts = detector.detect_sentiment_shift(sentiment_series, window_size=5, threshold=0.3)
        
        assert len(shifts) > 0

    def test_detect_coordinated_behavior(self, detector: AnomalyDetector):
        """测试协同行为检测"""
        posts = []
        base_time = datetime.utcnow()
        
        for i in range(10):
            posts.append({
                "created_at": base_time + timedelta(seconds=i * 30),
                "content": f"Coordinated post {i}",
            })
        
        results = detector.detect_coordinated_behavior(posts, time_window=3600)
        
        assert len(results) > 0

    def test_detect_bot_behavior(self, detector: AnomalyDetector):
        """测试机器人行为检测"""
        user_data = {
            "followers_count": 100,
            "following_count": 5000,
        }
        
        posts = []
        base_time = datetime.utcnow()
        for i in range(20):
            posts.append({
                "created_at": base_time + timedelta(minutes=i * 60),
                "content": "Same content repeated",
            })
        
        result = detector.detect_bot_behavior(user_data, posts)
        
        assert "bot_probability" in result
        assert "is_likely_bot" in result
        assert "indicators" in result

    def test_get_statistics(self, detector: AnomalyDetector):
        """测试获取统计信息"""
        stats = detector.get_statistics()
        
        assert "is_fitted" in stats
        assert "contamination" in stats


class TestDetectorComparison:
    """检测器比较测试"""

    @pytest.fixture
    def data_with_known_anomalies(self):
        """生成已知异常的数据"""
        np.random.seed(42)
        normal_data = np.random.randn(200, 5)
        anomaly_data = np.random.randn(20, 5) * 4 + 10
        
        X = np.vstack([normal_data, anomaly_data])
        y = np.array([False] * 200 + [True] * 20)
        
        return X, y

    def test_isolation_forest_performance(self, data_with_known_anomalies):
        """测试孤立森林性能"""
        X, y = data_with_known_anomalies
        
        detector = IsolationForestDetector(contamination=0.1, n_estimators=50)
        detector.fit(X)
        predictions = detector.predict(X)
        
        true_positives = np.sum(predictions[y] == True)
        false_positives = np.sum(predictions[~y] == True)
        
        assert true_positives > 0

    def test_zscore_performance(self, data_with_known_anomalies):
        """测试Z-Score性能"""
        X, y = data_with_known_anomalies
        
        detector = ZScoreDetector(threshold=3.0, method="any")
        detector.fit(X)
        predictions = detector.predict(X)
        
        true_positives = np.sum(predictions[y] == True)
        
        assert true_positives > 0

    def test_ensemble_performance(self, data_with_known_anomalies):
        """测试集成检测器性能"""
        X, y = data_with_known_anomalies
        
        if_detector = IsolationForestDetector(contamination=0.1, n_estimators=30)
        zscore_detector = ZScoreDetector(threshold=2.5)
        
        ensemble = EnsembleDetector(
            detectors=[if_detector, zscore_detector],
            method="weighted_average",
        )
        ensemble.fit(X)
        predictions = ensemble.predict(X)
        
        true_positives = np.sum(predictions[y] == True)
        
        assert true_positives > 0
