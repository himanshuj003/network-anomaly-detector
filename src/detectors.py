"""
Anomaly detection engines:
  1. Statistical (Z-score / robust MAD)
  2. Isolation Forest (unsupervised ML)
  3. Rule-based (known bad patterns)
  4. Hybrid ensemble
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from .features import prepare_features, NUMERIC_FEATURES

# Optional heavy deps (Isolation Forest)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    IsolationForest = None
    StandardScaler = None
    joblib = None


@dataclass
class DetectionResult:
    """Container for detection output."""
    labels: np.ndarray
    scores: np.ndarray
    method: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_anomalies(self) -> int:
        return int(self.labels.sum())

    def to_series(self, index=None) -> pd.Series:
        return pd.Series(self.labels, index=index, name=f"anomaly_{self.method}")


class StatisticalDetector:
    """Robust statistical anomaly detector using MAD and optional Z-score."""

    def __init__(self, threshold: float = 4.5, use_mad: bool = True, min_features: int = 2):
        self.threshold = threshold
        self.use_mad = use_mad
        self.min_features = min_features
        self.medians_ = None
        self.mads_ = None
        self.means_ = None
        self.stds_ = None
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "StatisticalDetector":
        X, names = prepare_features(df, feature_cols, log_transform=True)
        self.feature_names_ = names
        if self.use_mad:
            self.medians_ = X.median()
            self.mads_ = (X - self.medians_).abs().median().replace(0, 1e-9)
        else:
            self.means_ = X.mean()
            self.stds_ = X.std().replace(0, 1e-9)
        return self

    def predict(self, df: pd.DataFrame) -> DetectionResult:
        X, _ = prepare_features(df, self.feature_names_, log_transform=True)
        if self.use_mad:
            mad_z = 0.6745 * (X - self.medians_) / self.mads_
            abs_z = mad_z.abs()
        else:
            z = (X - self.means_) / self.stds_
            abs_z = z.abs()
        scores = abs_z.max(axis=1).values
        n_over = (abs_z > self.threshold).sum(axis=1).values
        labels = (n_over >= self.min_features).astype(int)
        return DetectionResult(
            labels=labels, scores=scores, method="statistical",
            details={"threshold": self.threshold, "use_mad": self.use_mad, "min_features": self.min_features},
        )

    def fit_predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> DetectionResult:
        return self.fit(df, feature_cols).predict(df)


class IsolationForestDetector:
    """Unsupervised ML anomaly detector. Requires scikit-learn."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for IsolationForestDetector. Install with: pip install scikit-learn")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_ = None
        self.scaler_ = None
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "IsolationForestDetector":
        X, names = prepare_features(df, feature_cols, log_transform=True)
        self.feature_names_ = names
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = IsolationForest(
            contamination=self.contamination, n_estimators=self.n_estimators,
            random_state=self.random_state, n_jobs=self.n_jobs,
        )
        self.model_.fit(X_scaled)
        return self

    def predict(self, df: pd.DataFrame) -> DetectionResult:
        X, _ = prepare_features(df, self.feature_names_, log_transform=True)
        X_scaled = self.scaler_.transform(X)
        raw = self.model_.predict(X_scaled)
        labels = (raw == -1).astype(int)
        scores = -self.model_.decision_function(X_scaled)
        return DetectionResult(
            labels=labels, scores=scores, method="isolation_forest",
            details={"contamination": self.contamination},
        )

    def fit_predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> DetectionResult:
        return self.fit(df, feature_cols).predict(df)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model_, "scaler": self.scaler_, "features": self.feature_names_}, path)

    def load(self, path: str) -> "IsolationForestDetector":
        data = joblib.load(path)
        self.model_ = data["model"]
        self.scaler_ = data["scaler"]
        self.feature_names_ = data["features"]
        return self


class RuleBasedDetector:
    """Explicit cybersecurity rules for common attack patterns."""

    def __init__(self):
        self.rules = [
            {"name": "port_scan", "desc": "High unique destination ports",
             "condition": lambda r: r.get("unique_dst_ports", 0) >= 30, "severity": 0.8},
            {"name": "syn_flood", "desc": "Excessive SYN packets",
             "condition": lambda r: r.get("syn_count", 0) >= 50, "severity": 0.9},
            {"name": "rst_storm", "desc": "High RST count",
             "condition": lambda r: r.get("rst_count", 0) >= 20, "severity": 0.7},
            {"name": "high_packet_rate", "desc": "Abnormally high packets/sec",
             "condition": lambda r: r.get("packets_per_sec", 0) >= 2000, "severity": 0.85},
            {"name": "large_transfer", "desc": "Very large byte transfer",
             "condition": lambda r: r.get("byte_count", 0) >= 5_000_000, "severity": 0.75},
            {"name": "suspicious_port", "desc": "Known suspicious destination port",
             "condition": lambda r: r.get("dst_port", 0) in {31337, 4444, 6667, 12345, 1337, 65535}, "severity": 0.6},
            {"name": "short_burst", "desc": "Very high rate in short duration",
             "condition": lambda r: (r.get("duration_sec", 999) < 1.0 and r.get("packet_count", 0) > 500), "severity": 0.8},
        ]

    def predict(self, df: pd.DataFrame) -> DetectionResult:
        labels = np.zeros(len(df), dtype=int)
        scores = np.zeros(len(df), dtype=float)
        triggered = [[] for _ in range(len(df))]
        records = df.to_dict(orient="records")
        for i, row in enumerate(records):
            max_sev = 0.0
            for rule in self.rules:
                try:
                    if rule["condition"](row):
                        triggered[i].append(rule["name"])
                        max_sev = max(max_sev, rule["severity"])
                except Exception:
                    continue
            if triggered[i]:
                labels[i] = 1
                scores[i] = max_sev
        return DetectionResult(
            labels=labels, scores=scores, method="rule_based",
            details={"triggered_rules": triggered, "n_rules": len(self.rules)},
        )

    def fit(self, df: pd.DataFrame = None, **kwargs) -> "RuleBasedDetector":
        return self

    def fit_predict(self, df: pd.DataFrame, **kwargs) -> DetectionResult:
        return self.predict(df)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


class HybridDetector:
    """Combines Statistical + Isolation Forest + Rule-based with majority vote."""

    def __init__(self, contamination: float = 0.05, z_threshold: float = 3.5, min_votes: int = 2):
        self.stat = StatisticalDetector(threshold=z_threshold)
        self.iforest = None
        if HAS_SKLEARN:
            try:
                self.iforest = IsolationForestDetector(contamination=contamination)
            except ImportError:
                self.iforest = None
        self.rules = RuleBasedDetector()
        self.min_votes = min_votes
        self._fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "HybridDetector":
        self.stat.fit(df, feature_cols)
        if self.iforest is not None:
            self.iforest.fit(df, feature_cols)
        self.rules.fit(df)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        r_stat = self.stat.predict(df)
        r_rule = self.rules.predict(df)
        if self.iforest is not None:
            r_if = self.iforest.predict(df)
            votes = r_stat.labels + r_if.labels + r_rule.labels
            scores = 0.35 * _normalize(r_stat.scores) + 0.40 * _normalize(r_if.scores) + 0.25 * r_rule.scores
            if_anom = int(r_if.labels.sum())
        else:
            votes = r_stat.labels + r_rule.labels
            scores = 0.6 * _normalize(r_stat.scores) + 0.4 * r_rule.scores
            if_anom = 0
        effective_min = self.min_votes if self.iforest is not None else min(self.min_votes, 2)
        labels = (votes >= effective_min).astype(int)
        return DetectionResult(
            labels=labels, scores=scores, method="hybrid",
            details={
                "votes": votes, "min_votes": effective_min,
                "stat_anomalies": int(r_stat.labels.sum()),
                "iforest_anomalies": if_anom,
                "rule_anomalies": int(r_rule.labels.sum()),
                "sklearn_available": self.iforest is not None,
            },
        )

    def fit_predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> DetectionResult:
        return self.fit(df, feature_cols).predict(df)


def get_detector(name: str, **kwargs):
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("stat", "statistical", "zscore", "mad"):
        return StatisticalDetector(**kwargs)
    if name in ("iforest", "isolation_forest", "isolation", "ml"):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for isolation_forest. Install with: pip install scikit-learn")
        return IsolationForestDetector(**kwargs)
    if name in ("rule", "rules", "rule_based"):
        return RuleBasedDetector()
    if name in ("hybrid", "ensemble", "all"):
        return HybridDetector(**kwargs)
    raise ValueError(f"Unknown detector: {name}. Choose: statistical, isolation_forest, rule_based, hybrid")
