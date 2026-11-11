"""Hourly CPU forecasting with LightGBM, evaluated against a seasonal-naive baseline."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 24, 48, 168]
FEATURES = [f"lag_{l}" for l in LAGS] + ["roll_24", "roll_168", "hour", "dow", "is_weekend", "instance_code"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling and calendar features, computed per instance without leakage."""
    df = df.sort_values(["instance_id", "timestamp"]).copy()
    g = df.groupby("instance_id")["cpu_pct"]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    df["roll_24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    df["roll_168"] = g.transform(lambda s: s.shift(1).rolling(168).mean())
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.day_of_week
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["instance_code"] = df["instance_id"].astype("category").cat.codes
    return df


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error. Unlike MAPE it stays meaningful for near-idle machines."""
    return float(np.abs(y - yhat).sum() / np.abs(y).sum() * 100)


@dataclass
class Evaluation:
    mae: float
    wape: float
    baseline_mae: float
    baseline_wape: float
    test_hours: int


class CpuForecaster:
    def __init__(self, **params):
        self.params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                           verbose=-1, random_state=0) | params
        self.model: lgb.LGBMRegressor | None = None
        self.codes: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "CpuForecaster":
        feats = add_features(df).dropna(subset=FEATURES)
        self.codes = dict(zip(feats["instance_id"], feats["instance_code"]))
        self.model = lgb.LGBMRegressor(**self.params).fit(feats[FEATURES], feats["cpu_pct"])
        return self

    def evaluate(self, df: pd.DataFrame, test_days: int = 7) -> Evaluation:
        """Train on everything before the last `test_days`, score one-step-ahead on the rest."""
        cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
        feats = add_features(df).dropna(subset=FEATURES)
        train, test = feats[feats["timestamp"] <= cutoff], feats[feats["timestamp"] > cutoff]
        model = lgb.LGBMRegressor(**self.params).fit(train[FEATURES], train["cpu_pct"])
        y, yhat = test["cpu_pct"].to_numpy(), model.predict(test[FEATURES])
        naive = test["lag_168"].to_numpy()  # same hour last week
        return Evaluation(
            mae=float(np.abs(y - yhat).mean()), wape=wape(y, yhat),
            baseline_mae=float(np.abs(y - naive).mean()), baseline_wape=wape(y, naive),
            test_hours=len(test),
        )

    def forecast(self, df: pd.DataFrame, instance_id: str, hours: int = 168) -> pd.DataFrame:
        """Recursive multi-step forecast for one instance."""
        if self.model is None:
            raise RuntimeError("Call fit() first")
        hist = df[df["instance_id"] == instance_id].sort_values("timestamp")[["timestamp", "cpu_pct"]]
        if hist.empty:
            raise KeyError(f"Unknown instance {instance_id}")
        values = hist["cpu_pct"].tolist()
        ts = hist["timestamp"].iloc[-1]
        code = self.codes.get(instance_id, -1)
        rows = []
        for _ in range(hours):
            ts = ts + pd.Timedelta(hours=1)
            x = {f"lag_{l}": values[-l] for l in LAGS}
            x |= {"roll_24": float(np.mean(values[-24:])), "roll_168": float(np.mean(values[-168:])),
                  "hour": ts.hour, "dow": ts.day_of_week, "is_weekend": int(ts.day_of_week >= 5),
                  "instance_code": code}
            pred = float(np.clip(self.model.predict(pd.DataFrame([x])[FEATURES])[0], 0, 100))
            values.append(pred)
            rows.append({"timestamp": ts, "cpu_pct_forecast": round(pred, 2)})
        return pd.DataFrame(rows)
