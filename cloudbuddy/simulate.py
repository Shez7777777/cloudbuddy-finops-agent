"""Synthetic hourly EC2 CPU utilisation.

Real per-instance AWS telemetry isn't publicly available, so this generator
reproduces patterns that show up in real fleets: daily and weekly cycles,
night-time batch jobs, slow growth, noise, and machines that sit idle.
Everything downstream (forecasting, cost, carbon) works the same way on real
CloudWatch exports with the same columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# role -> shape of its CPU curve (percent)
PROFILES = {
    "web-frontend": dict(base=32, daily=22, weekly=8, noise=4.0, trend=0.004, peak_hour=14),
    "api-backend": dict(base=48, daily=18, weekly=10, noise=5.0, trend=0.006, peak_hour=13),
    "batch-worker": dict(base=18, daily=40, weekly=2, noise=6.0, trend=0.0, peak_hour=2),
    "analytics-db": dict(base=22, daily=10, weekly=12, noise=3.0, trend=0.002, peak_hour=10),
    "dev-sandbox": dict(base=2.5, daily=1.5, weekly=1.0, noise=0.8, trend=0.0, peak_hour=11),
}

# instance_id, role, instance_type
FLEET = [
    ("i-0web01", "web-frontend", "m5.xlarge"),
    ("i-0web02", "web-frontend", "m5.xlarge"),
    ("i-0api01", "api-backend", "c5.xlarge"),
    ("i-0bat01", "batch-worker", "c5.2xlarge"),
    ("i-0db01", "analytics-db", "m5.2xlarge"),
    ("i-0dev01", "dev-sandbox", "t3.large"),
    ("i-0dev02", "dev-sandbox", "m5.large"),
]


def simulate_usage(days: int = 60, seed: int = 7, end: str = "2026-10-31 23:00") -> pd.DataFrame:
    """Return hourly rows: timestamp, instance_id, role, instance_type, cpu_pct."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(end=pd.Timestamp(end), periods=days * 24, freq="h")
    t = np.arange(len(hours))
    frames = []
    for instance_id, role, itype in FLEET:
        p = PROFILES[role]
        hour = hours.hour.to_numpy()
        dow = hours.day_of_week.to_numpy()
        daily = p["daily"] * np.cos(2 * np.pi * (hour - p["peak_hour"]) / 24)
        weekly = -p["weekly"] * (dow >= 5)  # quieter weekends
        trend = p["trend"] * t
        noise = rng.normal(0, p["noise"], len(t))
        spikes = (rng.random(len(t)) < 0.004) * rng.uniform(15, 35, len(t))
        cpu = np.clip(p["base"] + daily + weekly + trend + noise + spikes, 0.2, 100)
        frames.append(pd.DataFrame({
            "timestamp": hours, "instance_id": instance_id, "role": role,
            "instance_type": itype, "cpu_pct": cpu.round(2),
        }))
    return pd.concat(frames, ignore_index=True)
