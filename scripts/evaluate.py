"""Backtest the forecaster and print fleet recommendations.

python scripts/evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudbuddy.actions import ApprovalQueue, MockExecutor
from cloudbuddy.forecast import CpuForecaster
from cloudbuddy.simulate import simulate_usage
from cloudbuddy.tools import FinOpsTools

usage = simulate_usage(days=60)
fc = CpuForecaster()

ev = fc.evaluate(usage, test_days=7)
print(f"Backtest on last 7 days ({ev.test_hours} instance-hours)")
print(f"  LightGBM          MAE {ev.mae:5.2f}   WAPE {ev.wape:5.2f}%")
print(f"  same hour last wk MAE {ev.baseline_mae:5.2f}   WAPE {ev.baseline_wape:5.2f}%")

fc.fit(usage)
tools = FinOpsTools(usage, fc, ApprovalQueue(MockExecutor()))
total_usd = total_co2 = 0
print("\nRecommendations (next 7 days)")
for r in tools.recommend_actions():
    total_usd += r["monthly_savings_usd"]
    total_co2 += r["monthly_co2_saved_kg"]
    target = f" -> {r['target_type']}" if r["target_type"] else ""
    print(f"  {r['instance_id']:9} {r['instance_type']:11} {r['action']:8}{target:14} "
          f"p95 {r['forecast_p95_cpu']:5.1f}%  saves ${r['monthly_savings_usd']:7.2f}/mo")
print(f"\nTotal: ${total_usd:.2f}/month, {total_co2:.1f} kg CO2/month")
