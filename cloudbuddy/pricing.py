"""Cost, carbon and rightsizing rules.

Prices are approximate us-east-1 Linux on-demand rates; check the AWS pricing
page before using them for real decisions. Energy and carbon follow the Cloud
Carbon Footprint method: per-vCPU power interpolated between idle and full
load, times a regional grid emission factor.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

HOURS_PER_MONTH = 730

# type -> (vCPU, USD per hour)
INSTANCE_TYPES = {
    "t3.medium": (2, 0.0416), "t3.large": (2, 0.0832),
    "m5.large": (2, 0.096), "m5.xlarge": (4, 0.192), "m5.2xlarge": (8, 0.384),
    "c5.large": (2, 0.085), "c5.xlarge": (4, 0.17), "c5.2xlarge": (8, 0.34),
}
DOWNSIZE = {"m5.2xlarge": "m5.xlarge", "m5.xlarge": "m5.large", "c5.2xlarge": "c5.xlarge",
            "c5.xlarge": "c5.large", "t3.large": "t3.medium"}

WATTS_PER_VCPU_IDLE = 0.74   # Cloud Carbon Footprint AWS averages
WATTS_PER_VCPU_MAX = 3.5
PUE = 1.135                  # AWS power usage effectiveness used by CCF
GRID_KG_PER_KWH = 0.379      # us-east-1 (Virginia) grid factor used by CCF

IDLE_P95 = 5.0          # below this, the machine is effectively idle
DOWNSIZE_P95 = 40.0     # below this, half the vCPUs should still be enough
TARGET_MAX_P95 = 80.0   # projected p95 after downsizing must stay under this


def monthly_cost(instance_type: str) -> float:
    return INSTANCE_TYPES[instance_type][1] * HOURS_PER_MONTH


def monthly_co2_kg(instance_type: str, avg_cpu_pct: float) -> float:
    vcpu = INSTANCE_TYPES[instance_type][0]
    watts = vcpu * (WATTS_PER_VCPU_IDLE + (WATTS_PER_VCPU_MAX - WATTS_PER_VCPU_IDLE) * avg_cpu_pct / 100)
    kwh = watts * HOURS_PER_MONTH / 1000 * PUE
    return kwh * GRID_KG_PER_KWH


@dataclass
class Recommendation:
    instance_id: str
    instance_type: str
    action: str             # "stop", "downsize" or "keep"
    target_type: str | None
    forecast_p95_cpu: float
    monthly_savings_usd: float
    monthly_co2_saved_kg: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def recommend(instance_id: str, instance_type: str, forecast_cpu: list[float]) -> Recommendation:
    """Rule-based decision from a usage forecast. The LLM explains it; it doesn't invent it."""
    s = sorted(forecast_cpu)
    p95 = s[int(0.95 * (len(s) - 1))]
    avg = sum(s) / len(s)
    cost, co2 = monthly_cost(instance_type), monthly_co2_kg(instance_type, avg)

    if p95 < IDLE_P95:
        return Recommendation(instance_id, instance_type, "stop", None, round(p95, 1),
                              round(cost, 2), round(co2, 2),
                              f"Forecast p95 CPU is {p95:.1f}%, so the instance is effectively idle.")
    target = DOWNSIZE.get(instance_type)
    if target and p95 < DOWNSIZE_P95:
        ratio = INSTANCE_TYPES[instance_type][0] / INSTANCE_TYPES[target][0]
        projected = p95 * ratio
        if projected < TARGET_MAX_P95:
            new_avg = min(avg * ratio, 100)
            return Recommendation(instance_id, instance_type, "downsize", target, round(p95, 1),
                                  round(cost - monthly_cost(target), 2),
                                  round(co2 - monthly_co2_kg(target, new_avg), 2),
                                  f"Forecast p95 CPU is {p95:.1f}%; on {target} it would be about {projected:.0f}%.")
    return Recommendation(instance_id, instance_type, "keep", None, round(p95, 1), 0.0, 0.0,
                          f"Forecast p95 CPU is {p95:.1f}%, so the current size is justified.")
