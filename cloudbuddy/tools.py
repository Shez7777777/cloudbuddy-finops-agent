"""Tools exposed to the LLM. They only read data or queue requests -
nothing here touches infrastructure directly."""
from __future__ import annotations

import json

import pandas as pd

from .actions import ApprovalQueue
from .forecast import CpuForecaster
from .pricing import monthly_cost, recommend


class FinOpsTools:
    def __init__(self, usage: pd.DataFrame, forecaster: CpuForecaster, queue: ApprovalQueue, horizon_hours=168):
        self.usage = usage
        self.forecaster = forecaster
        self.queue = queue
        self.horizon = horizon_hours
        self.fleet = usage.groupby("instance_id")[["role", "instance_type"]].first()
        self._forecasts: dict[str, list[float]] = {}

    def list_instances(self):
        return [{"instance_id": i, "role": r.role, "instance_type": r.instance_type,
                 "monthly_cost_usd": round(monthly_cost(r.instance_type), 2)}
                for i, r in self.fleet.iterrows()]

    def get_usage_summary(self, instance_id, days=7):
        self._check(instance_id)
        u = self.usage
        recent = u[(u.instance_id == instance_id) & (u.timestamp > u.timestamp.max() - pd.Timedelta(days=days))]
        cpu = recent.cpu_pct
        return {"instance_id": instance_id, "days": days, "avg_cpu_pct": round(cpu.mean(), 1),
                "p95_cpu_pct": round(cpu.quantile(0.95), 1), "max_cpu_pct": round(cpu.max(), 1)}

    def forecast_usage(self, instance_id):
        s = pd.Series(self.get_forecast(instance_id))
        return {"instance_id": instance_id, "horizon_hours": self.horizon,
                "avg_cpu_pct": round(s.mean(), 1), "p95_cpu_pct": round(s.quantile(0.95), 1),
                "max_cpu_pct": round(s.max(), 1)}

    def recommend_actions(self):
        recs = [recommend(i, r.instance_type, self.get_forecast(i)).to_dict() for i, r in self.fleet.iterrows()]
        return sorted(recs, key=lambda x: -x["monthly_savings_usd"])

    def request_action(self, instance_id, action, reason, target_type=None):
        self._check(instance_id)
        req = self.queue.submit(instance_id, action, reason, target_type)
        return {"request_id": req.id, "status": req.status,
                "note": "waiting for human approval, nothing has been changed yet"}

    def get_forecast(self, instance_id):
        self._check(instance_id)
        if instance_id not in self._forecasts:
            fc = self.forecaster.forecast(self.usage, instance_id, self.horizon)
            self._forecasts[instance_id] = fc.cpu_pct_forecast.tolist()
        return self._forecasts[instance_id]

    def _check(self, instance_id):
        if instance_id not in self.fleet.index:
            raise KeyError(f"unknown instance_id {instance_id} - call list_instances first")

    def call(self, name, arguments):
        """Run a tool and return JSON. Errors go back to the model as JSON too so it can retry."""
        fns = {"list_instances": self.list_instances, "get_usage_summary": self.get_usage_summary,
               "forecast_usage": self.forecast_usage, "recommend_actions": self.recommend_actions,
               "request_action": self.request_action}
        if name not in fns:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            return json.dumps(fns[name](**args), default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})


INSTANCE_ID = {"type": "string", "description": "EC2 instance id, e.g. i-0web01"}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_instances",
        "description": "List all EC2 instances with role, type and monthly on-demand cost.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_usage_summary",
        "description": "Recent CPU stats (avg, p95, max) for one instance.",
        "parameters": {"type": "object", "properties": {
            "instance_id": INSTANCE_ID,
            "days": {"type": "integer", "description": "look-back window in days"}},
            "required": ["instance_id"]}}},
    {"type": "function", "function": {
        "name": "forecast_usage",
        "description": "LightGBM CPU forecast for the next 7 days for one instance.",
        "parameters": {"type": "object", "properties": {"instance_id": INSTANCE_ID}, "required": ["instance_id"]}}},
    {"type": "function", "function": {
        "name": "recommend_actions",
        "description": "Stop/downsize/keep recommendations for the whole fleet with monthly USD and CO2 savings.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "request_action",
        "description": "Queue a stop or downsize for human approval. Does not execute anything.",
        "parameters": {"type": "object", "properties": {
            "instance_id": INSTANCE_ID,
            "action": {"type": "string", "enum": ["stop", "downsize"]},
            "target_type": {"type": "string", "description": "needed for downsize, e.g. m5.large"},
            "reason": {"type": "string", "description": "short justification shown to the approver"}},
            "required": ["instance_id", "action", "reason"]}}},
]
