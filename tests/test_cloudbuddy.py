import json

import pytest

from cloudbuddy.actions import ApprovalQueue, MockExecutor
from cloudbuddy.agent import FinOpsAgent
from cloudbuddy.forecast import CpuForecaster, add_features
from cloudbuddy.pricing import recommend
from cloudbuddy.simulate import simulate_usage
from cloudbuddy.tools import FinOpsTools


@pytest.fixture(scope="module")
def usage():
    return simulate_usage(days=21, seed=1)


@pytest.fixture(scope="module")
def tools(usage):
    fc = CpuForecaster(n_estimators=50).fit(usage)
    return FinOpsTools(usage, fc, ApprovalQueue(MockExecutor()), horizon_hours=24)


def test_features_have_no_future_leak(usage):
    f = add_features(usage)
    one = f[f.instance_id == "i-0web01"].reset_index(drop=True)
    assert one.loc[200, "lag_1"] == one.loc[199, "cpu_pct"]
    assert one.loc[200, "lag_168"] == one.loc[32, "cpu_pct"]


def test_idle_machine_gets_stopped():
    rec = recommend("i-x", "m5.large", [1.0] * 168)
    assert rec.action == "stop" and rec.monthly_savings_usd > 0


def test_downsize_only_when_smaller_type_fits():
    assert recommend("i-x", "m5.xlarge", [20.0] * 168).action == "downsize"
    # 39% on 4 vCPU would be ~78% on 2 vCPU, ok; 45% would not be
    assert recommend("i-x", "m5.xlarge", [45.0] * 168).action == "keep"


def test_forecast_is_bounded(tools):
    out = tools.forecast_usage("i-0api01")
    assert 0 <= out["avg_cpu_pct"] <= 100


def test_request_action_does_not_execute(tools):
    res = json.loads(tools.call("request_action", {"instance_id": "i-0dev01", "action": "stop", "reason": "idle"}))
    assert res["status"] == "pending"
    assert tools.queue.executor.log == []
    tools.queue.approve(res["request_id"])
    assert tools.queue.executor.log == ["stop i-0dev01"]


def test_bad_tool_input_returns_error_json(tools):
    out = json.loads(tools.call("forecast_usage", {"instance_id": "i-nope"}))
    assert "error" in out


class ScriptedLLM:
    """Plays back fixed replies so the loop can be tested without an API key."""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools):
        return self.replies.pop(0)


def test_agent_runs_tools_then_answers(tools):
    llm = ScriptedLLM([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "list_instances", "arguments": "{}"}}]},
        {"role": "assistant", "content": "You have 7 instances."},
    ])
    reply = FinOpsAgent(llm, tools).ask("how many machines do I have?")
    assert reply.answer == "You have 7 instances."
    assert reply.steps[0]["tool"] == "list_instances"
