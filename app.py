import os

import pandas as pd
import streamlit as st

from cloudbuddy.actions import ApprovalQueue, Boto3Executor, MockExecutor
from cloudbuddy.agent import FinOpsAgent, GroqChat
from cloudbuddy.forecast import CpuForecaster
from cloudbuddy.simulate import simulate_usage
from cloudbuddy.tools import FinOpsTools

st.set_page_config(page_title="CloudBuddy", page_icon="☁️", layout="wide")


@st.cache_resource
def load():
    usage = simulate_usage(days=60)
    fc = CpuForecaster()
    evaluation = fc.evaluate(usage)
    fc.fit(usage)
    return usage, fc, evaluation


usage, forecaster, evaluation = load()

if "tools" not in st.session_state:
    live = os.getenv("CLOUDBUDDY_EXECUTOR") == "boto3"
    executor = Boto3Executor(dry_run=os.getenv("CLOUDBUDDY_DRY_RUN", "1") != "0") if live else MockExecutor()
    st.session_state.tools = FinOpsTools(usage, forecaster, ApprovalQueue(executor))
    st.session_state.chat = []
tools: FinOpsTools = st.session_state.tools

st.title("CloudBuddy")
st.caption("Forecast EC2 usage, find waste, and let an agent queue fixes for you to approve. "
           "Demo data is simulated.")

c1, c2, c3 = st.columns(3)
recs = pd.DataFrame(tools.recommend_actions())
c1.metric("Potential savings", f"${recs.monthly_savings_usd.sum():,.0f}/mo")
c2.metric("CO2 avoided", f"{recs.monthly_co2_saved_kg.sum():.1f} kg/mo")
c3.metric("Forecast WAPE", f"{evaluation.wape:.1f}%", f"{evaluation.wape - evaluation.baseline_wape:.1f} vs naive",
          delta_color="inverse")

tab_fc, tab_recs, tab_agent, tab_queue = st.tabs(["Forecast", "Recommendations", "Ask the agent", "Approvals"])

with tab_fc:
    instance = st.selectbox("Instance", tools.fleet.index.tolist())
    hist = usage[usage.instance_id == instance].set_index("timestamp").cpu_pct.tail(24 * 14)
    fc = forecaster.forecast(usage, instance, 168).set_index("timestamp").cpu_pct_forecast
    st.line_chart(pd.DataFrame({"actual": hist, "forecast": fc}))
    st.json(tools.forecast_usage(instance))

with tab_recs:
    st.dataframe(recs, hide_index=True, use_container_width=True)
    for r in recs[recs.action != "keep"].itertuples():
        if st.button(f"Queue {r.action} for {r.instance_id}", key=f"q-{r.instance_id}"):
            tools.queue.submit(r.instance_id, r.action, r.reason, r.target_type)
            st.success("Added to approvals")

with tab_agent:
    if not os.getenv("LLM_API_KEY"):
        st.info("Set LLM_API_KEY (a free Groq key works) to chat with the agent.")
    else:
        if "agent" not in st.session_state:
            st.session_state.agent = FinOpsAgent(GroqChat(), tools)
        for role, text in st.session_state.chat:
            st.chat_message(role).write(text)
        if q := st.chat_input("e.g. Which machines can I shut down, and how much would I save?"):
            st.chat_message("user").write(q)
            with st.spinner("thinking..."):
                reply = st.session_state.agent.ask(q)
            with st.chat_message("assistant"):
                st.write(reply.answer)
                with st.expander(f"{len(reply.steps)} tool calls"):
                    for s in reply.steps:
                        st.code(f"{s['tool']}({s['args']})\n-> {s['result'][:600]}")
            st.session_state.chat += [("user", q), ("assistant", reply.answer)]

with tab_queue:
    reqs = list(tools.queue.requests.values())
    if not reqs:
        st.write("Nothing queued yet.")
    for r in reqs:
        label = f"#{r.id} {r.action} {r.instance_id}" + (f" -> {r.target_type}" if r.target_type else "")
        with st.container(border=True):
            st.write(f"**{label}** — {r.status}")
            st.caption(r.reason)
            if r.status == "pending":
                a, b = st.columns(2)
                if a.button("Approve", key=f"a{r.id}"):
                    tools.queue.approve(r.id)
                    st.rerun()
                if b.button("Reject", key=f"r{r.id}"):
                    tools.queue.reject(r.id)
                    st.rerun()
            elif r.result:
                st.code(r.result)
