# CloudBuddy – FinOps agent for EC2

My final-year project at FAST was CloudBuddy, a tool that forecasts AWS EC2 usage and
suggests ways to cut cost and carbon. This repo is a rebuild of the parts I worked on,
cleaned up and extended with an LLM agent that can explain the numbers and queue fixes.

The main idea: **the model never touches infrastructure.** It can read data, run the
forecast and *request* a change. A person approves it, and only then does an executor
run, by default as an AWS dry run.

```
            ┌──────────────┐   tool calls   ┌──────────────────────┐
 question → │  LLM agent   │ ─────────────► │ list / forecast /    │
            │ (Llama 3.3)  │ ◄───────────── │ recommend / request  │
            └──────────────┘    JSON        └─────────┬────────────┘
                                                      │ request_action
                                                      ▼
                                         ┌──────────────────────┐  approve  ┌──────────────┐
                                         │   approval queue     │ ────────► │ executor     │
                                         └──────────────────────┘           │ mock | boto3 │
                                                                            └──────────────┘
```

## What it does

- **Forecasting** – LightGBM on lag, rolling-window and calendar features, predicting
  hourly CPU per instance and rolled forward 7 days.
- **Recommendations** – simple, explainable rules on the forecast: stop idle machines
  (p95 < 5%), downsize one step when the smaller type would still stay under 80% p95.
- **Cost and carbon** – on-demand prices plus a Cloud Carbon Footprint style energy
  estimate (per-vCPU watts × PUE × grid factor for us-east-1).
- **Agent** – a plain tool-calling loop over any OpenAI-compatible API. I use Groq with
  `llama-3.3-70b-versatile`. Tool errors go back to the model as JSON so it can recover.
- **Human in the loop** – `request_action` only queues; approve or reject in the UI.
  `Boto3Executor` defaults to `DryRun=True`.
- **Streamlit UI** – forecast chart, recommendations, chat, approvals.

## Results

The demo runs on simulated data (7 instances, 60 days, hourly). I couldn't publish real
account data, so the generator copies patterns I saw in practice: daily/weekly cycles,
night batch jobs, slow growth and forgotten dev boxes. Swap in a CloudWatch export with
the same columns to use real data.

Backtest on the last 7 days (one-step-ahead, 1,176 instance-hours):

| model | MAE (CPU %) | WAPE |
|---|---|---|
| same hour last week | 3.55 | 14.4% |
| LightGBM | **2.96** | **12.0%** |

I report WAPE instead of MAPE because MAPE blows up on near-idle machines.

On the simulated fleet the rules find about **$271/month** of savings (two idle dev
boxes to stop, one database server to downsize). That number only means something for
this fake fleet, it's there to show the pipeline end to end.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/evaluate.py      # backtest + recommendations in the terminal
pytest -q                       # tests use a scripted LLM, no API key needed

cp .env.example .env            # add your Groq key for the chat tab
streamlit run app.py
```

Or with Docker:

```bash
docker build -t cloudbuddy .
docker run -p 8501:8501 --env-file .env cloudbuddy
```

To point it at a real account set `CLOUDBUDDY_EXECUTOR=boto3`. It stays a dry run until you
also set `CLOUDBUDDY_DRY_RUN=0`.

## Layout

```
cloudbuddy/
  simulate.py   synthetic fleet data
  forecast.py   features, LightGBM, backtest, recursive forecast
  pricing.py    prices, carbon estimate, rightsizing rules
  tools.py      tools + JSON schemas for the agent
  agent.py      tool-calling loop
  actions.py    approval queue, mock and boto3 executors
app.py          Streamlit UI
scripts/evaluate.py
tests/
```

## Things I'd like to add

- read real metrics from CloudWatch instead of the simulator
- memory and network in the rightsizing rules, not just CPU
- quantile forecasts so the p95 comes from the model instead of the point forecast
- Savings Plans / spot suggestions
- an MCP server so the same tools work from other agents
