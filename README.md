# Quant Agents

An autonomous quant research desk built from LLM agents. An **Ideator** proposes a trading hypothesis, a **Coder** turns it into vectorized pandas code, a **Sandbox** backtests it out-of-sample with transaction costs and a deterministic lookahead test, an **Auditor** enforces risk gates and critiques the code, and a **Packager** writes a handoff report for the execution team.

It runs on any OpenAI-compatible endpoint: our own vLLM server on AMD Instinct GPUs (ROCm), AMD Token Factory, or another hosted API. The default model is `Qwen/Qwen3.6-35B-A3B`.

## Architecture

```
Ideator ──▶ Coder ──▶ Sandbox ──▶ Auditor ──(all gates pass)──▶ Packager ──▶ END
   ▲          ▲      (subprocess)    │
   │          └──── revise ──────────┤  gates fail, iteration < MAX_ITERATIONS
   └────────── new idea ─────────────┤  idea exhausted, idea_count < MAX_IDEAS
                                     └──▶ END                    MAX_IDEAS reached
```

| Component | File | What it does |
|---|---|---|
| Graph | `core/graph.py` | LangGraph state machine. Each idea gets `MAX_ITERATIONS` coder attempts. The run ends after `MAX_IDEAS` ideas (recursion limit 100). |
| Sandbox | `core/sandbox.py`, `core/sandbox_worker.py` | Executes `generate_signals(df)` in a separate Python process with a 30 s timeout. Positions are `signal.shift(1)`, clipped to [-1, 1], and charged 1 bp per unit of turnover. Metrics are reported for a 70/30 train/test split against buy & hold. |
| Lookahead test | `core/sandbox.py` | Re-runs the strategy on `df.iloc[:k]` at about 9 checkpoints and requires the last signal to equal the full-history signal at bar `k`. Also scans the code for `shift(-n)`. Any mismatch fails the run with an exact date and values. |
| Gates | `agents/auditor.py` | Decided in code, not by the LLM: **no execution errors AND OOS Sharpe > `BENCHMARK_SHARPE` AND OOS max drawdown < `MAX_DRAWDOWN_LIMIT` AND lookahead test passes**. The LLM only writes the critique. |
| Runner | `core/runner.py` | `run_pipeline(params)` yields one event per node, `{node, payload, ts, latency_s, tokens}`, and appends each event to `runs/<run_id>.jsonl`. |
| UI | `app.py` | Streamlit dashboard with a live agent timeline, gate checklist, equity curve, metric cards, code diffs, the handoff report download, and a "Powered by AMD" strip (model, endpoint, tokens/s, latency per call, LLM calls). |
| Data | `data/fetcher.py` | Daily OHLCV from yfinance (`auto_adjust=True`). Falls back to `data/<TICKER>.csv` when yfinance is unavailable. `data/SPY.csv` ships 2 years of SPY. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then point it at your endpoint
streamlit run app.py                                 # UI on http://localhost:8501
python main.py --demo                                # or headless CLI
```

If the endpoint is unreachable, the UI shows a banner, disables live runs, and replays the most recent recorded run from `runs/`.

### Option A: vLLM on AMD GPUs (ROCm)

`./run_amd_vllm.sh` starts the official `vllm/vllm-openai-rocm:v0.31.0` image. It serves `Qwen/Qwen3.6-35B-A3B` with `--reasoning-parser qwen3 --max-model-len 32768` and requires the API key from `.env`. The port is published on `127.0.0.1:8000` only. For a full one-command setup, see [Deploy on AMD Developer Cloud](#deploy-on-amd-developer-cloud) below.

`--reasoning-parser qwen3` moves the model's `<think>` output out of the answer. Without it, the agents strip `<think>` blocks themselves. Set `LLM_DISABLE_THINKING=1` to turn thinking off for faster iterations.

### Option B: AMD Token Factory or any OpenAI-compatible API

```env
LLM_BASE_URL=https://<provider>/v1
LLM_API_KEY=<your key>
LLM_MODEL=<provider's model id>
```

### Docker (the app only)

```bash
docker build -t quant-agents .
docker run -p 8501:8501 --env-file .env quant-agents
```

`.env` is never baked into the image (see `.dockerignore`). Pass it at runtime.

## Deploy on AMD Developer Cloud

Use an MI300X droplet with the Ubuntu + ROCm image. SSH in, then run:

```bash
git clone https://github.com/saifshbeita/quant-agents.git
cd quant-agents
cp .env.example .env
./deploy_amd.sh
```

`deploy_amd.sh` is safe to re-run. It:

1. Checks that `rocm-smi` works, `/dev/kfd` exists, and Docker is running.
2. Writes a random `LLM_API_KEY` into `.env` if none is set.
3. Starts vLLM with `run_amd_vllm.sh` and waits for `/v1/models`, printing load progress. The first start downloads about 70 GB of weights into `~/.cache/huggingface`; later starts reuse them. A vLLM server that is already healthy is left running.
4. Builds the app image and runs it with `--network host` at `http://0.0.0.0:8501`, pointed at `http://127.0.0.1:8000/v1`. Both containers use `--restart unless-stopped`.
5. Sends one smoke-test completion through the app's own LLM client.
6. Prints the public URL `http://<public-ip>:8501` and the `rocm-smi` output.

vLLM listens on `127.0.0.1:8000` only and needs the API key, so only the Streamlit app on port 8501 is public. The app has no login, so share the URL only with your judges and team.

Useful follow-ups:

```bash
./deploy_amd.sh --app-only        # rebuild/restart the app after editing .env or pulling code
./deploy_amd.sh --restart-vllm    # recreate the vLLM container (e.g. after changing VLLM_IMAGE)
docker logs -f vllm-server        # model loading, request throughput
docker logs -f quant-agents       # app logs
./capture_gpu_stats.sh            # record rocm-smi during a demo run (Ctrl-C to stop)
./capture_gpu_stats.sh 600 1      # ...or for 600 s at 1 s intervals
```

`capture_gpu_stats.sh` writes `gpu_stats/<timestamp>/`, which contains `snapshot.txt`, `samples.log`, `samples.csv`, and `vllm_throughput.log` (vLLM's own prompt and generation tokens/s for the window). Keep this folder as GPU evidence for the demo.

**If calls are too slow because of thinking:** Qwen3.6 reasons before answering by default, which can add tens of seconds per agent call. Set `LLM_DISABLE_THINKING=1` in `.env`, then restart the app with `./deploy_amd.sh --app-only`. `docker restart` alone does not reload `.env`. vLLM does not need a restart, because the app sends `enable_thinking=false` per request.

**Recording a demo replay:** after a good live run, copy its log to keep it as the offline fallback: `cp runs/<run_id>.jsonl runs/demo-<name>.jsonl`, then commit it.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL |
| `LLM_API_KEY` | `EMPTY` | API key (vLLM accepts any value) |
| `LLM_MODEL` | `Qwen/Qwen3.6-35B-A3B` | Model id (`MODEL_NAME` is accepted as a legacy alias) |
| `LLM_TEMPERATURE` | `0.6` | Sampling temperature |
| `LLM_TIMEOUT_S` | `180` | Per-call timeout |
| `LLM_DISABLE_THINKING` | `false` | Send `enable_thinking=False` to Qwen3 chat templates |
| `LLM_MOCK` | `false` | Use the deterministic offline fake LLM (rehearsals and tests) |
| `TARGET_TICKER` | `SPY` | Default ticker |
| `MAX_IDEAS` | `3` | Ideas per run before giving up |
| `MAX_ITERATIONS` | `5` | Coder attempts per idea |
| `RECURSION_LIMIT` | `100` | LangGraph recursion limit (raised automatically if limits need more) |
| `SANDBOX_TIMEOUT_S` | `30` | Kill strategies that run longer than this |
| `TRANSACTION_COST_BPS` | `1` | Cost per unit of turnover, in basis points |
| `TRAIN_FRACTION` | `0.7` | Train/test split |
| `BENCHMARK_SHARPE` | `1.5` | Minimum OOS Sharpe |
| `MAX_DRAWDOWN_LIMIT` | `0.15` | Maximum OOS drawdown |
| `DEMO_MODE` | `false` | Use the looser gates below (also a toggle in the UI) |
| `DEMO_BENCHMARK_SHARPE` | `0.5` | OOS Sharpe gate in demo mode |
| `DEMO_MAX_DRAWDOWN_LIMIT` | `0.25` | OOS drawdown gate in demo mode |
| `RUNS_DIR` | `./runs` | Where run logs are written and replays are read |
| `HF_TOKEN` | (empty) | Hugging Face token for vLLM downloads (optional for Qwen) |
| `VLLM_IMAGE` | `vllm/vllm-openai-rocm:v0.31.0` | vLLM ROCm image used by `run_amd_vllm.sh` |
| `TENSOR_PARALLEL_SIZE` | `1` | GPUs per vLLM server (one MI300X holds the BF16 model) |
| `VLLM_MAX_MODEL_LEN` | `32768` | vLLM context length |
| `VLLM_READY_TIMEOUT` | `3600` | Seconds `deploy_amd.sh` waits for the first model load |

## Recorded runs and replay

Every run is logged to `runs/<run_id>.jsonl`. Only files named `runs/demo-*.jsonl` are committed (see `.gitignore`), so rename a good run to keep it as the offline fallback. `runs/demo-mock-offline-SPY.jsonl` was produced by the mock LLM and is labelled as such in the UI.

## Tests

```bash
python test_system.py     # endpoint ping, data fetch, sandbox metrics, lookahead detection
python test_e2e_mock.py   # fake-LLM end-to-end: graph terminates, gates enforced, endpoint errors, UI loads
```

## Notes

- The sandbox gives process isolation and a timeout, not a security boundary. Only run it against models you control.
- Generated strategies only see the ticker's own daily OHLCV and returns. The prompts say this explicitly so the model doesn't invent VIX or other external data.
