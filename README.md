---
title: Incident Response RL Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Incident Response RL Environment

This environment simulates a real production incident response workflow. An agent inspects logs, metrics, and alerts, diagnoses the outage, takes corrective actions, and attempts to restore service health without making the situation worse.

## Motivation

Incident response is a real operational task with noisy signals, partial progress, delayed effects, and meaningful penalties for bad decisions. This environment is intended for evaluating agent reliability on troubleshooting work that humans actually do in production systems.

## Action Space

The action model is `IncidentAction`:

- `analyze_logs`
- `restart_service`
- `rollback_deployment`
- `scale_up`
- `ignore`
- `escalate`

Optional `target` is used for service-level actions and defaults to `api` in the built-in tasks.

## Observation Space

The observation model is `IncidentObservation`:

- `logs: list[str]`
- `metrics: dict[str, float]`
- `alerts: list[str]`
- `system_status: healthy | degraded | critical`
- `step_count: int`
- OpenEnv fields:
  - `done`
  - `reward`
  - `metadata`

Observation metadata includes:

- `scenario_id`
- `difficulty`
- `incident_family`
- `task_score`
- `terminal_grade`
- per-step `info`

## Tasks

The built-in tasks are deterministic and graded from `0.0` to `1.0`:

- `high_latency_easy`
  - Goal: detect capacity saturation and scale the API service.
  - Expected path: `scale_up`
- `service_crash_medium`
  - Goal: recover a crash-looping API with a misleading signal present.
  - Expected path: `restart_service`
- `bad_deployment_hard`
  - Goal: unwind a bad deploy with cascading symptoms and delayed recovery.
  - Expected path: `rollback_deployment` then `restart_service`

## Reward And Grading

Dense reward shaping is used during the trajectory:

- `+0.2` diagnosis improvement
- `+0.3` correct remedial action
- `+0.3` measurable system improvement
- `+0.5` full incident resolution
- `-0.2` wrong or harmful action
- `-0.1` repeated ineffective action
- `-0.3` failed escalation

Deterministic task graders are separate from dense reward:

- `1.0` full recovery
- `0.5` partial recovery
- `0.0` failure

Per-step partial task progress is exposed through `metadata.task_score`.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev,platform]
```

Required environment variables for the baseline runner:

```bash
HF_TOKEN=...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=openai/gpt-oss-20b
ENV_BASE_URL=http://127.0.0.1:7860
```

The submission baseline uses the OpenAI Python client pointed at `API_BASE_URL` and authenticated with `HF_TOKEN`.

## Local Usage

Run tests:

```bash
python -m pytest
```

Run the OpenEnv server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Validate the environment:

```bash
openenv validate .
```

Or with the helper script on Windows:

```powershell
.\run.ps1 -Mode test
.\run.ps1 -Mode api
.\run.ps1 -Mode validate
```

For local baseline runs, point `ENV_BASE_URL` at the local server:

```powershell
$env:HF_TOKEN="your_token"
$env:ENV_BASE_URL="http://127.0.0.1:7860"
.\.venv\Scripts\python inference.py
```

## Baseline Inference

The submission inference script is the root [`inference.py`](/D:/RL/inference.py). It uses the OpenAI Python client with:

- `base_url=API_BASE_URL`
- `api_key=HF_TOKEN`
- `model=MODEL_NAME`

The official hackathon runtime path is the Hugging Face router plus the deployed OpenEnv environment.

For a deployed Hugging Face Space run:

```powershell
$env:HF_TOKEN="your_token"
$env:ENV_BASE_URL="https://saransh24-incident-response-rl.hf.space"
.\.venv\Scripts\python inference.py
```

Equivalent shell form:

```bash
export HF_TOKEN=your_token
export ENV_BASE_URL=https://saransh24-incident-response-rl.hf.space
python inference.py
```

This evaluates all 3 tasks against the target environment and writes a reproducible report to `artifacts/baseline_scores.json`.

The script emits structured stdout logs in this order:

- `[START]` once with `task`, `env`, `model`
- `[STEP]` once per environment step with `step`, `action`, `reward`, `done`, `error`
- `[END]` once with `success`, `steps`, `score`, `rewards`

## Latest Recorded Baseline Scores

The checked-in artifact at `artifacts/baseline_scores.json` currently records this live HF-router run for `openai/gpt-oss-20b`:

- `average_score`: `1.0`
- `high_latency_easy`: score `1.0`, terminal grade `1.0`, steps `1`, total reward `1.3`, successful actions `["scale_up"]`
- `service_crash_medium`: score `1.0`, terminal grade `1.0`, steps `1`, total reward `1.3`, successful actions `["restart_service"]`
- `bad_deployment_hard`: score `1.0`, terminal grade `1.0`, steps `2`, total reward `2.1`, successful actions `["rollback_deployment", "restart_service"]`

The baseline report includes:

- `model_name`
- `router_base_url`
- `average_score`
- per-task `scenario_id`, `difficulty`, `score`, `terminal_grade`, `steps_taken`, `total_reward`, `successful_actions`

Re-run `python inference.py` after changing the model or prompt policy to refresh these numbers before final submission.

## Docker

Build:

```bash
docker build -t incident-response-rl:latest .
```

Run:

```bash
docker run --rm -p 7860:7860 incident-response-rl:latest
```

Then validate the running service:

```bash
openenv validate --url http://127.0.0.1:7860
```

## Hugging Face Spaces Deployment

Log in first:

```bash
huggingface-cli login
```

Then push the environment:

```bash
openenv push
```

The Space should be a Docker Space and include the `openenv` tag. After deployment, validate the live service with:

```bash
openenv validate --url https://<your-space>.hf.space
```
