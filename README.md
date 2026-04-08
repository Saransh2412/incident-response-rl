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

The environment exposes 3 public task families with deterministic graders, and task scores are exported strictly inside `(0, 1)`:

- `high_latency_easy`
  - Goal: diagnose API latency caused by capacity or burst-amplification issues.
  - Typical path: `scale_up`, with some seeded variants requiring `analyze_logs` first.
- `service_crash_medium`
  - Goal: recover a crash-looping API while distinguishing direct restarts from config or dependency-rooted failures.
  - Typical path: `restart_service`, with diagnosis-first variants that punish premature restarts.
- `bad_deployment_hard`
  - Goal: unwind a bad release with delayed recovery and multi-step dependencies.
  - Typical path: `rollback_deployment` then `restart_service`, with harder seeded variants requiring `analyze_logs` before rollback.

When `scenario_id` is omitted, the environment samples from these task families using the provided seed. The family name stays stable while the evidence, noise, and required sequence can vary by seed.

The service also exposes validator-facing task and grading surfaces:

- `GET /info` exposes environment metadata, task registry, max steps, and JSON schemas
- `GET /tasks` returns the 3 public tasks as a bare machine-readable list for validator compatibility
- `GET /tasks_wrapped` returns the same tasks under `{"tasks": [...]}` for convenience
- `POST /grader` grades a proposed trajectory for a given `task_id`
- `POST /grade` is a simpler compatibility alias for task grading
- `POST /baseline` runs the built-in deterministic heuristic over one or all public tasks
- `POST /reset` accepts either `scenario_id` or `task_id`

Task registry and explicit grader mapping:

- `high_latency_easy` -> `incident_response_grade_high_latency`
- `service_crash_medium` -> `incident_response_grade_service_crash`
- `bad_deployment_hard` -> `incident_response_grade_bad_deployment`

A static machine-readable copy of this registry is also available at the repo root in [`task_registry.json`](/D:/RL/task_registry.json).
For compatibility with static validators, the same public task list is also mirrored at the repo root in [`tasks.py`](/D:/RL/tasks.py), and the named grader entrypoints are re-exported from [`graders.py`](/D:/RL/graders.py).

Every grader-visible result includes 5 explicit grading components:

- `diagnosis`
- `sequence`
- `effectiveness`
- `efficiency`
- `safety`

## Reward And Grading

Dense reward shaping is deterministic but trajectory-sensitive:

- diagnosis earns a smaller reward unless it is a required step in the recovery sequence
- correct remedial actions earn reward before full resolution, enabling partial-progress learning
- measurable system improvement uses improvement bands, so stronger recoveries earn more than marginal ones
- premature or unnecessary actions incur larger penalties in diagnosis-first variants
- escalation ends the episode with a low score

Deterministic task graders are separate from dense reward:

- scores are exported strictly inside `(0, 1)` for validator compatibility
- perfect direct recoveries land in the high band, typically around `0.92`
- perfect diagnosis-first recoveries can reach the ceiling at `0.99`
- partial recovery paths land in the mid band, for example around `0.63`
- clearly wrong or escalated failure paths stay near the floor, for example `0.01–0.21`

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

List public tasks directly:

```bash
curl http://127.0.0.1:7860/tasks
```

Grade a trajectory directly:

```bash
curl -X POST http://127.0.0.1:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id":"service_crash_medium","trajectory":[{"action_type":"restart_service","target":"api"}]}'
```

Run the built-in validator-facing baseline:

```bash
curl -X POST http://127.0.0.1:7860/baseline \
  -H "Content-Type: application/json" \
  -d '{}'
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

The script emits structured stdout logs in the sample one-line format:

- `[START] task=<task> env=<env> model=<model>`
- `[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>`

`action` is emitted as a plain-text action string such as `scale_up api`.

## Latest Recorded Baseline Scores

The latest live HF-router run after the seeded-variant depth upgrade produced:

- `average_score`: `1.0`
- `high_latency_easy`: solved in `1` step with reward `1.10`
- `service_crash_medium`: solved in `2` steps with rewards `0.25, 1.00`
- `bad_deployment_hard`: solved in `3` steps with rewards `0.25, 0.50, 0.90`

A representative evaluator-facing stdout transcript now looks like:

```text
[START] task=all_tasks env=incident-response-rl model=openai/gpt-oss-20b
[STEP] step=1 action=scale_up api reward=1.10 done=true error=null
[STEP] step=1 action=analyze_logs reward=0.25 done=false error=null
[STEP] step=2 action=restart_service api reward=1.00 done=true error=null
[STEP] step=1 action=analyze_logs reward=0.25 done=false error=null
[STEP] step=2 action=rollback_deployment api reward=0.50 done=false error=null
[STEP] step=3 action=restart_service api reward=0.90 done=true error=null
[END] success=true steps=6 score=1.000 rewards=1.10,0.25,1.00,0.25,0.50,0.90
```

Representative terminal grades from the current continuous grader are:

- `high_latency_easy` direct-fix path: `0.92`
- `high_latency_easy` diagnosis-first path: `0.99`
- `service_crash_medium` direct-fix path: `0.92`
- `service_crash_medium` diagnosis-first path: `0.99`
- `bad_deployment_hard` direct rollback-then-restart path: `0.92`
- `bad_deployment_hard` diagnosis-first path: `0.99`
- partial hard-task recovery after rollback only: about `0.63`
- repeated wrong-action path: about `0.21`
- escalation failure: about `0.16`

Re-run `python inference.py` after deploying the latest environment to refresh the public baseline numbers. The report still includes:

- `model_name`
- `router_base_url`
- `average_score`
- per-task `scenario_id`, `difficulty`, `score`, `terminal_grade`, `steps_taken`, `total_reward`, `successful_actions`

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
