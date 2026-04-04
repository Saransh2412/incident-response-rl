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

## Baseline Inference

The mandatory inference script uses the Hugging Face router, not an OpenAI API key.

Run it after the environment server is up:

```bash
uv run inference.py
```

Equivalent:

```bash
python inference.py
```

This evaluates all 3 tasks and writes a reproducible report to `artifacts/baseline_scores.json`.

## Expected Baseline Output

The baseline report contains:

- `model_name`
- `router_base_url`
- `average_score`
- per-task `scenario_id`, `difficulty`, `score`, `terminal_grade`, `steps_taken`, `total_reward`

Baseline scores depend on the chosen HF model, so record the final numbers after your submission model is fixed.

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
