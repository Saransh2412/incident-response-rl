# Project Context: Incident Response RL

This file is a full handoff for another engineer or LLM working on this repository on the same laptop.

It is written to be pasted directly into an LLM as project context so work can continue immediately.

## 1. Project Summary

- Project name: `incident-response-rl`
- Local path: `D:\RL`
- GitHub repo: [incident-response-rl](https://github.com/Saransh2412/incident-response-rl)
- Hugging Face Space: [Saransh24/incident-response-rl](https://huggingface.co/spaces/Saransh24/incident-response-rl)
- Live app URL: [saransh24-incident-response-rl.hf.space](https://saransh24-incident-response-rl.hf.space)
- Main goal: submit a real-world OpenEnv environment for the Meta PyTorch Hackathon x Scaler School of Technology
- Domain: production incident response simulation

The environment simulates a human-realistic operations workflow:
- inspect logs, alerts, and metrics
- diagnose a production issue
- take recovery actions
- restore service health without making the incident worse

This is not a game. It is intended as a real-world agent-evaluation environment.

## 2. Hackathon Requirements

The important hackathon requirements are:

- real-world task simulation
- full OpenEnv compliance
- minimum 3 tasks with graders
- meaningful reward shaping
- baseline inference script
- deploy to Hugging Face Spaces
- working Dockerfile
- README with documentation and baseline scores

Phase breakdown:

- Phase 1: automated validation
- Phase 2: agentic evaluation
- Phase 3: human review

Current pain point:

- Phase 1 passes
- Phase 2 keeps failing at `Task Validation`
- exact error: `Not enough tasks with graders`

## 3. Current Evaluation State

Latest repeated evaluation pattern:

- `Docker Build Creation`: passed
- `inference.py Execution`: passed
- `Output Parsing`: passed
- `LLM Criteria Check`: passed
- `Task Validation`: failed

The repeated failure text is:

```text
Not enough tasks with graders
Your submission must include at least 3 tasks with graders.
```

Important conclusion:

- the environment is not broadly broken
- the issue is almost certainly task/grader discovery by the hidden validator
- this is likely a static or schema-discovery mismatch, not a runtime logic failure

## 4. Current Best Diagnosis

The project clearly has 3 public tasks and graders in a human/runtime sense.

But the hidden validator still does not recognize them.

Most likely reason:

- the validator is doing a rigid scan for task/grader declarations
- it may prefer root-level modules and static manifests over custom runtime behavior
- it may not fully trust or inspect the custom `/tasks`, `/info`, `/grader`, `/baseline` endpoints

Because of this, the project has been progressively made more explicit and redundant in how tasks and graders are declared.

## 5. Public Tasks

The 3 public tasks are:

1. `high_latency_easy`
   - Name: `API Latency Recovery`
   - Difficulty: `easy`
   - Grader: `incident_response_grade_high_latency`

2. `service_crash_medium`
   - Name: `Crash Loop Recovery`
   - Difficulty: `medium`
   - Grader: `incident_response_grade_service_crash`

3. `bad_deployment_hard`
   - Name: `Bad Deployment Remediation`
   - Difficulty: `hard`
   - Grader: `incident_response_grade_bad_deployment`

Each task has hidden seeded variants, but these 3 are the public canonical task ids.

Canonical seeds used for stable validator-facing grading:

- `high_latency_easy`: `5`
- `service_crash_medium`: `3`
- `bad_deployment_hard`: `1`

## 6. Current Grading Model

The project now uses a richer continuous grading model.

Every task result has:

- one final `score` strictly inside `(0, 1)`
- one `breakdown` object with 5 grading components

The 5 grading components are:

- `diagnosis`
- `sequence`
- `effectiveness`
- `efficiency`
- `safety`

Current component weights:

- `diagnosis`: `0.20`
- `sequence`: `0.25`
- `effectiveness`: `0.25`
- `efficiency`: `0.15`
- `safety`: `0.15`

Export policy:

- minimum exported score floor: `0.01`
- maximum exported score ceiling: `0.99`

Representative grade bands:

- strong direct success: around `0.92`
- diagnosis-first perfect success: `0.99`
- partial hard recovery: around `0.63`
- wrong path: around `0.21`
- escalation failure: around `0.16`

## 7. Reward Shaping

Dense reward shaping is separate from final grading.

The environment gives partial progress rewards for:

- diagnosis
- correct partial remediation
- measurable system improvement

It penalizes:

- wrong actions
- unnecessary actions
- premature harmful actions in diagnosis-first variants
- escalation

The reward function is intended to give learning signal across the trajectory, not just at the terminal step.

## 8. Architecture

Main code locations:

- environment logic: [src/incident_response_rl/env.py](/D:/RL/src/incident_response_rl/env.py)
- scenario generation: [src/incident_response_rl/scenarios.py](/D:/RL/src/incident_response_rl/scenarios.py)
- transition logic: [src/incident_response_rl/transition.py](/D:/RL/src/incident_response_rl/transition.py)
- models: [src/incident_response_rl/models.py](/D:/RL/src/incident_response_rl/models.py)
- graders: [src/incident_response_rl/graders.py](/D:/RL/src/incident_response_rl/graders.py)
- task models/registry: [src/incident_response_rl/tasks.py](/D:/RL/src/incident_response_rl/tasks.py)
- baseline logic: [src/incident_response_rl/inference.py](/D:/RL/src/incident_response_rl/inference.py)
- FastAPI app: [server/app.py](/D:/RL/server/app.py)
- environment wrapper: [server/incident_response_environment.py](/D:/RL/server/incident_response_environment.py)
- root inference entrypoint: [inference.py](/D:/RL/inference.py)

Root-level compatibility shims added for validator discovery:

- [tasks.py](/D:/RL/tasks.py)
- [graders.py](/D:/RL/graders.py)

Static metadata / manifests:

- [openenv.yaml](/D:/RL/openenv.yaml)
- [task_registry.json](/D:/RL/task_registry.json)
- [README.md](/D:/RL/README.md)
- this file: [context.md](/D:/RL/context.md)

## 9. Important Runtime Endpoints

OpenEnv core endpoints:

- `GET /health`
- `GET /metadata`
- `GET /schema`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /mcp`

Validator-facing custom endpoints:

- `GET /info`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`

Expected purpose:

- `/info`: machine-readable environment summary including tasks and grading components
- `/tasks`: machine-readable task registry
- `/grader`: score a submitted trajectory for a given task
- `/baseline`: run built-in heuristic over one or all tasks

## 10. Current Task/Grader Redundancy

Task/grader mapping is intentionally declared in multiple places because the hidden validator appears brittle.

Currently declared in:

1. runtime API:
   - `/info`
   - `/tasks`

2. static manifest:
   - [task_registry.json](/D:/RL/task_registry.json)

3. OpenEnv metadata:
   - [openenv.yaml](/D:/RL/openenv.yaml)

4. root-level Python shims:
   - [tasks.py](/D:/RL/tasks.py)
   - [graders.py](/D:/RL/graders.py)

5. package code:
   - [src/incident_response_rl/tasks.py](/D:/RL/src/incident_response_rl/tasks.py)
   - [src/incident_response_rl/graders.py](/D:/RL/src/incident_response_rl/graders.py)

This redundancy was added specifically to try to satisfy Phase 2 task discovery.

## 11. Major Changes Already Made

This section is important because a lot of work has already been done.

### A. Inference format fixed

The baseline logs were changed to strict one-line hackathon format:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

This now passes:

- `inference.py Execution`
- `Output Parsing`

### B. Task depth upgraded

The environment was upgraded from simpler one-step flows to more realistic seeded variants:

- `high_latency_easy` variants
  - traffic surge
  - cache degradation
  - queue pressure

- `service_crash_medium` variants
  - OOM/crash-loop
  - bad config
  - dependency timeout / diagnosis-first variants

- `bad_deployment_hard` variants
  - schema mismatch
  - feature-flag regression
  - rollout regression

This introduced:

- multi-step dependencies
- diagnosis-first variants
- delayed effects
- less trivial hard task behavior

### C. Task scores moved to open interval

Hackathon feedback said exact `0.0` and `1.0` were invalid.

So exported scores were changed to stay strictly inside `(0, 1)`.

### D. Continuous grading added

The original coarse buckets were replaced with a richer weighted continuous grading model.

### E. Explicit task/grader contract added

The app was extended to expose:

- `/info`
- `/tasks`
- `/grader`
- `/baseline`

with explicit task ids and grader info.

### F. Unseeded reset discovery added

When `reset()` is called with no `seed` and no `scenario_id`, it now cycles deterministically through:

1. `high_latency_easy`
2. `service_crash_medium`
3. `bad_deployment_hard`

This was added because we suspected the validator might discover tasks by repeatedly calling plain `/reset`.

### G. Root-level shims added

Most recent compatibility change:

- root-level [tasks.py](/D:/RL/tasks.py)
- root-level [graders.py](/D:/RL/graders.py)

These are thin shims to make static task/grader discovery easier for validators that only scan root modules.

## 12. Current openenv.yaml Intent

[openenv.yaml](/D:/RL/openenv.yaml) now explicitly declares:

- the 3 public tasks
- each task’s exact grader id
- a top-level `graders` section

The grader implementations now point at root-level import paths:

- `graders:incident_response_grade_high_latency`
- `graders:incident_response_grade_service_crash`
- `graders:incident_response_grade_bad_deployment`

This was changed from package-internal import paths to make static discovery easier.

## 13. Root-Level Compatibility Files

### [tasks.py](/D:/RL/tasks.py)

Exports:

- `TASKS`
- `PUBLIC_TASKS`
- `CANONICAL_TASK_SEEDS`
- `get_public_tasks`

`TASKS` is a simple list of dicts with:

- `id`
- `name`
- `description`
- `difficulty`
- `grader`
- `num_scenarios`
- `canonical_seed`

### [graders.py](/D:/RL/graders.py)

Re-exports:

- `incident_response_grade_high_latency`
- `incident_response_grade_service_crash`
- `incident_response_grade_bad_deployment`
- `grade_episode`
- `grading_components`
- `score_state`

This file exists mainly for validator visibility.

## 14. Local Verification Status

Latest verified local checks:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\openenv.exe validate .
```

Latest observed results:

- `53 passed`
- `openenv validate .` -> `[OK] : Ready for multi-mode deployment`

These passing checks mean:

- code compiles
- tests pass
- OpenEnv validation passes locally

## 15. Git State / Important Commits

Recent important commits:

- `9f87310` `fix: add root-level task and grader shims`
- `c257c27` `fix: declare explicit graders in openenv metadata`
- `a04afd2` `fix: make task grader mappings explicit`
- `33d6200` `feat: expose five grading components per task`
- `b26755b` `feat: match validator task and grader contract`
- `2f55468` `feat: add explicit task and grader endpoints`
- `74cb1f7` `fix: expose all public tasks to validators`

Latest pushed commit expected on GitHub:

- `9f87310`

## 16. What Still Fails

Even after all the above:

- Phase 2 still says `Not enough tasks with graders`

Current belief:

- this is a hidden validator contract problem
- the validator still is not finding or counting the 3 tasks with graders from the place or shape it expects

## 17. Most Likely Remaining Hypotheses

If another engineer or LLM continues debugging this, these are the best hypotheses left:

1. the hackathon validator is reading only a very specific root-level static format
2. it may expect a file name or object name we still have not matched
3. it may ignore custom endpoints and inspect only `openenv.yaml` plus root-level Python
4. it may expect a particular top-level symbol such as `TASKS`, `GRADERS`, or explicit callable mappings
5. it may expect grader declarations in a format closer to their sample/reference implementation than current custom code

## 18. Practical Next Steps If Continuing

If continuing from here, the next best things to inspect or try are:

1. verify the live Space after rebuild:
   - `GET /info`
   - `GET /tasks`
   - `POST /grader`
   - `POST /baseline`

2. inspect live OpenAPI:
   - make sure `/tasks`, `/info`, `/grader`, `/baseline` are visible as expected

3. compare a successful hackathon Space against this one:
   - exact endpoint shapes
   - exact static file names
   - exact `openenv.yaml` task/grader format
   - exact root-level module layout

4. if still blocked, add one more explicit root-level manifest such as:
   - `graders_manifest.json`
   - `tasks_manifest.json`
   - or a top-level `GRADERS` dict in [graders.py](/D:/RL/graders.py)

5. if organizer contact is possible, ask:
   - what exact file/endpoint their Phase 2 task validation reads for “3 tasks with graders”

## 19. Useful Commands

Run tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

Local OpenEnv validation:

```powershell
.\.venv\Scripts\openenv.exe validate .
```

Live OpenEnv validation:

```powershell
.\.venv\Scripts\openenv.exe validate --url https://saransh24-incident-response-rl.hf.space
```

Run baseline:

```powershell
$env:HF_TOKEN="your_token"
$env:ENV_BASE_URL="https://saransh24-incident-response-rl.hf.space"
.\.venv\Scripts\python inference.py
```

Quick live checks:

```powershell
Invoke-RestMethod -Method Get -Uri "https://saransh24-incident-response-rl.hf.space/info" | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Get -Uri "https://saransh24-incident-response-rl.hf.space/tasks" | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "https://saransh24-incident-response-rl.hf.space/grader" -ContentType "application/json" -Body '{"task_id":"service_crash_medium","trajectory":[{"action_type":"restart_service","target":"api"}]}' | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "https://saransh24-incident-response-rl.hf.space/baseline" -ContentType "application/json" -Body '{}' | ConvertTo-Json -Depth 10
```

## 20. What Not To Re-Do

Avoid wasting time re-solving already-solved areas unless something regressed:

- do not rework the inference stdout format
- do not rework Docker unless build regresses
- do not rework Phase 1 OpenEnv compliance
- do not redesign the environment from scratch
- do not remove the redundancy around tasks and graders

The main unresolved issue is discovery by the hidden Phase 2 task validator.

## 21. Short Summary For Another LLM

If you need a short prompt to continue work, use this:

```text
You are working in D:\RL on the incident-response-rl hackathon project. Phase 1 passes. Phase 2 repeatedly fails only at “Task Validation: Not enough tasks with graders.” The project already has 3 public tasks, explicit named graders, /info, /tasks, /grader, /baseline, openenv.yaml tasks+graders, task_registry.json, and root-level tasks.py and graders.py shims. Local pytest and openenv validate pass. The problem is likely hidden validator task/grader discovery. Focus only on making the 3-task-with-grader contract impossible for a rigid static validator to miss, without breaking the current runtime API or baseline flow.
```

