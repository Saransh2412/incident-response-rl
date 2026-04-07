from __future__ import annotations

from fastapi import Body
from openenv.core.env_server.http_server import create_app

from incident_response_rl.env import IncidentResponseEnv
from incident_response_rl.graders import grade_episode, score_state
from incident_response_rl.models import Action, Observation
from incident_response_rl.inference import choose_fallback_action, choose_runbook_action
from incident_response_rl.tasks import (
    BaselineRequest,
    BaselineResponse,
    CANONICAL_TASK_SEEDS,
    EnvironmentInfo,
    GraderRequest,
    GraderResponse,
    GraderResult,
    TaskInfo,
    TasksResponse,
    get_public_tasks,
    get_task_info,
)
from server.incident_response_environment import IncidentResponseEnvironment


app = create_app(
    IncidentResponseEnvironment,
    Action,
    Observation,
    env_name="incident-response-rl",
    max_concurrent_envs=4,
)


def _action_space_schema() -> dict:
    return Action.model_json_schema()


def _observation_space_schema() -> dict:
    return Observation.model_json_schema()


def _canonical_seed(task_id: str, seed: int | None) -> int:
    return CANONICAL_TASK_SEEDS[task_id] if seed is None else seed


def _feedback_for_result(score: float, resolved: bool, steps_taken: int) -> str:
    if resolved and score >= 0.95:
        return "Excellent - task solved efficiently."
    if resolved and score >= 0.80:
        return "Strong recovery - resolved with some inefficiency."
    if score >= 0.40:
        return "Partial progress - incident improved but was not handled optimally."
    if steps_taken == 0:
        return "No actions taken."
    return "Needs improvement - recovery path was incomplete or incorrect."


def _grade_task_result(task_id: str, seed: int | None, trajectory: list[dict]) -> GraderResult:
    env = IncidentResponseEnv()
    resolved_task = get_task_info(task_id)
    observation, _ = env.reset(seed=_canonical_seed(task_id, seed), scenario_id=task_id)
    for action_payload in trajectory:
        action = Action.model_validate(action_payload)
        observation, _, done, _ = env.step(action)
        if done:
            break

    state = env.state_data
    if state is None:
        raise RuntimeError("Environment state unavailable after grading run.")

    score = score_state(state)
    terminal_grade = grade_episode(state)
    breakdown = {
        "task_score": round(score, 3),
        "terminal_grade": round(terminal_grade, 3),
        "resolved_bonus": 0.15 if state.resolved else 0.0,
        "efficiency_signal": round(max(0.0, 1.0 - (state.step_count / max(1, state.scenario.max_steps))), 3),
    }
    return GraderResult(
        task_id=resolved_task.id,
        score=score,
        breakdown=breakdown,
        feedback=_feedback_for_result(score, state.resolved, state.step_count),
        steps_taken=state.step_count,
    )


def _run_baseline_task(task_id: str, seed: int) -> GraderResult:
    env = IncidentResponseEnv()
    observation, _ = env.reset(seed=seed, scenario_id=task_id)
    trajectory: list[dict] = []

    while not observation.done:
        action = choose_runbook_action(observation) or choose_fallback_action(observation)
        trajectory.append(action.model_dump(exclude={"metadata"}))
        observation, _, done, _ = env.step(action)
        if done:
            break

    return _grade_task_result(task_id, seed, trajectory)


@app.get("/info", response_model=EnvironmentInfo, tags=["Environment Info"])
def get_info() -> EnvironmentInfo:
    return EnvironmentInfo(
        tasks=get_public_tasks(),
        max_steps=10,
        action_space=_action_space_schema(),
        observation_space=_observation_space_schema(),
    )


@app.get("/tasks", response_model=TasksResponse, tags=["Environment Info"])
def list_tasks() -> TasksResponse:
    return TasksResponse(tasks=get_public_tasks())


@app.post("/grader", response_model=GraderResponse, tags=["Environment Control"])
def grade_task(request: GraderRequest = Body(...)) -> GraderResponse:
    return GraderResponse(result=_grade_task_result(request.task_id, request.seed, request.trajectory))


@app.post("/baseline", response_model=BaselineResponse, tags=["Environment Control"])
def run_baseline_endpoint(request: BaselineRequest | None = Body(default=None)) -> BaselineResponse:
    request = request or BaselineRequest()
    task_ids = [request.task_id] if request.task_id else [task.id for task in get_public_tasks()]
    results: list[GraderResult] = []
    for _ in range(max(1, request.num_episodes)):
        for task_id in task_ids:
            results.append(_run_baseline_task(task_id, CANONICAL_TASK_SEEDS[task_id]))

    aggregate_score = round(sum(result.score for result in results) / max(1, len(results)), 4)
    return BaselineResponse(results=results, aggregate_score=aggregate_score)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
