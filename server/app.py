from __future__ import annotations

from fastapi import Body
from openenv.core.env_server.http_server import create_app

from incident_response_rl.env import IncidentResponseEnv
from incident_response_rl.graders import grade_episode, score_state
from incident_response_rl.models import Action, Observation
from incident_response_rl.tasks import GradeTaskRequest, GradeTaskResponse, TaskDefinition, get_public_tasks
from server.incident_response_environment import IncidentResponseEnvironment


app = create_app(
    IncidentResponseEnvironment,
    Action,
    Observation,
    env_name="incident-response-rl",
    max_concurrent_envs=4,
)


@app.get("/tasks", response_model=list[TaskDefinition], tags=["Environment Info"])
def list_tasks() -> list[TaskDefinition]:
    return get_public_tasks()


@app.post("/grader", response_model=GradeTaskResponse, tags=["Environment Control"])
def grade_task(request: GradeTaskRequest = Body(...)) -> GradeTaskResponse:
    env = IncidentResponseEnv()
    observation, _ = env.reset(seed=request.seed, scenario_id=request.task_id)
    for action in request.actions:
        observation, _, done, _ = env.step(action)
        if done:
            break

    state = env.state_data
    if state is None:
        raise RuntimeError("Environment state unavailable after grading run.")

    return GradeTaskResponse(
        task_id=request.task_id,
        difficulty=observation.difficulty,
        score=score_state(state),
        terminal_grade=grade_episode(state),
        resolved=state.resolved,
        successful_actions=list(state.successful_actions),
        steps_taken=state.step_count,
    )


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
