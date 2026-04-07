from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import Action, Difficulty


CANONICAL_TASK_SEEDS = {
    "high_latency_easy": 5,
    "service_crash_medium": 3,
    "bad_deployment_hard": 1,
}


class TaskInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    difficulty: Difficulty
    num_scenarios: int


class TasksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[TaskInfo]


class EnvironmentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "incident-response-rl"
    version: str = "1.0.0"
    description: str = "Troubleshoot and remediate production incidents in a realistic service environment."
    tasks: list[TaskInfo]
    max_steps: int = 10
    action_space: dict
    observation_space: dict


class GraderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    trajectory: list[dict] = Field(default_factory=list)
    seed: int | None = None


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    score: float
    max_score: float = 1.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    steps_taken: int
    hints_used: int = 0


class GraderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: GraderResult


class BaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    num_episodes: int = 1


class BaselineResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[GraderResult]
    aggregate_score: float


PUBLIC_TASKS = [
    TaskInfo(
        id="high_latency_easy",
        name="API Latency Recovery",
        description="Restore API latency caused by capacity pressure or burst amplification.",
        difficulty="easy",
        num_scenarios=3,
    ),
    TaskInfo(
        id="service_crash_medium",
        name="Crash Loop Recovery",
        description="Recover a crash-looping API service while handling diagnosis-first variants.",
        difficulty="medium",
        num_scenarios=3,
    ),
    TaskInfo(
        id="bad_deployment_hard",
        name="Bad Deployment Remediation",
        description="Undo a bad deployment and fully restore service health through the required sequence.",
        difficulty="hard",
        num_scenarios=3,
    ),
]


def get_public_tasks() -> list[TaskInfo]:
    return [task.model_copy(deep=True) for task in PUBLIC_TASKS]


def get_task_info(task_id: str) -> TaskInfo:
    for task in PUBLIC_TASKS:
        if task.id == task_id:
            return task.model_copy(deep=True)
    valid = ", ".join(task.id for task in PUBLIC_TASKS)
    raise ValueError(f"Unknown task_id '{task_id}'. Valid tasks: {valid}")
