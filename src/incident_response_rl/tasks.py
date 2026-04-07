from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .models import Action, Difficulty


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    difficulty: Difficulty
    objective: str
    grader: str


class GradeTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    seed: int | None = None
    actions: list[Action]


class GradeTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    score: float
    terminal_grade: float
    resolved: bool
    successful_actions: list[str]
    steps_taken: int


PUBLIC_TASKS = [
    TaskDefinition(
        id="high_latency_easy",
        difficulty="easy",
        objective="Restore API latency caused by capacity pressure or burst amplification.",
        grader="grade_episode",
    ),
    TaskDefinition(
        id="service_crash_medium",
        difficulty="medium",
        objective="Recover a crash-looping API service while handling diagnosis-first variants.",
        grader="grade_episode",
    ),
    TaskDefinition(
        id="bad_deployment_hard",
        difficulty="hard",
        objective="Undo a bad deployment and fully restore service health through the required sequence.",
        grader="grade_episode",
    ),
]


def get_public_tasks() -> list[TaskDefinition]:
    return [task.model_copy(deep=True) for task in PUBLIC_TASKS]

