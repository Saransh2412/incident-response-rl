from incident_response_rl.tasks import CANONICAL_TASK_SEEDS, PUBLIC_TASKS, get_public_tasks

TASKS = [
    {
        "id": task.id,
        "name": task.name,
        "description": task.description,
        "difficulty": task.difficulty,
        "grader": task.grader,
        "num_scenarios": task.num_scenarios,
        "canonical_seed": CANONICAL_TASK_SEEDS[task.id],
    }
    for task in PUBLIC_TASKS
]

TASKS_WRAPPED = {"tasks": TASKS}

__all__ = ["CANONICAL_TASK_SEEDS", "PUBLIC_TASKS", "TASKS", "TASKS_WRAPPED", "get_public_tasks"]
