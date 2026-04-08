from __future__ import annotations

from .models import EnvironmentState

MIN_EXPORTED_SCORE = 0.01
MAX_EXPORTED_SCORE = 0.99
COMPONENT_WEIGHTS = {
    "diagnosis": 0.20,
    "sequence": 0.25,
    "effectiveness": 0.25,
    "efficiency": 0.15,
    "safety": 0.15,
}

GRADERS = {
    "high_latency_easy": "incident_response_grade_high_latency",
    "service_crash_medium": "incident_response_grade_service_crash",
    "bad_deployment_hard": "incident_response_grade_bad_deployment",
}


def _open_interval(score: float) -> float:
    if score <= 0.0:
        return MIN_EXPORTED_SCORE
    if score >= 1.0:
        return MAX_EXPORTED_SCORE
    return round(score, 3)


def _required_actions_without_diagnosis(state: EnvironmentState) -> list[str]:
    return [action for action in state.scenario.required_actions if action != "analyze_logs"]


def _successful_actions_without_diagnosis(state: EnvironmentState) -> list[str]:
    return [action for action in state.successful_actions if action != "analyze_logs"]


def _sequence_completion(state: EnvironmentState) -> float:
    required = _required_actions_without_diagnosis(state)
    if not required:
        return 1.0
    completed = _successful_actions_without_diagnosis(state)
    return min(len(completed), len(required)) / len(required)


def _sequence_score(state: EnvironmentState) -> float:
    required = state.scenario.required_actions
    if not required:
        return 1.0

    matched = 0
    history = [action.action_type for action in state.action_history]
    for expected in required:
        if matched < len(history) and history[matched] == expected:
            matched += 1
        elif expected in history[matched + 1 :]:
            break
    return matched / len(required)


def _wrong_action_count(state: EnvironmentState) -> int:
    return len(
        [
            action
            for action in state.action_history
            if action.action_type not in state.scenario.required_actions and action.action_type != "analyze_logs"
        ]
    )


def _repeated_action_count(state: EnvironmentState) -> int:
    seen: set[str] = set()
    repeated = 0
    for action in state.action_history:
        label = f"{action.action_type}:{action.target}"
        if label in seen:
            repeated += 1
        else:
            seen.add(label)
    return repeated


def _diagnosis_score(state: EnvironmentState) -> float:
    diagnosis_required = "analyze_logs" in state.scenario.required_actions
    if diagnosis_required:
        return 1.0 if state.analyzed and "analyze_logs" in state.successful_actions else 0.0
    return 1.0 if state.analyzed else 0.6


def _efficiency_score(state: EnvironmentState) -> float:
    max_steps = max(1, state.scenario.max_steps)
    step_budget_score = max(0.0, 1.0 - ((state.step_count - len(state.successful_actions)) / max_steps))
    wrong_penalty = min(0.7, 0.2 * _wrong_action_count(state))
    repeat_penalty = min(0.5, 0.08 * _repeated_action_count(state))
    escalation_penalty = 0.7 if state.terminated_by_escalation else 0.0
    return max(0.0, min(1.0, step_budget_score - wrong_penalty - repeat_penalty - escalation_penalty))


def _effectiveness_score(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return 1.0
    if state.system_status == "healthy":
        return 0.9
    if state.system_status == "degraded":
        return 0.55
    return 0.15


def _safety_score(state: EnvironmentState) -> float:
    wrong_penalty = min(0.7, 0.25 * _wrong_action_count(state))
    escalation_penalty = 0.8 if state.terminated_by_escalation else 0.0
    diagnosis_required = "analyze_logs" in state.scenario.required_actions
    premature_restart_penalty = 0.0
    if diagnosis_required and state.action_history:
        first_action = state.action_history[0].action_type
        if first_action != "analyze_logs":
            premature_restart_penalty = 0.25
    return max(0.0, min(1.0, 1.0 - wrong_penalty - escalation_penalty - premature_restart_penalty))


def grading_components(state: EnvironmentState) -> dict[str, float]:
    components = {
        "diagnosis": _diagnosis_score(state),
        "sequence": _sequence_score(state),
        "effectiveness": _effectiveness_score(state),
        "efficiency": _efficiency_score(state),
        "safety": _safety_score(state),
    }
    return {name: round(min(max(value, 0.0), 1.0), 3) for name, value in components.items()}


def _raw_terminal_grade(state: EnvironmentState) -> float:
    components = grading_components(state)
    weighted = sum(COMPONENT_WEIGHTS[name] * components[name] for name in COMPONENT_WEIGHTS)
    return min(max(weighted, 0.0), 1.0)


def score_state(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return MAX_EXPORTED_SCORE
    if state.terminated_by_escalation:
        return MIN_EXPORTED_SCORE

    components = grading_components(state)
    score = (
        (COMPONENT_WEIGHTS["diagnosis"] * components["diagnosis"])
        + (COMPONENT_WEIGHTS["sequence"] * components["sequence"])
        + (COMPONENT_WEIGHTS["effectiveness"] * components["effectiveness"])
        + (0.05 if state.partial_recovery else 0.0)
    )
    score = min(0.95, score)
    return _open_interval(score)


def grade_episode(state: EnvironmentState) -> float:
    return _open_interval(_raw_terminal_grade(state))


def incident_response_grade_high_latency(state: EnvironmentState) -> float:
    return grade_episode(state)


def incident_response_grade_service_crash(state: EnvironmentState) -> float:
    return grade_episode(state)


def incident_response_grade_bad_deployment(state: EnvironmentState) -> float:
    return grade_episode(state)


def grade(task_id: str, state: EnvironmentState) -> float:
    if task_id == "high_latency_easy":
        return incident_response_grade_high_latency(state)
    if task_id == "service_crash_medium":
        return incident_response_grade_service_crash(state)
    if task_id == "bad_deployment_hard":
        return incident_response_grade_bad_deployment(state)
    valid = ", ".join(GRADERS)
    raise ValueError(f"Unknown task_id '{task_id}'. Valid tasks: {valid}")


def grade_detailed(task_id: str, state: EnvironmentState) -> dict[str, float | dict[str, float]]:
    components = grading_components(state)
    return {
        "task_id": task_id,
        "score": grade(task_id, state),
        "breakdown": components,
    }
