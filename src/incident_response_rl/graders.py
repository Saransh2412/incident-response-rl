from __future__ import annotations

from .models import EnvironmentState

MIN_EXPORTED_SCORE = 0.01
MAX_EXPORTED_SCORE = 0.99


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


def _outcome_score(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return 1.0
    if state.system_status == "healthy":
        return 0.9
    if state.system_status == "degraded":
        return 0.55
    return 0.15


def _raw_terminal_grade(state: EnvironmentState) -> float:
    diagnosis = _diagnosis_score(state)
    sequence = _sequence_completion(state)
    efficiency = _efficiency_score(state)
    outcome = _outcome_score(state)
    weighted = (0.20 * diagnosis) + (0.35 * sequence) + (0.20 * efficiency) + (0.25 * outcome)
    return min(max(weighted, 0.0), 1.0)


def score_state(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return MAX_EXPORTED_SCORE
    if state.terminated_by_escalation:
        return MIN_EXPORTED_SCORE

    diagnosis = 0.20 * _diagnosis_score(state)
    sequence = 0.35 * _sequence_completion(state)
    recovery = 0.25 if state.partial_recovery else 0.0
    healthy_bonus = 0.10 if state.system_status == "healthy" else 0.0
    score = min(0.95, diagnosis + sequence + recovery + healthy_bonus)
    return _open_interval(score)


def grade_episode(state: EnvironmentState) -> float:
    return _open_interval(_raw_terminal_grade(state))
