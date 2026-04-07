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


def score_state(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return MAX_EXPORTED_SCORE
    if state.terminated_by_escalation:
        return MIN_EXPORTED_SCORE

    score = 0.0
    if state.analyzed:
        score += 0.15
    required_remedial = [action for action in state.scenario.required_actions if action != "analyze_logs"]
    completed_remedial = [action for action in state.successful_actions if action != "analyze_logs"]
    if required_remedial:
        score += 0.35 * min(len(completed_remedial), len(required_remedial)) / len(required_remedial)
    if state.partial_recovery:
        score += 0.25
    if state.system_status == "healthy":
        score += 0.15
    return _open_interval(min(score, 1.0))


def grade_episode(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return MAX_EXPORTED_SCORE
    if state.partial_recovery:
        return 0.5
    return MIN_EXPORTED_SCORE
