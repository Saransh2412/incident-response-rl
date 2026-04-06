from __future__ import annotations

from .models import EnvironmentState


def score_state(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return 1.0
    if state.terminated_by_escalation:
        return 0.0

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
    return round(min(score, 1.0), 3)


def grade_episode(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return 1.0
    if state.partial_recovery:
        return 0.5
    return 0.0
