from __future__ import annotations

from .models import EnvironmentState


def score_state(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return 1.0
    if state.terminated_by_escalation:
        return 0.0

    score = 0.0
    if state.analyzed:
        score += 0.2
    if state.successful_actions:
        score += 0.3 * min(len(state.successful_actions), len(state.scenario.required_actions)) / len(
            state.scenario.required_actions
        )
    if state.partial_recovery:
        score += 0.3
    if state.system_status == "healthy":
        score += 0.2
    return round(min(score, 1.0), 3)


def grade_episode(state: EnvironmentState) -> float:
    if state.resolved and state.system_status == "healthy":
        return 1.0
    if state.partial_recovery:
        return 0.5
    return 0.0
