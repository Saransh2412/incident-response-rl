from __future__ import annotations

from .graders import grade_episode
from .models import EnvironmentState, Reward


def build_reward(value: float, reasons: list[str]) -> Reward:
    text = "; ".join(reasons) if reasons else "no significant change"
    return Reward(value=round(value, 3), reason=text)


def grade_final_state(state: EnvironmentState) -> float:
    return grade_episode(state)
