from __future__ import annotations

import random
import threading
from typing import Any

from .graders import grade_episode, score_state
from .models import Action, EnvironmentState, IncidentState, Observation, StepResult
from .reward import build_reward
from .scenarios import SCENARIO_IDS, choose_scenario
from .transition import transition, update_system_status


ACTION_MAP = {
    0: Action(action_type="analyze_logs"),
    1: Action(action_type="restart_service", target="api"),
    2: Action(action_type="rollback_deployment", target="api"),
    3: Action(action_type="scale_up", target="api"),
    4: Action(action_type="ignore"),
    5: Action(action_type="escalate"),
}

_DISCOVERY_LOCK = threading.Lock()
_DISCOVERY_INDEX = 0


class IncidentResponseEnv:
    def __init__(self) -> None:
        self._rng = random.Random()
        self.state_data: EnvironmentState | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        scenario_id: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        options = options or {}
        selected_scenario_id = scenario_id or options.get("scenario_id")

        if selected_scenario_id is not None:
            chosen_seed = 0 if seed is None else seed
            self._rng = random.Random(chosen_seed)
            scenario = choose_scenario(selected_scenario_id, self._rng)
        elif seed is not None:
            self._rng = random.Random(seed)
            scenario = choose_scenario(None, self._rng)
        else:
            global _DISCOVERY_INDEX
            with _DISCOVERY_LOCK:
                scenario_id_for_reset = SCENARIO_IDS[_DISCOVERY_INDEX % len(SCENARIO_IDS)]
                _DISCOVERY_INDEX += 1
            self._rng = random.Random()
            scenario = choose_scenario(scenario_id_for_reset, self._rng)

        logs = list(scenario.initial_logs)
        alerts = list(scenario.initial_alerts)
        noisy = False
        if scenario.misleading_logs:
            logs.extend(scenario.misleading_logs)
            noisy = True
        if scenario.false_alerts:
            alerts.extend(scenario.false_alerts)
            noisy = True

        metrics = dict(scenario.initial_metrics)
        system_status = update_system_status(metrics)
        self.state_data = EnvironmentState(
            scenario=scenario,
            metrics=metrics,
            logs=logs,
            alerts=alerts,
            system_status=system_status,  # type: ignore[arg-type]
            noisy=noisy,
        )
        observation = self.observe()
        return observation, self.info()

    def step(self, action: int | Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.state_data is None:
            raise RuntimeError("Environment must be reset before stepping.")
        state = self.state_data
        resolved_action = ACTION_MAP[action] if isinstance(action, int) else action
        state.step_count += 1
        state.action_history.append(resolved_action)

        reward_value, reasons = transition(state, resolved_action)
        state.total_reward = round(state.total_reward + reward_value, 3)
        done = (
            state.resolved
            or state.terminated_by_escalation
            or state.step_count >= state.scenario.max_steps
        )
        observation = self.observe(done=done, reward=reward_value)
        info = self.info(done=done)
        return observation, reward_value, done, info

    def observe(self, done: bool | None = None, reward: float | None = None) -> Observation:
        if self.state_data is None:
            raise RuntimeError("Environment must be reset before reading state.")
        state = self.state_data
        task_score = score_state(state)
        terminal_grade = grade_episode(state) if done else None
        return Observation(
            logs=list(state.logs[-12:]),
            metrics=dict(state.metrics),
            alerts=list(state.alerts[-6:]),
            system_status=state.system_status,
            step_count=state.step_count,
            scenario_id=state.scenario.scenario_id,
            difficulty=state.scenario.difficulty,
            incident_family=state.scenario.family,
            last_transition_reason=state.last_transition_reason,
            task_score=task_score,
            terminal_grade=terminal_grade,
            done=state.resolved if done is None else done,
            reward=reward,
            metadata={
                "successful_actions": list(state.successful_actions),
            },
        )

    def step_result(self, action: Action) -> StepResult:
        observation, reward_value, done, info = self.step(action)
        reward_reason = info["last_transition_reason"]
        return StepResult(
            observation=observation,
            reward=build_reward(reward_value, [reward_reason]),
            done=done,
            info=info,
        )

    @property
    def state(self) -> IncidentState:
        if self.state_data is None:
            raise RuntimeError("Environment must be reset before reading state.")
        state = self.state_data
        done = (
            state.resolved
            or state.terminated_by_escalation
            or state.step_count >= state.scenario.max_steps
        )
        terminal_grade = grade_episode(state) if done else None
        return IncidentState(
            episode_id=state.scenario.scenario_id,
            step_count=state.step_count,
            scenario_id=state.scenario.scenario_id,
            incident_family=state.scenario.family,
            difficulty=state.scenario.difficulty,
            logs=list(state.logs[-12:]),
            metrics=dict(state.metrics),
            alerts=list(state.alerts[-6:]),
            system_status=state.system_status,
            last_transition_reason=state.last_transition_reason,
            successful_actions=list(state.successful_actions),
            partial_recovery=state.partial_recovery,
            noise_present=state.noisy,
            terminal_grade=terminal_grade,
        )

    def render_state(self) -> dict[str, Any]:
        return {"observation": self.observe().model_dump(), "state": self.state.model_dump()}

    def info(self, done: bool = False) -> dict[str, Any]:
        if self.state_data is None:
            return {}
        state = self.state_data
        return {
            "scenario_id": state.scenario.scenario_id,
            "incident_family": state.scenario.family,
            "difficulty": state.scenario.difficulty,
            "active_incident_id": state.scenario.scenario_id,
            "last_transition_reason": state.last_transition_reason,
            "partial_recovery": state.partial_recovery,
            "noise_present": state.noisy,
            "successful_actions": list(state.successful_actions),
            "max_steps": state.scenario.max_steps,
            "available_scenarios": list(SCENARIO_IDS),
            "task_score": score_state(state),
            "terminal_grade": grade_episode(state) if done else None,
            "total_reward": state.total_reward,
        }
