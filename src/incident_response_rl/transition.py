from __future__ import annotations

from copy import deepcopy

from .models import Action, ActionType, EnvironmentState, PendingEffect


def update_system_status(metrics: dict[str, float]) -> str:
    latency = metrics["latency_ms"]
    error_rate = metrics["error_rate"]
    if error_rate <= 0.04 and latency <= 160:
        return "healthy"
    if error_rate <= 0.20 and latency <= 260:
        return "degraded"
    return "critical"


def clamp_metrics(metrics: dict[str, float]) -> None:
    metrics["latency_ms"] = max(40.0, round(metrics["latency_ms"], 3))
    metrics["error_rate"] = min(1.0, max(0.0, round(metrics["error_rate"], 3)))
    metrics["cpu_pct"] = min(100.0, max(0.0, round(metrics["cpu_pct"], 3)))
    metrics["deployment_version"] = round(metrics["deployment_version"], 3)


def apply_metric_delta(metrics: dict[str, float], delta: dict[str, float]) -> None:
    for key, value in delta.items():
        metrics[key] = metrics.get(key, 0.0) + value
    clamp_metrics(metrics)


def advance_pending_effects(state: EnvironmentState) -> list[str]:
    reasons: list[str] = []
    remaining: list[PendingEffect] = []
    for effect in state.pending_effects:
        effect.remaining_delay -= 1
        if effect.remaining_delay <= 0:
            apply_metric_delta(state.metrics, effect.metrics_delta)
            state.logs.extend(effect.logs)
            state.alerts.extend(effect.alerts)
            reasons.append(effect.transition_reason or f"{effect.action_type} took effect")
        else:
            remaining.append(effect)
    state.pending_effects = remaining
    return reasons


def is_correct_next_action(state: EnvironmentState, action_type: ActionType) -> bool:
    expected_index = len(state.successful_actions)
    required = state.scenario.required_actions
    if expected_index >= len(required):
        return False
    return required[expected_index] == action_type


def action_target_matches(state: EnvironmentState, action: Action) -> bool:
    expected = state.scenario.action_targets.get(action.action_type)
    return expected is None or expected == action.target


def _worsen_for_wrong_action(state: EnvironmentState, action: Action) -> None:
    if action.action_type == "ignore":
        apply_metric_delta(state.metrics, {"latency_ms": 18.0, "error_rate": 0.03, "cpu_pct": 4.0})
        state.logs.append("WARN no mitigation applied while incident continued")
        return
    if action.action_type == "restart_service":
        apply_metric_delta(state.metrics, {"latency_ms": 18.0, "error_rate": 0.04})
        state.logs.append("WARN restart caused brief disruption without fixing root cause")
        return
    if action.action_type == "scale_up":
        apply_metric_delta(state.metrics, {"cpu_pct": 5.0, "latency_ms": 12.0})
        state.logs.append("WARN scale-up added capacity cost but did not resolve incident")
        return
    if action.action_type == "rollback_deployment":
        apply_metric_delta(state.metrics, {"latency_ms": 10.0, "error_rate": 0.02})
        state.logs.append("WARN rollback procedure failed to help in this context")
        return
    if action.action_type == "analyze_logs":
        state.logs.append("INFO additional log scan completed")


def metrics_improved(before: dict[str, float], after: dict[str, float]) -> bool:
    return (
        after["latency_ms"] < before["latency_ms"]
        or after["error_rate"] < before["error_rate"]
        or after["cpu_pct"] < before["cpu_pct"]
    )


def action_already_successful(state: EnvironmentState, action_type: ActionType) -> bool:
    return action_type in state.successful_actions


def transition(state: EnvironmentState, action: Action) -> tuple[float, list[str]]:
    reward_value = 0.0
    reasons: list[str] = []
    before_metrics = deepcopy(state.metrics)

    delayed_reasons = advance_pending_effects(state)
    reasons.extend(delayed_reasons)

    if action.action_type == "analyze_logs":
        if not state.analyzed:
            state.analyzed = True
            state.logs.extend(state.scenario.diagnosis_hints)
            reward_value += 0.2
            reasons.append("diagnosis improved")
        else:
            reward_value -= 0.1
            reasons.append("repeated ineffective action")
    elif action.action_type == "escalate":
        state.terminated_by_escalation = True
        reward_value -= 0.3
        reasons.append("early failed escalation")
    elif is_correct_next_action(state, action.action_type) and action_target_matches(state, action):
        state.successful_actions.append(action.action_type)
        reward_value += 0.3
        reasons.append("correct remedial action")
        immediate_delta = state.scenario.immediate_effects.get(action.action_type)
        if immediate_delta:
            apply_metric_delta(state.metrics, immediate_delta)
            state.logs.extend(state.scenario.improvement_logs.get(action.action_type, []))
        if action.action_type in state.scenario.delayed_effects:
            delay, delta = state.scenario.delayed_effects[action.action_type]
            effect = PendingEffect(
                action_type=action.action_type,
                remaining_delay=delay,
                metrics_delta=delta,
                logs=state.scenario.improvement_logs.get(action.action_type, []),
                transition_reason=f"{action.action_type} completed after delay",
            )
            if delay <= 0:
                apply_metric_delta(state.metrics, effect.metrics_delta)
                state.logs.extend(effect.logs)
                reasons.append(effect.transition_reason)
            else:
                state.pending_effects.append(
                    PendingEffect(
                        action_type=effect.action_type,
                        remaining_delay=effect.remaining_delay,
                        metrics_delta=effect.metrics_delta,
                        logs=effect.logs,
                        transition_reason=effect.transition_reason,
                    )
                )
        if not state.analyzed:
            reward_value += 0.2
            reasons.append("correct diagnosis")
    elif action_already_successful(state, action.action_type):
        reward_value -= 0.1
        reasons.append("repeated ineffective action")
        _worsen_for_wrong_action(state, action)
    else:
        reward_value -= 0.2
        reasons.append("wrong action")
        _worsen_for_wrong_action(state, action)

    state.system_status = update_system_status(state.metrics)
    state.partial_recovery = state.system_status != "critical"

    if metrics_improved(before_metrics, state.metrics):
        reward_value += 0.3
        reasons.append("system improved")

    if len(state.successful_actions) == len(state.scenario.required_actions) and not state.pending_effects:
        state.system_status = update_system_status(state.metrics)
        if state.system_status == "healthy":
            state.resolved = True
            reward_value += 0.5
            reasons.append("resolved issue")

    state.last_transition_reason = "; ".join(reasons) if reasons else "no-op"
    return reward_value, reasons
