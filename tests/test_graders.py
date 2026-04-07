from incident_response_rl.env import IncidentResponseEnv
from incident_response_rl.graders import grade_episode, score_state
from incident_response_rl.models import Action


def _run_actions(scenario_id: str, seed: int, actions: list[Action]):
    env = IncidentResponseEnv()
    env.reset(seed=seed, scenario_id=scenario_id)
    for action in actions:
        env.step(action)
    state = env.state_data
    assert state is not None
    return state


def test_grader_returns_open_interval_scores() -> None:
    state = _run_actions("high_latency_easy", 1, [])
    assert 0.0 < score_state(state) < 1.0
    assert 0.0 < grade_episode(state) < 1.0


def test_perfect_direct_recovery_scores_high_but_not_perfect() -> None:
    state = _run_actions(
        "high_latency_easy",
        5,
        [Action(action_type="scale_up", target="api")],
    )
    assert state.system_status == "healthy"
    assert 0.90 <= grade_episode(state) < 0.99


def test_perfect_diagnosis_first_recovery_can_reach_ceiling() -> None:
    state = _run_actions(
        "bad_deployment_hard",
        1,
        [
            Action(action_type="analyze_logs"),
            Action(action_type="rollback_deployment", target="api"),
            Action(action_type="restart_service", target="api"),
        ],
    )
    assert state.system_status == "healthy"
    assert grade_episode(state) == 0.99


def test_partial_recovery_scores_midrange_not_bucketed_half() -> None:
    state = _run_actions(
        "bad_deployment_hard",
        2,
        [Action(action_type="rollback_deployment", target="api")],
    )
    assert state.system_status == "degraded"
    assert 0.55 <= grade_episode(state) <= 0.75


def test_inefficient_but_successful_path_scores_below_perfect() -> None:
    state = _run_actions(
        "high_latency_easy",
        3,
        [
            Action(action_type="scale_up", target="api"),
            Action(action_type="analyze_logs"),
            Action(action_type="scale_up", target="api"),
        ],
    )
    assert state.system_status == "healthy"
    assert 0.90 <= grade_episode(state) < 0.99


def test_repeated_wrong_actions_score_low() -> None:
    state = _run_actions(
        "service_crash_medium",
        3,
        [
            Action(action_type="scale_up", target="api"),
            Action(action_type="scale_up", target="api"),
        ],
    )
    assert state.system_status == "critical"
    assert 0.01 <= grade_episode(state) <= 0.25


def test_escalation_failure_scores_near_floor() -> None:
    state = _run_actions(
        "service_crash_medium",
        3,
        [Action(action_type="escalate")],
    )
    assert state.terminated_by_escalation is True
    assert 0.01 <= grade_episode(state) <= 0.20
