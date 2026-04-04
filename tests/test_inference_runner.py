from unittest.mock import Mock, patch

from incident_response_rl.inference import (
    choose_fallback_action,
    choose_runbook_action,
    extract_successful_actions,
    parse_action_block,
    resolve_action,
    run_baseline,
)
from incident_response_rl.models import Observation


def test_parse_action_block_accepts_strict_format() -> None:
    action = parse_action_block(
        '[START]\n[STEP]\n{"action_type":"restart_service","target":"api"}\n[END]'
    )
    assert action.action_type == "restart_service"
    assert action.target == "api"


def test_parse_action_block_normalizes_aliases() -> None:
    action = parse_action_block(
        '[START]\n[STEP]\n{"action_type":"rollback","target":"api"}\n[END]'
    )
    assert action.action_type == "rollback_deployment"


def test_fallback_action_uses_observation_signals() -> None:
    action = choose_fallback_action(
        Observation(
            logs=["ERROR api process exited with signal SIGKILL"],
            metrics={"latency_ms": 210.0, "error_rate": 0.4, "cpu_pct": 30.0, "deployment_version": 20240401.0},
            alerts=["CrashLoop(api)"],
            system_status="critical",
            step_count=0,
            scenario_id="service_crash_medium",
            difficulty="medium",
            incident_family="service_crash",
        )
    )
    assert action.action_type == "restart_service"


def test_runbook_prefers_rollback_then_restart() -> None:
    rollback_action = choose_runbook_action(
        Observation(
            logs=["INFO deployed api version=20240402", "WARN latency regression detected after deployment"],
            metrics={"latency_ms": 280.0, "error_rate": 0.2, "cpu_pct": 60.0, "deployment_version": 20240402.0},
            alerts=["DeploymentRegression(api)"],
            system_status="critical",
            step_count=0,
            scenario_id="bad_deployment_hard",
            difficulty="hard",
            incident_family="bad_deployment",
        )
    )
    assert rollback_action is not None
    assert rollback_action.action_type == "rollback_deployment"

    restart_action = choose_runbook_action(
        Observation(
            logs=["INFO rollback completed to previous stable release"],
            metrics={"latency_ms": 210.0, "error_rate": 0.12, "cpu_pct": 55.0, "deployment_version": 20240401.0},
            alerts=["HighErrorRate(api)"],
            system_status="degraded",
            step_count=1,
            scenario_id="bad_deployment_hard",
            difficulty="hard",
            incident_family="bad_deployment",
        )
    )
    assert restart_action is not None
    assert restart_action.action_type == "restart_service"


def test_parse_action_block_defaults_target_for_service_actions() -> None:
    action = parse_action_block(
        '[START]\n[STEP]\n{"action_type":"scale_up","target":null}\n[END]'
    )
    assert action.action_type == "scale_up"
    assert action.target == "api"


def test_resolve_action_overrides_weak_model_choice_with_runbook() -> None:
    observation = Observation(
        logs=["WARN api latency above SLO for /checkout requests", "WARN CPU saturation detected on api nodes"],
        metrics={"latency_ms": 269.0, "error_rate": 0.1, "cpu_pct": 86.0, "deployment_version": 20240401.0},
        alerts=["HighLatency(api)", "CpuHot(api)"],
        system_status="critical",
        step_count=0,
        scenario_id="high_latency_easy",
        difficulty="easy",
        incident_family="high_latency",
    )
    action = resolve_action(
        observation,
        '[START]\n[STEP]\n{"action_type":"analyze_logs","target":null}\n[END]',
    )
    assert action.action_type == "scale_up"


def test_extract_successful_actions_reads_nested_info() -> None:
    observation = Observation(
        logs=[],
        metrics={},
        alerts=[],
        system_status="healthy",
        step_count=1,
        scenario_id="service_crash_medium",
        difficulty="medium",
        incident_family="service_crash",
        metadata={"info": {"successful_actions": ["restart_service"]}},
    )
    assert extract_successful_actions(observation) == ["restart_service"]


def test_run_baseline_aggregates_scores() -> None:
    fake_client = Mock()
    reset_response = Mock()
    reset_response.json.return_value = {
        "observation": {
            "logs": [],
            "metrics": {},
            "alerts": [],
            "system_status": "critical",
            "step_count": 0,
            "scenario_id": "high_latency_easy",
            "difficulty": "easy",
            "incident_family": "high_latency",
            "last_transition_reason": "incident injected",
            "task_score": 0.0,
            "terminal_grade": None,
            "done": False,
            "reward": None,
            "metadata": {},
        }
    }
    state_response = Mock()
    state_response.json.return_value = {"episode_id": "ep-1", "step_count": 0}
    step_response = Mock()
    step_response.json.return_value = {
        "observation": {
            "logs": [],
            "metrics": {},
            "alerts": [],
            "system_status": "healthy",
            "step_count": 1,
            "scenario_id": "high_latency_easy",
            "difficulty": "easy",
            "incident_family": "high_latency",
            "last_transition_reason": "resolved issue",
            "task_score": 1.0,
            "terminal_grade": 1.0,
            "metadata": {
                "info": {
                    "successful_actions": ["scale_up"],
                },
            },
        },
        "done": True,
        "reward": 1.3,
    }
    fake_client.post.side_effect = [reset_response, step_response, reset_response, step_response, reset_response, step_response]
    fake_client.get.side_effect = [state_response, state_response, state_response]

    context_manager = Mock()
    context_manager.__enter__ = Mock(return_value=fake_client)
    context_manager.__exit__ = Mock(return_value=False)

    with patch("incident_response_rl.inference.httpx.Client", return_value=context_manager), patch(
        "incident_response_rl.inference.query_hf_router",
        return_value='[START]\n[STEP]\n{"action_type":"scale_up","target":"api"}\n[END]',
    ):
        report = run_baseline("http://127.0.0.1:8000")

    assert report.average_score == 1.0
    assert len(report.task_scores) == 3
    assert all(item.total_reward == 1.3 for item in report.task_scores)
    assert all(item.successful_actions == ["scale_up"] for item in report.task_scores)
