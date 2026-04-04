from unittest.mock import Mock, patch

from incident_response_rl.inference import parse_action_block, run_baseline


def test_parse_action_block_accepts_strict_format() -> None:
    action = parse_action_block(
        '[START]\n[STEP]\n{"action_type":"restart_service","target":"api"}\n[END]'
    )
    assert action.action_type == "restart_service"
    assert action.target == "api"


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
            "done": True,
            "reward": 1.3,
            "metadata": {
                "successful_actions": ["scale_up"],
            },
        }
    }
    fake_client.post.side_effect = [reset_response, step_response, reset_response, step_response, reset_response, step_response]

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
