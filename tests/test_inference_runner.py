from types import SimpleNamespace
from unittest.mock import Mock, patch

from incident_response_rl.inference import (
    choose_fallback_action,
    choose_runbook_action,
    create_llm_client,
    extract_successful_actions,
    format_action_for_log,
    log_end,
    log_start,
    log_step,
    parse_action_block,
    query_hf_router,
    resolve_action,
    run_baseline,
)
from incident_response_rl.models import Action, BaselineEpisodeResult, BaselineRunReport, Observation


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


def test_format_action_for_log_returns_plain_text() -> None:
    assert format_action_for_log(Action(action_type="scale_up", target="api")) == "scale_up api"
    assert format_action_for_log(Action(action_type="analyze_logs")) == "analyze_logs"


def test_create_llm_client_uses_hf_router_env() -> None:
    with patch.dict(
        "os.environ",
        {
            "HF_TOKEN": "hf_test_token",
            "API_BASE_URL": "https://router.huggingface.co/v1",
        },
        clear=False,
    ), patch("incident_response_rl.inference.OpenAI") as openai_cls:
        create_llm_client()

    openai_cls.assert_called_once_with(
        base_url="https://router.huggingface.co/v1",
        api_key="hf_test_token",
    )


def test_query_hf_router_uses_openai_client_shape() -> None:
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='[START]\n[STEP]\n{"action_type":"scale_up","target":"api"}\n[END]'))]
    )
    observation = Observation(
        logs=["WARN api latency above SLO"],
        metrics={"latency_ms": 250.0, "error_rate": 0.1, "cpu_pct": 85.0, "deployment_version": 20240401.0},
        alerts=["HighLatency(api)"],
        system_status="critical",
        step_count=0,
        scenario_id="high_latency_easy",
        difficulty="easy",
        incident_family="high_latency",
    )

    with patch.dict("os.environ", {"MODEL_NAME": "openai/gpt-oss-20b"}, clear=False):
        result = query_hf_router(fake_client, observation)

    assert '"action_type":"scale_up"' in result
    fake_client.chat.completions.create.assert_called_once()
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-20b"
    assert kwargs["temperature"] == 0
    assert kwargs["stream"] is False
    assert kwargs["extra_body"] == {"seed": 7}


def test_structured_log_helpers_emit_expected_blocks(capsys) -> None:
    log_start(task="all_tasks", env="incident-response-rl", model="openai/gpt-oss-20b")
    log_step(
        step=1,
        action=Action(action_type="scale_up", target="api"),
        reward=1.3,
        done=True,
        error=None,
    )
    log_end(
        success=True,
        steps=1,
        score=1.0,
        rewards=[1.3],
    )

    stdout = capsys.readouterr().out.strip().splitlines()
    assert stdout == [
        "[START] task=all_tasks env=incident-response-rl model=openai/gpt-oss-20b",
        "[STEP] step=1 action=scale_up api reward=1.30 done=true error=null",
        "[END] success=true steps=1 score=1.000 rewards=1.30",
    ]


def test_run_baseline_aggregates_scores() -> None:
    fake_client = Mock()
    fake_openai_client = Mock()
    fake_openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='[START]\n[STEP]\n{"action_type":"scale_up","target":"api"}\n[END]'))]
    )
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
        "incident_response_rl.inference.create_llm_client",
        return_value=fake_openai_client,
    ):
        report = run_baseline("http://127.0.0.1:8000")

    assert report.average_score == 1.0
    assert len(report.task_scores) == 3
    assert all(item.total_reward == 1.3 for item in report.task_scores)
    assert all(item.successful_actions == ["scale_up"] for item in report.task_scores)


def test_run_baseline_stdout_matches_sample_style(capsys) -> None:
    fake_client = Mock()
    fake_openai_client = Mock()
    fake_openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='[START]\n[STEP]\n{"action_type":"scale_up","target":"api"}\n[END]'))]
    )
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
            "metadata": {"info": {"successful_actions": ["scale_up"]}},
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
        "incident_response_rl.inference.create_llm_client",
        return_value=fake_openai_client,
    ):
        run_baseline("http://127.0.0.1:8000")

    stdout = capsys.readouterr().out.strip().splitlines()
    assert stdout[0] == "[START] task=all_tasks env=incident-response-rl model=openai/gpt-oss-20b"
    assert stdout[-1] == "[END] success=true steps=3 score=1.000 rewards=1.30,1.30,1.30"
    step_lines = [line for line in stdout if line.startswith("[STEP] ")]
    assert len(step_lines) == 3
    for line in step_lines:
        assert " step=1 " in line
        assert " reward=1.30 " in line
        assert " done=true " in line
        assert line.endswith("error=null")
        assert any(action in line for action in ("action=scale_up api", "action=restart_service api"))
