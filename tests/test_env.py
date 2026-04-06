from incident_response_rl.env import IncidentResponseEnv
from incident_response_rl.models import Action
from incident_response_rl.reward import grade_final_state


def test_reset_is_reproducible() -> None:
    env_a = IncidentResponseEnv()
    env_b = IncidentResponseEnv()
    obs_a, info_a = env_a.reset(seed=42, scenario_id="high_latency_easy")
    obs_b, info_b = env_b.reset(seed=42, scenario_id="high_latency_easy")
    assert obs_a.model_dump() == obs_b.model_dump()
    assert info_a["scenario_id"] == info_b["scenario_id"]


def test_observation_schema_and_validation() -> None:
    env = IncidentResponseEnv()
    obs, _ = env.reset(seed=1, scenario_id="service_crash_medium")
    assert obs.system_status in {"degraded", "critical"}
    assert obs.done is False
    Action(action_type="restart_service", target="api")


def test_high_latency_happy_path_resolves() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=5, scenario_id="high_latency_easy")
    result = env.step_result(Action(action_type="scale_up", target="api"))
    assert result.done is True
    assert result.observation.system_status == "healthy"
    assert result.info["terminal_grade"] == 1.0


def test_service_crash_happy_path_resolves() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=3, scenario_id="service_crash_medium")
    result = env.step_result(Action(action_type="restart_service", target="api"))
    assert result.done is True
    assert result.observation.system_status == "healthy"


def test_wrong_action_regresses_or_fails_to_improve() -> None:
    env = IncidentResponseEnv()
    initial_obs, _ = env.reset(seed=5, scenario_id="service_crash_medium")
    result = env.step_result(Action(action_type="scale_up", target="api"))
    assert result.reward.value < 0
    assert result.observation.metrics["error_rate"] >= initial_obs.metrics["error_rate"]


def test_hard_bad_deployment_requires_rollback_then_restart() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=2, scenario_id="bad_deployment_hard")
    first = env.step_result(Action(action_type="rollback_deployment", target="api"))
    assert first.done is False
    second = env.step_result(Action(action_type="restart_service", target="api"))
    assert second.done is True
    assert second.observation.system_status == "healthy"
    assert second.info["terminal_grade"] == 1.0


def test_hard_bad_deployment_failing_seed_now_recovers() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=1, scenario_id="bad_deployment_hard")
    diagnosis = env.step_result(Action(action_type="analyze_logs"))
    assert diagnosis.done is False
    first = env.step_result(Action(action_type="rollback_deployment", target="api"))
    assert first.done is False
    second = env.step_result(Action(action_type="restart_service", target="api"))
    assert second.done is True
    assert second.observation.system_status == "healthy"
    assert second.info["terminal_grade"] == 1.0


def test_high_latency_variant_can_require_diagnosis_before_scale() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=3, scenario_id="high_latency_easy")
    premature = env.step_result(Action(action_type="scale_up", target="api"))
    assert premature.done is False
    assert premature.reward.value < 0
    diagnosed = env.step_result(Action(action_type="analyze_logs"))
    assert diagnosed.reward.value > 0
    resolved = env.step_result(Action(action_type="scale_up", target="api"))
    assert resolved.done is True
    assert resolved.observation.system_status == "healthy"


def test_service_crash_variant_requires_diagnosis_before_restart() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=4, scenario_id="service_crash_medium")
    premature = env.step_result(Action(action_type="restart_service", target="api"))
    assert premature.done is False
    assert premature.reward.value < 0
    diagnosed = env.step_result(Action(action_type="analyze_logs"))
    assert diagnosed.reward.value > 0
    resolved = env.step_result(Action(action_type="restart_service", target="api"))
    assert resolved.done is True
    assert resolved.observation.system_status == "healthy"


def test_noise_injection_is_seeded() -> None:
    env = IncidentResponseEnv()
    obs, info = env.reset(seed=9, scenario_id="bad_deployment_hard")
    assert info["noise_present"] is True
    assert any("cache miss ratio" in line for line in obs.logs)


def test_reward_and_partial_recovery_grading() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=2, scenario_id="bad_deployment_hard")
    first = env.step_result(Action(action_type="rollback_deployment", target="api"))
    assert first.reward.value > 0
    state = env.state_data
    assert state is not None
    assert grade_final_state(state) == 0.5


def test_discrete_action_mapping_works() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=5, scenario_id="high_latency_easy")
    _, reward, done, info = env.step(3)
    assert reward > 0
    assert done is True
    assert info["terminal_grade"] == 1.0


def test_seeded_variants_change_surface_signals() -> None:
    env_a = IncidentResponseEnv()
    env_b = IncidentResponseEnv()
    obs_a, _ = env_a.reset(seed=1, scenario_id="bad_deployment_hard")
    obs_b, _ = env_b.reset(seed=3, scenario_id="bad_deployment_hard")
    assert obs_a.logs != obs_b.logs
    assert obs_a.alerts != obs_b.alerts
