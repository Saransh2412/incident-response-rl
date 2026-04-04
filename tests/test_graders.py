from incident_response_rl.env import IncidentResponseEnv
from incident_response_rl.graders import grade_episode, score_state
from incident_response_rl.models import Action


def test_grader_returns_zero_to_one_scores() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=1, scenario_id="high_latency_easy")
    state = env.state_data
    assert state is not None
    assert 0.0 <= score_state(state) <= 1.0
    assert 0.0 <= grade_episode(state) <= 1.0


def test_full_recovery_scores_one() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=2, scenario_id="high_latency_easy")
    env.step(Action(action_type="scale_up", target="api"))
    state = env.state_data
    assert state is not None
    assert grade_episode(state) == 1.0


def test_partial_recovery_scores_half() -> None:
    env = IncidentResponseEnv()
    env.reset(seed=3, scenario_id="bad_deployment_hard")
    env.step(Action(action_type="rollback_deployment", target="api"))
    state = env.state_data
    assert state is not None
    assert grade_episode(state) == 0.5
