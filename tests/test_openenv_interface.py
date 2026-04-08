from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_openenv_health_and_schema() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    schema = client.get("/schema")
    assert schema.status_code == 200
    payload = schema.json()
    assert "action" in payload
    assert "observation" in payload
    assert "state" in payload

    metadata = client.get("/metadata")
    assert metadata.status_code == 200
    metadata_payload = metadata.json()
    description = metadata_payload["description"]
    assert "high_latency_easy" in description
    assert "service_crash_medium" in description
    assert "bad_deployment_hard" in description


def test_openenv_reset_and_step() -> None:
    reset = client.post("/reset", json={"scenario_id": "high_latency_easy", "seed": 5})
    assert reset.status_code == 200
    reset_payload = reset.json()
    assert reset_payload["done"] is False
    assert reset_payload["observation"]["scenario_id"] == "high_latency_easy"

    step = client.post("/step", json={"action": {"action_type": "scale_up", "target": "api"}})
    assert step.status_code == 200
    step_payload = step.json()
    assert step_payload["done"] is True
    assert step_payload["observation"]["system_status"] == "healthy"


def test_openenv_state_endpoint() -> None:
    client.post("/reset", json={"scenario_id": "service_crash_medium", "seed": 2})
    state = client.get("/state")
    assert state.status_code == 200
    payload = state.json()
    assert payload["step_count"] == 0


def test_unseeded_reset_cycles_three_public_tasks() -> None:
    scenarios = []
    for _ in range(3):
        reset = client.post("/reset", json={})
        assert reset.status_code == 200
        scenarios.append(reset.json()["observation"]["scenario_id"])

    assert scenarios == [
        "high_latency_easy",
        "service_crash_medium",
        "bad_deployment_hard",
    ]


def test_tasks_endpoint_lists_three_public_tasks() -> None:
    response = client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["tasks"]] == [
        "high_latency_easy",
        "service_crash_medium",
        "bad_deployment_hard",
    ]
    assert all("name" in item for item in payload["tasks"])
    assert all("description" in item for item in payload["tasks"])
    assert all("num_scenarios" in item for item in payload["tasks"])
    assert all("grader" in item for item in payload["tasks"])


def test_reset_accepts_task_id_alias() -> None:
    reset = client.post("/reset", json={"task_id": "service_crash_medium", "seed": 3})
    assert reset.status_code == 200
    payload = reset.json()
    assert payload["observation"]["scenario_id"] == "service_crash_medium"


def test_grader_endpoint_scores_task_trajectory() -> None:
    response = client.post(
        "/grader",
        json={
            "task_id": "high_latency_easy",
            "seed": 5,
            "trajectory": [{"action_type": "scale_up", "target": "api"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()["result"]
    assert payload["task_id"] == "high_latency_easy"
    assert 0.0 < payload["score"] < 1.0
    assert payload["steps_taken"] == 1
    assert set(payload["breakdown"]) == {
        "diagnosis",
        "sequence",
        "effectiveness",
        "efficiency",
        "safety",
    }


def test_info_endpoint_exposes_task_registry_and_spaces() -> None:
    response = client.get("/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "incident-response-rl"
    assert len(payload["tasks"]) == 3
    assert all("grader" in task for task in payload["tasks"])
    assert "properties" in payload["action_space"]
    assert "properties" in payload["observation_space"]
    assert set(payload["grading_components"]) == {
        "diagnosis",
        "sequence",
        "effectiveness",
        "efficiency",
        "safety",
    }


def test_task_registry_manifest_exists_and_matches_tasks() -> None:
    import json
    from pathlib import Path

    payload = json.loads(Path("task_registry.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["tasks"]] == [
        "high_latency_easy",
        "service_crash_medium",
        "bad_deployment_hard",
    ]
    assert [item["grader"] for item in payload["tasks"]] == [
        "incident_response_grade_high_latency",
        "incident_response_grade_service_crash",
        "incident_response_grade_bad_deployment",
    ]


def test_baseline_endpoint_returns_one_result_per_task() -> None:
    response = client.post("/baseline", json={})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 3
    assert 0.0 < payload["aggregate_score"] < 1.0
    for result in payload["results"]:
        assert set(result["breakdown"]) == {
            "diagnosis",
            "sequence",
            "effectiveness",
            "efficiency",
            "safety",
        }


def test_grader_endpoint_empty_trajectory_has_five_components() -> None:
    response = client.post(
        "/grader",
        json={
            "task_id": "service_crash_medium",
            "trajectory": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()["result"]
    assert set(payload["breakdown"]) == {
        "diagnosis",
        "sequence",
        "effectiveness",
        "efficiency",
        "safety",
    }
