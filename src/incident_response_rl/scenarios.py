from __future__ import annotations

import random

from .models import Difficulty, IncidentScenario


SCENARIO_IDS = [
    "high_latency_easy",
    "service_crash_medium",
    "bad_deployment_hard",
]


def create_scenario_catalog(rng: random.Random) -> dict[str, IncidentScenario]:
    return {
        "high_latency_easy": build_high_latency(rng, "easy"),
        "service_crash_medium": build_service_crash(rng, "medium"),
        "bad_deployment_hard": build_bad_deployment(rng, "hard"),
    }


def choose_scenario(scenario_id: str | None, rng: random.Random) -> IncidentScenario:
    catalog = create_scenario_catalog(rng)
    if scenario_id:
        try:
            return catalog[scenario_id]
        except KeyError as exc:
            valid = ", ".join(sorted(catalog))
            raise ValueError(f"Unknown scenario_id '{scenario_id}'. Valid options: {valid}") from exc
    return catalog[rng.choice(SCENARIO_IDS)]


def build_high_latency(rng: random.Random, difficulty: Difficulty) -> IncidentScenario:
    latency = float(rng.randint(230, 290))
    cpu = float(rng.randint(82, 94))
    error_rate = round(rng.uniform(0.05, 0.12), 3)
    misleading_logs = ["INFO cache warmer completed successfully for unrelated batch job"]
    false_alerts = []
    if difficulty != "easy":
        misleading_logs.append("WARN billing worker retry spike observed in another region")
        false_alerts.append("Disk usage warning on analytics-worker")

    return IncidentScenario(
        scenario_id=f"high_latency_{difficulty}",
        family="high_latency",
        difficulty=difficulty,
        description="Traffic surge exhausted API capacity and raised latency.",
        target_service="api",
        initial_metrics={
            "latency_ms": latency,
            "error_rate": error_rate,
            "cpu_pct": cpu,
            "deployment_version": 20240401.0,
        },
        initial_logs=[
            "WARN api latency above SLO for /checkout requests",
            "INFO autoscaler at max replica count=3 for api",
            "WARN CPU saturation detected on api nodes",
        ],
        initial_alerts=["HighLatency(api)", "CpuHot(api)"],
        required_actions=["scale_up"],
        action_targets={"scale_up": "api"},
        immediate_effects={"scale_up": {"latency_ms": -140.0, "cpu_pct": -35.0, "error_rate": -0.08}},
        delayed_effects={},
        improvement_logs={"scale_up": ["INFO scale-up completed for api replicas=5"]},
        misleading_logs=misleading_logs,
        false_alerts=false_alerts,
        max_steps=8,
        diagnosis_hints=["WARN api queue depth rising faster than workers can drain"],
    )


def build_service_crash(rng: random.Random, difficulty: Difficulty) -> IncidentScenario:
    error_rate = round(rng.uniform(0.38, 0.62), 3)
    latency = float(rng.randint(180, 240))
    cpu = float(rng.randint(20, 45))
    misleading_logs = ["INFO frontend deploy finished without errors"]
    false_alerts = ["MinorLatency(worker)"] if difficulty != "easy" else []

    return IncidentScenario(
        scenario_id=f"service_crash_{difficulty}",
        family="service_crash",
        difficulty=difficulty,
        description="The API process crashed and needs a restart.",
        target_service="api",
        initial_metrics={
            "latency_ms": latency,
            "error_rate": error_rate,
            "cpu_pct": cpu,
            "deployment_version": 20240401.0,
        },
        initial_logs=[
            "ERROR api process exited with signal SIGKILL",
            "ERROR readiness probe failed for api pod",
            "WARN request failures rising after pod restart loop",
        ],
        initial_alerts=["CrashLoop(api)", "HighErrorRate(api)"],
        required_actions=["restart_service"],
        action_targets={"restart_service": "api"},
        immediate_effects={"restart_service": {"error_rate": -0.70, "latency_ms": -120.0, "cpu_pct": 5.0}},
        delayed_effects={},
        improvement_logs={"restart_service": ["INFO api restarted cleanly and passed readiness probe"]},
        misleading_logs=misleading_logs,
        false_alerts=false_alerts,
        max_steps=8,
        diagnosis_hints=["ERROR crash dump points to exhausted process state on api"],
    )


def build_bad_deployment(rng: random.Random, difficulty: Difficulty) -> IncidentScenario:
    version = float(rng.choice([20240402.0, 20240403.0]))
    error_rate = round(rng.uniform(0.21, 0.34), 3)
    latency = float(rng.randint(260, 340))
    cpu = float(rng.randint(58, 76))

    return IncidentScenario(
        scenario_id=f"bad_deployment_{difficulty}",
        family="bad_deployment",
        difficulty=difficulty,
        description="A bad deployment introduced regressions; rollback then restart is required.",
        target_service="api",
        initial_metrics={
            "latency_ms": latency,
            "error_rate": error_rate,
            "cpu_pct": cpu,
            "deployment_version": version,
        },
        initial_logs=[
            f"INFO deployed api version={version:.0f}",
            "WARN latency regression detected after deployment",
            "ERROR null pointer exception in checkout handler",
        ],
        initial_alerts=["HighLatency(api)", "HighErrorRate(api)", "DeploymentRegression(api)"],
        required_actions=["rollback_deployment", "restart_service"],
        action_targets={"rollback_deployment": "api", "restart_service": "api"},
        immediate_effects={},
        delayed_effects={
            "rollback_deployment": (0, {"deployment_version": -1.0, "latency_ms": -100.0, "error_rate": -0.14}),
            "restart_service": (0, {"latency_ms": -80.0, "error_rate": -0.16, "cpu_pct": -18.0}),
        },
        improvement_logs={
            "rollback_deployment": ["INFO rollback completed to previous stable release"],
            "restart_service": ["INFO api restarted after rollback and error rate dropped"],
        },
        misleading_logs=[
            "WARN cache miss ratio rose on recommendation-service",
            "INFO batch exporter completed successfully",
        ],
        false_alerts=["CpuWarm(worker)"],
        max_steps=10,
        diagnosis_hints=[
            "ERROR failures began immediately after release promotion",
            "WARN restart alone will not remove the bad code path",
        ],
    )
