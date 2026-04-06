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
    scenario_id = f"high_latency_{difficulty}"
    variants = [
        IncidentScenario(
            scenario_id=scenario_id,
            family="high_latency",
            difficulty=difficulty,
            description="Traffic surge exhausted API capacity and raised latency.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(230, 290)),
                "error_rate": round(rng.uniform(0.05, 0.12), 3),
                "cpu_pct": float(rng.randint(82, 94)),
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
            immediate_effects={"scale_up": {"latency_ms": -135.0, "cpu_pct": -30.0, "error_rate": -0.07}},
            delayed_effects={},
            improvement_logs={"scale_up": ["INFO scale-up completed for api replicas=5"]},
            misleading_logs=["INFO cache warmer completed successfully for unrelated batch job"],
            false_alerts=[],
            max_steps=8,
            diagnosis_hints=["WARN api queue depth rising faster than workers can drain"],
        ),
        IncidentScenario(
            scenario_id=scenario_id,
            family="high_latency",
            difficulty=difficulty,
            description="Cache churn amplified request fan-out and made the API look underprovisioned.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(240, 305)),
                "error_rate": round(rng.uniform(0.08, 0.14), 3),
                "cpu_pct": float(rng.randint(76, 88)),
                "deployment_version": 20240401.0,
            },
            initial_logs=[
                "WARN checkout cache hit rate fell below target",
                "WARN api queue depth rising during cache refill bursts",
                "INFO replica count stable at 3 despite intermittent hot shards",
            ],
            initial_alerts=["HighLatency(api)", "QueuePressure(api)"],
            required_actions=["analyze_logs", "scale_up"],
            action_targets={"scale_up": "api"},
            immediate_effects={"scale_up": {"latency_ms": -170.0, "cpu_pct": -30.0, "error_rate": -0.14}},
            delayed_effects={},
            improvement_logs={"scale_up": ["INFO scale-up absorbed cache refill bursts on api"]},
            misleading_logs=["WARN billing worker retry spike observed in another region"],
            false_alerts=["Disk usage warning on analytics-worker"],
            max_steps=8,
            diagnosis_hints=[
                "INFO cache stampede widened API fan-out during warmup",
                "WARN latency is demand-side amplification, not a failed deployment",
            ],
        ),
        IncidentScenario(
            scenario_id=scenario_id,
            family="high_latency",
            difficulty=difficulty,
            description="Dependency backpressure is surfacing as API latency under burst load.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(250, 320)),
                "error_rate": round(rng.uniform(0.04, 0.10), 3),
                "cpu_pct": float(rng.randint(80, 92)),
                "deployment_version": 20240401.0,
            },
            initial_logs=[
                "WARN downstream checkout requests are backing up in api worker threads",
                "WARN api saturation detected while dependency pool waits grow",
                "INFO no recent deploys detected for api",
            ],
            initial_alerts=["HighLatency(api)", "CpuHot(api)", "PoolWait(api)"],
            required_actions=["analyze_logs", "scale_up"],
            action_targets={"scale_up": "api"},
            immediate_effects={"scale_up": {"latency_ms": -150.0, "cpu_pct": -32.0, "error_rate": -0.08}},
            delayed_effects={},
            improvement_logs={"scale_up": ["INFO scale-up restored headroom while dependency pool drained"]},
            misleading_logs=["INFO recommendation-service cache purge completed successfully"],
            false_alerts=["CpuWarm(worker)"],
            max_steps=8,
            diagnosis_hints=[
                "WARN thread pools are starved by dependency backpressure",
                "INFO more API workers reduce wait amplification quickly enough to recover",
            ],
        ),
    ]
    return rng.choice(variants)


def build_service_crash(rng: random.Random, difficulty: Difficulty) -> IncidentScenario:
    scenario_id = f"service_crash_{difficulty}"
    variants = [
        IncidentScenario(
            scenario_id=scenario_id,
            family="service_crash",
            difficulty=difficulty,
            description="The API process crashed and needs a restart.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(180, 240)),
                "error_rate": round(rng.uniform(0.38, 0.62), 3),
                "cpu_pct": float(rng.randint(20, 45)),
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
            misleading_logs=["INFO frontend deploy finished without errors"],
            false_alerts=["MinorLatency(worker)"] if difficulty != "easy" else [],
            max_steps=8,
            diagnosis_hints=["ERROR crash dump points to exhausted process state on api"],
        ),
        IncidentScenario(
            scenario_id=scenario_id,
            family="service_crash",
            difficulty=difficulty,
            description="A bad runtime config is causing restarts; diagnosis is needed before a safe restart.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(170, 235)),
                "error_rate": round(rng.uniform(0.35, 0.56), 3),
                "cpu_pct": float(rng.randint(24, 42)),
                "deployment_version": 20240401.0,
            },
            initial_logs=[
                "ERROR api exited after loading invalid runtime configuration",
                "WARN pod restart loop resumed after config reload",
                "ERROR readiness probe failed after bootstrap completed",
            ],
            initial_alerts=["CrashLoop(api)", "HighErrorRate(api)", "ConfigDrift(api)"],
            required_actions=["analyze_logs", "restart_service"],
            action_targets={"restart_service": "api"},
            immediate_effects={"restart_service": {"error_rate": -0.66, "latency_ms": -115.0, "cpu_pct": 8.0}},
            delayed_effects={},
            improvement_logs={"restart_service": ["INFO api restarted cleanly after config rollback hook ran"]},
            misleading_logs=["INFO frontend deploy finished without errors"],
            false_alerts=["MinorLatency(worker)"],
            max_steps=8,
            diagnosis_hints=[
                "ERROR config checksum mismatch detected in api bootstrap",
                "WARN restarting before identifying the config issue causes repeated user-visible disruption",
            ],
        ),
        IncidentScenario(
            scenario_id=scenario_id,
            family="service_crash",
            difficulty=difficulty,
            description="Dependency exhaustion surfaces as crashes and requires diagnosis before restart.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(190, 250)),
                "error_rate": round(rng.uniform(0.32, 0.52), 3),
                "cpu_pct": float(rng.randint(22, 40)),
                "deployment_version": 20240401.0,
            },
            initial_logs=[
                "ERROR api worker crashed after upstream connection pool exhaustion",
                "WARN watchdog restarted api after repeated timeout storms",
                "ERROR readiness probe failed while dependency sockets stayed stale",
            ],
            initial_alerts=["CrashLoop(api)", "UpstreamTimeout(api)"],
            required_actions=["analyze_logs", "restart_service"],
            action_targets={"restart_service": "api"},
            immediate_effects={"restart_service": {"error_rate": -0.60, "latency_ms": -105.0, "cpu_pct": 10.0}},
            delayed_effects={},
            improvement_logs={"restart_service": ["INFO api restarted after dependency pools were recycled"]},
            misleading_logs=["INFO batch exporter completed successfully"],
            false_alerts=["MinorLatency(worker)"],
            max_steps=8,
            diagnosis_hints=[
                "WARN stale upstream sockets survived the last crash loop",
                "INFO recycling api only works once the dependency pool clue is identified",
            ],
        ),
    ]
    return rng.choice(variants)


def build_bad_deployment(rng: random.Random, difficulty: Difficulty) -> IncidentScenario:
    scenario_id = f"bad_deployment_{difficulty}"
    version = float(rng.choice([20240402.0, 20240403.0]))
    variants = [
        IncidentScenario(
            scenario_id=scenario_id,
            family="bad_deployment",
            difficulty=difficulty,
            description="A bad deployment introduced regressions; rollback then restart is required.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(260, 340)),
                "error_rate": round(rng.uniform(0.21, 0.34), 3),
                "cpu_pct": float(rng.randint(58, 76)),
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
        ),
        IncidentScenario(
            scenario_id=scenario_id,
            family="bad_deployment",
            difficulty=difficulty,
            description="A feature-flag rollout caused a bad code path; diagnosis, rollback, then restart are required.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(255, 335)),
                "error_rate": round(rng.uniform(0.19, 0.31), 3),
                "cpu_pct": float(rng.randint(55, 74)),
                "deployment_version": version,
            },
            initial_logs=[
                f"INFO deployed api version={version:.0f}",
                "WARN checkout failures spiked after feature flag rollout",
                "ERROR fallback code path panicked under partial flag enablement",
            ],
            initial_alerts=["HighErrorRate(api)", "FeatureFlagDrift(api)", "DeploymentRegression(api)"],
            required_actions=["analyze_logs", "rollback_deployment", "restart_service"],
            action_targets={"rollback_deployment": "api", "restart_service": "api"},
            immediate_effects={},
            delayed_effects={
                "rollback_deployment": (0, {"deployment_version": -1.0, "latency_ms": -95.0, "error_rate": -0.12}),
                "restart_service": (0, {"latency_ms": -92.0, "error_rate": -0.21, "cpu_pct": -18.0}),
            },
            improvement_logs={
                "rollback_deployment": ["INFO rollback disabled the problematic feature-flag cohort"],
                "restart_service": ["INFO api restarted after rollback and stale flag state cleared"],
            },
            misleading_logs=[
                "WARN cache miss ratio rose on recommendation-service",
                "INFO background exporter finished normally",
            ],
            false_alerts=["CpuWarm(worker)"],
            max_steps=10,
            diagnosis_hints=[
                "WARN feature-flag cohort changed immediately before failures began",
                "ERROR restart alone preserves the bad runtime path until the rollout is undone",
            ],
        ),
        IncidentScenario(
            scenario_id=scenario_id,
            family="bad_deployment",
            difficulty=difficulty,
            description="A schema mismatch after release requires diagnosis, rollback, and restart.",
            target_service="api",
            initial_metrics={
                "latency_ms": float(rng.randint(270, 345)),
                "error_rate": round(rng.uniform(0.20, 0.33), 3),
                "cpu_pct": float(rng.randint(56, 75)),
                "deployment_version": version,
            },
            initial_logs=[
                f"INFO deployed api version={version:.0f}",
                "ERROR checkout handler rejected rows with incompatible schema version",
                "WARN latency rose after migration checks started failing open",
            ],
            initial_alerts=["HighLatency(api)", "SchemaMismatch(api)", "HighErrorRate(api)"],
            required_actions=["analyze_logs", "rollback_deployment", "restart_service"],
            action_targets={"rollback_deployment": "api", "restart_service": "api"},
            immediate_effects={},
            delayed_effects={
                "rollback_deployment": (0, {"deployment_version": -1.0, "latency_ms": -90.0, "error_rate": -0.11}),
                "restart_service": (0, {"latency_ms": -100.0, "error_rate": -0.24, "cpu_pct": -18.0}),
            },
            improvement_logs={
                "rollback_deployment": ["INFO rollback restored previous schema-compatible release"],
                "restart_service": ["INFO api restarted after rollback and schema cache reloaded"],
            },
            misleading_logs=[
                "INFO billing worker heartbeat recovered",
                "WARN cache miss ratio rose on recommendation-service",
            ],
            false_alerts=["CpuWarm(worker)"],
            max_steps=10,
            diagnosis_hints=[
                "ERROR failures correlate with schema version mismatch in freshly deployed handlers",
                "WARN rollback is required before restart can fully clear bad state",
            ],
        ),
    ]
    return rng.choice(variants)
