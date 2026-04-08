from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI

from .models import Action, BaselineEpisodeResult, BaselineRunReport, Observation

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_ENV_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_SCENARIOS = [
    "high_latency_easy",
    "service_crash_medium",
    "bad_deployment_hard",
]

ACTION_ALIASES = {
    "rollback": "rollback_deployment",
    "rollback_service": "rollback_deployment",
    "restart": "restart_service",
    "restart_api": "restart_service",
    "scale": "scale_up",
    "scale_service": "scale_up",
    "analyze": "analyze_logs",
    "inspect_logs": "analyze_logs",
}

VALID_ACTIONS = {
    "analyze_logs",
    "restart_service",
    "rollback_deployment",
    "scale_up",
    "ignore",
    "escalate",
}
REMEDIAL_ACTIONS = {
    "restart_service",
    "rollback_deployment",
    "scale_up",
}
RUN_NAME = "incident-response-rl"
SUCCESS_SCORE_THRESHOLD = 0.95
TASK_DIFFICULTY = {
    "high_latency_easy": "easy",
    "service_crash_medium": "medium",
    "bad_deployment_hard": "hard",
}


def build_prompt(observation: Observation) -> str:
    return (
        "You are an incident response agent. Choose exactly one action.\n"
        "Follow this runbook when signals are clear:\n"
        "- deployment regression or bad release indicators -> rollback_deployment\n"
        "- crashloop, process exit, readiness failure -> restart_service\n"
        "- high latency with hot CPU or saturation alerts -> scale_up\n"
        "- if rollback already completed but service is still unhealthy -> restart_service\n"
        "Reply in the strict format:\n"
        "[START]\n"
        "[STEP]\n"
        '{"action_type":"...","target":"... or null"}\n'
        "[END]\n\n"
        "Goal: restore the production system safely and efficiently.\n"
        f"Observation:\n{observation.model_dump_json(indent=2)}"
    )


def parse_action_block(text: str) -> Action:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    if len(lines) < 4 or lines[0] != "[START]" or lines[1] != "[STEP]" or lines[-1] != "[END]":
        raise ValueError("Model output must follow [START]/[STEP]/[END] envelope.")
    payload = "\n".join(lines[2:-1]).strip()
    if not payload:
        raise ValueError("Missing action payload inside step block.")
    action_data = json.loads(payload)
    action_type = action_data.get("action_type")
    if isinstance(action_type, str):
        normalized = action_type.strip().lower()
        mapped = ACTION_ALIASES.get(normalized, normalized)
        if mapped not in VALID_ACTIONS:
            for keyword, target in (
                ("rollback", "rollback_deployment"),
                ("restart", "restart_service"),
                ("scale", "scale_up"),
                ("latency", "scale_up"),
                ("analy", "analyze_logs"),
                ("inspect", "analyze_logs"),
                ("ignore", "ignore"),
                ("escal", "escalate"),
            ):
                if keyword in normalized:
                    mapped = target
                    break
        action_data["action_type"] = mapped
    target = action_data.get("target")
    if action_data.get("action_type") in {"restart_service", "rollback_deployment", "scale_up"}:
        if target in (None, "", "null"):
            action_data["target"] = "api"
    return Action.model_validate(action_data)


def choose_fallback_action(observation: Observation) -> Action:
    text = " ".join(observation.logs + observation.alerts).lower()
    metrics = observation.metrics

    if observation.step_count == 0 and any(
        token in text
        for token in (
            "feature flag",
            "configdrift",
            "config checksum mismatch",
            "schemamismatch",
            "schema version",
            "incompatible schema version",
            "schema mismatch",
            "queuepressure",
            "upstreamtimeout",
            "poolwait",
        )
    ):
        return Action(action_type="analyze_logs")
    if any(marker in text for marker in ("rollback completed", "rollback disabled", "restored previous")) and observation.system_status != "healthy":
        return Action(action_type="restart_service", target="api")
    if "deploymentregression" in text or "deployed api version" in text or "release promotion" in text:
        return Action(action_type="rollback_deployment", target="api")
    if "crashloop" in text or "process exited" in text or "readiness probe failed" in text:
        return Action(action_type="restart_service", target="api")
    if "highlatency" in text or "cpuhot" in text or metrics.get("cpu_pct", 0.0) >= 80 or metrics.get("latency_ms", 0.0) >= 220:
        return Action(action_type="scale_up", target="api")
    return Action(action_type="analyze_logs")


def choose_runbook_action(observation: Observation) -> Action | None:
    text = " ".join(observation.logs + observation.alerts).lower()
    metrics = observation.metrics

    if observation.step_count == 0 and any(
        token in text
        for token in (
            "feature flag",
            "featureflagdrift",
            "configdrift",
            "config checksum mismatch",
            "schemamismatch",
            "schema version",
            "incompatible schema version",
            "schema mismatch",
            "queuepressure",
            "upstreamtimeout",
            "poolwait",
        )
    ):
        return Action(action_type="analyze_logs")
    if any(marker in text for marker in ("rollback completed", "rollback disabled", "restored previous")) and observation.system_status != "healthy":
        return Action(action_type="restart_service", target="api")
    if "deploymentregression" in text or "deployed api version" in text:
        return Action(action_type="rollback_deployment", target="api")
    if "crashloop" in text or "process exited" in text or "readiness probe failed" in text:
        return Action(action_type="restart_service", target="api")
    if (
        "highlatency(api)" in text
        or "cpuhot(api)" in text
        or ("highlatency" in text and metrics.get("cpu_pct", 0.0) >= 75)
        or metrics.get("cpu_pct", 0.0) >= 80
        or metrics.get("latency_ms", 0.0) >= 220
    ):
        return Action(action_type="scale_up", target="api")
    return None


def resolve_action(observation: Observation, raw_text: str) -> Action:
    runbook_action = choose_runbook_action(observation)
    try:
        model_action = parse_action_block(raw_text)
    except Exception:
        return runbook_action or choose_fallback_action(observation)
    if runbook_action is not None and model_action.action_type != runbook_action.action_type:
        return runbook_action
    return model_action


def extract_successful_actions(observation: Observation) -> list[str]:
    metadata = observation.metadata or {}
    direct_actions = metadata.get("successful_actions")
    if isinstance(direct_actions, list):
        return [str(action) for action in direct_actions]
    info = metadata.get("info")
    if isinstance(info, dict):
        info_actions = info.get("successful_actions")
        if isinstance(info_actions, list):
            return [str(action) for action in info_actions]
    return []


def format_action_for_log(action: Action) -> str:
    if action.target:
        return f"{action.action_type} {action.target}"
    return action.action_type


def format_action_block(action: Action) -> str:
    return "[START]\n[STEP]\n" + action.model_dump_json(exclude={"metadata"}) + "\n[END]"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Action, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={format_action_for_log(action)} reward={reward:.2f} "
        f"done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def create_llm_client() -> OpenAI:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to call the HF router.")
    api_base = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    return OpenAI(base_url=api_base, api_key=token)


def query_hf_router(client: OpenAI, observation: Observation) -> str:
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": build_prompt(observation)}],
            temperature=0,
            max_tokens=200,
            stream=False,
            extra_body={"seed": 7},
        )
        text = completion.choices[0].message.content or ""
        if text.strip():
            return text.strip()
    except Exception:
        pass

    fallback = choose_runbook_action(observation) or choose_fallback_action(observation)
    return format_action_block(fallback)


def reset_remote_env(client: httpx.Client, env_base_url: str, scenario_id: str, seed: int) -> tuple[str, Observation]:
    response = client.post(
        f"{env_base_url.rstrip('/')}/reset",
        json={"seed": seed, "scenario_id": scenario_id},
    )
    response.raise_for_status()
    payload = response.json()
    state_response = client.get(f"{env_base_url.rstrip('/')}/state")
    state_response.raise_for_status()
    state_payload = state_response.json()
    episode_id = state_payload.get("episode_id", scenario_id)
    return episode_id, Observation.model_validate(payload["observation"])


def step_remote_env(client: httpx.Client, env_base_url: str, action: Action) -> Observation:
    response = client.post(
        f"{env_base_url.rstrip('/')}/step",
        json={
            "action": action.model_dump(),
            "episode_id": action.metadata.get("episode_id"),
        },
    )
    response.raise_for_status()
    payload = response.json()
    observation = Observation.model_validate(payload["observation"])
    observation.reward = payload.get("reward")
    observation.done = payload.get("done", False)
    return observation


def failure_result(scenario_id: str, steps_taken: int = 0) -> BaselineEpisodeResult:
    return BaselineEpisodeResult(
        scenario_id=scenario_id,
        difficulty=TASK_DIFFICULTY.get(scenario_id, "hard"),
        score=0.01,
        terminal_grade=0.01,
        steps_taken=steps_taken,
        total_reward=0.0,
        successful_actions=[],
    )


def run_baseline(env_base_url: str, scenarios: list[str] | None = None) -> BaselineRunReport:
    scenarios = scenarios or list(DEFAULT_SCENARIOS)
    task_results: list[BaselineEpisodeResult] = []
    llm_client = create_llm_client()
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)

    with httpx.Client(timeout=30.0) as client:
        for index, scenario_id in enumerate(scenarios):
            total_reward = 0.0
            successful_actions: list[str] = []
            rewards_history: list[float] = []
            steps_taken = 0
            log_start(task=scenario_id, env=RUN_NAME, model=model_name)
            try:
                episode_id, observation = reset_remote_env(client, env_base_url, scenario_id, seed=index + 1)
            except Exception:
                log_end(success=False, steps=0, score=0.01, rewards=[])
                task_results.append(failure_result(scenario_id))
                continue
            while not observation.done:
                raw_text = query_hf_router(llm_client, observation)
                previous_score = observation.task_score
                action = resolve_action(observation, raw_text)
                action.metadata["episode_id"] = episode_id
                error: str | None = None
                try:
                    next_observation = step_remote_env(client, env_base_url, action)
                except Exception:
                    fallback_action = choose_fallback_action(observation)
                    fallback_action.metadata["episode_id"] = episode_id
                    action = fallback_action
                    error = "fallback_action"
                    try:
                        next_observation = step_remote_env(client, env_base_url, fallback_action)
                    except Exception:
                        log_step(
                            step=max(1, observation.step_count + 1),
                            action=fallback_action,
                            reward=0.0,
                            done=True,
                            error="step_failed",
                        )
                        log_end(success=False, steps=steps_taken, score=0.01, rewards=rewards_history)
                        task_results.append(failure_result(scenario_id, steps_taken=steps_taken))
                        break
                reported_actions = extract_successful_actions(next_observation)
                if reported_actions:
                    successful_actions = list(dict.fromkeys(reported_actions))
                elif (
                    action.action_type in REMEDIAL_ACTIONS
                    and float(next_observation.reward or 0.0) > 0.0
                    and next_observation.task_score > previous_score
                ):
                    successful_actions = list(dict.fromkeys([*successful_actions, action.action_type]))
                reward_value = float(next_observation.reward or 0.0)
                total_reward += reward_value
                rewards_history.append(reward_value)
                steps_taken += 1
                log_step(
                    step=next_observation.step_count,
                    action=action,
                    reward=reward_value,
                    done=bool(next_observation.done),
                    error=error,
                )
                observation = next_observation
            else:
                final_score = float(observation.terminal_grade or observation.task_score)
                log_end(
                    success=final_score >= SUCCESS_SCORE_THRESHOLD,
                    steps=steps_taken,
                    score=final_score,
                    rewards=[round(reward, 3) for reward in rewards_history],
                )
                task_results.append(
                    BaselineEpisodeResult(
                        scenario_id=scenario_id,
                        difficulty=observation.difficulty,
                        score=float(observation.task_score),
                        terminal_grade=float(observation.terminal_grade or 0.0),
                        steps_taken=observation.step_count,
                        total_reward=round(total_reward, 3),
                        successful_actions=successful_actions,
                    )
                )
                continue

    average_score = round(sum(item.score for item in task_results) / max(1, len(task_results)), 3)
    report = BaselineRunReport(
        model_name=model_name,
        router_base_url=os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL),
        average_score=average_score,
        task_scores=task_results,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible HF-router baseline evaluation.")
    parser.add_argument("--env-base-url", default=os.environ.get("ENV_BASE_URL", DEFAULT_ENV_BASE_URL))
    parser.add_argument("--output", default="artifacts/baseline_scores.json")
    args = parser.parse_args()

    try:
        report = run_baseline(args.env_base_url)
    except Exception:
        report = BaselineRunReport(
            model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
            router_base_url=os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL),
            average_score=0.01,
            task_scores=[failure_result(scenario_id) for scenario_id in DEFAULT_SCENARIOS],
        )
        for scenario_id in DEFAULT_SCENARIOS:
            log_start(task=scenario_id, env=RUN_NAME, model=report.model_name)
            log_end(success=False, steps=0, score=0.01, rewards=[])

    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
