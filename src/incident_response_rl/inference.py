from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx

from .models import Action, BaselineEpisodeResult, BaselineRunReport, Observation

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_ENV_BASE_URL = "http://127.0.0.1:7860"
DEFAULT_SCENARIOS = [
    "high_latency_easy",
    "service_crash_medium",
    "bad_deployment_hard",
]


def build_prompt(observation: Observation) -> str:
    return (
        "You are an incident response agent. Choose exactly one action.\n"
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
    return Action.model_validate_json(payload)


def query_hf_router(observation: Observation) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to call the HF router.")
    api_base = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": build_prompt(observation)}],
        "temperature": 0,
        "seed": 7,
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def reset_remote_env(client: httpx.Client, env_base_url: str, scenario_id: str, seed: int) -> tuple[str, Observation]:
    response = client.post(
        f"{env_base_url.rstrip('/')}/reset",
        json={"seed": seed, "scenario_id": scenario_id},
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("episode_id", scenario_id), Observation.model_validate(payload["observation"])


def step_remote_env(client: httpx.Client, env_base_url: str, action: Action) -> Observation:
    response = client.post(
        f"{env_base_url.rstrip('/')}/step",
        json={"action": action.model_dump()},
    )
    response.raise_for_status()
    payload = response.json()
    return Observation.model_validate(payload["observation"])


def run_baseline(env_base_url: str, scenarios: list[str] | None = None) -> BaselineRunReport:
    scenarios = scenarios or list(DEFAULT_SCENARIOS)
    task_results: list[BaselineEpisodeResult] = []

    with httpx.Client(timeout=30.0) as client:
        for index, scenario_id in enumerate(scenarios):
            total_reward = 0.0
            episode_id, observation = reset_remote_env(client, env_base_url, scenario_id, seed=index + 1)
            while not observation.done:
                raw_text = query_hf_router(observation)
                action = parse_action_block(raw_text)
                observation = step_remote_env(client, env_base_url, action)
                total_reward += float(observation.reward or 0.0)

            task_results.append(
                BaselineEpisodeResult(
                    scenario_id=scenario_id,
                    difficulty=observation.difficulty,
                    score=float(observation.task_score),
                    terminal_grade=float(observation.terminal_grade or 0.0),
                    steps_taken=observation.step_count,
                    total_reward=round(total_reward, 3),
                    successful_actions=list(observation.metadata.get("successful_actions", [])),
                )
            )

    average_score = round(sum(item.score for item in task_results) / len(task_results), 3)
    return BaselineRunReport(
        model_name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        router_base_url=os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL),
        average_score=average_score,
        task_scores=task_results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible HF-router baseline evaluation.")
    parser.add_argument("--env-base-url", default=os.environ.get("ENV_BASE_URL", DEFAULT_ENV_BASE_URL))
    parser.add_argument("--output", default="artifacts/baseline_scores.json")
    args = parser.parse_args()

    report = run_baseline(args.env_base_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
