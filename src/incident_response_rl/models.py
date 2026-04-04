from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState
from pydantic import BaseModel, ConfigDict, Field


ActionType = Literal[
    "analyze_logs",
    "restart_service",
    "rollback_deployment",
    "scale_up",
    "ignore",
    "escalate",
]
Difficulty = Literal["easy", "medium", "hard"]
IncidentFamily = Literal["high_latency", "service_crash", "bad_deployment"]
SystemStatus = Literal["healthy", "degraded", "critical"]


class IncidentAction(OpenEnvAction):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    target: str | None = None


class IncidentObservation(OpenEnvObservation):
    model_config = ConfigDict(extra="forbid")

    logs: list[str]
    metrics: dict[str, float]
    alerts: list[str]
    system_status: SystemStatus
    step_count: int
    scenario_id: str
    difficulty: Difficulty
    incident_family: IncidentFamily
    last_transition_reason: str = ""
    task_score: float = 0.0
    terminal_grade: float | None = None


class IncidentState(OpenEnvState):
    model_config = ConfigDict(extra="allow")

    scenario_id: str | None = None
    incident_family: IncidentFamily | None = None
    difficulty: Difficulty | None = None
    logs: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    system_status: SystemStatus | None = None
    last_transition_reason: str = "incident injected"
    successful_actions: list[ActionType] = Field(default_factory=list)
    partial_recovery: bool = False
    noise_present: bool = False
    terminal_grade: float | None = None


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    reason: str


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: IncidentObservation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class BaselineEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    difficulty: Difficulty
    score: float
    terminal_grade: float
    steps_taken: int
    total_reward: float
    successful_actions: list[ActionType]


class BaselineRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    router_base_url: str
    average_score: float
    task_scores: list[BaselineEpisodeResult]


@dataclass(slots=True)
class PendingEffect:
    action_type: ActionType
    remaining_delay: int
    metrics_delta: dict[str, float]
    logs: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    transition_reason: str = ""


@dataclass(slots=True)
class IncidentScenario:
    scenario_id: str
    family: IncidentFamily
    difficulty: Difficulty
    description: str
    target_service: str
    initial_metrics: dict[str, float]
    initial_logs: list[str]
    initial_alerts: list[str]
    required_actions: list[ActionType]
    action_targets: dict[ActionType, str | None]
    immediate_effects: dict[ActionType, dict[str, float]]
    delayed_effects: dict[ActionType, tuple[int, dict[str, float]]]
    improvement_logs: dict[ActionType, list[str]]
    misleading_logs: list[str]
    false_alerts: list[str]
    max_steps: int
    diagnosis_hints: list[str]


@dataclass(slots=True)
class EnvironmentState:
    scenario: IncidentScenario
    metrics: dict[str, float]
    logs: list[str]
    alerts: list[str]
    system_status: SystemStatus
    step_count: int = 0
    analyzed: bool = False
    action_history: list[IncidentAction] = field(default_factory=list)
    successful_actions: list[ActionType] = field(default_factory=list)
    pending_effects: list[PendingEffect] = field(default_factory=list)
    last_transition_reason: str = "incident injected"
    partial_recovery: bool = False
    resolved: bool = False
    terminated_by_escalation: bool = False
    noisy: bool = False
    total_reward: float = 0.0


Action = IncidentAction
Observation = IncidentObservation
State = IncidentState
