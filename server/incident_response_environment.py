from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from incident_response_rl.env import IncidentResponseEnv
from incident_response_rl.models import Action, IncidentState, Observation


class IncidentResponseEnvironment(Environment[Action, Observation, IncidentState]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    _episodes: dict[str, IncidentResponseEnv] = {}
    _current_episode_id: str | None = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="IncidentResponseEnvironment",
            description=(
                "Incident response environment with 3 public tasks and deterministic graders: "
                "high_latency_easy (capacity and burst amplification), "
                "service_crash_medium (crash loop recovery), and "
                "bad_deployment_hard (rollback and restart recovery)."
            ),
            version="1.0.0",
        )

    def __init__(self) -> None:
        super().__init__()
        self._episode_id = self.__class__._current_episode_id or str(uuid4())

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> Observation:
        self._episode_id = episode_id or str(uuid4())
        simulator = IncidentResponseEnv()
        scenario_id = kwargs.get("scenario_id") or kwargs.get("task_id")
        observation, info = simulator.reset(seed=seed, scenario_id=scenario_id)
        self.__class__._episodes[self._episode_id] = simulator
        self.__class__._current_episode_id = self._episode_id
        observation.metadata.update(
            {
                "episode_id": self._episode_id,
                "info": info,
            }
        )
        return observation

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **kwargs,
    ) -> Observation:
        episode_id = kwargs.get("episode_id") or action.metadata.get("episode_id") or self.__class__._current_episode_id
        if not episode_id or episode_id not in self.__class__._episodes:
            raise RuntimeError("Unknown episode_id. Call reset first or pass a valid episode_id.")
        self._episode_id = episode_id
        simulator = self.__class__._episodes[episode_id]
        observation, reward, done, info = simulator.step(action)
        observation.reward = reward
        observation.done = done
        observation.metadata.update(
            {
                "episode_id": self._episode_id,
                "timeout_s": timeout_s,
                "info": info,
            }
        )
        if done:
            self.__class__._episodes.pop(self._episode_id, None)
            if self.__class__._current_episode_id == self._episode_id:
                self.__class__._current_episode_id = None
        return observation

    @property
    def state(self) -> IncidentState:
        episode_id = self.__class__._current_episode_id
        if not episode_id or episode_id not in self.__class__._episodes:
            raise RuntimeError("Environment must be reset before reading state.")
        state = self.__class__._episodes[episode_id].state
        state.episode_id = episode_id
        return state
