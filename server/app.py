from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from incident_response_rl.models import Action, Observation
from server.incident_response_environment import IncidentResponseEnvironment


app = create_app(
    IncidentResponseEnvironment,
    Action,
    Observation,
    env_name="incident-response-rl",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
