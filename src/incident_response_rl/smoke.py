from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient
import uvicorn

from .env import IncidentResponseEnv
from .models import Action
from server.app import app


def main() -> None:
    env = IncidentResponseEnv()
    observation, info = env.reset(seed=7, scenario_id="service_crash_medium")
    print("Local reset:", observation.model_dump(), info)
    step = env.step_result(Action(action_type="restart_service", target="api"))
    print("Local step:", step.model_dump())

    client = TestClient(app)
    reset_response = client.post("/reset", json={"scenario_id": "high_latency_easy", "seed": 11})
    print("API reset:", reset_response.json())
    step_response = client.post(
        "/step",
        json={"action": {"action_type": "scale_up", "target": "api"}},
    )
    print("API step:", step_response.json())

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.0)
    print("Remote validation target ready at http://127.0.0.1:8000")
    server.should_exit = True
    thread.join(timeout=5)


if __name__ == "__main__":
    main()
