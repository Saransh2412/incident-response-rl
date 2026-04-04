FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml inference.py ./
COPY server ./server
COPY src ./src

RUN pip install --no-cache-dir .

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
