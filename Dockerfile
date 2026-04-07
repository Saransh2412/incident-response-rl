FROM public.ecr.aws/docker/library/python:3.11-slim-bookworm

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml inference.py ./
COPY server ./server
COPY src ./src

RUN pip install --no-cache-dir .

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
