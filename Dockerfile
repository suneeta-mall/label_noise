FROM python:3.9-slim-buster as BASE

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry install --no-root

COPY . .

RUN poetry install

