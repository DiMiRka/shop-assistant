FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/app/data/tickets.db

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY tests ./tests

RUN useradd --create-home bot && mkdir -p /app/data && chown bot /app/data
USER bot

CMD ["python", "-m", "app.bot"]
