FROM python:3.12-slim

WORKDIR /app

COPY src/ /app

RUN pip install --no-cache-dir \
    fastapi==0.135.1 \
    uvicorn==0.41.0 \
    pydantic==2.12.5 \
    python-dotenv==1.2.2 \
    requests==2.32.5 \
    langchain==0.2.17 \
    langchain-openai==0.1.25 \
    duckduckgo-search==8.1.1 \
    APScheduler==3.11.2 \
    PyJWT==2.11.0 \
    click==8.3.1 \
    textual==8.0.1

RUN useradd -r -s /bin/false appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "__main__.py", "serve", "--host", "0.0.0.0", "--port", "5000"]
