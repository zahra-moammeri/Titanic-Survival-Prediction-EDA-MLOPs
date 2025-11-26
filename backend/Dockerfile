FROM python:3.12.9-slim-bullseye


ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy application code
COPY . /app

# Copy tests
COPY tests/ ./tests/

# Install testing dependencies
RUN pip install pytest pytest-asyncio httpx

# Run tests when container starts
CMD [ "python", "-m", "pytest", "tests/", "-v", "--tb=short" ]

# EXPOSE 8000

# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]