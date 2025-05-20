#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Prevent errors in a pipeline from being masked.
set -o pipefail

echo "[INFO] Starting Redis server..."
redis-server --daemonize yes

# Give Redis a moment to start
sleep 1
echo "[INFO] Redis server started."

echo "[INFO] Starting Gunicorn server for FastAPI application..."
# Use exec to replace the current shell process with the gunicorn process
# This ensures signals like SIGTERM from docker stop are passed correctly.
# API_PORT will be available as an environment variable (set in Dockerfile or passed by docker run).
# Defaulting to 8000 if API_PORT is not set.
# The gunicorn workers are set to 1, and uvicorn worker will handle async tasks.
exec gunicorn -k uvicorn.workers.UvicornWorker \
    --bind "0.0.0.0:${API_PORT:-8000}" \
    --workers 2 \
    coding_task.server.server:app
