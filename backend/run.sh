#!/usr/bin/env bash
# backend/run.sh
set -e
cd "$(dirname "$0")"
echo "Starting uvicorn on http://0.0.0.0:8600 ..."
uvicorn app:app --host 0.0.0.0 --port 8600 --reload