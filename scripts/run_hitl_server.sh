#!/bin/bash
# Script to launch the MiniSEAL HITL Dashboard

echo "=== STARTING MiniSEAL HITL DASHBOARD ==="
cd "$(dirname "$0")/.."

# Install fastapi and uvicorn if not present
python3 -c "import fastapi" 2>/dev/null || pip3 install fastapi uvicorn pydantic

# Start the server
python3 -m uvicorn src.hitl.api:app --host 0.0.0.0 --port 8543 --reload
