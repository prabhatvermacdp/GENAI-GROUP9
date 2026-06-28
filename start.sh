#!/bin/sh
set -e

# Start FastAPI backend
uvicorn backend:app --host 0.0.0.0 --port 8000 &

# Streamlit listens on the HF-default port 7860 and talks to the backend in-process
exec streamlit run streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
