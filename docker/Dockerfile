FROM python:3.10-slim

WORKDIR /workspace

# Install system utilities and development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create physical placeholders for dynamic mounting
RUN mkdir -p data/raw data/alignment data/self_adapt ckpts runs

EXPOSE 8001 8002 8003 8080
