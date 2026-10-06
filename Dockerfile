# ==========================================
# UNIVERSAL PRODUCTION DOCKERFILE FOR ML/DL
# ==========================================

# 1. Use an official, stable, and slim Python runtime as a parent image
FROM python:3.11-slim

# 2. Set environment variables to keep Python clean inside the container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies (needed for heavy ML libraries like XGBoost, LightGBM, or OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to take advantage of Docker's layer caching
COPY requirements.txt .

# 6. Upgrade pip and install Python packages without saving cache files (reduces image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the core application code and your trained Pickle model
COPY app.py .
COPY model.pkl .

# 8. Document the port the container is intended to listen on
EXPOSE 8000

# 9. UNIVERSAL RUN COMMAND:
# This dynamically reads the $PORT variable set by cloud providers (like Render or AWS).
# If no port is provided, it defaults to port 8000 for local laptop testing.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
