# Thai News Topic Classifier - Backend API
# Dockerfile with pre-bundled joblib model for Cloud Run

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY app.py .
COPY wsgi.py .
COPY models/ models/

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120"]
