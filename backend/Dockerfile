# Thai News Topic Classifier - Backend API
# Dockerfile with pre-downloaded model for Cloud Run

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY wsgi.py .

# Pre-download model from Hugging Face at build time (not runtime!)
# This makes cold starts much faster
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='farypor/my-thai-news-classifier-onnx', repo_type='model', local_dir='onnx_model', local_dir_use_symlinks=False)"

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 120"]
