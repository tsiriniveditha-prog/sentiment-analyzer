FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose default port
EXPOSE 5000

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run with gunicorn
CMD [gunicorn, --bind, 0.0.0.0:5000, --workers, 2, app:app]

