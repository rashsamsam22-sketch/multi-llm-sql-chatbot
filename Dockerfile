FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install deps with pip + wheels only (no source builds)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --only-binary=:all: -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]