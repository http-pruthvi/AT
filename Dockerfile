FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 7860

# Run the server with proxy header support for HuggingFace Spaces (HTTPS)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--forwarded-allow-ips", "*", "--proxy-headers"]
