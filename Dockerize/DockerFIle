# Use a lightweight base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install required packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python", "main.py"]
