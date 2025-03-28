FROM --platform=linux/amd64 python:3.10-slim-buster

# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies needed for your application
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files into the container (main.py and other source files)
COPY . .

# Expose the port that the app will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
