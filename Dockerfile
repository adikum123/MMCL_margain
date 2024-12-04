# Use an official Python base image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the local requirements.txt file to the container
COPY requirements.txt .

# Install pip (it's included in most Python base images, but just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the Python script/code to the container
COPY . .

# Set the default command to run the Python script
# Replace `your_script.py` with the actual script name
CMD ["python", "test.py"]
