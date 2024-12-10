# Use the official Python slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies for dlib and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the dlib wheel file from the local directory to the container
COPY libs/*.whl /app/libs/

# Copy the requirements file to the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir /app/libs/*.whl

# Copy the application code into the container
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
