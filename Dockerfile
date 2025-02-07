# Use a Python base image
FROM python:3.10.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application code
COPY main.py /app/
COPY settings.py /app/
COPY .env /app/

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (optional, if needed)
# ENV ENV_VAR_NAME=value

# Command to run your Python application
CMD ["python", "main.py"]

