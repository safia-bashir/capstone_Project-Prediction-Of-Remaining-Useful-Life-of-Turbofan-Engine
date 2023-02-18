# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements-backend.txt requirements.txt

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the code to the container
COPY . /app

# Set the environment variable for TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api_final:app", "--host", "0.0.0.0", "--port", "8000"]
