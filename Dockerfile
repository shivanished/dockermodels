# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Create directories for the models and ensure directory names do not have spaces
RUN mkdir -p /app/Model1 /app/Model2

# Copy model files to their respective directories
COPY ["Model 1/diet_recipe_rnn_model.h5", "/app/Model1/diet_recipe_rnn_model.h5"]
COPY ["Model 2/taste_trio_rnn_model.h5", "/app/Model2/taste_trio_rnn_model.h5"]

# Copy tokenizer word index files if they are part of your application
COPY ["Model 1/diet_tokenizer_word_index.json", "/app/Model1/diet_tokenizer_word_index.json"]
COPY ["Model 2/taste_trio_tokenizer_word_index.json", "/app/Model2/taste_trio_tokenizer_word_index.json"]

# Install system dependencies, including libraries needed by h5py
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libpq-dev \
    libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables
ENV PORT=8080
ENV HOST=0.0.0.0

# Command to run the application
CMD ["python", "app.py"]
