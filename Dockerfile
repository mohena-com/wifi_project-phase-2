FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy environment file and install dependencies
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate conda environment for future commands
SHELL ["conda", "run", "-n", "har_env", "/bin/bash", "-c"]

# Copy all project files into /app
COPY . .

# Default entrypoint (can be overridden)
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "har_env", "python", "src/real_time_inference.py"]
