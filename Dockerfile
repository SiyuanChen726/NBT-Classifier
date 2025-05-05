FROM continuumio/miniconda3:latest

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Create conda env
RUN mamba env create -f /app/environment.yml && \
    mamba clean -a

# Activate environment on container startup
SHELL ["bash", "-c"]
RUN echo "conda activate nbtclassifier" >> ~/.bashrc

# Automatically activate env in container sessions
ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "nbtclassifier" ]
CMD [ "/bin/bash" ]
