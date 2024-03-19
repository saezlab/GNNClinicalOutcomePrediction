# Use the base image with GPU support
FROM nvcr.io/nvidia/pyg:24.01-py3

WORKDIR /
# Clone your repository from GitHub
RUN git clone https://github.com/saezlab/GNNClinicalOutcomePrediction.git
WORKDIR /GNNClinicalOutcomePrediction
RUN mkdir -p /GNNClinicalOutcomePrediction/models/JacksonFischer
RUN mkdir -p /GNNClinicalOutcomePrediction/data/JacksonFischer

# Install the packages
RUN pip install scanpy==1.9.8
RUN pip install pytorch-lightning==2.1.2
RUN pip install leidenalg==0.10.2

# Set the working directory



# Optionally, you can copy any additional files required for your application
# COPY <local_path> <container_path>

# Command to run your script (assuming it's named train.py)
# COPY /home/rifaioglu/projects/GNNClinicalOutcomePrediction/data/JacksonFischer/month /GNNClinicalOutcomePrediction/


# Add tag
LABEL py_gem="true"

