# Use the base image with GPU support
FROM nvcr.io/nvidia/pyg:24.01-py3

WORKDIR /
# Clone your repository from GitHub
RUN git clone https://github.com/saezlab/GNNClinicalOutcomePrediction.git
WORKDIR /GNNClinicalOutcomePrediction
RUN mkdir -p /GNNClinicalOutcomePrediction/models/JacksonFischer
RUN mkdir -p /GNNClinicalOutcomePrediction/data/JacksonFischer
RUN mkdir -p /GNNClinicalOutcomePrediction/data/out_data/adatafiles
RUN mkdir -p /GNNClinicalOutcomePrediction/models/METABRIC
RUN mkdir -p /GNNClinicalOutcomePrediction/data/METABRIC
RUN mkdir -p /GNNClinicalOutcomePrediction/data/out_data/adatafiles

# Install the packages
RUN pip install scanpy==1.9.8
RUN pip install pytorch-lightning==2.1.2
RUN pip install leidenalg==0.10.2

# Set the working directory



# Copy JacksonFischer
COPY ./models/JacksonFischer/model_hyperparams.json  /GNNClinicalOutcomePrediction/models/JacksonFischer
COPY ./data/JacksonFischer/month/  /GNNClinicalOutcomePrediction/data/JacksonFischer/month
COPY ./data/JacksonFischer/raw/  /GNNClinicalOutcomePrediction/data/JacksonFischer/raw
COPY ./data/JacksonFischer/folds.json /GNNClinicalOutcomePrediction/data/JacksonFischer/
COPY ./data/JacksonFischer/*.csv /GNNClinicalOutcomePrediction/data/JacksonFischer/

# Copy METABRIC
COPY ./models/METABRIC/model_hyperparams.json  /GNNClinicalOutcomePrediction/models/METABRIC
COPY ./data/METABRIC/month/  /GNNClinicalOutcomePrediction/data/METABRIC/month
COPY ./data/METABRIC/raw/  /GNNClinicalOutcomePrediction/data/METABRIC/raw
COPY ./data/METABRIC/folds.json /GNNClinicalOutcomePrediction/data/METABRIC/
COPY ./data/METABRIC/*.csv /GNNClinicalOutcomePrediction/data/METABRIC/
COPY ./data/METABRIC/*.txt /GNNClinicalOutcomePrediction/data/METABRIC/
COPY ./data/METABRIC/*.xlsx /GNNClinicalOutcomePrediction/data/METABRIC/



# Add tag
LABEL py_gem="true"

