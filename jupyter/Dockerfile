FROM jupyter/datascience-notebook
LABEL authors="prft"

USER root

RUN apt-get update && \
    apt-get install -y pkg-config libmysqlclient-dev

USER jovyan

COPY ../requirements.txt ./
RUN pip install -r requirements.txt