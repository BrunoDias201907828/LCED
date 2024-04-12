FROM python:3
LABEL authors="prft"

WORKDIR ./home
COPY requirements.txt ./
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]