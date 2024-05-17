FROM python:3
LABEL authors="prft"

WORKDIR ./git-repos
COPY requirements.txt ./

RUN pip install -r requirements.txt
ENV PYTHONPATH "${PYTHONPATH}:/git-repos"

ENTRYPOINT ["/bin/bash"]