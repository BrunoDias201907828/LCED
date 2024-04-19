FROM ghcr.io/mlflow/mlflow
LABEL authors="prft"

RUN groupadd -r mlflowgroup && useradd -r -g mlflowgroup mlflowuser

# Create the /app directory and immediately set ownership
RUN mkdir -p /app && chown -R mlflowuser:mlflowgroup /app
RUN mkdir -p /app/mlflow && chown -R mlflowuser:mlflowgroup /app/mlflow
RUN mkdir -p /app/mlflow_artifacts && chown -R mlflowuser:mlflowgroup /app/mlflow_artifacts

USER mlflowuser
EXPOSE 5000

CMD mlflow server --serve-artifacts --host 0.0.0.0 --backend-store-uri file:///app/mlflow --artifacts-destination file:///app/mlflow_artifacts
