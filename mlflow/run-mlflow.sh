docker run -d \
  -p 5000:5000 \
  -v mlflow_artifacts:/app/mlflow_artifacts:Z \
  -v mlflow:/app/mlflow:Z \
  mlflow

