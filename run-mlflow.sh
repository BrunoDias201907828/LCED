docker run -d -v /home/lced1/mlflow:/data --network=host ghcr.io/mlflow/mlflow mlflow server --serve-artifacts --host 0.0.0.0 --backend-store-uri file:///data --default-artifact-root file:///data