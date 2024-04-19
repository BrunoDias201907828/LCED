import mlflow


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    mlflow.set_experiment("ptavares")
    with mlflow.start_run(run_name="test1"):
        mlflow.log_param("use_mysql", False)
        mlflow.log_dict({"user": "user1", "passwd": "bE97XnZzmF", "host": "lced-data.fe.up.pt", "database": "weg_a"}, "data.json")

