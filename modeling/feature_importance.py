from db_connection import DBConnection
import mlflow


if __name__ == "__main__":
    path = "mlflow-artifacts:/142205264287169719/a309cce1d995493294d94fbac8a4cee4/artifacts/best_model"
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.sklearn.load_model(path)
    rf = model["model"]
    feature_importance = rf.feature_importances_.tolist()

    db_connection = DBConnection()
    df = db_connection.get_dataframe()
    feature_names = df.columns

    from IPython import embed
    embed()
