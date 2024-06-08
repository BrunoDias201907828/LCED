from db_connection import DBConnection
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
import mlflow
import json
import os
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name="linear_boot")

def calculate_confidence_intervals(data, confidence_level=95):
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return lower_bound, upper_bound

if __name__ == "__main__":
    with mlflow.start_run(run_name="linear_boot"):
        db_connection = DBConnection()
        df = db_connection.get_dataframe()
        df.dropna(inplace=True)
        y = df["CustoIndustrial"].to_numpy(dtype=float)
        x = df['volume_estator'].to_numpy(dtype=float)[..., None]
        model = LinearRegression()

        n_iterations = 5000
        mae_scores = []
        mape_scores = []
        r2_scores = []
        rmse_scores = []
        results = []

        for i in range(n_iterations):

            x_train, y_train = resample(x, y, replace=True, n_samples=int(0.8 * len(x)))

            mask = ~np.isin(x, x_train).all(axis=1)
            x_test = x[mask]
            y_test = y[mask]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            mae_scores.append(mae)
            mape_scores.append(mape)
            r2_scores.append(r2)
            rmse_scores.append(rmse)

            results.append({
                "iteration": i,
                "mae": mae,
                "mape": mape,
                "r2": r2,
                "rmse": rmse
            })

        mae_lower, mae_upper = calculate_confidence_intervals(mae_scores)
        mape_lower, mape_upper = calculate_confidence_intervals(mape_scores)
        r2_lower, r2_upper = calculate_confidence_intervals(r2_scores)
        rmse_lower, rmse_upper = calculate_confidence_intervals(rmse_scores)

        mlflow.log_param("model", model)
        mlflow.log_metric("mean_mae", np.mean(mae_scores))
        mlflow.log_metric("mean_mape", np.mean(mape_scores))
        mlflow.log_metric("mean_r2", np.mean(r2_scores))
        mlflow.log_metric("mean_rmse", np.mean(rmse_scores))

        mlflow.log_metric("mae_ci_lower", mae_lower)
        mlflow.log_metric("mae_ci_upper", mae_upper)
        mlflow.log_metric("mape_ci_lower", mape_lower)
        mlflow.log_metric("mape_ci_upper", mape_upper)
        mlflow.log_metric("r2_ci_lower", r2_lower)
        mlflow.log_metric("r2_ci_upper", r2_upper)
        mlflow.log_metric("rmse_ci_lower", rmse_lower)
        mlflow.log_metric("rmse_ci_upper", rmse_upper)

        mlflow.sklearn.log_model(model, "model")

        results_df = pd.DataFrame(results)
        results_df.to_csv('results.csv', index=False)
        mlflow.log_artifact('results.csv')
