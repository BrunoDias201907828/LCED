from db_connection import DBConnection
from encoding import CATEGORICAL_COLUMNS
from modeling.utils import KMeansTransformer

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.utils import resample
from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import category_encoders as ce
import argparse
import pandas as pd
import mlflow
import json
import os
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
TOLERANCE = 0.05
MODEL_MAPPER = {
    "linear_regression": LinearRegression,  # TODO
    "elastic_net": ElasticNet,  # TODO - l1_ratio 0, 0.5, 1 (ridge, elastic, lasso)
    "decision_tree": DecisionTreeRegressor,  # TODO
    "bayesian_ridge": BayesianRidge,  # TODO
    "sgd": SGDRegressor,  # TODO

    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "svm": SVR,

    "bagging": BaggingRegressor(estimator=LinearRegression()),  # TODO - choose estimator
    "adaboost": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3)),  # TODO - choose estimator
}
ENCODING_MAPPER = {
    "BinaryEncoding": ce.BinaryEncoder(cols=CATEGORICAL_COLUMNS),
    "TargetEncoding": ce.TargetEncoder(cols=CATEGORICAL_COLUMNS)
}
FEATURES_KMEANS = [
    "QuantidadeComponente",
    "BitolaCaboAterramentoCarcaca [mm2]",
    "BitolaCabosDeLigacao [mm2]",
    "DiametroExternoEstator [mm]",
    "DiametroUsinadoRotor [mm]",
    "ComprimentoTotalPacote [mm]",
]

def calculate_confidence_intervals(data, confidence_level=95):
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return lower_bound, upper_bound


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model"     , choices=MODEL_MAPPER.keys()                 , type=str, help="Model to use"            )
    parser.add_argument("--imputation", action='store_true'                                   , help="Use imputation or not"   )
    parser.add_argument("--external"  , action='store_true'                                   , help="Include external data"   )
    parser.add_argument("--encoding"  , choices=("BinaryEncoding", "TargetEncoding"), type=str, help="Encoding method to use"  )
    parser.add_argument("--params_path"                                             , type=str, help="Parameters for the model")
    parser.add_argument("--run_name"                                                , type=str, help="Name of the run"         )
    parser.add_argument("--experiment_name"                                         , type=str, help="Name of the experiment"  )
    args = parser.parse_args()


    # python3 modeling/bootstrap.py --model xgboost --external --encoding BinaryEncoding --params modeling/params_bootstrap.json --run_name xgboost_best --experiment_name bootstrap
    # python3 modeling/cluster_evaluation.py --model xgboost --external --encoding BinaryEncoding --params modeling/params_bootstrap.json --run_name test_cluster --experiment_name test_cluster


    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        db_connection = DBConnection()
        df = db_connection.get_dataframe(include_external=args.external)
        df = df.rename(str, axis="columns")
        # Reorder columns so that the kmeans is performed with the correct columns
        no_kmeans_columns = [c for c in df.columns if c not in FEATURES_KMEANS]
        df = df[FEATURES_KMEANS + no_kmeans_columns]
        if not args.imputation:
            df = df.dropna()
        y = df["CustoIndustrial"]
        x = df.drop("CustoIndustrial", axis=1)

        steps = (
            [
                ("encoder", ENCODING_MAPPER[args.encoding]),  # output - dataframe with NaNs
                ("scaler", StandardScaler())  # output - numpy array with NaNs
            ] +
            ([("imputer", IterativeImputer(tol=TOLERANCE, estimator=None))] if args.imputation else []) +  # output - numpy array w/out NaNs
            [
                ("kmeans", KMeansTransformer(tuple(x.columns.get_loc(feat) for feat in FEATURES_KMEANS))),
                ("model", MODEL_MAPPER[args.model]())
            ]
        )
        pipeline = Pipeline(steps=steps)

        with open(args.params_path, "r") as f:
            params = json.load(f)
        param_grid = {"model__" + key: value for key, value in params.items()}
        if args.imputation:
            param_grid.update({"imputer__estimator": [RandomForestRegressor(), BayesianRidge()]})

        n_iterations = 5000
        mae_scores = {"all": [], "small": [], "big": []}
        mape_scores = {"all": [], "small": [], "big": []}
        r2_scores = {"all": [], "small": [], "big": []}
        rmse_scores = {"all": [], "small": [], "big": []}
        results = []

        for i in range(n_iterations):

            x_train, y_train = resample(x, y, replace=True, n_samples=int(0.8 * len(x)))

            x_test = x.loc[~x.index.isin(x_train.index)]
            y_test = y.loc[~y.index.isin(y_train.index)]

            pipeline.fit(x_train, y_train)
            pipeline_cluster = Pipeline(steps=steps[:-1])
            x_test_cluster = pipeline_cluster.transform(x_test)
            cluster = x_test_cluster[:, -1]
            y_pred = pipeline.predict(x_test)
            
            k_means = pipeline["kmeans"]
            centroids = k_means.kmeans.cluster_centers_

            if np.sum(centroids[0]) > np.sum(centroids[1]):
                cluster_names = ["small", "big"]
            else:
                cluster_names = ["big", "small"]

            for subset in ["all", "small", "big"]:
                if subset == "all":
                    _y_test = y_test
                    _y_pred = y_pred
                elif subset == "small":
                    index = cluster_names.index("small")
                    _y_test = y_test[cluster == index]
                    _y_pred = y_pred[cluster == index]
                elif subset == "big":
                    index = cluster_names.index("big")
                    _y_test = y_test[cluster == index]
                    _y_pred = y_pred[cluster == index]

                mae = mean_absolute_error(_y_test, _y_pred)
                mape = mean_absolute_percentage_error(_y_test, _y_pred)
                r2 = r2_score(_y_test, _y_pred)
                rmse = root_mean_squared_error(_y_test, _y_pred)

                mae_scores[subset].append(mae)
                mape_scores[subset].append(mape)
                r2_scores[subset].append(r2)
                rmse_scores[subset].append(rmse)

                results[subset].append({
                    "subset": subset,
                    "iteration": i,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "rmse": rmse
                })

            # mlflow.log_metric(f"mse_bootstrap_{i}", mae)
            # mlflow.log_metric(f"mape_bootstrap_{i}", mape)
            # mlflow.log_metric(f"r2_bootstrap_{i}", r2)
            # mlflow.log_metric(f"rmse_bootstrap_{i}", rmse)

        mlflow.log_param("model", args.model)
        mlflow.log_param("encoding", args.encoding)
        mlflow.log_param("imputation", args.imputation)
        mlflow.log_param("params", params)
        mlflow.sklearn.log_model(pipeline, "model")

        for subset in ["all", "small", "big"]:
            mae_lower, mae_upper = calculate_confidence_intervals(mae_scores[subset])
            mape_lower, mape_upper = calculate_confidence_intervals(mape_scores[subset])
            r2_lower, r2_upper = calculate_confidence_intervals(r2_scores[subset])
            rmse_lower, rmse_upper = calculate_confidence_intervals(rmse_scores[subset])

            mlflow.log_metric(f"mean_mse_{subset}", np.mean(mae_scores))
            mlflow.log_metric(f"mean_mape_{subset}", np.mean(mape_scores))
            mlflow.log_metric(f"mean_r2_{subset}", np.mean(r2_scores))
            mlflow.log_metric(f"mean_rmse_{subset}", np.mean(rmse_scores))

            mlflow.log_metric(f"mae_ci_lower_{subset}", mae_lower)
            mlflow.log_metric(f"mae_ci_upper_{subset}", mae_upper)
            mlflow.log_metric(f"mape_ci_lower_{subset}", mape_lower)
            mlflow.log_metric(f"mape_ci_upper_{subset}", mape_upper)
            mlflow.log_metric(f"r2_ci_lower_{subset}", r2_lower)
            mlflow.log_metric(f"r2_ci_upper_{subset}", r2_upper)
            mlflow.log_metric(f"rmse_ci_lower_{subset}", rmse_lower)
            mlflow.log_metric(f"rmse_ci_upper_{subset}", rmse_upper)

        results_df = pd.DataFrame(results)
        results_df.to_csv('results.csv', index=False)
        mlflow.log_artifact('results.csv')

        # print(f"Model: {args.model}, Mean MAE: {np.mean(mae_scores)}, Mean MAPE: {np.mean(mape_scores)} , Mean R²: {np.mean(r2_scores)}, Mean RMSE: {np.mean(rmse_scores)}")
        # print(f"95% CI for MAE: [{mae_lower}, {mae_upper}]")
        # print(f"95% CI for MAPE: [{mape_lower}, {mape_upper}]")
        # print(f"95% CI for R²: [{r2_lower}, {r2_upper}]")
        # print(f"95% CI for RMSE: [{rmse_lower}, {rmse_upper}]")