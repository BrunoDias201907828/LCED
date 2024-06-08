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
        mae_scores = []
        mape_scores = []
        r2_scores = []
        rmse_scores = []
        results = []

        for i in range(n_iterations):

            x_train, y_train = resample(x, y, replace=True, n_samples=int(0.8 * len(x)))

            x_test = x.loc[~x.index.isin(x_train.index)]
            y_test = y.loc[~y.index.isin(y_train.index)]

            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)

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

            # mlflow.log_metric(f"mse_bootstrap_{i}", mae)
            # mlflow.log_metric(f"mape_bootstrap_{i}", mape)
            # mlflow.log_metric(f"r2_bootstrap_{i}", r2)
            # mlflow.log_metric(f"rmse_bootstrap_{i}", rmse)

        mae_lower, mae_upper = calculate_confidence_intervals(mae_scores)
        mape_lower, mape_upper = calculate_confidence_intervals(mape_scores)
        r2_lower, r2_upper = calculate_confidence_intervals(r2_scores)
        rmse_lower, rmse_upper = calculate_confidence_intervals(rmse_scores)

        mlflow.log_param("model", args.model)
        mlflow.log_param("encoding", args.encoding)
        mlflow.log_param("imputation", args.imputation)
        mlflow.log_param("params", params)
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

        mlflow.sklearn.log_model(pipeline, "model")

        results_df = pd.DataFrame(results)
        results_df.to_csv('results.csv', index=False)
        mlflow.log_artifact('results.csv')

        print(f"Model: {args.model}, Mean MAE: {np.mean(mae_scores)}, Mean MAPE: {np.mean(mape_scores)} , Mean R²: {np.mean(r2_scores)}, Mean RMSE: {np.mean(rmse_scores)}")
        print(f"95% CI for MAE: [{mae_lower}, {mae_upper}]")
        print(f"95% CI for MAPE: [{mape_lower}, {mape_upper}]")
        print(f"95% CI for R²: [{r2_lower}, {r2_upper}]")
        print(f"95% CI for RMSE: [{rmse_lower}, {rmse_upper}]")


        mlflow.end_run()