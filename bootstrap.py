from db_connection import DBConnection
from encoding import CATEGORICAL_COLUMNS
from modeling.utils import KMeansTransformer

from sklearn.metrics import mean_squared_error, r2_score
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
    # example:
    # python modeling/train_script.py --model random_forest --imputation NoImputation --encoding TargetEncoding --params '{"n_estimators": [10, 100, 1000], "max_depth": [3, 5, 10]}' --run_name random_forest --experiment_name default



    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        db_connection = DBConnection()
        df = db_connection.get_dataframe(include_external=args.external)
        df = df.rename(str, axis="columns")
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


        n_iterations = 50
        mse_scores = []
        r2_scores = []

        for _ in range(n_iterations):
            x_sample, y_sample = resample(x, y, replace=True, random_state=42, n_samples=len(x))

            x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.2, random_state=42)

            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            r2_scores.append(r2)

            mlflow.log_metric("mse_bootstrap", mse)
            mlflow.log_metric("r2_bootstrap", r2)
        mlflow.log_param("model", args.model)
        mlflow.log_param("encoding", args.encoding)
        mlflow.log_param("imputation", args.imputation)
        mlflow.log_param("params", params)
        mlflow.log_metric("mean_mse", np.mean(mse_scores))
        mlflow.log_metric("mean_r2", np.mean(r2_scores))
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Model: {args.model}, Mean MSE: {np.mean(mse_scores)}, Mean R²: {np.mean(r2_scores)}")

        mlflow.end_run()