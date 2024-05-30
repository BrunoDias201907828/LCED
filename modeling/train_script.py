from db_connection import DBConnection
from encoding import CATEGORICAL_COLUMNS
from modeling.utils import KMeansTransformer

from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import category_encoders as ce
import argparse
import pandas as pd
import mlflow
import json
import os
pd.set_option('future.no_silent_downcasting', True)
TOLERANCE = 0.05
MODEL_MAPPER = {
    "linear_regression": LinearRegression,
    "elastic_net": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "bayesian_ridge": BayesianRidge,
    "sgd": SGDRegressor,

    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "svr": SVR,

    "bagging": BaggingRegressor(estimator=BayesianRidge(lambda_1=0.001, lambda_2=1e-6, alpha_1=1e-7, alpha_2=0.001)),
    "adaboost": AdaBoostRegressor(estimator=BayesianRidge(lambda_1=0.001, lambda_2=1e-6, alpha_1=1e-7, alpha_2=0.001)),
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
    parser.add_argument("--model"            , choices=MODEL_MAPPER.keys()                 , type=str, help="Model to use"             )
    parser.add_argument("--imputation"       , action='store_true'                                   , help="Use imputation or not"    )
    parser.add_argument("--external"         , action='store_true'                                   , help="Include external data"    )
    parser.add_argument("--feature_selection", action='store_true'                                   , help="Perform Feature Selection")
    parser.add_argument("--encoding"         , choices=("BinaryEncoding", "TargetEncoding"), type=str, help="Encoding method to use"   )
    parser.add_argument("--params_path"                                                    , type=str, help="Parameters for the model" )
    parser.add_argument("--run"                                                            , type=str, help="Name of the run"          )
    parser.add_argument("--experiment"                                                     , type=str, help="Name of the experiment"   )
    args = parser.parse_args()
    # example:
    # python modeling/train_script.py --model random_forest --imputation NoImputation --encoding TargetEncoding --params '{"n_estimators": [10, 100, 1000], "max_depth": [3, 5, 10]}' --run_name random_forest --experiment_name default

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=args.experiment)

    with mlflow.start_run(run_name=args.run):
        db_connection = DBConnection()
        df = db_connection.get_dataframe(include_external=args.external)
        df = df.rename(str, axis="columns")
        if not args.imputation:
            df = df.dropna()
        y = df["CustoIndustrial"]
        x = df.drop("CustoIndustrial", axis=1)

        feature_selector = SequentialFeatureSelector(
            estimator=BayesianRidge(lambda_1=0.001, lambda_2=1e-6, alpha_1=1e-7, alpha_2=0.001),
            tol=0.00005,
            direction="backward",
        )
        steps = (
            [
                ("encoder", ENCODING_MAPPER[args.encoding]),  # output - dataframe with NaNs
                ("scaler", StandardScaler())  # output - numpy array with NaNs
            ] +
            ([("imputer", IterativeImputer(tol=TOLERANCE, estimator=None))] if args.imputation else []) +  # output - numpy array w/out NaNs
            ([("feature_selector", feature_selector)] if args.feature_selection else []) +
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
            param_grid.update({"imputer__estimator": [RandomForestRegressor(max_depth=3), BayesianRidge()]})

        # search_method = HalvingGridSearchCV
        search_method = GridSearchCV
        search = search_method(
            estimator=pipeline,
            scoring='neg_root_mean_squared_error',
            param_grid=param_grid,
            n_jobs=-1,
            verbose=2
        ).fit(x, y)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("rmse", search.best_score_)
        best_model = search.best_estimator_

        signature = mlflow.models.infer_signature(x, best_model.predict(x))
        mlflow.sklearn.log_model(best_model, "best_model", signature=signature)

        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df.to_csv('cv_results.csv', index=False)
        mlflow.log_artifact('cv_results.csv')





