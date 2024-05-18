from db_connection import DBConnection
from encoding import target_encoding, binary_encoding

from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
import argparse
import pandas as pd
import mlflow
import json

MODEL_MAPPER = {
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
    "svm": SVR
}

IMPUTATION_MAPPER = {  # TODO
    "BayesianRidge": None,
    "RandomForest": None,
    "NoImputation": lambda _df: _df.dropna()
}

ENCODING_MAPPER = {
    "OneHotEncoding": binary_encoding,
    "TargetEncoding": target_encoding
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model"     , choices=MODEL_MAPPER.keys()                 , type=str, help="Model to use"            )
    parser.add_argument("--imputation", choices=IMPUTATION_MAPPER.keys()            , type=str, help="Imputation method to use")
    parser.add_argument("--encoding"  , choices=("OneHotEncoding", "TargetEncoding"), type=str, help="Encoding method to use"  )
    parser.add_argument("--params"                                                  , type=json.loads,
                        help=""" Parameters for the model. E.g, '{"learning_rate": [0.01, 0.1], "n_estimators": [50, 100]}' """             )
    parser.add_argument("--run_name"                                                , type=str, help="Name of the run"         )
    parser.add_argument("--experiment_name"                                         , type=str, help="Name of the experiment"  )
    args = parser.parse_args()
    # example:
    # python modeling/train_script.py --model random_forest --imputation NoImputation --encoding TargetEncoding --params '{"n_estimators": [10, 100, 1000], "max_depth": [3, 5, 10]}' --run_name random_forest --experiment_name default

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name=args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        seed = hash(args.run_name + args.experiment_name)
        mlflow.log_param("seed", seed)

        db_connection = DBConnection()
        df = db_connection.get_dataframe()
        df = IMPUTATION_MAPPER[args.imputation](df)  # TODO: Wait for Bruno
        df = ENCODING_MAPPER[args.encoding](df)
        y = df["CustoIndustrial"].to_numpy(dtype=float)
        x = df.drop("CustoIndustrial", axis=1).to_numpy(dtype=float)

        from IPython import embed
        embed()

        scaler = StandardScaler()
        model = MODEL_MAPPER[args.model](random_state=seed)
        pipeline = make_pipeline(scaler, model)
        search = HalvingGridSearchCV(
            estimator=pipeline,
            scoring=root_mean_squared_error,
            param_grid=args.params
        )
        results = search.fit(x, y)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("rmse", search.best_score_)
        best_model = search.best_estimator_

        signature = mlflow.models.infer_signature(x, best_model.predict(x))
        mlflow.sklearn.log_model(best_model, "best_model", signature=signature)

        cv_results_df = pd.DataFrame(results.cv_results_)
        cv_results_df.to_csv('cv_results.csv', index=False)
        mlflow.log_artifact('cv_results.csv')





