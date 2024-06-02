from db_connection import DBConnection
from encoding import target_encoding, binary_encoding
from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name="linear")

with mlflow.start_run(run_name="linear"):
    # seed = hash(args.run_name + args.experiment_name)
    db_connection = DBConnection()
    df = db_connection.get_dataframe()
    df.dropna(inplace=True)
    y = df["CustoIndustrial"].to_numpy(dtype=float)
    x = df['volume_estator'].to_numpy(dtype=float)[..., None]
    model = LinearRegression()
    scores = cross_val_score(model, x, y, cv = 5, scoring='neg_root_mean_squared_error')
    mlflow.log_metric("rmse", scores.mean())
    mlflow.log_metric("rmse_std", scores.std())

    columns = ["mean_test_score", "iter",
               "split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score", "split4_test_score"]
    row = [scores.mean(), 0, *scores]
    df_mlflow = pd.DataFrame([row], columns=columns)
    df_mlflow.to_csv("cv_results.csv")
    mlflow.log_artifact('cv_results.csv')
