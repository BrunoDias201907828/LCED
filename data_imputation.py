from db_connection import DBConnection
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import make_pipeline


def build_simple_imputers(strategies):
    return {
        strategy: SimpleImputer(strategy=strategy) for strategy in strategies
    }


def build_iterative_imputers(estimators):
    return {
        estimator.__name__: IterativeImputer(estimator=estimator()) for estimator in estimators
    }


def build_imputers():
    imputers = build_simple_imputers(strategies=["mean", "median"])
    iterative_imputers = build_iterative_imputers(estimators=[BayesianRidge, RandomForestRegressor])
    imputers.update(iterative_imputers)
    return imputers


if __name__ == "__main__":
    db_connection = DBConnection()
    df = db_connection._get_dataframe_cleaned()

    from IPython import embed
    embed()

    imputers = build_imputers()