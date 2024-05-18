from db_connection import DBConnection
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
from encoding import target_encoding


missing_col = ['BitolaCaboAterramentoCarcaca [mm2]','ChoqueTermico','DiametroAnelCurto [mm]','DiametroExternoEstator [mm]','DiametroUsinadoRotor [mm]','LarguraAnelCurto [mm]','NrTotalFiosEnrol']

def impute_with_random_forest(df):
    df = target_encoding(df)
    df_changed = df.copy()
    imputer = IterativeImputer(estimator=RandomForestRegressor())
    complete_data = imputer.fit_transform(df_changed[df.columns].to_numpy(dtype='float64'))
    complete_data_df = pd.DataFrame(complete_data, columns=df.columns)
    return complete_data_df

def impute_with_bayesian_ridge(df):
    df = target_encoding(df)
    df_changed = df.copy()
    imputer = IterativeImputer(estimator=BayesianRidge())
    complete_data = imputer.fit_transform(df_changed[df.columns].to_numpy(dtype='float64'))
    complete_data_df = pd.DataFrame(complete_data, columns=df.columns)
    return complete_data_df


if __name__ == "__main__":
    db_connection = DBConnection()
    df = db_connection.get_dataframe()
    df = impute_with_random_forest(df)
    print(df.isna().sum().sum())
    #df = impute_with_bayesian_ridge(df, df.columns)
    #from IPython import embed; embed()