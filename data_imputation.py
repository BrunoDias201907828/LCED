from db_connection import DBConnection
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

string_columns = ["DescricaoComponente", "CabosProtecaoTermica", "CarcacaPlataformaEletricaRaw", 
    "CarcacaPlataformaEletricaComprimento", "CodigoDesenhoEstatorCompleto", "CodigoDesenhoDiscoEstator", 
    "CodigoDesenhoDiscoRotor", "EsquemaBobinagem", "GrupoCarcaca", "LigacaoDosCabos01", "MaterialChapa", 
    "MaterialIsolFio01Enrol01", "PolaridadeChapa", "PotenciaCompletaCv01", 
    "TipoLigacaoProtecaoTermica", "PassoEnrolamento01", "TipoDeImpregnacao"]


def apply_ordinal_encoding(df):
    encoder = OrdinalEncoder(min_frequency=10)
    for col in string_columns:
        missing_mask = df[col].isna()
        df[col] = df[col].astype(str)         
        df[col] = encoder.fit_transform(df[[col]])  
        df.loc[missing_mask, col] = np.nan 

    return df

def impute_with_random_forest(df, columns):
    from IPython import embed; embed()
    imputer = IterativeImputer(estimator=RandomForestRegressor())
    df_changed = apply_ordinal_encoding(df)
    complete_data = imputer.fit_transform(df_changed[columns].to_numpy())
    embed()
    df[string_columns] = complete_data[string_columns]
    return df


#ACRESCENTAR AQUI FUNCOES TODAS PARA INT


def impute_with_random_forest(df, columns):
    # Identify datetime columns
    datetime_columns = df.select_dtypes(include=[np.datetime64]).columns
    non_datetime_columns = [col for col in columns if col not in datetime_columns]
    
    # Impute datetime columns (choose a strategy)
    for col in datetime_columns:
        df[col].fillna(pd.Timestamp.min, inplace=True)  # Or your preferred method
    
    # Apply ordinal encoding on the rest of the data
    df_changed = apply_ordinal_encoding(df)

    # Impute with RandomForestRegressor using only non datetime columns
    imputer = IterativeImputer(estimator=RandomForestRegressor())
    complete_data = imputer.fit_transform(df_changed[non_datetime_columns].to_numpy())
    df[non_datetime_columns] = complete_data 

    return df

#def impute_with_bayesian_ridge(df, columns):
    imputer = IterativeImputer(estimator=BayesianRidge())
    df[columns] = imputer.fit_transform(df[columns])
    return df


if __name__ == "__main__":
    db_connection = DBConnection()
    df = db_connection.get_dataframe()

    df = impute_with_random_forest(df, df.columns)
    #df = impute_with_bayesian_ridge(df, df.columns)
    from IPython import embed; embed()