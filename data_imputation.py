from db_connection import DBConnection
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from encoding import target_encoding


missing_col = ['BitolaCaboAterramentoCarcaca [mm2]','ChoqueTermico','DiametroAnelCurto [mm]','DiametroExternoEstator [mm]','DiametroUsinadoRotor [mm]','LarguraAnelCurto [mm]','NrTotalFiosEnrol']

# def convert_to_int(df):
#     columns = ['ChoqueTermico', 'ClassIsolamento', 'CabosLigacaoEmParalelo', 'NumeroEnrolamentoMotor', 'UsoDoTerminal', 'TipoEstatorBobinado', 'TipoDeImpregnacao']
#     for col in columns:
#         df[col] = df[col].astype('Int64')
#     return df

# def apply_ordinal_encoding(df):
#     string_columns = ["DescricaoComponente", "CabosProtecaoTermica", "CarcacaPlataformaEletricaRaw", 
#         "CarcacaPlataformaEletricaComprimento",
#         "CodigoDesenhoDiscoRotor", "EsquemaBobinagem", "GrupoCarcaca", "LigacaoDosCabos01", "MaterialChapa", 
#         "MaterialIsolFio01Enrol01", "PolaridadeChapa", 
#         "TipoLigacaoProtecaoTermica", "PassoEnrolamento01"]
#     encoder = OrdinalEncoder(min_frequency=10)
#     for col in string_columns:
#         missing_mask = df[col].isna()
#         df[col] = df[col].astype(str)         
#         df[col] = encoder.fit_transform(df[[col]])  
#         df.loc[missing_mask, col] = np.nan
#     return df

# def get_missing_values_info(df):
#     missing_values = df.isna().sum()
#     missing_values = missing_values[missing_values > 0]
#     missing_values_info = pd.DataFrame({
#         'column_type': df[missing_values.index].dtypes,
#         'missing_values': missing_values
#     })
#     return missing_values_info

def impute_with_random_forest(df):
    df_changed = df.copy()
    # df_changed = apply_ordinal_encoding(df)
    # df_changed = convert_to_int(df_changed)
    df_changed = target_encoding(df_changed)

    from IPython import embed; embed()
    imputer = IterativeImputer(estimator=RandomForestRegressor())
    complete_data = imputer.fit_transform(df_changed[df.columns].to_numpy(dtype='float64'))
    complete_data_df = pd.DataFrame(complete_data, columns=df.columns)
    df[missing_col] = complete_data_df[missing_col]
    return df

def impute_with_bayesian_ridge(df):
    df_changed = df.copy()
    df_changed = target_encoding(df_changed)
    imputer = IterativeImputer(estimator=RandomForestRegressor())
    complete_data = imputer.fit_transform(df_changed[df.columns].to_numpy(dtype='float64'))
    complete_data_df = pd.DataFrame(complete_data, columns=df.columns)
    df[missing_col] = complete_data_df[missing_col]
    return df


if __name__ == "__main__":
    db_connection = DBConnection()
    df = db_connection.get_dataframe()
    #print(get_missing_values_info(df))
    df = impute_with_random_forest(df)
    #df = impute_with_bayesian_ridge(df, df.columns)
    from IPython import embed; embed()