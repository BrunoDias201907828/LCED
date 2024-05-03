import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from db_connection import DBConnection


def get_duplicate_rows(df):
    duplicate_rows = df.duplicated(subset='CodigoMaterial', keep=False)
    df = df[duplicate_rows]
    return df

def get_rows_with_less_null_values(df):
    df = df.copy()
    missing_values_per_row = df.isnull().sum(axis=1)
    df['missing_values'] = missing_values_per_row
    df = df.sort_values(by='missing_values')
    return df

def remove_duplicated_rows(df):

    df_duplicated = get_duplicate_rows(df)
    distinct_values_list = df_duplicated['CodigoMaterial'].unique().tolist()
    new_df = pd.DataFrame()

    for cod in distinct_values_list:
        less_null = get_rows_with_less_null_values(df_duplicated[df_duplicated['CodigoMaterial'] == cod])
        new_df = pd.concat([new_df, less_null.iloc[[0]]])  

    df_keep = new_df.drop(columns=['missing_values'])

    merged_df = pd.merge(df_duplicated, df_keep, how='outer', indicator=True)
    df_to_drop = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    merged_df2 = pd.merge(df, df_to_drop, how = 'outer', indicator=True)
    df_distinct = merged_df2[merged_df2['_merge'] == 'left_only'].drop(columns=['_merge'])

    return df_distinct

def print_value_counts(df):
    for column in df.columns:
        print(f"Column: {column}")
        print(df[column].value_counts())
        print("\n")

def drop_columns(df):
    columns_to_remove = ['NumeroDeFases', 'ProcessoFabricacao', 'NivelRendEficiencia', 'CodigoMaterial', 'CodigoMaterialFio01Enrol01', 'NumeroDesenho', 'TerminalLigacao', 'IdEstatortInsertado']
    df = df.drop(columns_to_remove, axis=1)
    return df

def replace_strings(df, column, str1, str2):
    df[column] = df[column].replace({str1: '0', str2: '1'})
    df[column] = df[column].astype('Int64')
    return df

def replace_values(df, column, val1, val2):
    df[column] = df[column].replace({val1: 0, val2: 1})
    return df

def convert_cols_to_int(df):
    df = replace_strings(df, 'TipoEstatorBobinado', 'DE LINHA ESPECIAL', 'DE TABELA DE VALORES')
    df = replace_strings(df, 'ClassIsolamento', 'F', 'H')
    df = replace_values(df, 'NumeroEnrolamentoMotor', 1, 2) 
    return df

def convert_to_boolean(df, column):
    df[column] = df[column].astype('boolean')
    return df

def convert_cols_to_boolean(df):
    df = convert_to_boolean(df, 'TipoEstatorBobinado')
    df = convert_to_boolean(df, 'ClassIsolamento')
    df = convert_to_boolean(df, 'CabosLigacaoEmParalelo')
    df = convert_to_boolean(df, 'NumeroEnrolamentoMotor')
    return df

def replace_single_occurrences(df):
    columns = df.columns.tolist()
    columns.remove('CustoIndustrial')
    for column in columns:
        counts = df[column].value_counts()
        df[column] = df[column].map(lambda x: x if pd.isna(x) else ('Outros' if counts[x] == 1 else x))
    return df

def termica_solved(df):
    #df['CabosProtecaoTermica']=df['CabosProtecaoTermica'].astype(str)
    df['TipoLigacaoProtecaoTermica'] = df['TipoLigacaoProtecaoTermica'].fillna('Nao Aplicavel')
    df['CabosProtecaoTermica'] = df['CabosProtecaoTermica'].astype('string').fillna('Nao Aplicavel')
    return df

def missing_values_heatmap(df): #only for columns and rows where there are missing values, not whole df
    missing_rows = df[df.isnull().any(axis=1)]
    missing_cols = missing_rows.loc[:, missing_rows.isnull().any()]

    plt.figure(figsize=(8, 6))
    sns.heatmap(missing_cols.isnull(), cmap='viridis', cbar=False)
    plt.title('Missing Values Heatmap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def df_changed(df):
    df = remove_duplicated_rows(df)
    df = drop_columns(df)    
    df = df.drop([2478,3882])
    df = convert_cols_to_int(df)
    df = convert_cols_to_boolean(df)
    df = termica_solved(df)
    #df = replace_single_occurrences(df)

    return df

def select_columns(df, column_names):
    return df[column_names]


if __name__ == '__main__':

    db = DBConnection()
    df = db.get_dataframe_with_extracted_features()
    
    #selected_columns_df = select_columns(df, ['Descricao', 'CabosProtecaoTermica', 'TipoLigacaoProtecaoTermica'])
    #na_rows_df = selected_columns_df[selected_columns_df['CabosProtecaoTermica'].isna() | selected_columns_df['TipoLigacaoProtecaoTermica'].isna()]

    df_change = df_changed(df)

    # ChoqueTermico DiametroAnelCurto [mm] DiametroUsinadoRotor [mm] LarguraAnelCurto [mm] NrTotalFiosEnrol TipoDeImpregnacao

    #df_change = df_change.drop(columns=['TipoLigacaoProtecaoTermica', 'CabosProtecaoTermica'])

    #Fazer imputaçao a 'BitolaCaboAterramentoCarcaca [mm2]' e a 'DiametroExternoEstator [mm]'

    from IPython import embed; embed()
    