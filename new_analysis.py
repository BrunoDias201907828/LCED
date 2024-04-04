import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from db_connection import DBConnection


def missing_values_column(df):
    missing_values_per_column = df.isnull().sum(axis=1)
    missing_percentage_per_column = ((missing_values_per_column / len(df)) * 100).round(2)

    missing_info = pd.concat([missing_values_per_column, missing_percentage_per_column], axis=1)
    missing_info.columns = ['Missing Values Count', 'Percentage']

    return missing_info

def missing_values_row(df):
    missing_values_per_row = df.isnull().sum(axis=1)
    missing_percentage_per_row = (missing_values_per_row / len(df.columns)) * 100

    missing_info = pd.concat([missing_values_per_row, missing_percentage_per_row], axis=1)
    missing_info.columns = ['Missing Values Count', 'Percentage']

    return missing_info

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
    columns_to_remove = ['NumeroDeFases', 'ProcessoFabricacao', 'NivelRendEficiencia', 'CodigoMaterial', 'CodigoMaterialFio01Enrol01', 'NumeroDesenho', 'TerminalLigacao']
    df = df.drop(columns_to_remove, axis=1)
    return df

def replace_strings(df, column, str1, str2):
    df[column] = df[column].replace({str1: '0', str2: '1'})
    df[column] = df[column].astype('Int64')
    return df

def convert_cols_to_int(df):
    df = replace_strings(df, 'TipoEstatorBobinado', 'DE LINHA ESPECIAL', 'DE TABELA DE VALORES')
    df = replace_strings(df, 'TipoLigacaoProtecaoTermica', 'INDEPENDENTE', 'SERIE')
    df = replace_strings(df, 'ClassIsolamento', 'F', 'H')
    return df

def convert_to_boolean(df, column):
    df[column] = df[column].astype('boolean')
    return df

def convert_cols_to_boolean(df):
    df = convert_to_boolean(df, 'TipoEstatorBobinado')
    df = convert_to_boolean(df, 'TipoLigacaoProtecaoTermica')
    df = convert_to_boolean(df, 'ClassIsolamento')
    df = convert_to_boolean(df, 'CabosLigacaoEmParalelo')
    return df

if __name__ == '__main__':

    db = DBConnection()
    df = db.get_dataframe()


    df = remove_duplicated_rows(df)
    df = drop_columns(df)    
    df = df.drop(3882)
    df = convert_cols_to_int(df)

    print(f"Column: {'CabosLigacaoEmParalelo'}")
    print(df['CabosLigacaoEmParalelo'].value_counts())
    df = convert_cols_to_boolean(df)
    print(f"Column: {'CabosLigacaoEmParalelo'}")
    print(df['CabosLigacaoEmParalelo'].value_counts())

