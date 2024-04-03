import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def missing_values_column(df):
    missing_values_per_column = df.isnull().sum(axis=1)
    missing_percentage_per_column = (missing_values_per_column / len(df)) * 100

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
    duplicate_rows = df.duplicated(subset='Cod_do_Material', keep=False)
    df = df[duplicate_rows]
    return df


def get_rows_with_less_null_values(df):
    df = df.copy()
    missing_values_per_row = df.isnull().sum(axis=1)
    df['missing_values'] = missing_values_per_row
    df = df.sort_values(by='missing_values')
    return df

df = pd.read_csv("ListaEBs.csv")

df_duplicated = get_duplicate_rows(df)

distinct_values_list = df_duplicated['Cod_do_Material'].unique().tolist()

new_df = pd.DataFrame()

for cod in distinct_values_list:
    less_null = get_rows_with_less_null_values(df_duplicated[df_duplicated['Cod_do_Material'] == cod])
    new_df = pd.concat([new_df, less_null.iloc[[0]]])
print(df_duplicated)
print(new_df)