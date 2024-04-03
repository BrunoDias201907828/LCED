import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def missing_values_column(df):
    missing_values_per_column = df.isnull().sum
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
    duplicate_rows = df.duplicated(subset='Cod_do_Material', keep=False)
    df = df[duplicate_rows]
    return df


def get_rows_with_less_null_values(df):
    df = df.copy()
    missing_values_per_row = df.isnull().sum(axis=1)
    df['missing_values'] = missing_values_per_row
    df = df.sort_values(by='missing_values')
    return df

def convert_value_to_double(value):
    if pd.isnull(value):
        return np.nan  
    
    value = value.replace(",", ".")

    if value.count('.') > 1:
        last_dot_index = value.rindex('.')
        cleaned_value = value[:last_dot_index].replace('.', '') + value[last_dot_index:]
    else:
        cleaned_value = value

    cleaned_value = re.sub(r'[^\d.]+', '', cleaned_value)
    double_value = float(cleaned_value)
    
    return double_value

def convert_to_double(df, columns):
    for column in columns:
        df[column] = df[column].apply(convert_value_to_double)
    return df

def convert_value_to_int(value):
    if pd.isnull(value):
        return np.nan  
    
    cleaned_value = re.sub(r'\D+', '', value)
    int_value = int(cleaned_value)
    
    return int_value

def convert_to_int(df, columns):
    for column in columns:
        df[column] = df[column].apply(convert_value_to_int)
    return df

df = pd.read_csv("ListaEBs.csv")

df_duplicated = get_duplicate_rows(df)

distinct_values_list = df_duplicated['Cod_do_Material'].unique().tolist()

new_df = pd.DataFrame()

for cod in distinct_values_list:
    less_null = get_rows_with_less_null_values(df_duplicated[df_duplicated['Cod_do_Material'] == cod])
    new_df = pd.concat([new_df, less_null.iloc[[0]]])

#print(new_df)
#print(df_duplicated)

df_keep = new_df.drop(columns=['missing_values'])
#print(df_keep)

#rows ro remove
merged_df = pd.merge(df_duplicated, df_keep, how='outer', indicator=True)
df_to_drop = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

#df without duplicates
merged_df2 = pd.merge(df, df_to_drop, how = 'outer', indicator=True)
df_distinct = merged_df2[merged_df2['_merge'] == 'left_only'].drop(columns=['_merge'])
#print(df_distinct)

distinct_value_counts = df_distinct.nunique()
#print(distinct_value_counts)

column_types = df_distinct.dtypes
#print(column_types)

columns_to_double = ['DIAMETRO_ANEL_CURTO', 'DIAMETRO_EXTERNO_ESTATOR','DIAMETRO_USINADO_ROTOR','INCLINACAO_ROTOR',
                      'LARGURA_ANEL_CURTO','COMPRIMENTO_TOTAL_PACOTE']

columns_to_int = ['BITOLA_CABO_ATERRAMEN_CAR CACA','BITOLA_CABOS_DE_LIGACAO']

df_aux = convert_to_int(df_distinct, columns_to_int)
df_converted = convert_to_double(df_aux, columns_to_double)
column_types = df_converted.dtypes
#print(column_types)

numeric_columns = df_converted.select_dtypes(include=['int64', 'float64'])
numeric_columns = numeric_columns.drop(columns=['Cod_do_Material'])

correlation_matrix = numeric_columns.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Columns')
plt.show()