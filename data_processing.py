import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    columns_to_remove = ['NumeroDeFases', 'ProcessoFabricacao', 'NivelRendEficiencia', 'CodigoMaterial', 'CodigoMaterialFio01Enrol01', 'NumeroDesenho', 'TerminalLigacao', 'IdEstatortInsertado', 'Descricao', 'MotorCompleto']
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
    df['CodigoDesenhoEstatorCompleto'] = df['CodigoDesenhoEstatorCompleto'].astype(int)
    df['CodigoDesenhoDiscoEstator'] = df['CodigoDesenhoDiscoEstator'].astype(int)
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

def termica_solved(df):
    df['TipoLigacaoProtecaoTermica'] = df['TipoLigacaoProtecaoTermica'].fillna('Nao Aplicavel')
    df['CabosProtecaoTermica'] = df['CabosProtecaoTermica'].astype('string').fillna('Nao Aplicavel')
    return df

def replace_single_occurrences(df):
    columns = ['CodigoComponente', 'DescricaoComponente', 'CarcacaPlataformaEletricaRaw', 'CarcacaPlataformaEletricaComprimento', 'CodigoDesenhoEstatorCompleto', 'CodigoDesenhoDiscoEstator', 'CodigoDesenhoDiscoRotor', 'EsquemaBobinagem', 'GrupoCarcaca', 'LigacaoDosCabos01', 'PolaridadeChapa', 'PassoEnrolamento01', 'TipoDeImpregnacao']
    
    original_dtypes = df.dtypes
    
    for column in columns:
        counts = df[column].value_counts()
        
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].map(lambda x: -1 if counts[x] <= 10 else x)
        else:
            df[column] = df[column].map(lambda x: x if pd.isna(x) else ('Outros' if counts[x] <= 10 else x))

    for column in columns:
        df[column] = df[column].astype(original_dtypes[column])
    
    return df

def get_distinct_values(df):
    columns = ["DescricaoComponente", "CabosProtecaoTermica", "CarcacaPlataformaEletricaRaw", 
                     "CarcacaPlataformaEletricaComprimento", "CodigoDesenhoEstatorCompleto", "CodigoDesenhoDiscoEstator", 
                     "CodigoDesenhoDiscoRotor", "EsquemaBobinagem", "GrupoCarcaca", "LigacaoDosCabos01", "MaterialChapa", 
                     "MaterialIsolFio01Enrol01", "PolaridadeChapa", "PotenciaCompletaCv01", 
                     "TipoLigacaoProtecaoTermica", "PassoEnrolamento01", "TipoDeImpregnacao"]    
    distinct_values = df[columns].nunique()
    return distinct_values


def print_unique_values_with_counts(df, column_name):
    value_counts = df[column_name].value_counts()
    print(f"Unique values and their counts in column '{column_name}':")
    for value, count in value_counts.items():
        print(f"{value}: {count}")

def df_changed(df):
    df = remove_duplicated_rows(df)
    df = drop_columns(df)    
    df = df.drop([2478,3882,2641])
    df = df.loc[df['MaterialIsolFio01Enrol01'] != 'COBRE WG6GR3']
    df = df.loc[~df['TipoDeImpregnacao'].isin(['VERNIZ F', 'GOTEJAMENTO + VPI'])]
    df = convert_cols_to_int(df)
    df = convert_cols_to_boolean(df)
    df = termica_solved(df)
    df = replace_single_occurrences(df)

    return df


    

    