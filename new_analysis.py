import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def missing_values(df):
    missing_values_per_column = df.isnull().sum()
    missing_percentage_per_column = (missing_values_per_column / len(df)) * 100

    missing_info = pd.concat([missing_values_per_column, missing_percentage_per_column], axis=1)
    missing_info.columns = ['Missing Values Count', 'Percentage']

    return missing_info

df = pd.read_csv("ListaEBs.csv")

missing = missing_values(df)

from IPython import embed; embed()

missing_values_per_row = df.isnull().sum(axis=1)
rows_with_missing_values = missing_values_per_row[missing_values_per_row > 0]
#print("Rows with missing values greater than 0:")
#print(rows_with_missing_values)

distinct_value_counts = df.nunique()
#print(distinct_value_counts)
