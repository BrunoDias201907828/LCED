import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ListaEBs_202404011800.csv")

#print(df)

#missing_values_per_columns = df.isnull().sum()
#print(missing_values_per_columns)

# Calculate total number of rows
total_rows = len(df)

# Calculate percentage of missing values for each column
missing_values_per_column = df.isnull().sum()
missing_percentage_per_column = (missing_values_per_column / total_rows) * 100

# Combine the count and percentage of missing values for each column
missing_info = pd.concat([missing_values_per_column, missing_percentage_per_column], axis=1)
missing_info.columns = ['Missing Values Count', 'Percentage']

# Print the missing values information
print(missing_info)


missing_values_per_row = df.isnull().sum(axis=1)
rows_with_missing_values = missing_values_per_row[missing_values_per_row > 0]
#print("Rows with missing values greater than 0:")
#print(rows_with_missing_values)

distinct_value_counts = df.nunique()
#print(distinct_value_counts)
