from db_connection_v2 import DBConnection
import pandas as pd
import matplotlib.pyplot as plt

#pd.set_option('display.max_columns', None)

db = DBConnection()
df = db.get_dataframe()

#print(df)
missing_values_per_columns = df.isnull().sum()
#print(missing_values_per_columns)

missing_values_per_row = df.isnull().sum(axis=1)
rows_with_missing_values = missing_values_per_row[missing_values_per_row > 1]
#print("Rows with missing values greater than 0:")
#print(rows_with_missing_values)

df_cleaned = df.drop(rows_with_missing_values.index)
#print("DataFrame after dropping rows with more than 1 missing value:")
#print(df_cleaned)

missing_values = df_cleaned.isna()
missing_values_per_columns = missing_values.sum()
#print(missing_values_per_columns)

missing_values_per_row = df_cleaned.isnull().sum(axis=1)
rows_with_missing_values = missing_values_per_row[missing_values_per_row > 0]
#print("Rows with missing values greater than 0:")
#print(rows_with_missing_values)

df_missing_values = df_cleaned.loc[rows_with_missing_values.index]
#print("DataFrame with rows containing missing values:")
#print(df_missing_values)


distinct_value_counts = df.nunique()
print(distinct_value_counts)

nr_total_fios_enrol = df['NR_TOTAL_FIOS_ENROL']

min = nr_total_fios_enrol.min()
max = nr_total_fios_enrol.max()
distinct = nr_total_fios_enrol.nunique()
#print(f"Minimum value: {min}")
#print(f"Maximum value: {max}")
#print(f"Number of distinct values: {distinct}")


#plt.figure(figsize=(10, 6))
#plt.hist(nr_total_fios_enrol.dropna(), bins=14, color='skyblue', edgecolor='black')  # Drop NaN values before plotting
#plt.title('Histogram of NR_TOTAL_FIOS_ENROL Column')
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.grid(True)
#plt.show()



equals_count = (df["TERMINAL_LIGACAO"] == df["USO_DO_TERMINAL"]).sum()
#print("Equals:", equals_count)
diff_count = (df["TERMINAL_LIGACAO"] != df["USO_DO_TERMINAL"]).sum()
#print("Different:", diff_count)

different_row_index = df[df["TERMINAL_LIGACAO"] != df["USO_DO_TERMINAL"]].index
#print("Row where values are different:")
#print(df.loc[different_row_index])


#from IPython import embed; embed()

