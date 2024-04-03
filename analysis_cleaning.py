from db_connection_v2 import DBConnection
import pandas as pd
import matplotlib.pyplot as plt

db = DBConnection()
df = db.get_dataframe()

missing_values = df.isna()
#print(missing_values)
missing_values_per_columns = missing_values.sum()
#print(missing_values_per_columns)

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

