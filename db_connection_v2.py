import mysql.connector
import pandas as pd


class DBConnection:

    def __init__(self):
    # Establish a connection
        self.conn = mysql.connector.connect(user="user1", password="bE97XnZzmF", host="lced-data.fe.up.pt", database="weg_a")
        self.cursor=self.conn.cursor()

    def close(self):
        self.conn.close()

    def get_dataframe(self):
        df = pd.read_sql('SELECT * FROM ListaEBsWPTV2_changed', self.conn)
        return df


# Create a cursor object
db=DBConnection()
cursor = db.cursor
df = db.get_dataframe()

from IPython import embed; embed()