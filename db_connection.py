from new_analysis import df_changed as clean_dataframe
from external_data import FeatureExtractor
from sqlalchemy import create_engine
try:
    import mysql.connector
    MYSQLCONNECTOR = True
except ImportError:
    MYSQLCONNECTOR = False
import pandas as pd


class DBConnection:
    def __init__(self, use_mysql=False):
        self.use_mysql = use_mysql
        self.user = 'user1'
        self.passwd = 'bE97XnZzmF'
        self.host = "10.227.243.130"
        self.database = "weg_a"
        if not use_mysql:
            self.url = f'mysql://{self.user}:{self.passwd}@{self.host}:3306/{self.database}'
            self.engine = create_engine(self.url, pool_pre_ping=True)

    def get_connection(self):
        if self.use_mysql:
            return mysql.connector.connect(user=self.user, password=self.passwd, host=self.host, database=self.database)
        return self.engine.connect()

    def _get_dataframe_raw(self):
        conn = self.get_connection()
        if self.use_mysql:
            df = pd.read_sql('SELECT * FROM DatasetWithCost', conn)
        else:
            df = pd.read_sql('DatasetWithCost', conn)
        conn.close()
        df = df.convert_dtypes()
        float_columns = [
            "DiametroExternoEstator [mm]", "ComprimentoExternoCabosLigacao", "CustoIndustrial",
            "ComprimentoTotalPacote [mm]"
        ]
        bool_columns = ["UsoDoTerminal", "ChoqueTermico", "CabosProtecaoTermica"]
        df[float_columns] = df[float_columns].astype("Float64")
        df[bool_columns] = df[bool_columns].astype("boolean")
        return df

    def _get_dataframe_cleaned(self):
        return clean_dataframe(self._get_dataframe_raw())

    def get_dataframe(self):
        df = self._get_dataframe_cleaned()
        extractor = FeatureExtractor()
        df_ext = extractor.extract_features()
        return (
            pd.merge(df, df_ext, left_on="DataCriacao", right_index=True, how="left").
            drop(columns=["DataCriacao"])
        )


if __name__ == "__main__":
    db = DBConnection()
    df = db.get_dataframe()

    from IPython import embed; embed()
