from sqlalchemy import create_engine
import mysql.connector
import pandas as pd


class DBConnection:
    def __init__(self, use_mysql=False):
        self.use_mysql = use_mysql
        self.user = 'user1'
        self.passwd = 'bE97XnZzmF'
        self.host = "lced-data.fe.up.pt"
        self.database = "weg_a"
        if not use_mysql:
            self.url = f'mysql://{self.user}:{self.passwd}@{self.host}:3306/{self.database}'
            self.engine = create_engine(self.url, pool_pre_ping=True)

    def get_connection(self):
        if self.use_mysql:
            return mysql.connector.connect(user=self.user, password=self.passwd, host=self.host, database=self.database)
        return self.engine.connect()

    def get_dataframe(self):
        conn = self.get_connection()
        df = pd.read_sql('dataset', conn)
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

    def get_dataframe_cleaned(self):
        raise NotImplementedError


if __name__ == "__main__":
    db = DBConnection()
    df = db.get_dataframe()
    from IPython import embed; embed()