from sqlalchemy import create_engine
import pandas as pd


class DBConnection:
    def __init__(self):
        user = 'user1'
        passwd = 'bE97XnZzmF'
        self.url = f'mysql://{user}:{passwd}@lced-data.fe.up.pt:3306/weg_a'
        self.engine = create_engine(self.url, pool_pre_ping=True)

    def get_connection(self):
        return self.engine.connect()

    def get_dataframe(self):
        df = pd.read_sql('ListaEBs', self.get_connection())
        return df


if __name__ == "__main__":
    db = DBConnection()
    df = db.get_dataframe()
    from IPython import embed; embed()
