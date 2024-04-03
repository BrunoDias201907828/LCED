from db_connection_v2 import DBConnection

db = DBConnection()
df = db.get_dataframe()

from IPython import embed; embed()