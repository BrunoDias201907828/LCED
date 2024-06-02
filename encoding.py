from sklearn.preprocessing import TargetEncoder
import category_encoders as ce

CATEGORICAL_COLUMNS = [
    "DescricaoComponente",
    "GrupoCarcaca",
    "CodigoComponente",
    "CodigoDesenhoEstatorCompleto",
    "CodigoDesenhoDiscoEstator",
    "CodigoDesenhoDiscoRotor",
    "EsquemaBobinagem",
    "LigacaoDosCabos01",
    "MaterialChapa",
    "MaterialIsolFio01Enrol01",
    "PolaridadeChapa",
    "TipoLigacaoProtecaoTermica",
    "PassoEnrolamento01",
    "CabosProtecaoTermica",
    "CarcacaPlataformaEletricaRaw",
    "CarcacaPlataformaEletricaComprimento",
]


def target_encoding(df):
    encoder = TargetEncoder(target_type="continuous")

    x = df[CATEGORICAL_COLUMNS].to_numpy()
    y = df["CustoIndustrial"].to_numpy()
    x_transformed = encoder.fit_transform(x, y)

    df_transformed = df.copy()
    df_transformed[CATEGORICAL_COLUMNS] = x_transformed
    return df_transformed


def binary_encoding(df):
    encoder = ce.BinaryEncoder(cols=CATEGORICAL_COLUMNS, return_df=True)

    df_encoded = encoder.fit_transform(df)
    return df_encoded
    

if __name__ == "__main__":
    from db_connection import DBConnection
    db = DBConnection()
    df = db.get_dataframe()
    from IPython import embed; embed()
