import functools
import pandas as pd
import yfinance as yf


def _load_data(ticker):
    start_date = "2023-03-01"
    end_date = "2024-03-01"
    interval = "1d"
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data[["Open", 'High', 'Low', 'Close']]


load_data_copper = functools.partial(_load_data, ticker="HG=F")
load_data_iron   = functools.partial(_load_data, ticker="IRON")


def extract_mean(data):
    means = data.mean(axis=0).to_frame().T
    return means


class FeatureExtractor:
    def __init__(self):
        self.copper_data = load_data_copper()
        self.iron_data = load_data_iron()

    def extract_features(self):
        fn_list = [extract_mean]  # TODO: insert more feature extraction functions
        return pd.concat([fn(self.copper_data) for fn in fn_list], axis=0)


if __name__ == '__main__':

    data_iron = load_data_iron()
    data_copper = load_data_copper()

    from IPython import embed
    embed()

