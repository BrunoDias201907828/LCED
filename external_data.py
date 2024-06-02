import functools
import pandas as pd
import yfinance as yf


def _load_data(ticker):
    start_date = "2020-01-01"
    end_date = "2024-03-30"
    interval = "1d"
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data = data[['Close']]
    idx = pd.date_range(start_date, end_date)
    data = data.reindex(idx)
    data[['Close']] = data[['Close']].ffill()
    return data


load_data_copper = functools.partial(_load_data, ticker="HG=F")
load_data_iron   = functools.partial(_load_data, ticker="IRON")


def add_exponential_moving_average(data):
    data['EMA30'] = data['Close'].ewm(span=30, adjust=False).mean()


def add_moving_average(data, window):
    data[f'MA{window}'] = data['Close'].rolling(window=window).mean()


def add_standard_deviation(data):
    data[f'STD30'] = data['Close'].rolling(window=30).std()


add_moving_average_30 = functools.partial(add_moving_average, window=30)
add_moving_average_90 = functools.partial(add_moving_average, window=90)


class FeatureExtractor:
    fn_list = [add_moving_average_30, add_moving_average_90, add_exponential_moving_average, add_standard_deviation]

    def __init__(self):
        self.copper_data = load_data_copper()
        self.iron_data = load_data_iron()

    def extract_features(self):
        copper_data = self.copper_data.copy()
        iron_data = self.iron_data.copy()
        for fn in self.fn_list:
            fn(copper_data)
            fn(iron_data)
        return pd.merge(
            copper_data, iron_data, left_index=True, right_index=True, suffixes=('_copper', '_iron')
        )


if __name__ == '__main__':
    df_ext = FeatureExtractor().extract_features()
