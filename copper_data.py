import pandas as pd
import yfinance as yf

if __name__ == '__main__':
    ticker = "HG=F"
    start_date = "2020-01-01"
    end_date = "2024-03-13"
    interval = "1d"

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    from IPython import embed
    embed()

