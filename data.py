'''
load stock data and create dataset
'''
import random
import numpy as np
from torch.utils.data.dataset import Dataset
import yfinance as yf


class StockDataset(Dataset):
    '''
    stock features and returns dataset
    '''

    def __init__(self, dates, features_x, targets_y):
        super().__init__()
        self._dates, self._features_x, self._targets_y = dates, features_x, targets_y

    def __len__(self):
        return len(self._features_x)

    def __getitem__(self, item):
        return self._features_x[item], self._targets_y[item]


def load_dataset(ticker: str, length: int, interval: str,
                 period: str, shuffle_y_for_unittest: bool = False):
    '''
    download stock quotes from yahoo finance
    :param shuffle_y_for_unittest: only for unittest randomly shuffles y,
     such returns are unpredictable
    :return:
    '''
    stock = yf.Ticker(ticker)

    # get historical market data
    hist = stock.history(period=period, interval=interval)

    vals = hist.Close.values
    dates = hist.index.values
    returns = vals[1:] / vals[:-1] - 1
    dates = dates[1:]
    features_x = []
    targets_y = []
    all_dates = []
    for i in range(length, len(returns)):
        features_x.append(returns[i - length:i])
        targets_y.append(returns[i])
        all_dates.append(dates[i])
    if shuffle_y_for_unittest:
        random.shuffle(targets_y)

    return np.array(all_dates), np.array(features_x, dtype=np.float32), \
           np.array(targets_y, dtype=np.float32)
