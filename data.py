import random

import numpy as np
from torch.utils.data.dataset import Dataset

import yfinance as yf


class StockDataset(Dataset):
    def __init__(self, d, x, y):
        super().__init__()
        self._d, self._x, self._y = d, x, y

    def __len__(self):
        return len(self._x)

    def __getitem__(self, item):
        return self._x[item], self._y[item]


def load_dataset(ticker: str, length: int, shuffle_y_for_unittest: bool = False):
    stock = yf.Ticker(ticker)

    # get historical market data
    hist = stock.history(period="2y", interval='1h')

    vals = hist.Close.values
    dates = hist.index.values
    returns = vals[1:] / vals[:-1] - 1
    dates = dates[1:]
    X = []
    Y = []
    D = []
    for i in range(length, len(returns)):
        X.append(returns[i - length:i])
        Y.append(returns[i])
        D.append(dates[i])
    if shuffle_y_for_unittest:
        random.shuffle(Y)

    return np.array(D), np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
