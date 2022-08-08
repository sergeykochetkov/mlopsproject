from prefect.deployments import DeploymentYAML
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import mlflow
from urllib.parse import urlparse

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import numpy as np


def validate(model, val_dataloader, criterion, device):
    model.eval()
    profit = []
    losses=[]
    eps = 1e-6
    for x, y in val_dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        losses.append(criterion(pred,y).item())
        profit.extend((y * pred).detach().cpu().numpy())

    return np.mean(profit) / (np.std(profit) + eps), np.mean(losses)


def kelly_loss(output, target):
    profit = output * target
    m = torch.mean(profit)

    loss = - (m - 0.5 * torch.std(profit) ** 2)
    return loss


import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, length: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.d_model = d_model
        self.length = length
        self.decoder = nn.Linear(d_model * length, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src.transpose(0, 1)  # [seq_len, batch_size]
        src = torch.unsqueeze(src, dim=-1)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.transpose(0, 1)  # [batch_size, seq_len]
        output = output.flatten(-2, -1)
        output = self.decoder(output)
        return torch.squeeze(torch.sigmoid(output)*2-1, dim=-1)

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


def load_dataset(ticker: str, length: int, interval: str, period: str, shuffle_y_for_unittest: bool = False):
    stock = yf.Ticker(ticker)

    # get historical market data
    hist = stock.history(period=period, interval=interval)

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

@task()
def create_loaders(ticker, batch_size, length, period, interval):
    d, x, y = load_dataset(ticker, length, period, interval)
    val_idx = int(0.8 * len(x))
    train_dataset = StockDataset(d[:val_idx], x[:val_idx], y[:val_idx])
    val_dataset = StockDataset(d[val_idx:], x[val_idx:], y[val_idx:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


@task()
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, grad_norm):
    model.train()
    avg_loss = []
    for x, y in train_loader:
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred, y)
        avg_loss.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
    profit, val_loss = validate(model, val_loader, criterion, device)
    avg_loss = float(np.mean(avg_loss))
    return profit, val_loss, avg_loss


@task()
def register_model(model):
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.pytorch.log_model(model, "model", registered_model_name="StockTransformer")
    else:
        mlflow.pytorch.log_model(model, "model")


@flow(task_runner=SequentialTaskRunner())
def main():
    device = 'cuda:1'
    batch_size = 100
    grad_norm = 0.5
    epochs = 100
    lr = 1e-2
    length = 15
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.5  # dropout probability
    criterion = kelly_loss  # nn.L1Loss()
    ticker = 'AAPL'
    period = '2y'
    interval = '1h'

    with mlflow.start_run():

        for p in ['ticker', 'batch_size', 'grad_norm', 'epochs', 'lr', 'length', 'emsize', 'd_hid', 'nlayers', 'nhead',
                  'dropout', 'criterion', 'period', 'interval']:
            mlflow.log_param(p, eval(p))

        train_loader, val_loader = create_loaders(ticker, batch_size, length, period, interval)

        model = TransformerModel(length, emsize, nhead, d_hid, nlayers, dropout).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(epochs):
            profit, val_loss, avg_loss = train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                                                            device, grad_norm)
            print(f' epoch={epoch} profit={profit} avg_loss={avg_loss}')

            mlflow.log_metric("val_profit", profit)
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("avg_loss", avg_loss)

        register_model(model)


