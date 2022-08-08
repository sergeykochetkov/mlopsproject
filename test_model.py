import random

import numpy as np
import torch
from torch import nn
import unittest
from torch.utils.data.dataloader import DataLoader
from data import StockDataset, load_dataset
from model import TransformerModel
from loss import kelly_loss
from validate import validate


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = 'cuda:1'
        self.grad_norm = 0.5
        self.batch_size = 5

        self.epochs = 10
        lr = 0.01
        self.length = 5
        self.emsize = 128  # embedding dimension
        self.d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 2  # number of heads in nn.MultiheadAttention
        self.dropout = 0.2  # dropout probability
        self.period = '2y'
        self.interval = '1d'

        d, x, y = load_dataset('AAPL', self.length, self.interval, self.period)

        train_dataset = StockDataset(d, x, y)
        self.train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=self.batch_size,
                                                                   shuffle=True,
                                                                   num_workers=0)

        self.model = self._build_fresh_model()

        self.criterion = nn.L1Loss()
        self.lr = lr

    def _build_fresh_model(self):
        return TransformerModel(self.length, self.emsize, self.nhead, self.d_hid, self.nlayers, self.dropout).to(
            self.device)

    def _train_and_eval(self, train_loader, val_loader):
        loss_vs_epoch = []
        profit_vs_epoch = []
        val_loss_vs_epoch = []
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            avg_loss = 0
            self.model.train()
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)

                loss = self.criterion(pred, y)
                avg_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                optimizer.step()
            loss_vs_epoch.append(avg_loss)
            profit, val_loss = validate(self.model, val_loader, self.criterion, self.device)
            val_loss_vs_epoch.append(val_loss)
            profit_vs_epoch.append(profit)
        return loss_vs_epoch, profit_vs_epoch, val_loss_vs_epoch

    def test_overfit(self):

        self.model = self._build_fresh_model()
        loss_vs_epoch, profit_vs_epoch, val_loss_vs_epoch = self._train_and_eval(self.train_loader, self.train_loader)

        self.assertGreater(loss_vs_epoch[0], loss_vs_epoch[-1])
        self.assertGreater(val_loss_vs_epoch[0], val_loss_vs_epoch[-1])

    def test_no_leak(self):

        random.seed(0)

        d, x, y = load_dataset('MSFT', self.length, self.interval, self.period, shuffle_y_for_unittest=True)

        med = len(x) // 2

        train_dataset = StockDataset(d[:med], x[:med], y[:med])
        train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                              num_workers=0)

        val_dataset = StockDataset(d[med:], x[med:], y[med:])
        val_loader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                                            num_workers=0)
        self.model = self._build_fresh_model()
        loss_vs_epoch, profit_vs_epoch, val_loss_vs_epoch = self._train_and_eval(train_loader, val_loader)
        y = torch.from_numpy(y)
        base_loss = self.criterion(y, torch.zeros_like(y)).item()

        self.assertLess(base_loss, val_loss_vs_epoch[-1])
