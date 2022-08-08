'''
unittests for model
'''
import unittest
import random
import torch
from torch import nn
from data import StockDataset, load_dataset
from model import TransformerModel
from validate import validate


class TestModel(unittest.TestCase):
    '''
    tests model basic functionality, ability to learn
    '''

    def setUp(self) -> None:
        self.device = 'cuda:1'
        self.grad_norm = 0.5
        self.batch_size = 5

        self.epochs = 10
        self.criterion = nn.L1Loss()
        self.learning_rate = 0.01
        self.length = 5
        self.emsize = 128  # embedding dimension
        self.d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
        self.nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.nhead = 2  # number of heads in nn.MultiheadAttention
        self.dropout = 0.2  # dropout probability
        self.period = '2y'
        self.interval = '1d'

        dates, features_x, target_y = \
            load_dataset('AAPL', self.length, self.interval, self.period)

        train_dataset = StockDataset(dates, features_x, target_y)
        self.train_loader = torch.utils.data.dataloader.DataLoader(train_dataset,
                                                                   batch_size=self.batch_size,
                                                                   shuffle=True,
                                                                   num_workers=0)

        self.model = self._build_fresh_model()

    def _build_fresh_model(self):
        return TransformerModel(self.length, self.emsize, self.nhead,
                                self.d_hid, self.nlayers, self.dropout).to(
            self.device)

    def _train_and_eval(self, train_loader, val_loader):
        loss_vs_epoch = []
        profit_vs_epoch = []
        val_loss_vs_epoch = []
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            avg_loss = 0
            self.model.train()
            for features_x, targets_y in train_loader:
                features_x = features_x.to(self.device)
                targets_y = targets_y.to(self.device)

                pred = self.model(features_x)

                loss = self.criterion(pred, targets_y)
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
        '''
        tests if we can reduce loss in training data
        '''

        self.model = self._build_fresh_model()
        loss_vs_epoch, _, _ = self._train_and_eval(self.train_loader, self.train_loader)

        self.assertGreater(loss_vs_epoch[0], loss_vs_epoch[-1])

    def test_no_leak(self):
        '''
        tests of there is leak between train and val dataset parts
        '''

        random.seed(0)

        dates, features_x, target_y = load_dataset('MSFT', self.length,
                                                   self.interval,
                                                   self.period,
                                                   shuffle_y_for_unittest=True)

        med = len(features_x) // 2

        train_dataset = StockDataset(dates[:med], features_x[:med], target_y[:med])
        train_loader = torch.utils.data.dataloader.DataLoader(train_dataset,
                                                              batch_size=self.batch_size,
                                                              shuffle=True,
                                                              num_workers=0)

        val_dataset = StockDataset(dates[med:], features_x[med:], target_y[med:])
        val_loader = torch.utils.data.dataloader.DataLoader(val_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            num_workers=0)
        self.model = self._build_fresh_model()
        _, _, val_loss_vs_epoch = self._train_and_eval(train_loader, val_loader)
        target_y = torch.from_numpy(target_y)
        base_loss = self.criterion(target_y, torch.zeros_like(target_y)).item()

        self.assertLess(base_loss, val_loss_vs_epoch[-1])
