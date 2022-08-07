import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from data import StockDataset, load_dataset
from model import TransformerModel
from loss import kelly_loss
from validate import validate

if __name__ == "__main__":
    device = 'cuda:1'
    batch_size = 100
    grad_norm = 0.5
    epochs = 100
    lr = 0.001
    length = 5
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability

    d, x, y = load_dataset('AAPL', length)
    val_idx = int(0.8 * len(x))

    train_dataset = StockDataset(d[:val_idx], x[:val_idx], y[:val_idx])
    val_dataset = StockDataset(d[val_idx:], x[val_idx:], y[val_idx:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TransformerModel(length, emsize, nhead, d_hid, nlayers, dropout).to(device)

    criterion = kelly_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            dbg = 1
        profit = validate(model, val_loader, device)
        print(f' epoch={epoch} profit={profit}')
