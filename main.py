import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import mlflow
from urllib.parse import urlparse
from data import StockDataset, load_dataset
from model import TransformerModel
from loss import kelly_loss
from validate import validate
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


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


if __name__ == "__main__":
    main()
