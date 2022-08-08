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
