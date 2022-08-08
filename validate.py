'''
utils for validation model by trading metrics
'''
import numpy as np


def validate(model, val_dataloader, criterion, device):
    '''
    calculate Sharp ratio of trading returns, and criterion (loss) on validation dataset
    '''
    model.eval()
    profit = []
    losses = []
    eps = 1e-6
    for features_x, targets_y in val_dataloader:
        features_x = features_x.to(device)
        targets_y = targets_y.to(device)

        pred = model(features_x)
        losses.append(criterion(pred, targets_y).item())
        profit.extend((targets_y * pred).detach().cpu().numpy())

    return np.mean(profit) / (np.std(profit) + eps), np.mean(losses)
