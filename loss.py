import torch


def kelly_loss(output, target):
    profit = output * target
    m = torch.mean(profit)

    loss = - (m - 0.5 * torch.std(profit) ** 2)
    return loss
