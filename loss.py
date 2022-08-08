'''
trading specific losses
'''
import torch


def kelly_loss(output, target):
    '''
    loss accounting for risks
    '''
    profit = output * target
    mean_profit = torch.mean(profit)

    loss = - (mean_profit - 0.5 * torch.std(profit) ** 2)
    return loss
