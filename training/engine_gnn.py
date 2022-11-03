import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import accuracy


__all__ = ['train_step', 'val_step']


def train(
    model,
    optimizer,
    data,
    mask,
    metric = accuracy,
):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask])
    metric_value  = metric(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item(), metric_value


@torch.no_grad()
def test(
    model,
    data,
    mask,
    metric = accuracy,
):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask])
    metric_value  = metric(out[mask], data.y[mask])
    return loss.item(), metric_value
