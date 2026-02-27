import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def batch_to_torch(batch, device):
    torch_batch = {}
    for key in batch:
        torch_batch[key] = torch.from_numpy(batch[key]).to(device)
    return torch_batch


def mse_loss(val, target):
    return torch.mean(torch.square(val - target))


def cross_ent_loss(logits, target):
    # target 必须是 Long 类型，值是类别索引（0 或 1）
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.long, device=logits.device)
    else:
        target = target.long()

    # print("logits shape:", logits.shape, "dtype:", logits.dtype)
    # print("labels shape:", target.shape, "dtype:", target.dtype, "unique:", target.unique())

    return F.cross_entropy(input=logits, target=target)



