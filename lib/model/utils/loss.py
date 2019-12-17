import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.config import cfg

gamma = 2
alpha = 0.75

def focal_loss(inputs, targets, weight, softmax=True, size_average=False):
    if softmax:
        P = F.sigmoid(inputs)
    else:
        P = inputs
    pt = P * targets + (1 - P) * (1 - targets)

    w = alpha * targets + (1 - alpha) * (1 - targets)
    # w = 0.25
    w = w * (1 - pt).pow(gamma) * weight

    loss = F.binary_cross_entropy_with_logits(inputs, targets, w, size_average=size_average)
    return loss

def focal_loss_all(inputs, targets):
    P = inputs[:, 1]
    pt = P * targets + (1-P) * (1 - targets)
    w = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - pt).pow(gamma)
    # w = 0.25

    loss = F.binary_cross_entropy_with_logits(P, targets, w, size_average=True)
    return loss