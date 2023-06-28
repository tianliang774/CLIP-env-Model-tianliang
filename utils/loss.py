import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum


class Multi_L1Loss(nn.Module):
    """
    Applied L1Loss into Multi-modal model
    """

    def __init__(self, ignore_index=None, num_classes=5, **kwargs):
        super(Multi_L1Loss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.L1Loss()

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            tname = name[b]
            method_index = TEMPLATE[tname]
            l1 = self.criterion(predict[b, method_index], target[b, 0])
            total_loss.append(l1)
        total_loss = torch.stack(total_loss)

        return total_loss.sum() / total_loss.shape[0]
