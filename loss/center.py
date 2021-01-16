#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class CenterLoss(nn.Module):
    """
    Center Loss
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes(int)
        feat_dim(int)

    """

    def __init__(self, device, num_classes=10, feat_dim=256):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        print('center loss device:', self.device)
        if self.device == 'cuda': # 有点问题
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        feature_size = x.size(1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        # distmat.addmm_(x, self.centers.t(),  1, -2)

        classes = torch.arange(self.num_classes).long()
        if self.device == 'cuda':
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
