# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 05:50:43 2024

@author: chaoyue.sun
"""
#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
gcn.py: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv,GATConv


class Net(nn.Module):
    def __init__(self, args, n_iter=5, num_heads=4):
        super(Net, self).__init__()
        in_feats = 33
        num_classes = 2
        h_feats = 64
        self.n_iter = n_iter
        
        self.node_encoder = torch.nn.ModuleList()
        for layer in range(2):
            self.node_encoder.append(nn.Linear(in_feats, h_feats))
            self.node_encoder.append(nn.BatchNorm1d(h_feats))
            self.node_encoder.append(nn.LeakyReLU())
            in_feats = h_feats


        self.conv = torch.nn.ModuleList()
        self.conv.append(GATConv(h_feats, h_feats, num_heads))
        for layer in range(n_iter-2):
            self.conv.append(GATConv(h_feats*num_heads, h_feats, num_heads))
        self.conv.append(GATConv(h_feats*num_heads, h_feats, 1))
        
        self.fc = nn.Linear(h_feats, num_classes)
        
    def forward(self, g):
        info = dict()
        h = g.ndata["feat"]
        for layer in self.node_encoder:
            h = layer(h)
            
        for i in range(self.n_iter):
            h = self.conv[i](g, h)
            h = F.leaky_relu(h)
            h = h.flatten(1)
        h = self.fc(h)
        
        return h, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}

    