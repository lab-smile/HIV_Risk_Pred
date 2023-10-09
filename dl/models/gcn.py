#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
gcn.py: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        in_feats = 33
        num_classes = 2
        h_feat = 32

        self.conv1 = GraphConv(in_feats, h_feat)
        self.conv2 = GraphConv(h_feat, h_feat)
        self.conv3 = GraphConv(h_feat, h_feat)
        self.conv4 = GraphConv(h_feat, h_feat)
        self.conv5 = GraphConv(h_feat, h_feat)
        
        self.conv6 = GraphConv(h_feat, h_feat)
        self.conv7 = GraphConv(h_feat, h_feat)
        self.conv8 = GraphConv(h_feat, h_feat)
        self.conv9 = GraphConv(h_feat, h_feat)
        self.conv10 = GraphConv(h_feat, h_feat)
        
        self.fc = nn.Linear(h_feat, num_classes)


    def forward(self, g):
        info = dict()
        node_feat = g.ndata["feat"]

        h = self.conv1(g, node_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        h = F.relu(h)
        
        h = self.conv6(g, h)
        h = F.relu(h)
        h = self.conv7(g, h)
        h = F.relu(h)
        h = self.conv8(g, h)
        h = F.relu(h)
        h = self.conv9(g, h)
        h = F.relu(h)
        h = self.conv10(g, h)
        h = F.relu(h)
        
        h = self.fc(h)

        
        return h, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}