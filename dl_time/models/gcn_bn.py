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
        self.bn1 = nn.BatchNorm1d(h_feat)
        self.conv2 = GraphConv(h_feat, h_feat)
        self.bn2 = nn.BatchNorm1d(h_feat)
        self.conv3 = GraphConv(h_feat, h_feat)
        self.bn3 = nn.BatchNorm1d(h_feat)
        self.conv4 = GraphConv(h_feat, h_feat)
        self.bn4 = nn.BatchNorm1d(h_feat)
        self.conv5 = GraphConv(h_feat, h_feat)
        self.bn5 = nn.BatchNorm1d(h_feat)
        '''
        self.conv6 = GraphConv(h_feat, h_feat)
        self.bn6 = nn.BatchNorm1d(h_feat)
        self.conv7 = GraphConv(h_feat, h_feat)
        self.bn7 = nn.BatchNorm1d(h_feat)
        self.conv8 = GraphConv(h_feat, h_feat)
        self.bn8 = nn.BatchNorm1d(h_feat)
        self.conv9 = GraphConv(h_feat, h_feat)
        self.bn9 = nn.BatchNorm1d(h_feat)
        self.conv10 = GraphConv(h_feat, h_feat)
        self.bn10 = nn.BatchNorm1d(h_feat)
        '''
        self.fc = nn.Linear(h_feat, num_classes)


    def forward(self, g):
        info = dict()
        node_feat = g.ndata["feat"]

        h = self.bn1(self.conv1(g, node_feat))
        h = F.leaky_relu(h)
        h = self.bn2(self.conv2(g, h))
        h = F.leaky_relu(h)
        h = self.bn3(self.conv3(g, h))
        h = F.leaky_relu(h)
        h = self.bn4(self.conv4(g, h))
        h = F.leaky_relu(h)
        h = self.bn5(self.conv5(g, h))
        h = F.leaky_relu(h)
        '''
        h = self.bn6(self.conv6(g, h))
        h = F.leaky_relu(h)
        h = self.bn7(self.conv7(g, h))
        h = F.leaky_relu(h)
        h = self.bn8(self.conv8(g, h))
        h = F.leaky_relu(h)
        h = self.bn9(self.conv9(g, h))
        h = F.leaky_relu(h)
        h = self.bn10(self.conv10(g, h))
        h = F.leaky_relu(h)
        '''
        h = self.fc(h)

        
        return h, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}