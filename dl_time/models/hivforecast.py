# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:03:20 2024

@author: chaoyue.sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout=0.0):
        super(GraphTransformerLayer, self).__init__()
        self.fc_q = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.fc_k = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.fc_v = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.fc_o = nn.Linear(out_feats * num_heads, out_feats, bias=False)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_feats)

    def forward(self, g, h):
        # Multi-head attention mechanism
        Q = self.fc_q(h).view(h.shape[0], self.num_heads, -1)
        K = self.fc_k(h).view(h.shape[0], self.num_heads, -1)
        V = self.fc_v(h).view(h.shape[0], self.num_heads, -1)

        g.ndata['Q'] = Q
        g.ndata['K'] = K
        g.ndata['V'] = V

        # Attention mechanism (scaled dot-product attention)
        g.apply_edges(dgl.function.u_mul_v('Q', 'K', 'score'))
        g.edata['score'] = g.edata['score'].sum(dim=-1) / (K.shape[-1] ** 0.5)
        g.edata['attn'] = F.softmax(g.edata['score'], dim=1).unsqueeze(-1)

        g.update_all(dgl.function.u_mul_e('V', 'attn', 'm'), dgl.function.sum('m', 'h_new'))
        h_new = g.ndata.pop('h_new')

        # Output projection and residual connection
        h_proj = self.fc_o(h_new.view(h.shape[0],-1))
        h_proj = self.dropout(h_proj)
        h_proj = self.norm(h_proj + h)  # Residual connection
        return h_proj

class Net(nn.Module):
    def __init__(self, args, n_iter=2):
        super(Net, self).__init__()
        in_feats = 33
        num_classes = 2
        h_feats = 64
        num_heads = 4
        dropout = 0.5
        
        self.node_encoder = torch.nn.ModuleList()
        for layer in range(10):
            self.node_encoder.append(nn.Linear(in_feats, h_feats))
            self.node_encoder.append(nn.BatchNorm1d(h_feats))
            self.node_encoder.append(nn.LeakyReLU())
            in_feats = h_feats
        
        self.layers = nn.ModuleList()
        for _ in range(n_iter):
            self.layers.append(GraphTransformerLayer(h_feats, h_feats, num_heads, dropout))

        self.fc = nn.Linear(h_feats, num_classes)
        
    def forward(self, g):
        info = dict()
        h = g.ndata["feat"]
        for layer in self.node_encoder:
            h = layer(h)
            
        for layer in self.layers:
            h = layer(g, h)
        return self.fc(h), info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}





