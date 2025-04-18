# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 01:38:22 2023

@author: chaoyue.sun
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim=100):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
    
class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, dims, expansion_factor, dropout=0, use_single_layer=False):
        super().__init__()

        self.dims = dims
        self.use_single_layer = use_single_layer
        
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if self.use_single_layer==False:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.use_single_layer==False:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, per_graph_size, dims, 
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 dropout=0, 
                 module_spec=None, use_single_layer=False):
        super().__init__()
        
        if module_spec == None:
            self.module_spec = ['token', 'channel']
        else:
            self.module_spec = module_spec.split('+')

        if 'token' in self.module_spec:
            self.token_layernorm = nn.LayerNorm(dims)
            self.token_forward = FeedForward(per_graph_size, token_expansion_factor, dropout, use_single_layer)
            
        if 'channel' in self.module_spec:
            self.channel_layernorm = nn.LayerNorm(dims)
            self.channel_forward = FeedForward(dims, channel_expansion_factor, dropout, use_single_layer)
        

    def reset_parameters(self):
        if 'token' in self.module_spec:
            self.token_layernorm.reset_parameters()
            self.token_forward.reset_parameters()

        if 'channel' in self.module_spec:
            self.channel_layernorm.reset_parameters()
            self.channel_forward.reset_parameters()
        
    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x
    
    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        if 'token' in self.module_spec:
            x = x + self.token_mixer(x)
        if 'channel' in self.module_spec:
            x = x + self.channel_mixer(x)
        return x
    
class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)

class MLPMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size

        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for ell in range(num_layers):
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer)
                )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer)
                )



        # init
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x = torch.zeros((batch_size * self.per_graph_size, 
                         edge_time_feats.size(1))).to(edge_feats.device)
        x[inds] = x[inds] + edge_time_feats     
        x = torch.split(x, self.per_graph_size)
        x = torch.stack(x)
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x
    
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        num_classes = 2
        self.f_size = 34
        self.h_size = 64
        
        self.mlpmixer = MLPMixer()
        self.time_encoder = TimeEncode(self.h_size)
        
        self.fc = nn.Linear(2*self.h_size, num_classes)
      
    def node_udf(self, edges):
        msg = edges.src['feat']
        m,n = msg.shape
        time_enc = torch.zeros([m, self.h_size])
        for i in range(m):                
            if edges.data['time'][i] == 1:
                time_enc[i,:] = self.time_encoder(0)
            elif edges.dst['time'][i] >= edges.data['time'][i]:
                time_enc[i,:] = self.time_encoder(edges.dst['time'][i]-edges.src['time'][i])
            else:
                msg[i,:] = torch.zeros([n])
        x = torch.cat((msg, time_enc), 1)
        return {'mixer': x}
    
    def reducer(self, nodes):
        return {'output': self.mlpmixer(nodes.mailbox['mixer'])}
        
    def forward(self, g):
        info = dict()
        
        g.update_all(self.node_udf, self.reducer)
        
        output = self.fc(g.ndata['output'])
        return output, info

    def ce_loss(self, y_pred, y_true, weight=None):
        # print(y_pred.shape, y_true.shape, weight.shape)
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}    