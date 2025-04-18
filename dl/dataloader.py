# -*- coding: utf-8 -*-
"""
@author: Suncy

dataloader for Graph NN Models
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from torchvision import transforms, utils
import os.path as osp
import dgl
import torch
import json
from dl import logger
from dgl.data import DGLDataset
from collections import Counter
from torch.utils.data import Dataset

# dataloader for graph models
class Dataset(DGLDataset):
    def __init__(self, args, phase, device):
        self.device = device
        ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)
        cat_feat = ['Gender', 'Race', 'Ethnicity', 'CD_MaritalStatus', 'HaveNeedlePartners',
                    'SyphilisTestResult', 'SexualOrientation','HistoryOfPriorSTD', 'MSM', 'EHE']
        val_feat = ['NO_SexPartners', 'Age']
        
        self.phase = phase
        if phase in ["train", "valid", "test"]:
            self.node_df = pd.read_csv(f"{ds_folder}/{phase}.csv", low_memory=False)
            self.edge_df = pd.read_csv(f"{ds_folder}/{phase}_edge.csv", low_memory=False)
            if args.permutation_test:
                if args.permutation_feat != "None":
                    if args.permutation_feat in cat_feat:
                        feat = args.permutation_feat
                        cols = [item for item in self.node_df.columns[-33:] if feat in item]
                        print(cols)
                        shuffle_cols = pd.DataFrame(self.node_df, columns=cols).values
                        np.random.shuffle(shuffle_cols)
                        for i in range(len(cols)):
                            self.node_df[feat+'_'+str(i)] = shuffle_cols[:,i]
                    else:
                        feat = args.permutation_feat
                        col = [item for item in self.node_df.columns[-33:] if feat in item]
                        print(col)
                        shuffle_col = self.node_df[col].values
                        np.random.shuffle(shuffle_col)
                        self.node_df[col] = shuffle_col
        else:
            raise NotImplementedError

        self.net_ids = self.node_df["network_id"].unique()  # num of trees

        self.node_feat_cols = self.node_df.columns[-33:]
        self.node_label_cols = 'status_cat'


        self.add_self_loop = args.add_self_loop
        self.bidirection = args.bidirection

    def process(self):        
        pass

    def __getitem__(self, index):

        net_id = self.net_ids[index]  # tree of index

        # dgl tree of index
        one_node_df = self.node_df[self.node_df['network_id'] == net_id]
        one_edge_df = self.edge_df[self.edge_df['network_id'] == net_id]
        src_ids = torch.tensor(one_edge_df['from'].values)
        dst_ids = torch.tensor(one_edge_df['to'].values)
        if len(src_ids) == 0 and len(dst_ids) == 0:
            g = dgl.DGLGraph()
            g.add_nodes(1) 
        else:
            g = dgl.graph((src_ids, dst_ids))  # create dgl
        sorted_one_node_df = one_node_df.sort_values(by='node')
        
        # assign features and labels for background nodes
        node_feat = sorted_one_node_df[self.node_feat_cols].values
        node_label = sorted_one_node_df[self.node_label_cols].values
        node_time = sorted_one_node_df['ReportYear'].values
        edge_time = np.array([int(a.split('-')[0]) for a in one_edge_df['ReportDate'].values])

        # assign features for nodes and edges, assign labels
        g.ndata["feat"] = torch.tensor(node_feat, dtype=torch.float32)
        g.ndata["label"] = torch.tensor(node_label, dtype=torch.int64)
        g.ndata["time"] = torch.tensor(node_time, dtype=torch.int64)
        g.edata["time"] = torch.tensor(edge_time, dtype=torch.int64)

        if self.add_self_loop:
            g = dgl.add_self_loop(g)  
        if self.bidirection:
            g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)

        g = g.to(self.device)
        return g
        
    def __len__(self):
        return len(self.net_ids)   # number of trees


# create batch of trees(aggregate multiples trees to a single tree)
def collate_fn(batch_graphs):
    g = dgl.batch(batch_graphs)
    return g


def gen_label_weight(args):
    # Get the weights for the unbalanced sample based on the positive sample
    # weights inversely proportional to class frequencies in the training data
    ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)
    node_df = pd.read_csv(f'{ds_folder}/train.csv')

    node_label = node_df[args.node_label_cols].values
    label_counter = Counter(node_label)
    n_samples = len(node_label)
    n_classes = len(label_counter)

    label_weights = [n_samples / (n_classes * label_counter[i]) for i in range(n_classes)]
    #label_weights[1] = label_weights[1]*2


    return label_weights





