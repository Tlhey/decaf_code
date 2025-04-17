# From HNN_GAD/data/lodaer.py https://github.com/Jing-DS/HNN_GAD
"""Data utils functions for pre-processing and data loading."""
import os
import math
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy
import pickle as pkl
import sys
from sklearn import preprocessing
import json

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import coalesce
from torch_geometric.io import read_npz
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected

# from ogb.nodeproppred import PygNodePropPredDataset

from scipy.stats import ortho_group
# import matplotlib.pyplot as plt

cpu = torch.device("cpu")


def preprocess_features(features, method="l2"):
    """Row-normalize feature matrix and convert to tuple representation"""
    if method=="l2":
        rowsum = np.array(np.sqrt((features**2).sum(1)))
    elif method=="l1":
        rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

def preset_parameters(args):
    """preset parameters for outlier node generation"""
    
    # 不知道哪里来的这个m和n: 论文C.2 appendix 里找到了。n是Nv的0.05左右，m是graph degree的两倍
    # num_outlier_dict = {"squirrel":280, "chameleon":120, "actor":390, "cora":140, "citeseer":180, "pubmed":1000, "amazon":700, "flickr":4480, "ogbn-arxiv":8400}
    # m_dict = {"squirrel":70, "chameleon":30, "actor":15, "cora":10, "citeseer":10, "pubmed":10, "amazon":70, "flickr":20, "ogbn-arxiv":30}
    # args.outlier_num = num_outlier_dict[args.dataset_name]
    # args.struc_clique_size = m_dict[args.dataset_name]
    args.struc_drop_prob = 0.2
    # args.sample_size = m_dict[args.dataset_name]
    args.dice_ratio = 0.5
    return args

def gen_structural_outliers_cf(data, data_cf, m, n, p=0.2, y_outlier=None, random_state=None):
    
    if y_outlier is not None:
        node_set = set(list(np.where(y_outlier==0)[0]))
    else:
        y_outlier = np.zeros(data.num_nodes)
        node_set = set(range(data.num_nodes))   
        
    r = np.random.RandomState(random_state)
    outlier_idx = r.choice(list(node_set), size=int(m * n), replace=False)
    y_outlier[outlier_idx] = 1
    
    def add_edges(data_, outlier_idx, m, n, p):
        new_edges = []
        for i in range(0, n):
            for j in range(m * i, m * (i + 1)):
                for k in range(m * i, m * (i + 1)):
                    if j != k:
                        node1, node2 = outlier_idx[j], outlier_idx[k]
                        new_edges.append(torch.tensor([[node1, node2]], dtype=torch.long))
        new_edges = torch.cat(new_edges)

        if p != 0:
            indices = torch.randperm(len(new_edges))[:int((1 - p) * len(new_edges))]
            new_edges = new_edges[indices]

        data_.edge_index = torch.cat([data_.edge_index, new_edges.T], dim=1)
        return data_

    
    data = add_edges(data, outlier_idx, m, n, p)
    data_cf = add_edges(data_cf, outlier_idx, m, n, p)

    return data, data_cf, y_outlier

import torch
import numpy as np

def gen_contextual_outliers_cf(data, data_cf, n, k, random_state=None):
    node_set = set(range(data.num_nodes))
    r = np.random.RandomState(random_state)
    outlier_idx = r.choice(list(node_set), size=n, replace=False)
    candidate_set = node_set.difference(set(outlier_idx))
    candidate_idx = r.choice(list(candidate_set), size=int(n * k))
    y_outlier = np.zeros(data.num_nodes)
    y_outlier[outlier_idx] = 1

    def apply_contextual_outliers(data_, outlier_idx, candidate_idx, k):
        for i, idx in enumerate(outlier_idx):
            cur_candidates = candidate_idx[k * i: k * (i + 1)]
            euclidean_dist = torch.cdist(data_.x[idx].unsqueeze(0), data_.x[list(cur_candidates)])
            max_dist_idx = torch.argmax(euclidean_dist)
            max_dist_node = list(cur_candidates)[max_dist_idx]
            with torch.no_grad():
                data_.x[idx] = data_.x[max_dist_node]
        return data_

    data = apply_contextual_outliers(data, outlier_idx, candidate_idx, k)
    data_cf = apply_contextual_outliers(data_cf, outlier_idx, candidate_idx, k)

    return data, data_cf, y_outlier





def gen_dice_outliers_cf(data, data_cf, n, r_perturb, y_outlier=None, random_state=None):
    r = np.random.RandomState(random_state)
    edge_index = data.edge_index
    labels = torch.from_numpy(data.y.reshape(-1,))
    
    undirected_edge_index = to_undirected(edge_index)
    is_symmetric = torch.equal(undirected_edge_index, undirected_edge_index.t())
    
    if y_outlier is not None:
        y_outlier = y_outlier.copy()
        node_set = set(list(np.where(y_outlier == 0)[0]))
    else:
        y_outlier = np.zeros(data.num_nodes)
        node_set = set(range(data.num_nodes)) 
    outlier_idx = r.choice(list(node_set), size=n, replace=False)
    y_outlier[outlier_idx] = 1

    def apply_dice_outliers(data, outlier_idx, r_perturb, node_set, labels, is_symmetric):
        edge_index = data.edge_index
        drop_list = []
        add_list = []
        
        for node1 in outlier_idx:
            node_drop_list = edge_index[:, edge_index[0] == node1][1]
            no_edge = (node_set - set(node_drop_list.tolist()))
            label_diff = torch.nonzero(labels != labels[node1]).squeeze()
            node_add_list = torch.tensor(list(no_edge & set(label_diff.tolist())))

            drop_num = int(np.ceil(r_perturb * np.min((len(node_drop_list), data.num_nodes - len(node_drop_list)))))
            
            edge_drop = r.choice(node_drop_list, size=drop_num, replace=False)
            for node2 in edge_drop:
                drop_list.append(torch.tensor([[node1, node2]], dtype=torch.long))
                if is_symmetric:
                    drop_list.append(torch.tensor([[node2, node1]], dtype=torch.long))

            edge_add = r.choice(node_add_list, size=drop_num, replace=False)
            for node2 in edge_add:
                add_list.append(torch.tensor([[node1, node2]], dtype=torch.long))
                if is_symmetric:
                    add_list.append(torch.tensor([[node2, node1]], dtype=torch.long))

        drop_list = torch.unique(torch.cat(drop_list), dim=0)
        add_list = torch.unique(torch.cat(add_list), dim=0)

        for ed in drop_list:
            edge_mask = (edge_index[0] != ed[0]) | (edge_index[1] != ed[1])
            edge_index = edge_index[:, edge_mask]
        edge_index = torch.cat([edge_index, add_list.T], dim=1)
        data.edge_index = edge_index
        
        return data


    data = apply_dice_outliers(data, outlier_idx, r_perturb, node_set, labels, is_symmetric)
    data_cf = apply_dice_outliers(data_cf, outlier_idx, r_perturb, node_set, labels, is_symmetric)

    return data, data_cf, y_outlier



def gen_path_outliers_cf(data, data_cf, n, k, random_state=None):
    graph = to_networkx(data)
    graph = graph.to_undirected()
        
    y_outlier = np.zeros(data.num_nodes)
    r = np.random.RandomState(random_state)
    
    while sum(y_outlier) < n:
        node_set = set(list(np.where(y_outlier == 0)[0]))
        outlier_idx = r.choice(list(node_set), size=n, replace=False)
        candidate_set = node_set.difference(set(outlier_idx))
        candidate_idx = r.choice(list(candidate_set), size=int(n * k))
    
        def apply_path_outliers(data, outlier_idx, candidate_idx, k):
            for i, idx in enumerate(outlier_idx):
                cur_candidates = candidate_idx[k * i: k * (i + 1)]
                path = []
                path_len = []
                for can in cur_candidates:
                    try:
                        shortest_path = nx.shortest_path(graph, source=idx, target=can)
                        len_shortest_path = len(shortest_path)
                        path.append(shortest_path)
                        path_len.append(len_shortest_path)
                    except:
                        path_len.append(-1)

                if np.sum(np.array(path_len) > 0) > (k // 2):
                    max_path_idx = np.random.choice(np.where(path_len == np.max(path_len))[0])
                    max_path_node = list(cur_candidates)[max_path_idx]
                    with torch.no_grad():
                        data.x[idx] = data.x[max_path_node]
                    y_outlier[idx] = 1

                if sum(y_outlier) >= n:
                    break
            return data
        data = apply_path_outliers(data, outlier_idx, candidate_idx, k)
        data_cf = apply_path_outliers(data_cf, outlier_idx, candidate_idx, k)

    return data, data_cf, y_outlier

def gen_cont_struc_outliers_cf(data, data_cf, n, sample_size, struc_clique_size, clique_num, struc_drop_prob, random_state=None):
    data, data_cf, y_outlier = gen_contextual_outliers_cf(data, data_cf, n//2, sample_size, random_state)
    data, data_cf, y_outlier = gen_structural_outliers_cf(data, data_cf, struc_clique_size, clique_num, struc_drop_prob, y_outlier, random_state)
    return data, data_cf, y_outlier

def gen_path_dice_outliers_cf(data, data_cf, n, sample_size, dice_ratio, random_state=None):
    data, data_cf, y_outlier = gen_path_outliers_cf(data, data_cf, n//2, sample_size, random_state)
    data, data_cf, y_outlier = gen_dice_outliers_cf(data, data_cf, n//2, dice_ratio, y_outlier, random_state)
    return data, data_cf, y_outlier

def outlier_injection_cf(args, data, data_cf):  
    
    # args = preset_parameters(args)
    
    if args.outlier_type=='structural':
        clique_num = math.ceil(args.outlier_num / args.struc_clique_size)
        data, data_cf, y_outlier = gen_structural_outliers_cf(data, data_cf, args.struc_clique_size, clique_num, args.struc_drop_prob, None, args.outlier_seed)
        print('Generating structural outliers', int(np.sum(y_outlier)))
        
    elif args.outlier_type=='contextual':
        data, data_cf, y_outlier = gen_contextual_outliers_cf(data, data_cf, args.outlier_num, args.sample_size, args.outlier_seed)
        print('Generating contextual outliers', int(np.sum(y_outlier)))
        
    elif args.outlier_type=='dice':
        data, data_cf, y_outlier = gen_dice_outliers_cf(data, data_cf, args.outlier_num, args.dice_ratio, None, args.outlier_seed)
        print('Generating dice outliers', int(np.sum(y_outlier)))
  
    elif args.outlier_type=='path':
        data, data_cf, y_outlier = gen_path_outliers_cf(data, data_cf, args.outlier_num, args.sample_size, args.outlier_seed)
        print('Generating path outliers', int(np.sum(y_outlier)))
        
    elif args.outlier_type=='cont_struc':
        clique_num = math.ceil(args.outlier_num /2 / args.struc_clique_size)
        data, data_cf, y_outlier = gen_cont_struc_outliers_cf(data, data_cf, args.outlier_num, args.sample_size, args.struc_clique_size, clique_num, args.struc_drop_prob, args.outlier_seed)
        print('Generating contextual and structural outliers', int(np.sum(y_outlier)))  
        
    elif args.outlier_type=='path_dice':
        data, data_cf, y_outlier = gen_path_dice_outliers_cf(data, data_cf, args.outlier_num, args.sample_size, args.dice_ratio, args.outlier_seed)
        print('Generating path and dice outliers', int(np.sum(y_outlier)))  
                
    return data, data_cf, y_outlier
    


