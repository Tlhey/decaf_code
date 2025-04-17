import sys
import numpy as np
import torch
import random
import argparse
import math
import os

from utils.Preprocessing import load_data
from utils.loader_HNN_GAD import *

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import pickle



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german', help='Dataset to use')
    args = parser.parse_args()
    
    adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, raw_data_info = load_data(path_root='', dataset=args.dataset)
    label_number = len(labels) 
    
    # Ensure all data is in torch.Tensor format
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    sens = torch.tensor(sens, dtype=torch.float)

    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    data = Data(x=features, edge_index=edge_index, y=labels)

    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0
    args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
    args.sample_size = args.struc_clique_size
    args.outlier_num = math.ceil(data.num_nodes * 0.05)
    
    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    # for outlier_type in outlier_types:
    #     args.outlier_type = outlier_type
    #     data_mod, y_outlier = outlier_injection(args, data)
    #     data_mod.y = y_outlier

    #     filename = f"injected_data/{args.dataset}/{args.dataset}_{outlier_type}.pkl"
    #     with open(filename, 'wb') as f:
    #         pickle.dump(data_mod, f)
    #     print(f"Data with {outlier_type} outliers has been generated and saved as {filename}.")


# import os
# import math
# import numpy as np
# import random
# from scipy import sparse
# import scipy.io as scio
# import argparse 
# import optuna
# import pickle
# from sklearn.preprocessing import normalize
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# from torch_geometric.utils import from_scipy_sparse_matrix
# from torch_geometric.data import Data
# from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DOMINANT, DONE, GAAN, GADNR, GAE, GUIDE
# from utils.outlier_inject_cf import *

# def generate_and_save_outlier_data(args, data, data_cf, outlier_types, flip_rate, save_dir='injected_data'):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     for outlier_type in outlier_types:
#         args.outlier_type = outlier_type
#         data_mod, data_mod_cf, y_outlier = outlier_injection_cf(args, data, data_cf)
#         data_mod.y = y_outlier

#         # adj_mod = sparse.csr_matrix((np.ones(data_mod.edge_index.shape[1]), (data_mod.edge_index[0], data_mod.edge_index[1])), shape=(data_mod.num_nodes, data_mod.num_nodes))
#         # adj_mod_cf = sparse.csr_matrix((np.ones(data_mod_cf.edge_index.shape[1]), (data_mod_cf.edge_index[0], data_mod_cf.edge_index[1])), shape=(data_mod_cf.num_nodes, data_mod_cf.num_nodes))

#         # filename = f"{save_dir}/synthetic_{flip_rate}_{outlier_type}.pkl"
#         # scio.savemat(filename, {
#         #     'x': data_mod.x,
#         #     'edge_index': data_mod.edge_index,
#         #     'y': data_mod.y,
#         #     'x_cf': data_mod_cf.x,
#         #     'adj_cf': adj_mod_cf,
#         #     'y_cf': data_mod_cf.y
#         # })
#         filename = f"{save_dir}/synthetic/synthetic_{flip_rate}_{outlier_type}.pkl"
#         data_dict = {
#             'data_mod': data_mod,
#             'data_mod_cf': data_mod_cf
#         }
#         with open(filename, 'wb') as f:
#             pickle.dump(data_dict, f)
#         print(f"Data with {outlier_type} outliers has been generated and saved as {filename}.")
        
# def load_synthetic(path, label_number=1000):
#     data = scio.loadmat(path)
#     features = data['x']
#     features_cf = data['x_cf']
#     adj = data['adj']
#     adj_cf = data['adj_cf']
#     labels = data['y']
#     labels_cf = data['y_cf']
#     sens = data['sens'][0]
#     sens_cf = data['sens_cf'][0]
#     features = np.concatenate([sens.reshape(-1,1), features], axis=1)
#     features_cf = np.concatenate([sens_cf.reshape(-1,1), features_cf], axis=1)
#     raw_data_info = {}
#     raw_data_info['adj'] = adj
#     raw_data_info['w'] = data['w']
#     raw_data_info['w_s'] = data['w_s'][0][0]
#     raw_data_info['z'] = data['z']
#     raw_data_info['v'] = data['v']
#     raw_data_info['feat_idxs'] = data['feat_idxs'][0]
#     raw_data_info['alpha'] = data['alpha'][0][0]
#     features = torch.FloatTensor(features)
#     features_cf = torch.FloatTensor(features_cf)
#     return sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     for flip_rate in [0.0, 0.5, 1.0]:
#         path_synthetic = f'synthetic/synthetic_{flip_rate}.mat'
#         sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info = load_synthetic(path=path_synthetic)
        
#         edge_index, edge_attr = from_scipy_sparse_matrix(adj)
#         edge_index_cf, edge_attr_cf = from_scipy_sparse_matrix(adj_cf)
        
#         data = Data(x=features, edge_index=edge_index, y=labels)
#         data_cf = Data(x=features_cf, edge_index=edge_index_cf, y=labels_cf)

#         args.struc_drop_prob = 0.2
#         args.dice_ratio = 0.5
#         args.outlier_seed = 0
#         args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
#         args.sample_size = args.struc_clique_size
#         args.outlier_num = math.ceil(data.num_nodes * 0.05)
        
#         outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
#         generate_and_save_outlier_data(args, data, data_cf, outlier_types, flip_rate)
