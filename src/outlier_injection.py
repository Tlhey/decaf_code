#!/usr/bin/env python
import os
import math
import argparse
import pickle
import scipy.io as scio
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

from utils.Preprocessing import load_data
from utils.loader_HNN_GAD import outlier_injection
from utils.outlier_inject_cf import outlier_injection_cf

# Directory setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "..", "data", "injected_data")


def save_injected_data(dataset, flip_rate, outlier_type, data_mod, data_mod_cf=None):
    save_dir = os.path.join(OUTPUT_ROOT, dataset)
    os.makedirs(save_dir, exist_ok=True)

    if dataset == 'synthetic':
        filename = f"{dataset}_{flip_rate:.1f}_{outlier_type}.pkl"
        data_dict = {'data_mod': data_mod, 'data_mod_cf': data_mod_cf}
    else:
        filename = f"{dataset}_{outlier_type}.pkl"
        data_dict = {'data_mod': data_mod}

    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved injected data to {filepath}")


def process_non_synthetic(dataset, outlier_type, args):
    adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, raw_data_info = \
        load_data('../', dataset)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    data = Data(x=features, edge_index=edge_index, y=labels)

    args.outlier_type = outlier_type
    data_mod, y_outlier = outlier_injection(args, data)
    data_mod.y = y_outlier
    save_injected_data(dataset, None, outlier_type, data_mod)


def load_synthetic_data(flip_rate):
    mat_path = os.path.join(
        SCRIPT_DIR, "..", "data", "dataset", "synthetic",
        f"synthetic_{flip_rate:.1f}.mat"
    )
    mat = scio.loadmat(mat_path)

    x = mat['x']
    edge_index = torch.tensor(mat['edge_index'], dtype=torch.long)
    labels = torch.tensor(mat['labels'].reshape(-1), dtype=torch.long)
    data = Data(x=torch.FloatTensor(x), edge_index=edge_index, y=labels)

    x_cf = mat['x_cf']
    edge_index_cf = torch.tensor(mat['edge_index_cf'], dtype=torch.long)
    data_cf = Data(x=torch.FloatTensor(x_cf), edge_index=edge_index_cf, y=labels)

    return data, data_cf


def process_synthetic(flip_rate, outlier_type, args):
    data, data_cf = load_synthetic_data(flip_rate)

    args.outlier_type = outlier_type
    N = data.num_nodes
    args.outlier_num = math.ceil(0.05 * N)

    # 🌸 add back required internal parameters (not from CLI)
    args.struc_clique_size = 10
    args.sample_size = 5
    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0

    data_mod, data_mod_cf, y_outlier = outlier_injection_cf(args, data, data_cf)
    data_mod.y = y_outlier
    data_mod_cf.y = y_outlier
    save_injected_data('synthetic', flip_rate, outlier_type, data_mod, data_mod_cf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inject outliers into a single dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to process: german, credit, bail, pokec_z, pokec_n, synthetic')
    parser.add_argument('--outlier_type', type=str,
                        choices=['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice'],
                        required=True, help='Single outlier type to inject')
    parser.add_argument('--flip_rate', type=float, default=0.0,
                        help='Flip rate for synthetic data (only one value now)')

    args = parser.parse_args()

    if args.dataset == 'synthetic':
        process_synthetic(args.flip_rate, args.outlier_type, args)
    else:
        process_non_synthetic(args.dataset, args.outlier_type, args)
