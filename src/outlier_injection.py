import os
import copy
import numpy as np
import torch
import math
import argparse
from utils.loader_HNN_GAD import *
from utils.outlier_inject_cf import *
import ray
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import tuned_config
from utils.Preprocessing import *
from utils.CFGT import Subgraph
from collections import defaultdict

def generate_synthetic_data(path, n, z_dim, p, q, alpha, beta, threshold, dim):    
    n = 2000
    z_dim = 50
    p = 0.4
    q = 0.3
    alpha = 0.01  
    beta = 0.01
    threshold = 0.6
    dim = 32
    sens = np.random.binomial(n=1, p=p, size=n)
    sens_repeat = np.repeat(sens.reshape(-1, 1), z_dim, axis=1)
    sens_embedding = np.random.normal(loc=sens_repeat, scale=1, size=(n, z_dim))
    labels = np.random.binomial(n=1, p=q, size=n)
    labels_repeat = np.repeat(labels.reshape(-1, 1), z_dim, axis=1)
    labels_embedding = np.random.normal(loc=labels_repeat, scale=1, size=(n, z_dim))
    features_embedding = np.concatenate((sens_embedding, labels_embedding), axis=1)
    weight = np.random.normal(loc=0, scale=1, size=(z_dim*2, dim))
    # features = np.matmul(features_embedding, weight)
    features = np.matmul(features_embedding, weight) + np.random.normal(loc=0, scale=1, size=(n, dim))

    adj = np.zeros((n, n))
    sens_sim = np.zeros((n, n))
    labels_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim[i][j] = -1
                labels_sim[i][j] = -1
                continue
            sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])
            labels_sim[i][j] = labels_sim[j][i] = (labels[i] == labels[j])
            # sim_ij = 1 - spatial.distance.cosine(embedding[i], embedding[j])  # [-1, 1]
            # adj[i][j] = adj[j][i] = sim_ij + alpha * (sens[i] == sens[j])

    similarities = cosine_similarity(features_embedding)  # n x n
    similarities[np.arange(n), np.arange(n)] = -1
    adj = similarities + alpha * sens_sim + beta * labels_sim
    print('adj max: ', adj.max(), ' min: ', adj.min())
    adj[np.where(adj >= threshold)] = 1
    adj[np.where(adj < threshold)] = 0
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj, dtype=torch.float))
    edge_num = adj.sum()
    # adj = sparse.csr_matrix(adj)
    # features = np.concatenate((sens.reshape(-1,1), features), axis=1)

    # generate counterfactual
    sens_flip = 1 - sens
    sens_flip_repeat = np.repeat(sens_flip.reshape(-1, 1), z_dim, axis=1)
    # sens_flip_embedding = np.random.normal(loc=sens_flip_repeat, scale=1, size=(n, z_dim))
    sens_flip_embedding = sens_embedding
    features_embedding = np.concatenate((sens_flip_embedding, labels_embedding), axis=1)
    features_cf = np.matmul(features_embedding, weight) + np.random.normal(loc=0, scale=1, size=(n, dim))

    adj_cf = np.zeros((n, n))
    sens_cf_sim = np.zeros((n, n))
    labels_cf_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sens_cf_sim[i][j] = -1
                labels_cf_sim[i][j] = -1
                continue
            sens_cf_sim[i][j] = sens_cf_sim[j][i] = (sens_flip[i] == sens_flip[j])
            labels_cf_sim[i][j] = labels_cf_sim[j][i] = (labels[i] == labels[j])
    
    similarities_cf = cosine_similarity(features_cf)  # n x n
    similarities_cf[np.arange(n), np.arange(n)] = -1
    adj_cf = similarities_cf + alpha * sens_cf_sim + beta * labels_cf_sim
    print('adj_cf max', adj_cf.max(), ' min: ', adj_cf.min())
    adj_cf[np.where(adj_cf >= threshold)] = 1
    adj_cf[np.where(adj_cf < threshold)] = 0
    edge_index_cf, edge_attr_cf = dense_to_sparse(torch.tensor(adj_cf, dtype=torch.float))
    # edge_index_cf = torch.nonzero(torch.from_numpy(adj_cf)).t().contiguous()
    # adj_cf = sparse.csr_matrix(adj_cf)
    # features_cf = np.concatenate((sens_flip.reshape(-1,1), features_cf), axis=1)

    # statistics
    # pre_analysis(adj, labels, sens)
    # print('edge num: ', edge_num)
    data = {'x': features, 'edge_index': edge_index, 'labels': labels, 'sens': sens, 'x_cf': features_cf, 'edge_index_cf': edge_index_cf, "edge_num": edge_num}
    scio.savemat(path, data)
    # print("(labels_cf_sim - labels_sim", (labels_cf_sim - labels_sim.sum()))
    # print('data saved in ', path)
    return data

def evaluate_outlier_detection(y_true, y_pred, sens):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(sens, torch.Tensor):
        sens = sens.numpy()

    acc = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    f1_s = f1_score(y_true, y_pred)
    dp = np.abs(np.mean(y_pred[sens == 0]) - np.mean(y_pred[sens == 1]))
    eoo = np.abs(np.mean(y_pred[(sens == 0) & (y_true == 1)]) - np.mean(y_pred[(sens == 1) & (y_true == 1)]))
    eval_results = {'dp': dp, 'eoo': eoo, 'auc': auc_roc, 'f1': f1_s, 'acc': acc}
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['credit', 'german', 'bail', 'pokec_z', 'pokec_n', 'UCSD34', 'synthetic'], required=True, help='Dataset to use')
    parser.add_argument('--model', type=str,  choices=[ 'anomalous', 'adone', 'cola', 'conad', 'dominant', 'dmgd', 'done', 'gaan', 'gadnr', 'gae', 'guide', 'ocgnn', 'one', 'radar', 'scan'], required=False, help='Model to use')
    parser.add_argument('--outlier_type', type=str, choices=['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice'] , required=True, help='Model to use')
    args = parser.parse_args()
    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0
    
    dataset = args.dataset
    path_root = ""
    

    adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, raw_data_info = load_data(path_root, dataset)
    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    data = Data(x=features, edge_index=edge_index, y=labels)

    args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
    args.sample_size = args.struc_clique_size
    args.outlier_num  = math.ceil(data.num_nodes * 0.05)
    data_mod, data_mod.y = outlier_injection(args, data)
    config = tuned_config.get_config(args)
    model = tuned_config.get_model(args.model, config)
    
    model.fit(data_mod)
    scores = model.decision_function(data_mod)
    threshold = np.percentile(scores.cpu().detach().numpy(), 95)
    detected_outliers = (scores > threshold).to(torch.int).cpu().detach().numpy()
    eval_results = evaluate_outlier_detection(data_mod.y, detected_outliers, sens)
    
    
    sens_rate = 0.05
    if dataset == 'synthetic':
        path_cf_ag = os.path.join("data", "injected_data", "synthetic", f"{dataset}_cf_aug_{sens_rate}.pt")
        data_cf = torch.load(path_cf_ag)['data_cf']
        data_mod_, data_mod_cf_, data_mod.y = outlier_injection_cf(args, data, data_cf)
        data_mod, data_mod_cf = data_mod_.detach().clone(), data_mod_cf_.detach().clone()
        model = tuned_config.get_model(args.model, config)
        model_cf = tuned_config.get_model(args.model, config)
        
        model.fit(data_mod)
        model_cf.fit(data_mod_cf)
        scores = model.decision_function(data_mod)
        threshold = np.percentile(scores.cpu().detach().numpy(), 95)
        detected_outliers = (scores > threshold).to(torch.int).cpu().detach().numpy()

        eval_results = evaluate_outlier_detection(data_mod.y, detected_outliers, sens)
        scores_cf = model_cf.decision_function(data_mod_cf)
        threshold_cf = np.percentile(scores_cf.cpu().detach().numpy(), 95)
        detected_outliers_cf = (scores_cf > threshold_cf).to(torch.int).cpu().detach().numpy()
        
        cf = 1 - (np.sum(detected_outliers == detected_outliers_cf) / data.num_nodes)
        eval_results['cf'] = cf
        eval_results_cf = evaluate_outlier_detection(data_mod.y, detected_outliers_cf, sens)
    print(args.dataset, args.model, args.outlier_type, eval_results)
    print(args.dataset, args.model, args.outlier_type, "cf:", eval_results_cf)
    
    