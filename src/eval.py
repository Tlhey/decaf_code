import sys
import os
import numpy as np
import torch
import random
import argparse
import math
import pickle
import yaml

from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DOMINANT, DONE, GAAN, GAE, GUIDE, ONE
from pygod_mod.pygod.detector.gadnr import GADNR
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_recall_curve, auc


def evaluate_outlier_detection(y_true, y_pred, sens):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(sens, torch.Tensor):
        sens = sens.numpy()

    acc = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    pr_auc = auc(recall, precision)  
    f1_s = f1_score(y_true, y_pred)
    dp = np.abs(np.mean(y_pred[sens == 0]) - np.mean(y_pred[sens == 1]))
    eoo = np.abs(np.mean(y_pred[(sens == 0) & (y_true == 1)]) - np.mean(y_pred[(sens == 1) & (y_true == 1)]))
    eval_results = {'dp': dp, 'eoo': eoo, 'auc': auc_roc, 'f1': f1_s, 'acc': acc, 'prauc': pr_auc, 'ap': ap}
    return eval_results


def train_and_predict(data, model_class, model_params, contamination=0.05, threshold_percentile=95, random_seed=42):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    model = model_class(contamination=contamination, gpu=0, **model_params)
    model.fit(data)
    y_score = np.array(model.decision_score_).ravel()
    threshold = np.percentile(y_score, threshold_percentile)
    y_pred = (y_score > threshold).astype(int)
    return y_score, y_pred, threshold

def load_data_mod(dataset, outlier_type):
    filename = f"injected_data/{dataset}/{dataset}_{outlier_type}.pkl"
    with open(filename, 'rb') as f:
        data_mod = pickle.load(f)
    return data_mod

import scipy.io as scio
def load_synthetic_mod(outlier_type, flip_rate, dir_path='injected_data/synthetic'):
    path = f'{dir_path}/synthetic_{flip_rate}_{outlier_type}.pkl'
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)

    data_mod = data_dict['data_mod']
    data_mod_cf = data_dict['data_mod_cf']
    return data_mod, data_mod_cf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--flip_rate', type=str, required=False, default = None)
    parser.add_argument('--outlier_type', type=str, required=True)

    args = parser.parse_args()
    args.config = args.dataset + "_params.yml"
    
    model_map = {
        'AdONE': AdONE, 'ANOMALOUS': ANOMALOUS, 'AnomalyDAE': AnomalyDAE, 'CoLA': CoLA, 
        'CONAD': CONAD, 'DMGD': DMGD, 'DOMINANT': DOMINANT, 'DONE': DONE, 'GAAN': GAAN, 
        'GADNR': GADNR, 'GAE': GAE, 'GUIDE': GUIDE, 'ONE': ONE
    }
    if args.dataset == 'synthetic':
        model_class = model_map[args.model]
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) 
        model_params = config[args.model][args.outlier_type][str(args.flip_rate)]
        data, data_cf = load_synthetic_mod(outlier_type = args.outlier_type, flip_rate = args.flip_rate)
        sens, sens_cf = data.x[:, 0], data_cf.x[:, 0]
        
        results = []
        results_cf = []
        for i in range(10):
            print(i)
            random_seed = random.randint(0, 10000)
            y_score, y_pred, threshold = train_and_predict(data, model_class, model_params, random_seed=random_seed)
            y_score_cf, y_pred_cf, threshold_cf = train_and_predict(data_cf, model_class, model_params, random_seed=random_seed)          
            eval_result = evaluate_outlier_detection(data.y, y_pred, sens)
            eval_result_cf = evaluate_outlier_detection(data_cf.y.reshape(-1), y_pred_cf, sens_cf)
            cf = 1 - (np.sum(y_pred_cf == y_pred) / 2000)
            eval_result = {**eval_result, 'cf': cf}
            eval_result_cf = {**eval_result_cf, 'cf': cf}
            results.append(eval_result)   
            results_cf.append(eval_result_cf)   
            
        avg_results = {key: np.mean([res[key] for res in results]) * 100 for key in results[0]}
        std_results = {key: np.std([res[key] for res in results]) * 100 for key in results[0]}
        formatted_results = {key: f"{avg_results[key]:.6f} ± {std_results[key]:.6f}" for key in avg_results}  
        print(f"{args.dataset} {args.model} {args.outlier_type} {args.flip_rate}: {formatted_results}") 
        
        avg_results_cf = {key: np.mean([res[key] for res in results_cf]) * 100 for key in results_cf[0]}
        std_results_cf = {key: np.std([res[key] for res in results_cf]) * 100 for key in results_cf[0]}
        formatted_results_cf = {key: f"{avg_results_cf[key]:.6f} ± {std_results_cf[key]:.6f}" for key in avg_results_cf}
        print(f"{args.dataset} {args.model} {args.outlier_type} {args.flip_rate} cf : {formatted_results_cf}") 

        
    else:
        model_class = model_map[args.model]
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) 
        model_params = config[args.model][args.outlier_type]
        data = load_data_mod(args.dataset, args.outlier_type)
        sens = data.x[:, 0]
        if args.dataset == 'credit':
            sens = data.x[:, 1]

        results = []
        for i in range(10):
            random_seed = random.randint(0, 10000)
            if args.dataset == 'credit':
                model_params['batch_size'] = 2048
            y_score, y_pred, threshold = train_and_predict(data, model_class, model_params, random_seed=random_seed)
            eval_result = evaluate_outlier_detection(data.y, y_pred, sens)
            results.append(eval_result)
        
        avg_results = {key: np.mean([res[key] for res in results]) * 100 for key in results[0]}
        std_results = {key: np.std([res[key] for res in results]) * 100 for key in results[0]}
        
        formatted_results = {key: f"{avg_results[key]:.6f} ± {std_results[key]:.6f}" for key in avg_results}
        
        print(f"{args.dataset} {args.model} {args.outlier_type}: {formatted_results}")