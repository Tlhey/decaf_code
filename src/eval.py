import sys
import os
import argparse
import random
import yaml
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, precision_recall_curve, auc

# Import detectors
from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DOMINANT, DONE, GAAN, GAE, GUIDE, ONE
# If custom GAD implementations exist under pygod_mod

# from pygod_mod.pygod.detector.gadnr import GADNR
MODEL_MAP = {
    'AdONE': AdONE,
    'ANOMALOUS': ANOMALOUS,
    'AnomalyDAE': AnomalyDAE,
    'CoLA': CoLA,
    'CONAD': CONAD,
    'DMGD': DMGD,
    'DOMINANT': DOMINANT,
    'DONE': DONE,
    'GAAN': GAAN,
    'GAE': GAE,
    'GUIDE': GUIDE,
    'ONE': ONE,
    # 'GADNR': GADNR,
}


def evaluate_outlier_detection(y_true, y_pred, sens):
    """
    Compute fairness and performance metrics.
    """
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
    dp = abs(np.mean(y_pred[sens==0]) - np.mean(y_pred[sens==1]))
    eoo = abs(np.mean(y_pred[(sens==0)&(y_true==1)]) - np.mean(y_pred[(sens==1)&(y_true==1)]))
    return {
        'acc': acc,
        'auc': auc_roc,
        'prauc': pr_auc,
        'ap': ap,
        'f1': f1_s,
        'dp': dp,
        'eoo': eoo,
    }


def train_and_predict(data, model_class, params, contamination=0.05, threshold_percentile=95, random_seed=None):
    """
    Train the model and predict outliers.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    model = model_class(contamination=contamination, gpu=0, **params)
    model.fit(data)
    scores = np.array(model.decision_score_).ravel()
    threshold = np.percentile(scores, threshold_percentile)
    preds = (scores > threshold).astype(int)
    return scores, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GAD models on datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of dataset (e.g., credit, german, synthetic)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name matching keys in MODEL_MAP')
    parser.add_argument('--outlier_type', type=str, required=True,
                        help='Outlier injection type')
    parser.add_argument('--flip_rate', type=str, default=None,
                        help='Flip rate for synthetic dataset')
    args = parser.parse_args()

    # Determine config file
    base_cfg = f'configs/baseline_{args.dataset}.yml'
    custom_cfg = f'configs/custom_{args.dataset}.yml'
    cfg_path = custom_cfg if os.path.exists(custom_cfg) else base_cfg

    # Load hyperparameters
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # params under cfg[model][outlier_type] or cfg[model][outlier_type][flip_rate]
    model_cfg = cfg[args.model][args.outlier_type]
    if args.dataset == 'synthetic':
        model_cfg = model_cfg[str(args.flip_rate)]

    # Load data
    from eval import load_synthetic_mod, load_data_mod  # assume these util functions available
    if args.dataset == 'synthetic':
        data, data_cf = load_synthetic_mod(args.outlier_type, args.flip_rate)
        sens = data.x[:, 0]
        sens_cf = data_cf.x[:, 0]
        # evaluate multiple runs
        results, results_cf = [], []
        for i in range(10):
            seed = random.randint(0, 10000)
            _, pred = train_and_predict(data, MODEL_MAP[args.model], model_cfg, random_seed=seed)
            _, pred_cf = train_and_predict(data_cf, MODEL_MAP[args.model], model_cfg, random_seed=seed)
            res = evaluate_outlier_detection(data.y, pred, sens)
            res_cf = evaluate_outlier_detection(data_cf.y.reshape(-1), pred_cf, sens_cf)
            # counterfactual fairness score
            cf_score = 1 - (pred_cf == pred).mean()
            res['cf'] = cf_score
            res_cf['cf'] = cf_score
            results.append(res)
            results_cf.append(res_cf)
        # average and std
        avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
        std = {k: np.std([r[k] for r in results]) for k in results[0]}
        print(f"Dataset={args.dataset}, Model={args.model}, Type={args.outlier_type}, Flip={args.flip_rate}")
        print({k: f"{avg[k]:.4f} ± {std[k]:.4f}" for k in avg})

    else:
        data = load_data_mod(args.dataset, args.outlier_type)
        # determine sens column index
        sens = data.x[:, 0] if args.dataset != 'credit' else data.x[:, 1]
        results = []
        for i in range(10):
            seed = random.randint(0, 10000)
            _, pred = train_and_predict(data, MODEL_MAP[args.model], model_cfg, random_seed=seed)
            res = evaluate_outlier_detection(data.y, pred, sens)
            results.append(res)
        avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
        std = {k: np.std([r[k] for r in results]) for k in results[0]}
        print(f"Dataset={args.dataset}, Model={args.model}, Type={args.outlier_type}")
        print({k: f"{avg[k]:.4f} ± {std[k]:.4f}" for k in avg})
