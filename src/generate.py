import os
import numpy as np
import argparse
import shutil
from utils.loader_HNN_GAD import *
from utils.outlier_inject_cf import *
from utils.Preprocessing import *
from torch_geometric.utils import dense_to_sparse

def generate_synthetic_data(path, n=2000, z_dim=50, p=0.4, q=0.3, alpha=0.01, beta=0.01, threshold=0.6, dim=32):    
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scio.savemat(path, data)
    # print("(labels_cf_sim - labels_sim", (labels_cf_sim - labels_sim.sum()))
    # print('data saved in ', path)
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic.mat and flip_rate-labelled copies"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/dataset/synthetic',
        help='Directory to save synthetic datasets'
    )
    parser.add_argument(
        '--flip_rates',
        nargs='+',
        type=float,
        default=[0.0, 0.5, 1.0],
        help='List of flip rates for which to label as counterfactual distrubance'
    )
    args = parser.parse_args()

    # ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) generate the base synthetic dataset (contains both original & CF graphs)
    base_path = os.path.join(args.output_dir, 'synthetic.mat')
    print(f"Generating base synthetic dataset → {base_path}")
    generate_synthetic_data(base_path)

    # 2) for each flip_rate, make a copy named synthetic_{rate}.mat
    #    (NOTE: upstream generate_synthetic_data currently always does full flip;
    #     these copies are identical placeholders)
    for rate in args.flip_rates:
        dest_path = os.path.join(args.output_dir, f'synthetic_{rate}.mat')
        shutil.copyfile(base_path, dest_path)
        print(f"Copied base dataset for flip_rate={rate} → {dest_path}")

if __name__ == '__main__':
    main()
