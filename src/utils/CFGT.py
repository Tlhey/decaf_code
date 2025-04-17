import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch import optim
import utils.Preprocessing as dpp
from utils.CFDA import CFDA
import os 
import torch
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch

class CFGT(nn.Module):
    def __init__(self, h_dim, input_dim, adj):
        super(CFGT, self).__init__()
        self.h_dim = h_dim
        # A
        self.base_gcn = GraphConvSparse(input_dim, h_dim, adj)
        self.gcn_mean = GraphConvSparse(h_dim, h_dim, adj, activation=lambda x: x)
        self.pred_a = nn.Sequential(nn.Linear(h_dim, adj.shape[1]))

        # S
        self.sf = nn.Sequential(nn.Linear(1, 1))  # n x 1, parameter: 1

    def encode_A(self, X):
        mask_X = X
        hidden = self.base_gcn(mask_X)
        mean = self.gcn_mean(hidden)
        sampled_z = mean
        return sampled_z

    def pred_adj(self, Z, S):
        A_pred = self.pred_a(Z)  # n x n
        S_rep_f = self.sf(S)
        S_rep_cf = self.sf(1 - S)

        s_match = (torch.matmul(S_rep_f, S_rep_f.t()) + torch.matmul(S_rep_cf, S_rep_cf.t())) / 2
        A_pred = F.sigmoid(A_pred + s_match)
        return A_pred

    def encode(self, X):
        Z_a = self.encode_A(X)
        return Z_a

    def pred_graph(self, Z_a, S):
        A_pred = self.pred_adj(Z_a, S)
        return A_pred

    def forward(self, X, sen_idx):
        # encoder: X\S, adj -> Z
        # decoder: Z + S' -> A'
        S = X[:, sen_idx].view(-1, 1)
        X_ns = X.clone()
        X_ns[:, sen_idx] = 0.  # mute this dim

        Z_a = self.encode(X_ns)
        A_pred = self.pred_graph(Z_a, S)
        return A_pred

    def loss_function(self, adj, A_pred):
        # loss_reconst
        weighted = True
        if weighted:
            weights_0 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
            weights_1 = 1 - weights_0
            assert (weights_0 > 0 and weights_1 > 0)
            weight = torch.ones_like(A_pred).reshape(-1) * weights_0  # (n x n), weight 0
            idx_1 = adj.to_dense().reshape(-1) == 1
            weight[idx_1] = weights_1

            loss_bce = nn.BCELoss(weight=weight, reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))
        else:
            loss_bce = nn.BCELoss(reduction='mean')
            loss_reconst_a = loss_bce(A_pred.reshape(-1), adj.to_dense().reshape(-1))

        loss_result = {'loss_reconst_a': loss_reconst_a}
        return loss_result

    def train_model(self, X, adj, sen_idx, dataset, model_path='', lr=0.0001, weight_decay=1e-5):
        rate_1 = torch.sparse.sum(adj) / (adj.shape[0] * adj.shape[1])
        print('adj=1: ', rate_1)

        optimizer = optim.Adam([{'params': self.parameters(), 'lr': lr}], weight_decay=weight_decay)

        self.train()
        n = X.shape[0]

        print("start training counterfactual augmentation module!")
        for epoch in range(2000):
            optimizer.zero_grad()

            A_pred = self.forward(X, sen_idx)
            loss_result = self.loss_function(adj, A_pred)

            # backward propagation
            loss_reconst_a = loss_result['loss_reconst_a']
            loss_reconst_a.backward()
            optimizer.step()

            if epoch % 100 == 0:
                self.eval()
                eval_result = self.test(X, adj, sen_idx)
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_reconst_a: {:.4f}'.format(loss_reconst_a.item()),
                      'acc_a_pred: {:.4f}'.format(eval_result['acc_a_pred'].item()),
                      'acc_a_pred_0: {:.4f}'.format(eval_result['acc_a_pred_0'].item()),
                      'acc_a_pred_1: {:.4f}'.format(eval_result['acc_a_pred_1'].item()),
                      )
                # save model
                save_model = True
                if save_model and epoch > 0:
                    save_model_path = model_path + f'weights_CFGT_{dataset}' + '.pt'
                    torch.save(self.state_dict(), save_model_path)
                    print('saved model weight in: ', save_model_path)
                self.train()
        return

    def test(self, X, adj, sen_idx):
        self.eval()
        A_pred = self.forward(X, sen_idx)
        loss_result = self.loss_function(adj, A_pred)
        eval_result = loss_result

        A_pred_binary = (A_pred > 0.5).float()  # binary
        adj_size = A_pred_binary.shape[0] * A_pred_binary.shape[1]

        sum_1 = torch.sparse.sum(adj)
        correct_num_1 = torch.sparse.sum(sparse_dense_mul(adj, A_pred_binary))  # 1
        correct_num_0 = (adj_size - (A_pred_binary + adj).sum() + correct_num_1)
        acc_a_pred = (correct_num_1 + correct_num_0) / adj_size
        acc_a_pred_0 = correct_num_0 / (adj_size - sum_1)
        acc_a_pred_1 = correct_num_1 / sum_1

        eval_result['acc_a_pred'] = acc_a_pred
        eval_result['acc_a_pred_0'] = acc_a_pred_0
        eval_result['acc_a_pred_1'] = acc_a_pred_1
        return eval_result

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


class PPR:
    #Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
        
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else :
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        
        return neighbor
    
    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            print ('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else :
            print ('File of node {} exists.'.format(seed))
    
    def search_all(self, node_num, path):
        neighbor  = {}
        if os.path.isfile(path+'_neighbor') and os.stat(path+'_neighbor').st_size != 0:
            print ("Exists neighbor file")
            neighbor = torch.load(path+'_neighbor')
        else :
            print ("Extracting subgraphs")
            os.system('mkdir {}'.format(path))
            with mp.Pool() as pool:
                list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
                
            print ("Finish Extracting")
            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
            torch.save(neighbor, path+'_neighbor')
            os.system('rm -r {}'.format(path))
            print ("Finish Writing")
        return neighbor

class Subgraph:
    # Class for subgraph extraction

    def __init__(self, x, edge_index, path, maxsize=50, n_order=10):
        self.x = x
        self.path = path
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize

        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])),
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)

        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}

    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def adjust_edge(self, idx):
        # Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i

        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            # edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        # Generate node features for subgraphs
        return self.x[idx]

    def build(self, postfix):
        # # Extract subgraphs for all nodes
        # if os.path.isfile(self.path + '_subgraph') and os.stat(self.path + '_subgraph').st_size != 0:
        #     print("Exists subgraph file")
        #     self.subgraph = torch.load(self.path + '_subgraph')
        #     return

        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x, edge)
        torch.save(self.subgraph, self.path + '_subgraph' + postfix)

    def search(self, node_list):
        # Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        return batch, index