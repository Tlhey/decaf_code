import os
# import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
import scipy.io as scio

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    # print('building edge relationship complete')
    idx_map =  np.array(idx_map)
    
    return idx_map

def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

#    # Normalize MaxBillAmountOverLast6Months
#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
#
#    # Normalize MaxPaymentAmountOverLast6Months
#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
#
#    # Normalize MostRecentBillAmount
#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
#
#    # Normalize MostRecentPaymentAmount
#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
#
#    # Normalize TotalMonthsOverdue
#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]

    idx_train_list = []
    idx_val_list = []
    idx_test_list = []
    for i in range(10):
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        idx_train_list.append(idx_train)
        idx_val_list.append(idx_val)
        idx_test_list.append(idx_test)

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)

    return adj, features, labels, idx_train_list, idx_val_list, idx_test_list, sens


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]

    idx_train_list = []
    idx_val_list = []
    idx_test_list = []
    for i in range(10):
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        idx_train_list.append(idx_train)
        idx_val_list.append(idx_val)
        idx_test_list.append(idx_test)

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    
    return adj, features, labels, idx_train_list, idx_val_list, idx_test_list, sens

def load_synthetic(path, label_number=1000):
    data = scio.loadmat(path)
    features = data['x']
    adj = data['adj']
    labels = data['labels'][0]
    sens = data['sens'][0]
    features = np.concatenate([sens.reshape(-1,1), features], axis=1)
    raw_data_info = {}
    raw_data_info['adj'] = adj
    raw_data_info['w'] = data['w']
    raw_data_info['w_s'] = data['w_s'][0][0]
    raw_data_info['z'] = data['z']
    raw_data_info['v'] = data['v']
    raw_data_info['feat_idxs'] = data['feat_idxs'][0]
    raw_data_info['alpha'] = data['alpha'][0][0]

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    idx_train_list = []
    idx_val_list = []
    idx_test_list = []
    for i in range(10):
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        idx_train_list.append(idx_train)
        idx_val_list.append(idx_val)
        idx_test_list.append(idx_test)

    sens = torch.FloatTensor(sens)
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    return adj, features, labels, idx_train_list, idx_val_list, idx_test_list, sens, raw_data_info


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="../dataset/german/", label_number=1000):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

#    for i in range(idx_features_labels['PurposeOfLoan'].unique().shape[0]):
#        val = idx_features_labels['PurposeOfLoan'].unique()[i]
#        idx_features_labels['PurposeOfLoan'][idx_features_labels['PurposeOfLoan'] == val] = i

#    # Normalize LoanAmount
#    idx_features_labels['LoanAmount'] = 2*(idx_features_labels['LoanAmount']-idx_features_labels['LoanAmount'].min()).div(idx_features_labels['LoanAmount'].max() - idx_features_labels['LoanAmount'].min()) - 1
#
#    # Normalize Age
#    idx_features_labels['Age'] = 2*(idx_features_labels['Age']-idx_features_labels['Age'].min()).div(idx_features_labels['Age'].max() - idx_features_labels['Age'].min()) - 1
#
#    # Normalize LoanDuration
#    idx_features_labels['LoanDuration'] = 2*(idx_features_labels['LoanDuration']-idx_features_labels['LoanDuration'].min()).div(idx_features_labels['LoanDuration'].max() - idx_features_labels['LoanDuration'].min()) - 1
#
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]

    idx_train_list = []
    idx_val_list = []
    idx_test_list = []
    for i in range(10):
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)

        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                              label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                            label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        idx_train_list.append(idx_train)
        idx_val_list.append(idx_val)
        idx_test_list.append(idx_test)

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
   
    return adj, features, labels, idx_train_list, idx_val_list, idx_test_list, sens





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
