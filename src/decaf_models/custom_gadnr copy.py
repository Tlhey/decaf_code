# -*- coding: utf-8 -*-
"""GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction (GADNR)
   The code is partially from the original implementation in 
   https://github.com/Graph-COM/GAD-NR"""
# Author: Yingtong Dou <ytongdou@gmail.com>
# License: BSD 2 clause

import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCN
from torch_geometric import compile

from pygod.detector.base import DeepDetector
from pygod.utils import validate_device, logger
import math
import random
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from torch_geometric.nn import GCN, SAGEConv, PNAConv
from torch_geometric.utils import to_undirected, add_self_loops

from utils.loss_function import *
from utils.flip_sensitive_attributes import  *
from pygod.nn.nn import MLP_generator, FNN_GAD_NR
from pygod.nn.functional import KL_neighbor_loss, W2_neighbor_loss


class GADNRBase(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 encoder_layers=1,
                 deg_dec_layers=4,
                 fea_dec_layers=3,
                 sample_size=2,
                 sample_time=1,
                 neighbor_num_list=None,
                 neigh_loss='KL',
                 lambda_loss1=1e-2,
                 lambda_loss2=1e-3,
                 lambda_loss3=1e-4,
                 full_batch=True,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 device='cpu',
                 **kwargs):
        super(GADNRBase, self).__init__()
        
        self.linear = nn.Linear(in_dim, hid_dim)
        self.out_dim = hid_dim
        self.sample_time = sample_time
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3
        self.full_batch = full_batch
        self.neigh_loss = neigh_loss
        self.device = device

        if self.full_batch: # full batch mode
            self.neighbor_num_list = neighbor_num_list
            self.tot_node = len(neighbor_num_list)

            # the normal distrubution used during 
            # neighborhood distribution recontruction
            self.m_fullbatch = torch.distributions.Normal(
                                            torch.zeros(sample_size,
                                                        self.tot_node,
                                                        hid_dim),
                                            torch.ones(sample_size,
                                                       self.tot_node,
                                                       hid_dim))
            self.mean_agg = SAGEConv(hid_dim, hid_dim,
                                 aggr='mean', normalize = False)
            self.std_agg = PNAConv(hid_dim, hid_dim, aggregators=["std"],
                               scalers=["identity"], deg=neighbor_num_list)
        else: # mini batch mode
            self.m_minibatch = torch.distributions.Normal(
                                            torch.zeros(sample_size,
                                                        hid_dim),
                                            torch.ones(sample_size,
                                                        hid_dim))

        self.mlp_mean = nn.Linear(hid_dim, hid_dim)
        self.mlp_sigma = nn.Linear(hid_dim, hid_dim)
        self.mlp_gen = MLP_generator(hid_dim, hid_dim)

        # Encoder
        backbone_kwargs = {k: v for k, v in kwargs.items() if k not in ['tot_nodes']}
        self.shared_encoder = backbone(in_channels=hid_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **backbone_kwargs)

        # Decoder
        self.degree_decoder = FNN_GAD_NR(hid_dim//2, hid_dim//2, 1, deg_dec_layers)
        # feature decoder does not reconstruct the raw feature 
        # but the embeddings obtained by the ``self.linear``` layer 
        self.feature_decoder_s = FNN_GAD_NR(hid_dim//2, hid_dim//2,
                                          hid_dim//2, fea_dec_layers)
        self.feature_decoder_ns = FNN_GAD_NR(hid_dim//2, hid_dim//2,
                                          hid_dim//2, fea_dec_layers)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        if self.neigh_loss == "KL": 
            self.neighbor_loss = KL_neighbor_loss
        elif self.neigh_loss == 'W2': 
            self.neighbor_loss = W2_neighbor_loss
        else:
            raise ValueError(self.neigh_loss, 'should be either KL or W2')
        self.pool = mp.Pool(4)
        self.in_dim = in_dim
        self.sample_size = sample_size 
        self.emb = None

    # def sample_neighbors(self, input_id, neighbor_dict,
    #                      id_mapping, gt_embeddings):
    #     """ Sample neighbors from neighbor set, if the length of neighbor set
    #         less than the sample size, then do the padding.
    #     """
    #     sampled_embeddings_list = []
    #     mask_len_list = []
    #     for index in input_id:
    #         sampled_embeddings = []
    #         neighbor_indexes = neighbor_dict[index]
    #         if len(neighbor_indexes) < self.sample_size:
    #             mask_len = len(neighbor_indexes)
    #             sample_indexes = neighbor_indexes
    #         else:
    #             sample_indexes = random.sample(neighbor_indexes,
    #                                            self.sample_size)
    #             mask_len = self.sample_size
    #         for index in sample_indexes:
    #             sampled_embeddings.append(gt_embeddings[
    #                                         id_mapping[index]].tolist())
    #         if len(sampled_embeddings) < self.sample_size:
    #             for _ in range(self.sample_size - len(sampled_embeddings)):
    #                 sampled_embeddings.append(torch.zeros(self.out_dim
    #                                                       ).tolist())
    #         sampled_embeddings_list.append(sampled_embeddings)
    #         mask_len_list.append(mask_len)
        
    #     return sampled_embeddings_list, mask_len_list
    def sample_neighbors(self, input_id, neighbor_dict, id_mapping, gt_embeddings):
        sampled_embeddings_list = []
        mask_len_list = []
        for index in input_id:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict.get(index, [])
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, self.sample_size)
                mask_len = self.sample_size
            for idx in sample_indexes:
                if idx in id_mapping:
                    sampled_embeddings.append(gt_embeddings[id_mapping[idx]].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mask_len_list.append(mask_len)

        return sampled_embeddings_list, mask_len_list

    def full_batch_neigh_recon(self, h1, h0, edge_index):
        """Computing the target neighbor distribution and 
        reconstructed neighbor distribution using full batch of the data.
        """
                
        mean_neigh = self.mean_agg(h0, edge_index).detach()
        std_neigh = self.std_agg(h0, edge_index).detach()
        
        cov_neigh = torch.bmm(std_neigh.unsqueeze(dim=-1),
                              std_neigh.unsqueeze(dim=1))
        
        target_mean = mean_neigh
        target_cov = cov_neigh
        
        self_embedding = h1
        self_embedding = self_embedding.unsqueeze(0)
        self_embedding = self_embedding.repeat(self.sample_size, 1, 1)
        generated_mean = self.mlp_mean(self_embedding)
        generated_sigma = self.mlp_sigma(self_embedding)

        std_z = self.m_fullbatch.sample().to(self.device)
        var = generated_mean + generated_sigma.exp() * std_z
        nhij = self.mlp_gen(var)
        
        generated_mean = torch.mean(nhij, dim=0)
        generated_std = torch.std(nhij, dim=0)
        generated_cov = torch.bmm(generated_std.unsqueeze(dim=-1),
                                  generated_std.unsqueeze(dim=1))/ \
                                  self.sample_size
           
        tot_nodes = h1.shape[0]
        h_dim = h1.shape[1]
        
        single_eye = torch.eye(h_dim).to(self.device)
        single_eye = single_eye.unsqueeze(dim=0)
        batch_eye = single_eye.repeat(tot_nodes,1,1)
        
        target_cov = target_cov + batch_eye
        generated_cov = generated_cov + batch_eye

        det_target_cov = torch.linalg.det(target_cov) 
        det_generated_cov = torch.linalg.det(generated_cov) 
        trace_mat = torch.matmul(torch.inverse(generated_cov), target_cov)
             
        x = torch.bmm(torch.unsqueeze(generated_mean - target_mean,dim=1),
                      torch.inverse(generated_cov))
        y = torch.unsqueeze(generated_mean - target_mean,dim=-1)
        z = torch.bmm(x,y).squeeze()

        # the information needed for loss computation
        recon_info = [det_target_cov, det_generated_cov, h_dim, trace_mat, z]
    
        return recon_info   

    def mini_batch_neigh_recon(self, h1, h0, input_id,
                               neighbor_dict, id_mapping):
        """Computing the target neighbor distribution and 
        reconstructed neighbor distribution using mini_batch of the data
        and neighbor sampling.
        """
        gen_neighs, tar_neighs = [], []
        
        sampled_embeddings_list, mask_len_list = \
                                        self.sample_neighbors(input_id,
                                                              neighbor_dict,
                                                              id_mapping,
                                                              h0)
        for index, neighbor_embeddings in enumerate(sampled_embeddings_list):
            # Generating h^k_v, reparameterization trick
            # the center node embeddings start from first row
            # in the h1 embedding matrix
            mean = h1[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = h1[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m_minibatch.sample().to(self.device)
            var = mean + sigma.exp() * std_z
            nhij = self.mlp_gen(var)
            
            generated_neighbors = nhij
            sum_neighbor_norm = 0
            
            for _, generated_neighbor in enumerate(generated_neighbors):
                sum_neighbor_norm += \
                    torch.norm(generated_neighbor) / math.sqrt(self.out_dim)
            generated_neighbors = \
                torch.unsqueeze(generated_neighbors, dim=0).to(self.device)
            target_neighbors = \
                torch.unsqueeze(torch.FloatTensor(neighbor_embeddings), 
                                dim=0).to(self.device)
            
            gen_neighs.append(generated_neighbors)
            tar_neighs.append(target_neighbors)
        
        # the information needed for loss computation
        recon_info = [gen_neighs, tar_neighs, mask_len_list]

        return recon_info

    def forward(self,
                x,
                edge_index,
                input_id=None,
                neighbor_dict=None,
                id_mapping=None):
        if id_mapping is not None:
            self.id_mapping = id_mapping        
        # feature projection
        h0 = self.linear(x)

        # encode feature matrix
        # h1 = self.shared_encoder(h0, edge_index)
        h = self.shared_encoder(h0, edge_index)
        h_s, h_ns = h.chunk(2, dim=-1)

        # compute cf
        x_cf = flip_sensitive_attributes(x, 0)
        h0_cf = self.linear(x_cf)
        h_cf = self.shared_encoder(h0_cf, edge_index)
        h_s_cf, h_ns_cf = h_cf.chunk(2, dim=-1)

        
        if self.full_batch:
            center_h0 = h0
            # center_h1 = h1
            center_h_s, center_h_ns = h_s, h_ns
            center_h_s_cf = h_s_cf
        else: # mini-batch mode
            center_h0 = h0[[self.id_mapping[i] for i in input_id], :]
            center_h_s = h_s[[self.id_mapping[i] for i in input_id], :]
            center_h_ns = h_ns[[self.id_mapping[i] for i in input_id], :]
            center_h_s_cf = h_s_cf[[self.id_mapping[i] for i in input_id], :]

        # save embeddings
        self.emb = (center_h_s, center_h_ns)
    
        # decode node degree
        degree_logits = F.relu(self.degree_decoder(center_h_ns))

        # decode the node feature and neighbor distribution
        feat_recon_list_s, feat_recon_list_ns, feat_recon_list_s_cf = [], [], []
        neigh_recon_list = []
        # sample multiple times to remove noises
        for _ in range(self.sample_time):
            h0_prime_s = self.feature_decoder_s(center_h_s)
            h0_prime_ns = self.feature_decoder_ns(center_h_ns)
            h0_prime_s_cf = self.feature_decoder_s(center_h_s_cf)
            feat_recon_list_s.append(h0_prime_s)
            feat_recon_list_ns.append(h0_prime_ns)
            feat_recon_list_s_cf.append(h0_prime_s_cf)
            
            if self.full_batch: # full batch mode TODO?
                neigh_recon_info = self.full_batch_neigh_recon(h,
                                                               h0,
                                                               edge_index)
            else: # mini batch mode
                neigh_recon_info = self.mini_batch_neigh_recon(h,
                                                               h0,
                                                               input_id,
                                                               neighbor_dict,
                                                               id_mapping)
            neigh_recon_list.append(neigh_recon_info)

        return center_h0, degree_logits, feat_recon_list_s, feat_recon_list_ns, feat_recon_list_s_cf, neigh_recon_list

    def loss_func(self,
                  h0,
                  degree_logits,
                  feat_recon_list,
                  neigh_recon_list,
                  ground_truth_degree_matrix):
 

        
        batch_size = h0.shape[0]

        # degree reconstruction loss
        ground_truth_degree_matrix = \
            torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits,
                                            ground_truth_degree_matrix.float())
        degree_loss_per_node = \
            (degree_logits-ground_truth_degree_matrix).pow(2)
        
        h_loss = 0
        feature_loss = 0
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for t in range(self.sample_time):
            # feature reconstruction loss 
            h0_prime = feat_recon_list[t]
            feature_losses_per_node = (h0-h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)
            
            # neigbor distribution reconstruction loss
            if self.full_batch:
                # full batch neighbor reconstruction
                det_target_cov, det_generated_cov, h_dim, trace_mat, z = \
                                                        neigh_recon_list[t]
                KL_loss = 0.5 * (torch.log(det_target_cov / 
                                           det_generated_cov) - \
                        h_dim + trace_mat.diagonal(offset=0, dim1=-1, 
                                                    dim2=-2).sum(-1) + z)
                local_index_loss = torch.mean(KL_loss)
                local_index_loss_per_node = KL_loss
            else: # mini batch neighbor reconstruction
                local_index_loss = 0
                local_index_loss_per_node = []
                gen_neighs, tar_neighs, mask_lens = neigh_recon_list[t] 
                for generated_neighbors, target_neighbors, mask_len in \
                                        zip(gen_neighs, tar_neighs, mask_lens):
                    temp_loss = self.neighbor_loss(generated_neighbors,
                                                   target_neighbors,
                                                   mask_len,
                                                   self.device)
                    local_index_loss += temp_loss
                    local_index_loss_per_node.append(temp_loss)
                                        
                local_index_loss_per_node = \
                    torch.stack(local_index_loss_per_node)
            
            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)
            
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        
        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node, dim=0)
        
        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list),
                                           dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))
                
        h_loss_per_node = h_loss_per_node.reshape(batch_size, 1)
        degree_loss_per_node = degree_loss_per_node.reshape(batch_size, 1)
        feature_loss_per_node = feature_loss_per_node.reshape(batch_size, 1)
        
        loss = self.lambda_loss1 * h_loss \
            + degree_loss * self.lambda_loss3 \
            + self.lambda_loss2 * feature_loss
        
        loss_per_node = self.lambda_loss1 * h_loss_per_node \
            + degree_loss_per_node * self.lambda_loss3 \
                + self.lambda_loss2 * feature_loss_per_node
        
        return loss, loss_per_node, h_loss_per_node, \
            degree_loss_per_node, feature_loss_per_node

    @staticmethod
    def process_graph(data, input_id=None):
        
        # row normalize
        data.x = F.normalize(data.x, p=1, dim=1)
        # convert to undirected graph
        data.edge_index = to_undirected(data.edge_index)
        # add self loops
        new_edge_index, _= add_self_loops(data.edge_index)
        data.edge_index = new_edge_index 

        out_nodes = data.edge_index[0,:]
        in_nodes = data.edge_index[1,:]
        id_mapping = {}
        
        if input_id is None: # full batch of the data
            input_id = torch.unique(data.edge_index).tolist()
        else:  # reindexing the node id for mini-batch
            for edge_id, node_id in enumerate(data.n_id.tolist()):
                id_mapping[node_id] = edge_id
            in_nodes = [data.n_id[i] for i in in_nodes]
            out_nodes = [data.n_id[i] for i in out_nodes]

        neighbor_dict = {}
        for in_node, out_node in zip(in_nodes, out_nodes):
            if in_node.item() not in neighbor_dict:
                neighbor_dict[in_node.item()] = []
            neighbor_dict[in_node.item()].append(out_node.item())

        neighbor_num_list = []
        for i in input_id:
            neighbor_num_list.append(len(neighbor_dict.get(i, [])))

        neighbor_num_list = torch.tensor(neighbor_num_list)

        return data, neighbor_dict, neighbor_num_list, id_mapping

# -------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
# -------------------------------------------------
class CustomGADNR(DeepDetector, nn.Module):
    def __init__(self,
                 hid_dim=64,
                 num_layers=1,
                 deg_dec_layers=4,
                 fea_dec_layers=3,
                 backbone=GCN,
                 sample_size=2,
                 sample_time=3,
                 neigh_loss='KL',
                 lambda_loss1=1e-2,
                 lambda_loss2=1e-1,
                 lambda_loss3=8e-1,
                 real_loss=True,
                 lr=1e-2,
                 epoch=20,
                 dropout=0.,
                 weight_decay=3e-4,
                 act=F.relu,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 contamination=0.1,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 use_adv_loss=True,
                 use_ds_loss=True,
                 use_cf_loss=True,
                 rec_weight = 1.0,
                 cf_weight = 1.0,
                 ds_weight = 0.5,
                 adv_weight = 0.5,
                 **kwargs):

        super(CustomGADNR, self).__init__(hid_dim=hid_dim,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    backbone=backbone,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    save_emb=save_emb,
                                    compile_model=compile_model,
                                    **kwargs)
        nn.Module.__init__(self)

        self.encoder_layers = num_layers
        self.deg_dec_layers = deg_dec_layers
        self.fea_dec_layers = fea_dec_layers
        self.sample_size = sample_size
        self.sample_time = sample_time
        self.neigh_loss = neigh_loss
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3
        self.real_loss = real_loss
        self.neighbor_num_list = None
        self.neighbor_dict = None
        self.id_mapping = None
        self.full_batch = None
        self.tot_nodes = 0
        self.verbose = verbose

        self.hid_dim = hid_dim
        self.device = validate_device(gpu)
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.backbone = backbone
        self.kwargs = kwargs
        self.rec_weight = rec_weight
        self.cf_weight = cf_weight
        self.ds_weight = ds_weight
        self.use_adv_loss = use_adv_loss
        self.use_ds_loss = use_ds_loss
        self.use_cf_loss = use_cf_loss
        #add ad loss 
        self.discriminator = Discriminator(hid_dim//2, hid_dim, 1).to(self.device)
        self.adv_weight = adv_weight  # Weight for adversarial loss

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        self.discriminator = self.discriminator.to(device)
        return self

    # def process_graph(self, data):
    #     if self.batch_size != data.x.shape[0]: # mini-batch
    #         data, neighbor_dict, neighbor_num_list, id_mapping = \
    #                             GADNRBase.process_graph(data,
    #                                                     data.input_id.tolist())
    #     else: # full batch
    #         data, neighbor_dict, neighbor_num_list, id_mapping = \
    #                             GADNRBase.process_graph(data)
    #         self.tot_nodes = data.x.shape[0]

    #     self.neighbor_num_list = neighbor_num_list.to(self.device)
    #     self.neighbor_dict = neighbor_dict
    #     self.id_mapping = id_mapping

    #     return data
    def process_graph(self, data):
        if self.batch_size != data.x.shape[0]:  # mini-batch
            data, neighbor_dict, neighbor_num_list, id_mapping = \
                GADNRBase.process_graph(data, data.n_id.tolist())
        else:  # full batch
            data, neighbor_dict, neighbor_num_list, id_mapping = \
                GADNRBase.process_graph(data)
            self.tot_nodes = data.x.shape[0]

        self.neighbor_num_list = neighbor_num_list.to(self.device)
        self.neighbor_dict = neighbor_dict
        self.id_mapping = id_mapping

        return data

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
                         
        return GADNRBase(in_dim=self.in_dim, hid_dim=self.hid_dim,
                         encoder_layers=self.encoder_layers,
                         deg_dec_layers=self.deg_dec_layers,
                         fea_dec_layers=self.fea_dec_layers,
                         sample_size=self.sample_size,
                         sample_time=self.sample_time, 
                         neighbor_num_list=self.neighbor_num_list,
                         tot_nodes=self.tot_nodes,
                         neigh_loss=self.neigh_loss,
                         lambda_loss1=self.lambda_loss1,
                         lambda_loss2=self.lambda_loss2,
                         lambda_loss3=self.lambda_loss3,
                         full_batch=self.full_batch,
                         backbone=self.backbone,
                         device=self.device).to(self.device)

    def forward_model(self, data):
        data = data.to(self.device)
        if not self.full_batch: # mini-batch training
            h0, degree_logits, feat_recon_list_s, feat_recon_list_ns, neigh_recon_list = \
                                            self.model(data.x,
                                                       data.edge_index,
                                                       data.input_id.tolist(),
                                                       self.neighbor_dict,
                                                       self.id_mapping)
        else: # full batch training
            h0, degree_logits, feat_recon_list_s, feat_recon_list_ns, neigh_recon_list = \
                                                self.model(data.x,
                                                           data.edge_index)
        # Split the embeddings into sensitive and non-sensitive parts
        h0_s, h0_ns = h0.chunk(2, dim=-1)

        # Combine sensitive and non-sensitive reconstructions
        feat_recon_list = [torch.cat((s, ns), dim=-1) for s, ns in zip(feat_recon_list_s, feat_recon_list_ns)]
        # rec_loss
        loss, loss_per_node, h_loss, degree_loss, feature_loss = \
                                self.model.loss_func(h0,
                                                     degree_logits,
                                                     feat_recon_list,
                                                     neigh_recon_list,
                                                     self.neighbor_num_list)
        rec_loss = loss
        # Compute counterfactual fairness loss
        cf_loss = torch.tensor(0.0).to(self.device)
        if self.use_cf_loss:
            x_cf = flip_sensitive_attributes(data.x, 0)  # Assume sensitive attribute is at index 0
            _, _, feat_recon_list_s_cf, _,  _ = self.model(x_cf, data.edge_index)
            cf_loss = counterfactual_fairness_loss(feat_recon_list_s[0][:, :h0_s.size(1)],
                                                        feat_recon_list_s_cf[0][:, :h0_s.size(1)])
            
        # Compute disentanglement loss
        ds_loss = torch.tensor(0.0).to(self.device)
        if self.use_ds_loss:
            ds_loss = disentanglement_loss(h0_s, h0_ns)

        # Compute adversarial loss
        adv_loss = torch.tensor(0.0).to(self.device)
        if self.use_adv_loss:
            sensitive_attr = data.x[:, 0].float().unsqueeze(1).to(self.device)  # Assume sensitive attribute is at index 0
            sensitive_pred = self.discriminator(h0_ns)
            adv_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)

        # Combine all losses with mean operation
        total_loss = (self.rec_weight * rec_loss + 
                      self.cf_weight * cf_loss + 
                      self.ds_weight * ds_loss - 
                      self.adv_weight * adv_loss).mean()

        return total_loss, loss_per_node.cpu().detach(), h_loss.cpu().detach(), \
            degree_loss.cpu().detach(), feature_loss.cpu().detach()
    
    def comp_decision_score(self,
                            loss_per_node,
                            h_loss,
                            degree_loss,
                            feature_loss,
                            h_loss_weight,
                            degree_loss_weight,
                            feature_loss_weight):
        """Compute the decision score based on orginal loss or weighted loss.
        """
        if self.real_loss:
            # the orginal decision score from the loss
            comp_loss = loss_per_node
        else:
            # the weighted decision score
            h_loss_norm = h_loss / (torch.max(h_loss) - 
                                    torch.min(h_loss))
            degree_loss_norm = degree_loss / \
                (torch.max(degree_loss) - torch.min(degree_loss))
            feature_loss_norm = feature_loss / \
                (torch.max(feature_loss) - torch.min(feature_loss))
            comp_loss = h_loss_weight * h_loss_norm \
                + degree_loss_weight *  degree_loss_norm \
                    + feature_loss_weight * feature_loss_norm
        return comp_loss

    def train_step(self, batch, optimizer, discriminator_optimizer):
        if self.full_batch:
            h0, degree_logits, feat_recon_list_s, feat_recon_list_ns, feat_recon_list_s_cf, neigh_recon_list = self.model(batch.x, batch.edge_index)
        else:
            input_id = batch.n_id.tolist()
            h0, degree_logits, feat_recon_list_s, feat_recon_list_ns, feat_recon_list_s_cf, neigh_recon_list = self.model(
                batch.x, batch.edge_index, 
                input_id=input_id,
                neighbor_dict=self.neighbor_dict,
                id_mapping=self.id_mapping
            )
        
        h0_s, h0_ns = h0.chunk(2, dim=-1)

        feat_recon_list = [torch.cat((s, ns), dim=-1) for s, ns in zip(feat_recon_list_s, feat_recon_list_ns)]
        rec_loss, loss_per_node, h_loss, degree_loss, feature_loss = self.model.loss_func(
            h0, degree_logits, feat_recon_list, neigh_recon_list, self.neighbor_num_list
        )

        total_loss = self.rec_weight * rec_loss

        # Rest of your loss computations
        if self.use_cf_loss:
            cf_loss = counterfactual_fairness_loss(feat_recon_list_s[0][:, :h0_s.size(1)],
                                                feat_recon_list_s_cf[0][:, :h0_s.size(1)])
            total_loss += self.cf_weight * cf_loss

        if self.use_ds_loss:
            ds_loss = disentanglement_loss(h0_s, h0_ns)
            total_loss += self.ds_weight * ds_loss

        if self.use_adv_loss:
            sensitive_attr = batch.x[:, 0].float().unsqueeze(1).to(self.device)
            sensitive_pred = self.discriminator(h0_ns)
            adv_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)
            total_loss -= self.adv_weight * adv_loss


        # optimizer.zero_grad()
        # discriminator_optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
        # discriminator_optimizer.step()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if self.use_adv_loss:
            discriminator_optimizer.zero_grad()
            sensitive_pred = self.discriminator(h0_ns.detach())
            discriminator_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        return total_loss, loss_per_node.detach().cpu()
            
    def fit(self,
        data,
        label=None,
        h_loss_weight=1.0,
        degree_loss_weight=0.,
        feature_loss_weight=2.5,
        loss_step=20,
        *args, 
        **kwargs
        ):
        print("QWQ")
        print("batch_size: ", self.batch_size)
        print("epoch: ", self.epoch)
        self.num_nodes, self.in_dim = data.x.shape
        
        if self.batch_size == 0:  # full batch training
            self.batch_size = data.x.shape[0]
            data = self.process_graph(data)
            self.full_batch = True
        else:  # mini batch training
            loader = NeighborLoader(data,
                                    num_neighbors=self.num_neigh,
                                    batch_size=self.batch_size,
                                    shuffle=True)
            self.full_batch = False

        self.model = self.init_model(**self.kwargs)
        self.model = self.model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        if self.compile_model:
            self.model = compile(self.model)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.model.train()
        self.decision_score_ = torch.zeros(self.num_nodes)
        
        for epoch in range(1, self.epoch+1, 1):
            start_time = time.time()
            epoch_loss = 0
            
            if self.full_batch:
                total_loss, loss_per_node = self.train_step(data, optimizer, discriminator_optimizer)
                # print("loss_per_node: ", sum(loss_per_node))
                self.decision_score_[:data.x.size(0)] = loss_per_node.squeeze(1).detach().cpu()
                epoch_loss = total_loss.item() * self.batch_size
            else:
                for batch in loader:
                    batch = batch.to(self.device)
                    batch = self.process_graph(batch)
                    total_loss, loss_per_node = self.train_step(batch, optimizer, discriminator_optimizer)
                    batch_size = batch.batch_size
                    self.decision_score_[batch.n_id[:batch_size]] = loss_per_node[:batch_size].squeeze(1).detach().cpu()

                    epoch_loss += total_loss.item() * batch_size

            avg_loss = epoch_loss / self.num_nodes
            # print("self.decision_score_s: ", sum(self.decision_score_))
            
            logger(epoch=epoch,
                loss=avg_loss,
                score=self.decision_score_,
                target=label,
                time=time.time() - start_time,
                verbose=self.verbose,
                train=True)

        self._process_decision_score()
        return self