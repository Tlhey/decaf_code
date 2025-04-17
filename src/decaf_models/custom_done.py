# -*- coding: utf-8 -*-
"""Deep Outlier Aware Attributed Network Embedding (DONE)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import warnings

from pygod.detector.base import DeepDetector
import math
import torch
from torch import nn
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_adj

from pygod.nn.conv import NeighDiff
import time
import math
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.loss_function import *
from utils.flip_sensitive_attributes import  *
from pygod.utils import validate_device, logger

class DONEBase(nn.Module):
    """
    Deep Outlier Aware Attributed Network Embedding

    DONE consists of an attribute autoencoder and a structure
    autoencoder. It estimates five losses to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and a
    combination loss. It calculates three outlier scores, and averages
    them as an overall scores. This model is transductive only.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    x_dim : int
        Input dimension of attribute.
    s_dim : int
        Input dimension of structure.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    w1 : float, optional
        Weight of structure proximity loss. Default: ``0.2``.
    w2 : float, optional
        Weight of structure homophily loss. Default: ``0.2``.
    w3 : float, optional
        Weight of attribute proximity loss. Default: ``0.2``.
    w4 : float, optional
        Weight of attribute homophily loss. Default: ``0.2``.
    w5 : float, optional
        Weight of combination loss. Default: ``0.2``.
    **kwargs
        Other parameters for the backbone.
    """

    def __init__(self,
                 x_dim,
                 s_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 **kwargs):
        super(DONEBase, self).__init__()

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.attr_encoder = MLP(in_channels=x_dim,
                                hidden_channels=hid_dim,
                                out_channels=hid_dim,
                                num_layers=encoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)
        #TODO define two different decoder for z_s and z_ns
        self.attr_decoder1 = MLP(in_channels=hid_dim//2,
                                hidden_channels=hid_dim,
                                out_channels=x_dim,
                                num_layers=decoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)
        self.attr_decoder2 = MLP(in_channels=hid_dim//2,
                                hidden_channels=hid_dim,
                                out_channels=x_dim,
                                num_layers=decoder_layers,
                                dropout=dropout,
                                act=act,
                                **kwargs)

        self.struct_encoder = MLP(in_channels=s_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.struct_decoder = MLP(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  out_channels=s_dim,
                                  num_layers=decoder_layers,
                                  dropout=dropout,
                                  act=act,
                                  **kwargs)

        self.neigh_diff = NeighDiff()
        self.emb = None
        self.ns_projector = nn.Linear(hid_dim//2, hid_dim)  # 新增一个线性层来调整z_ns的维度

    def forward(self, x, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.
        """
        # h_a = self.attr_encoder(x)
        # x_ = self.attr_decoder(h_a)
        # dna = self.neigh_diff(h_a, edge_index).squeeze()
        # h_s = self.struct_encoder(s)
        # s_ = self.struct_decoder(h_s)
        # dns = self.neigh_diff(h_s, edge_index).squeeze()
        # self.emb = (h_a, h_s)

        # return x_, s_, h_a, h_s, dna, dns
        z = self.attr_encoder(x)
        x_cf = flip_sensitive_attributes(x, 0) #asume the sensitivity index is 0
        z_cf = self.attr_encoder(x_cf)

        z_s, z_ns = z[:, :z.size(1)//2], z[:, z.size(1)//2:]  # Split into two halves
        z_s_cf, z_ns_cf = z_cf[:, :z_cf.size(1)//2], z_cf[:, z_cf.size(1)//2:]  # Split into two halves

        x_s_hat = self.attr_decoder1(z_s)
        x_ns_hat = self.attr_decoder2(z_ns)
        x_s_cf_hat = self.attr_decoder1(z_s_cf)
        x_ns_cf_hat = self.attr_decoder2(z_ns_cf)

        z_ns_full = self.ns_projector(z_ns)  # 调整z_ns的维度
        dna = self.neigh_diff(z_ns_full, edge_index).squeeze()

        # dna = self.neigh_diff(z_ns, edge_index).squeeze()

        s_h = self.struct_encoder(s)
        s_ = self.struct_decoder(s_h)
        dns = self.neigh_diff(s_h, edge_index).squeeze()
        self.emb = (z_ns_full, s_h)

        return z_s, z_ns, x_s_hat, x_ns_hat,x_s_cf_hat, z_ns_full, s_h, s_, dna, dns
        



    # def loss_func(self, x, x_, s, s_, h_a, h_s, dna, dns):
    def loss_func(self, x, x_ns_hat, s, s_, z_ns_full, s_h, dna, dns):
        """
        Loss function for DONE.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s : torch.Tensor
            Input structure embeddings.
        s_ : torch.Tensor
            Reconstructed structure embeddings.
        h_a : torch.Tensor
            Attribute hidden embeddings.
        h_s : torch.Tensor
            Structure hidden embeddings.
        dna : torch.Tensor
            Attribute neighbor distance.
        dns : torch.Tensor
            Structure neighbor distance.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        oa : torch.Tensor
            Attribute outlier scores.
        os : torch.Tensor
            Structure outlier scores.
        oc : torch.Tensor
            Combined outlier scores.
        """
        # equation 9 is based on the official implementation, and it
        # is slightly different from the paper
        dx = torch.sum(torch.pow(x - x_ns_hat, 2), 1)
        tmp = self.w3 * dx + self.w4 * dna
        oa = tmp / torch.sum(tmp)

        # equation 8 is based on the official implementation, and it
        # is slightly different from the paper
        ds = torch.sum(torch.pow(s - s_, 2), 1)
        tmp = self.w1 * ds + self.w2 * dns
        os = tmp / torch.sum(tmp)

        # equation 10
        dc = torch.sum(torch.pow(z_ns_full - s_h, 2), 1)
        oc = dc / torch.sum(dc)

        # equation 4
        loss_prox_a = torch.mean(torch.log(torch.pow(oa, -1)) * dx)

        # equation 5
        loss_hom_a = torch.mean(torch.log(torch.pow(oa, -1)) * dna)

        # equation 2
        loss_prox_s = torch.mean(torch.log(torch.pow(os, -1)) * ds)

        # equation 3
        loss_hom_s = torch.mean(torch.log(torch.pow(os, -1)) * dns)

        # equation 6
        loss_c = torch.mean(torch.log(torch.pow(oc, -1)) * dc)

        # equation 7
        loss = self.w3 * loss_prox_a + \
               self.w4 * loss_hom_a + \
               self.w1 * loss_prox_s + \
               self.w2 * loss_hom_s + \
               self.w5 * loss_c

        return loss, oa, os, oc

    @staticmethod
    def process_graph(data):
        """
        Obtain the dense adjacency matrix of the graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph.
        """
        data.s = to_dense_adj(data.edge_index)[0]
# --------------------
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
    
#------------------------------

class CustomDONE(DeepDetector, nn.Module):
    """
    Deep Outlier Aware Attributed Network Embedding

    DONE consists of an attribute autoencoder and a structure
    autoencoder. It estimates five losses to optimize the model,
    including an attribute proximity loss, an attribute homophily loss,
    a structure proximity loss, a structure homophily loss, and a
    combination loss. It calculates three outlier scores, and averages
    them as an overall scores.

    .. note::
        This detector is transductive only. Using ``predict`` with
        unseen data will train the detector from scratch.

    See :cite:`bandyopadhyay2020outlier` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. A half (floor) of the layers
        are for the encoder, the other half (ceil) of the layers are for
        decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of DONE is fixed to be MLP. Changing of this
        parameter will not affect the model. Default: ``None``.
    w1 : float, optional
        Weight of structure proximity loss. Default: ``0.2``.
    w2 : float, optional
        Weight of structure homophily loss. Default: ``0.2``.
    w3 : float, optional
        Weight of attribute proximity loss. Default: ``0.2``.
    w4 : float, optional
        Weight of attribute homophily loss. Default: ``0.2``.
    w5 : float, optional
        Weight of combination loss. Default: ``0.2``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs
        Other parameters for the backbone model.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    attribute_score_ : torch.Tensor
        Attribute outlier score.
    structural_score_ : torch.Tensor
        Structural outlier score.
    combined_score_ : torch.Tensor
        Combined outlier score.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=None,
                 w1=0.2,
                 w2=0.2,
                 w3=0.2,
                 w4=0.2,
                 w5=0.2,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
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

        if backbone is not None:
            warnings.warn("Backbone is not used in DONE.")

        super(CustomDONE, self).__init__(hid_dim=hid_dim,
                                   num_layers=1,
                                   dropout=dropout,
                                   weight_decay=weight_decay,
                                   act=act,
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

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.num_layers = num_layers

        self.attribute_score_ = None
        self.structural_score_ = None
        self.combined_score_ = None

        #add
        self.device = validate_device(gpu)
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

    def process_graph(self, data):
        DONEBase.process_graph(data)

    def init_model(self, **kwargs):
        self.attribute_score_ = torch.zeros(self.num_nodes)
        self.structural_score_ = torch.zeros(self.num_nodes)
        self.combined_score_ = torch.zeros(self.num_nodes)

        if self.save_emb:
            self.emb = (torch.zeros(self.num_nodes, self.hid_dim),
                        torch.zeros(self.num_nodes, self.hid_dim))

        return DONEBase(x_dim=self.in_dim,
                        s_dim=self.num_nodes,
                        hid_dim=self.hid_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        act=self.act,
                        w1=self.w1,
                        w2=self.w2,
                        w3=self.w3,
                        w4=self.w4,
                        w5=self.w5,
                        **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)

        # x_, s_, h_a, h_s, dna, dns = self.model(x, s, edge_index)
        z_s, z_ns, x_s_hat, x_ns_hat,x_s_cf_hat, z_ns_full, s_h, s_, dna, dns = self.model(x, s, edge_index)
        # def loss_func(self, x, x_ns_hat, s, s_, z_ns, s_h, dna, dns):
        x_hat = x_ns_hat + x_s_hat
        loss, oa, os, oc = self.model.loss_func(x[:batch_size],
                                                x_hat[:batch_size],
                                                s[:batch_size],
                                                s_[:batch_size],
                                                z_ns_full[:batch_size],
                                                s_h[:batch_size],
                                                dna[:batch_size],
                                                dns[:batch_size])

        self.attribute_score_[node_idx[:batch_size]] = oa.detach().cpu()
        self.structural_score_[node_idx[:batch_size]] = os.detach().cpu()
        self.combined_score_[node_idx[:batch_size]] = oc.detach().cpu()

        rec_loss = loss

        cf_loss = counterfactual_fairness_loss(x_s_hat[:batch_size], 
                                           x_s_cf_hat[:batch_size],
                                           dist_mode='L2')
        ds_loss = disentanglement_loss(z_s[:batch_size],
                                       z_ns[:batch_size])
        # Adversarial prediction
        sensitive_pred = self.discriminator(z_ns[:batch_size])
        # Adversarial loss
        sensitive_attr = x[:batch_size, 0].float().unsqueeze(1).to(self.device)  # sensitive attribute index tobe 0
        adv_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)

        total_loss = (self.cf_weight * cf_loss + self.ds_weight * ds_loss + self.adv_weight * adv_loss).mean() + self.rec_weight * rec_loss

        return total_loss, ((oa + os + oc) / 3).detach().cpu()

    def fit(self, data, label=None, *args, **kwargs):
            # print("batch_size:", self.batch_size)
            loss_history = {
                'epoch': [],
                'rec_loss': [],
                'cf_loss': [],
                'ds_loss': [],
                'adv_loss': [],
                'disc_loss': [],
                'total_loss': []
            }
            self.process_graph(data)
            self.num_nodes, self.in_dim = data.x.shape
            if self.batch_size == 0:
                self.batch_size = data.x.shape[0]

            loader = NeighborLoader(data,
                                    self.num_neigh,
                                    batch_size=self.batch_size)
            self.model = self.init_model(**self.kwargs) 

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

            self.model.train()
            self.decision_score_ = torch.zeros(data.x.shape[0])
            for epoch in range(self.epoch):
                start_time = time.time()
                epoch_loss = 0
                for batch in loader:
                    batch_size = batch.batch_size
                    node_idx = batch.n_id

                    optimizer.zero_grad()
                    for param in self.discriminator.parameters():
                        param.requires_grad = False
                    
                    # z_s, z_ns, x_s_hat, x_ns_hat, x_cf_hat, s_ = self.model(batch.x, batch.edge_index)
                    z_s, z_ns, x_s_hat, x_ns_hat,x_s_cf_hat, z_ns_full, s_h, s_, dna, dns = self.model(batch.x, batch.s, batch.edge_index)
                    x_hat = x_ns_hat + x_s_hat
                    loss, oa, os, oc = self.model.loss_func(batch.x[:batch_size],
                                                x_hat[:batch_size],
                                                batch.s[:batch_size],
                                                s_[:batch_size],
                                                z_ns_full[:batch_size],
                                                s_h[:batch_size],
                                                dna[:batch_size],
                                                dns[:batch_size])

                    rec_loss = loss
                    self.decision_score_[node_idx[:batch_size]] =  ((oa + os + oc) / 3).detach().cpu()
                    # print(self.decision_score_[:10])

                    total_loss = self.rec_weight * rec_loss

                    if self.use_cf_loss:
                        cf_loss = counterfactual_fairness_loss(x_s_hat[:batch_size], x_s_cf_hat[:batch_size], dist_mode='L2').mean()
                        total_loss += self.cf_weight * cf_loss

                    if self.use_ds_loss:
                        ds_loss = disentanglement_loss(z_s[:batch_size], z_ns[:batch_size]).mean()
                        total_loss += self.ds_weight * ds_loss

                    if self.use_adv_loss:
                        sensitive_attr = batch.x[:batch_size, 0].float().unsqueeze(1).to(self.device)
                        sensitive_pred = self.discriminator(z_ns[:batch_size])
                        adv_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)
                        total_loss -= self.adv_weight * adv_loss

                    # for name, param in self.model.named_parameters():
                    #     if not param.requires_grad:
                    #         print(f"Parameter {name} does not require gradients")
                    total_loss.backward()
                    optimizer.step()
                    
                    for param in self.discriminator.parameters():
                        param.requires_grad = True

                    # Train the discriminator
                    if self.use_adv_loss:
                        discriminator_optimizer.zero_grad()
                        # for param in self.model.parameters():
                        #     param.requires_grad = False
                        for param in list(self.model.parameters()):
                            param.requires_grad = False
                        sensitive_pred = self.discriminator(z_ns[:batch_size].detach())
                        discriminator_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)
                        discriminator_loss.backward()
                        discriminator_optimizer.step()
                        for param in self.model.parameters():
                            param.requires_grad = True

                    epoch_loss += total_loss.item() * len(node_idx)

                    # Update loss history every 20 epochs or at the start/end of training
                    if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == self.epoch - 1:
                        loss_history['epoch'].append(epoch + 1)
                        loss_history['rec_loss'].append(rec_loss.item())
                        loss_history['cf_loss'].append(cf_loss.item() if self.use_cf_loss else 0)
                        loss_history['ds_loss'].append(ds_loss.item() if self.use_ds_loss else 0)
                        loss_history['adv_loss'].append(adv_loss.item() if self.use_adv_loss else 0)
                        loss_history['disc_loss'].append(discriminator_loss.item() if self.use_adv_loss else 0)
                        loss_history['total_loss'].append(total_loss.item())

                loss_value = epoch_loss / data.x.shape[0]
                logger(epoch=epoch,
                    loss=loss_value,
                    score=self.decision_score_,
                    target=label,
                    time=time.time() - start_time,
                    verbose=self.verbose,
                    train=True)
                
                # Log individual losses if needed
                if self.verbose > 1:
                    print(f"Epoch {epoch + 1}/{self.epoch}")
                    print(f"Rec Loss: {rec_loss.item():.4f}, CF Loss: {cf_loss.item() if self.use_cf_loss else 0:.4f}, DS Loss: {ds_loss.item() if self.use_ds_loss else 0:.4f}, Adv Loss: {adv_loss.item() if self.use_adv_loss else 0:.4f}, Total Loss: {total_loss.item():.4f}")

            self._process_decision_score()
            return loss_history


    # def decision_function(self, data, label=None):
    #     if data is not None:
    #         warnings.warn("This detector is transductive only. "
    #                     "Training from scratch with the input data.")
    #         self.fit(data, label)
    #     return self.decision_score_
