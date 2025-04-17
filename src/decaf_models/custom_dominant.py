# -*- coding: utf-8 -*-
"""Deep Anomaly Detection on Attributed Networks (DOMINANT)"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
from torch_geometric.nn import GCN
import numpy as np

from pygod.detector.base import DeepDetector
from pygod.nn import DOMINANTBase

import time
import math
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss
from utils.loss_function import *
from utils.flip_sensitive_attributes import  *
from pygod.utils import validate_device, logger

class DOMINANTBase(nn.Module):
    """
    Deep Anomaly Detection on Attributed Networks

    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int
       Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
       Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
       Activation function if not None.
       Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs : optional
        Additional arguments for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(DOMINANTBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)
        #TODO define two different decoder for z_s and z_ns
        self.attr_decoder1 = backbone(in_channels=hid_dim//2,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)
        self.attr_decoder2 = backbone(in_channels=hid_dim//2,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.struct_decoder = DotProductDecoder(in_dim=hid_dim//2,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.loss_func = double_recon_loss
        self.z = None

    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        # encode feature matrix
        
        self.z = self.shared_encoder(x, edge_index)

        x_cf = flip_sensitive_attributes(x, 0) #aasume the sensitivity index is 0

        self.z_cf = self.shared_encoder(x_cf, edge_index)

        #TODO split z_s and z_ns, then return 
        z_s, z_ns = self.z[:, :self.z.size(1)//2], self.z[:, self.z.size(1)//2:]  # Split into two halves
        z_s_cf, z_ns_cf = self.z_cf[:, :self.z_cf.size(1)//2], self.z_cf[:, self.z_cf.size(1)//2:]  # Split into two halves

        x_s_hat = self.attr_decoder1(z_s, edge_index)
        x_ns_hat = self.attr_decoder2(z_ns, edge_index)
        x_s_cf_hat = self.attr_decoder1(z_s_cf, edge_index)
        x_ns_cf_hat = self.attr_decoder2(z_ns_cf, edge_index)

        # decode adjacency matrix 
        s_ = self.struct_decoder(z_ns, edge_index)

        return z_s, z_ns, x_s_hat, x_ns_hat,x_s_cf_hat, s_

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



class CustomDOMINANT(DeepDetector, nn.Module):
    """
    Deep Anomaly Detection on Attributed Networks

    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
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
    weight : float, optional
        Weight between reconstruction of node feature and structure.
        Default: ``0.5``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs : optional
        Additional arguments for the backbone.

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
    z : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``z`` is
        ``None``. When the detector has multiple embeddings,
        ``z`` is a tuple of torch.Tensor.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 contamination=0.1,
                 lr=1e-3,
                 epoch=50,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5,
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

        super(CustomDOMINANT, self).__init__(hid_dim=hid_dim,
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

        # model param
        self.in_dim = None
        self.num_nodes = None
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.weight = weight
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.backbone = backbone
        self.kwargs = kwargs

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.sigmoid_s = sigmoid_s
        self.rec_weight = rec_weight
        self.cf_weight = cf_weight
        self.ds_weight = ds_weight
        self.use_adv_loss = use_adv_loss
        self.use_ds_loss = use_ds_loss
        self.use_cf_loss = use_cf_loss
        #add ad loss 
        self.discriminator = Discriminator(hid_dim//2, hid_dim, 1).to(self.device)
        self.adv_weight = adv_weight  # Weight for adversarial loss

        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        self.discriminator = self.discriminator.to(device)
        return self
        
    def process_graph(self, data):
        DOMINANTBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.z = torch.zeros(self.num_nodes, self.hid_dim)
        return DOMINANTBase(in_dim=self.in_dim,
                            hid_dim=self.hid_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            act=self.act,
                            sigmoid_s=self.sigmoid_s,
                            backbone=self.backbone,
                            **kwargs).to(self.device)
    
    def _process_decision_score(self):
        """Internal function to calculate key attributes:
        - threshold_: used to decide the binary label
        - label_: binary labels of training data
        """

        self.threshold_ = np.percentile(self.decision_score_,
                                        100 * (1 - self.contamination))
        self.label_ = (self.decision_score_ > self.threshold_).long()

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id
        data = data.to(self.device)
        x = data.x
        s = data.s
        edge_index = data.edge_index

        #TODO
        # x_, s_ = self.model(x, edge_index)
        z_s, z_ns, x_s_hat, x_ns_hat,x_s_cf_hat, s_ = self.model(x, edge_index)
        x_hat = x_ns_hat + x_s_hat
        rec_loss = self.model.loss_func(x[:batch_size],
                                       x_hat[:batch_size],
                                       s[:batch_size, node_idx],
                                       s_[:batch_size],
                                       self.weight)
        
        cf_loss = counterfactual_fairness_loss(x_s_hat[:batch_size], 
                                           x_s_cf_hat[:batch_size],
                                           dist_mode='L2')
        
        ds_loss = disentanglement_loss(z_s[:batch_size],
                                       z_ns[:batch_size])
        
        # return center_h0, degree_logits, feat_recon_list_s, feat_recon_list_ns, neigh_recon_list
        # Adversarial prediction
        sensitive_pred = self.discriminator(z_ns[:batch_size])
        # Adversarial loss
        sensitive_attr = x[:batch_size, 0].float().unsqueeze(1).to(self.device)  # sensitive attribute index tobe 0
        adv_loss = nn.BCEWithLogitsLoss()(sensitive_pred, sensitive_attr)
        
        total_loss = (self.rec_weight * rec_loss + self.cf_weight * cf_loss + self.ds_weight * ds_loss + self.adv_weight * adv_loss).mean()
        sample_score = rec_loss
        # print("score: ", sample_score)

        # score = total_loss
        
        # Log individual losses if needed
        if self.verbose > 1:
            print(f"Rec Loss: {rec_loss.item():.4f}, DS Loss: {ds_loss.item():.4f}")

        return total_loss, sample_score.detach().cpu()
        # return total_loss, rec_loss.item(), cf_loss.item(), ds_loss.item(), adv_loss.item()   

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
                
                z_s, c, x_s_hat, x_ns_hat, x_s_cf_hat, s_ = self.model(batch.x, batch.edge_index)
                x_hat = x_ns_hat + x_s_hat
                score = self.model.loss_func(batch.x[:batch_size], 
                                                x_hat[:batch_size], 
                                                batch.s[:batch_size, node_idx], 
                                                s_[:batch_size], 
                                                self.weight)
                rec_loss = score.mean()
                self.decision_score_[node_idx[:batch_size]] = score.detach().cpu()
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

                total_loss.backward()
                optimizer.step()
                
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                # Train the discriminator
                if self.use_adv_loss:
                    discriminator_optimizer.zero_grad()
                    for param in self.model.parameters():
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
            # print("decision_score: ", sum(self.decision_score_))
            # print("loss_value: ", loss_value)
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
    