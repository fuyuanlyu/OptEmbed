import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *

class Basic(torch.nn.Module):
    def __init__(self, opt):
        super(Basic, self).__init__()
        self.device = torch.device("cuda:0" if opt.get('use_cuda') else "cpu")
        self.latent_dim = opt['latent_dim']
        self.field_num = len(opt['field_dim'])
        self.feature_num = sum(opt['field_dim'])
        self.field_dim = opt['field_dim']
        self.retrain = opt['retrain']
        self.method = opt['method']
        self.embedding = self.init_embedding()
        if not self.retrain:
            if self.method in ["optembed", "optembed-e"]:
                self.threshold = self.init_threshold()
                self.sigma = BinaryStep.apply
            if self.method in ["optembed", "optembed-d"]:
                self.potential_dim_masks = self.pre_potential_dim_mask()

    def init_embedding(self):
        e = nn.Parameter(torch.rand([self.feature_num, self.latent_dim]))
        torch.nn.init.xavier_uniform_(e)
        return e

    def init_threshold(self, t_init=0):
        t = nn.Parameter(torch.empty(self.field_num, 1))
        nn.init.constant_(t, -t_init)
        return t
    
    def pre_potential_dim_mask(self):
        masks = []
        for i in range(self.latent_dim):
            zeros = np.zeros(self.latent_dim - i - 1)
            ones = np.ones(i + 1)
            mask = torch.from_numpy(np.concatenate((ones, zeros), axis=0)).unsqueeze(0)
            masks.append(mask)
        total_masks = torch.cat(masks, dim=0).float().to(self.device)
        return total_masks

    # Loading static masks used for re-training
    def load_mask(self, dim_mask=None, embed_mask=None):
        # Dimension mask
        if dim_mask is None:
            dim_mask = torch.ones((self.field_num,), dtype=torch.int64, device=self.device)
            dim_mask = torch.mul(dim_mask, self.latent_dim) - 1
        total_masks = self.pre_potential_dim_mask()
        self.dim_mask = F.embedding(dim_mask, total_masks)
        self.dim_mask.requires_grad_(False)

        # Embedding mask
        if embed_mask is None:
            embed_mask = torch.ones((self.feature_num,), device=self.device)
        self.embed_mask = embed_mask
        self.embed_mask.requires_grad_(False)

    def get_random_dim_mask(self, batch_size, phase):
        if self.method in ['optembed', 'optembed-d'] and phase == "train":
            indexes = torch.randint(low=0, high=self.latent_dim, size=(batch_size, self.field_num)).to(self.device)
            return F.embedding(indexes, self.potential_dim_masks)
        elif self.method in ['optembed-e', 'none'] or phase == "test":
            return torch.ones((batch_size, self.field_num, self.latent_dim), device=self.device, requires_grad=False)

    def calc_embed_mask(self, xv=None):
        if self.method in ['optembed', 'optembed-e']:
            if xv == None:
                embedding_norm = torch.norm(self.embedding, 1, dim=1, keepdim=True)
                k = []
                for i, dim in enumerate(self.field_dim):
                    ti = self.threshold[i]
                    ki = ti.expand(dim)
                    k.append(ki)
                final_threshold = torch.cat(k, dim=0).unsqueeze(1)
                embed_mask = self.sigma(embedding_norm - final_threshold)
                return embed_mask.squeeze()
            else:
                xv_norm = torch.norm(xv, 1, dim=2).unsqueeze(2)
                return self.sigma(xv_norm - self.threshold)
        elif self.method in ['optembed-d', 'none']:
            if xv == None:
                return torch.ones(self.feature_num, device=self.device, requires_grad=False)
            else:
                return torch.ones(self.field_dim, device=self.device, requires_grad=False)

    def calc_sparsity(self):
        base = self.feature_num * self.latent_dim
        if self.retrain:
            embed_mask = self.embed_mask.cpu().numpy()
            dim_mask = self.dim_mask.cpu().numpy().sum(axis=1)
            base = self.feature_num * self.latent_dim
            offset = 0
            params = 0
            for i, num_i in enumerate(self.field_dim):
                f_i = np.sum(embed_mask[offset:offset+num_i])
                params += f_i * dim_mask[i]
                offset += num_i
            percentage = 1 - (params / base)
        else:
            params = torch.nonzero(self.calc_embed_mask()).size(0) * self.latent_dim
            percentage = 1 - (params / base)
        return percentage, int(params)

    def calc_input(self, x, phase):
        xv = F.embedding(x, self.embedding)
        if self.retrain:
            mask_e = F.embedding(x, self.embed_mask).unsqueeze(2)
            mask_d = self.dim_mask
        else:
            mask_e = self.calc_embed_mask(xv)
            mask_d = self.get_random_dim_mask(xv.shape[0], phase)
        xe = torch.mul(mask_d, torch.mul(mask_e, xv))
        return xe

class FM(Basic):
    def __init__(self, opt):
        super(FM, self).__init__(opt)
        self.linear = FeaturesLinear(opt['field_dim'])
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, phase="test"):
        linear_score = self.linear.forward(x)
        xe = self.calc_input(x, phase)
        fm_score = self.fm.forward(xe)
        score = linear_score + fm_score
        return score.squeeze(1)

class DeepFM(FM):
    def __init__(self, opt):
        super(DeepFM, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, phase='test'):
        linear_score = self.linear.forward(x)
        xe = self.calc_input(x, phase)
        fm_score = self.fm.forward(xe)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)

class FNN(Basic):
    def __init__(self, opt):
        super(FNN, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, phase='test'):
        xv = self.calc_input(x, phase)
        score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        return score.squeeze(1)

class IPNN(Basic):
    def __init__(self, opt):
        super(IPNN, self).__init__(opt)      
        self.embed_output_dim = self.field_num * self.latent_dim
        self.product_output_dim = int(self.field_num * (self.field_num - 1) / 2)
        self.dnn_input_dim = self.embed_output_dim + self.product_output_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.dnn_input_dim, self.mlp_dims, dropout=self.dropout)

        # Create indexes
        rows = []
        cols = []
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                rows.append(i)
                cols.append(j)
        self.rows = torch.tensor(rows, device=self.device)
        self.cols = torch.tensor(cols, device=self.device)

    def calc_product(self, xe):
        batch_size = xe.shape[0]
        trans = torch.transpose(xe, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding

    def forward(self, x, phase='test'):
        xv = self.calc_input(x, phase)
        product = self.calc_product(xv)
        xv = xv.view(-1, self.embed_output_dim)
        xe = torch.cat((xv, product), 1)
        score = self.mlp.forward(xe)
        return score.squeeze(1)

class DCN(Basic):
    def __init__(self, opt):
        super(DCN, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, output_layer=False)
        self.cross = CrossNetwork(self.embed_output_dim, opt['cross_layer_num'])
        self.combine = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x, phase='test'):
        xe = self.calc_input(x, phase)
        dnn_score = self.mlp.forward(xe.view(-1, self.embed_output_dim))
        cross_score = self.cross.forward(xe.view(-1, self.embed_output_dim))
        stacked = torch.cat((dnn_score, cross_score), 1)
        logit = self.combine(stacked)
        return logit.squeeze(1)
