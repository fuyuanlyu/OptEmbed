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
        self.method = opt['method']
        self.embedding = nn.Parameter(torch.rand([self.feature_num, self.latent_dim]))
        if self.method in ['optembed']:
            self.threshold = nn.Parameter(torch.empty(self.field_num, 1))
            self.sigma = BinaryStep.apply
        self.potential_dim_masks = self.pre_potential_dim_mask()

    def pre_potential_dim_mask(self):
        masks = []
        for i in range(self.latent_dim):
            zeros = np.zeros(self.latent_dim - i - 1)
            ones = np.ones(i + 1)
            mask = torch.from_numpy(np.concatenate((ones, zeros), axis=0)).unsqueeze(0)
            masks.append(mask)
        total_masks = torch.cat(masks, dim=0).float().to(self.device)
        return total_masks

    def prepare_sparse_embedding(self):
        self.sparse_embedding = torch.mul(self.embedding, self.calc_embed_mask().unsqueeze(1))
        self.embed_mask = self.calc_embed_mask()

    def calc_embed_mask(self):
        if self.method in ['optembed']:
            embedding_norm = torch.norm(self.embedding, 1, dim=1, keepdim=True)
            k = []
            for i, dim in enumerate(self.field_dim):
                ti = self.threshold[i]
                ki = ti.expand(dim)
                k.append(ki)
            final_threshold = torch.cat(k, dim=0).unsqueeze(1)
            embed_mask = self.sigma(embedding_norm - final_threshold)
            return embed_mask.squeeze()
        elif self.method in ['optembed-d']:
            return torch.ones(self.feature_num, device=self.device, requires_grad=False)

    def calc_sparsity(self, cand=None):
        embed_mask = self.embed_mask.cpu().detach().numpy()
        base = self.feature_num * self.latent_dim
        if cand is None:
            params = np.sum(embed_mask) * self.latent_dim
        else:
            offset = 0
            params = 0
            for i, num_i in enumerate(self.field_dim):
                f_i = np.sum(embed_mask[offset:offset+num_i])
                params += f_i * cand[i]
                offset += num_i
        percentage = 1 - (params / base)
        return percentage, int(params)

    def calc_input(self, x, cand):
        xe = F.embedding(x, self.sparse_embedding)
        mask_e = F.embedding(cand, self.potential_dim_masks)
        xe = torch.mul(mask_e, xe)
        return xe

class FM(Basic):
    def __init__(self, opt):
        super(FM, self).__init__(opt)
        self.linear = FeaturesLinear(opt['field_dim'])  # linear part
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x, cand):
        linear_score = self.linear.forward(x)
        xv = self.calc_input(x, cand)
        fm_score = self.fm.forward(xv)
        score = linear_score + fm_score
        return score.squeeze(1)

class DeepFM(FM):
    def __init__(self, opt):
        super(DeepFM, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, cand):
        linear_score = self.linear.forward(x)
        xv = self.calc_input(x, cand)
        fm_score = self.fm.forward(xv)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)

class FNN(Basic):
    def __init__(self, opt):
        super(FNN, self).__init__(opt)
        self.embed_output_dim = self.field_num * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.dropout = opt['mlp_dropout']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout)

    def forward(self, x, cand):
        xv = self.calc_input(x, cand)
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

    def calculate_product(self, xe):
        batch_size = xe.shape[0]
        trans = torch.transpose(xe, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding

    def forward(self, x, cand):
        xv = self.calc_input(x, cand)
        product = self.calculate_product(xv)
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

    def forward(self, x, cand):
        xv = self.calc_input(x, cand)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        cross_score = self.cross.forward(xv.view(-1, self.embed_output_dim))
        stacked = torch.cat((dnn_score, cross_score), 1)
        logit = self.combine(stacked)
        return logit.squeeze(1)
