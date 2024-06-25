import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm

from .positional_encoding import PositionalEncoding


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.dk = hidden_dim // num_heads
        self.Wq = nn.Parameter(torch.empty((hidden_dim, hidden_dim)), requires_grad=True)
        self.Wk = nn.Parameter(torch.empty((hidden_dim, hidden_dim)), requires_grad=True)
        self.Wv = nn.Parameter(torch.empty((hidden_dim, hidden_dim)), requires_grad=True)
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x):
        # x: bsz, Vs, d
        bsz, V, d = x.size()
        # device = x.device
        q = torch.matmul(x, self.Wq).view(bsz, V, self.num_heads, self.dk)
        q = q / np.sqrt(self.dk)
        k = torch.matmul(x, self.Wk).view(bsz, V, self.num_heads, self.dk)
        k = k / np.sqrt(self.dk)
        v = torch.matmul(x, self.Wv)
        # v = v.view(bsz, V, self.num_heads, self.dk)
        A = torch.einsum('bthd,blhd->bhtl', q, k) # bsz, h, V, V
        # A = F.softmax(A, dim=-1)
        # A = F.dropout(A, self.dropout)
        # x = torch.einsum('bhvu,bvhd->bhvd', A, v)
        # x = self.Wo(x.reshape((bsz, T, d)))
        return A, v


class VariableGT(nn.Module):
    def __init__(self, temp_dim, static_dim, max_len, hidden_dim, num_heads, mha_dropout) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.temp_dim = temp_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.temp_encoder = nn.Conv1d(temp_dim, temp_dim*hidden_dim, 1, groups=temp_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        self.static = static_dim > 0
        if self.static:
            self.emb = nn.Linear(static_dim, temp_dim*hidden_dim)
        
        self.mha_dropout = nn.Dropout(mha_dropout)
        self.mha = MultiHeadAttention(hidden_dim, num_heads, mha_dropout)
        self.mlp = nn.Sequential(
            nn.Conv1d(temp_dim*hidden_dim, hidden_dim*4, 1),
            nn.BatchNorm1d(hidden_dim*4),
            nn.GELU(),
            nn.Conv1d(hidden_dim*4, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, Xtemp: Tensor, Xtimes: Tensor, Xstatic: Tensor, mask: Tensor):
        Xtemp = Xtemp.permute((1, 2, 0))
        mask = mask.permute((1, 2, 0))
        # Xtemp: bcs, V, seq_len
        # mask:  bcs, V, seq_len
        seq_len = Xtemp.shape[-1]
        Htemp = self.temp_encoder(Xtemp)
        # Htemp: bcs, V*D, seq_len
        Htemp = Htemp.unflatten(1, (self.temp_dim, self.hidden_dim))
        Htemp = Htemp.permute((0, 2, 1, 3))

        Hpos = self.pos_encoder(Xtimes)
        # mask:  seq_len, bcs, D
        Hpos = Hpos.permute((1, 2, 0)).unsqueeze(2)

        if Xstatic is not None:
            emb = self.emb(Xstatic).view((-1, self.temp_dim, self.hidden_dim))
            emb = emb.permute((0, 2, 1)).unsqueeze(-1)
            H = Htemp + Hpos + emb
        else:
            H = Htemp + Hpos

        mask = mask.unsqueeze(1)
        h1 = H * mask
        h2 = h1.sum(-1)
        len = mask.sum(-1) + 1
        h3 = h2 / len
        h4 = h3.permute((0, 2, 1))
        # h3: bcs, V, hidden_dim
        A, mean_v = self.mha(h4)
        # v: bcs, V, h, d
        A2 = A.unsqueeze(2)
        A3 = torch.repeat_interleave(A2, seq_len, 2)
        # bsz, h, seq_len, V, V
        mask2 = mask.permute((0, 1, 3, 2))
        # mask3 = mask2.unsqueeze(-1) # bcs, 1, seq_len, V, 1
        mask3 = torch.ones_like(mask2).unsqueeze(-1) # bcs, 1, seq_len, V, 1
        mask4 = mask2.unsqueeze(3)  # bcs, 1, seq_len, 1, V
        mask5 = mask3 * mask4       # bcs, 1, seq_len, V, V
        A4 = A3 - 1e6 * (1 - mask5)
        S = F.softmax(A4, dim=-1)
        S = self.mha_dropout(S)

        H2 = H.unflatten(1, (self.num_heads, self.hidden_dim // self.num_heads))
        v2 = torch.einsum('bhlvu,bhdul->bhdvl', S, H2)
        v3 = v2.flatten(1, 2)
        v4 = v3 + mean_v.permute((0, 2, 1)).unsqueeze(3)
        v5 = v4.flatten(1, 2)
        # v5: bcs, hidden_dim * V, seq_len
        
        z = self.mlp(v5)
        return z


class TemporalGT(nn.Module):
    def __init__(self, hidden_dim, nhead, nlayers, dropout):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, x, lengths):
        # mask out the all-zero rows
        maxlen = x.shape[0]
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(x.device)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        # lengths = lengths.permute(1, 0).unsqueeze(2)
        output = torch.sum(output * (1 - mask2), dim=0) / (lengths + 1)
        return output


class SVMPT_Model(nn.Module):
    """ Transformer model with context embedding, aggregation
    Inputs:
        temp_dim: dimension of temporal features
        pe_dim: dimension of positional encoding
        static_dim: dimension of positional encoding
        nhead = number of heads in multihead-attention
        hidden_dim: dimension of feedforward network model
        dropout = dropout rate (default 0.1)
        max_len = maximum sequence length
        n_classes = number of classes
    """
    def __init__(self, temp_dim, static_dim, hidden_dim, nhead, nlayers, dropout, max_len, aggreg, n_classes):
        super(SVMPT_Model, self).__init__()
        self.model_type = 'SVMPT_Model'

        self.variable_gt = VariableGT(temp_dim, static_dim, max_len, hidden_dim, nhead, dropout)
        self.temporal_gt = TemporalGT(hidden_dim, nhead, nlayers, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )
        self.aggreg = aggreg
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Xtemp, Xtimes, Xstatic, lengths):
        maxlen, _, _ = Xtemp.shape
        vmask = Xtemp[:, :, int(Xtemp.shape[2] / 2):]
        Xtemp = Xtemp[:, :, :int(Xtemp.shape[2] / 2)]

        # mask out the all-zero rows
        mask = torch.arange(maxlen)[None, :] >= (lengths.cpu()[:, None])
        mask = mask.squeeze(1).to(Xtemp.device)
        mask2 = mask.permute(1, 0).unsqueeze(2).long()
        
        x = self.variable_gt(Xtemp, Xtimes, Xstatic, vmask)
        x = x.permute((2, 0, 1))
        output = self.temporal_gt(x, lengths)
        
        if self.aggreg == 'mean':
            output = torch.sum(output * (1 - mask2), dim=0) / (lengths + 1)
        elif self.aggreg == 'max':
            output, _ = torch.max(output * ((mask2 == 0) * 1.0 + (mask2 == 1) * -10.0), dim=0)

        output = self.dropout(output)
        output = self.mlp(output)
        return output
