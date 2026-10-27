"""
Stub for BiFormer routing components — used only for compute overhead measurement.
QKVLinear, TopkRouting, KVGather are simplified versions that preserve parameter count.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        # x: (n, p^2, h, w, c)
        qkv = self.qkv(x)
        q, kv = qkv[..., :self.qk_dim], qkv[..., self.qk_dim:]
        return q, kv


class TopkRouting(nn.Module):
    def __init__(self, qk_dim, qk_scale=None, topk=4, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.emb = nn.Linear(qk_dim, qk_dim)

    def forward(self, q, k):
        # q, k: (n, p^2, c)
        q = self.emb(q)
        k = self.emb(k)
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (n, p^2, p^2)
        r_weight, r_idx = attn.topk(self.topk, dim=-1)  # (n, p^2, topk)
        r_weight = r_weight.softmax(dim=-1)
        return r_weight, r_idx


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        self.mul_weight = mul_weight

    def forward(self, r_idx, r_weight, kv):
        # kv: (n, p^2, w^2, c)  r_idx: (n, p^2, topk)
        n, p2, w2, c = kv.shape
        topk = r_idx.shape[-1]
        # gather topk kv windows for each query window
        idx = r_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, w2, c)
        kv_expanded = kv.unsqueeze(2).expand(-1, -1, topk, -1, -1)
        # simple gather: for overhead estimation correctness is not required
        kv_sel = torch.zeros(n, p2, topk, w2, c, device=kv.device, dtype=kv.dtype)
        for i in range(topk):
            idx_i = r_idx[:, :, i]  # (n, p2)
            for b in range(n):
                kv_sel[b, :, i, :, :] = kv[b, idx_i[b], :, :]
        if self.mul_weight == 'soft':
            kv_sel = kv_sel * r_weight.unsqueeze(-1).unsqueeze(-1)
        return kv_sel  # (n, p^2, topk, w^2, c)
