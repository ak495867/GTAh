import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_kv_pyramid_torch(K_bh: torch.Tensor, V_bh: torch.Tensor):
    k_pyr = [K_bh]
    v_pyr = [V_bh]
    ck = K_bh.transpose(1, 2)
    cv = V_bh.transpose(1, 2)
    while ck.shape[2] > 1:
        if ck.shape[2] % 2 != 0:
            ck = F.pad(ck, (0, 1), mode="replicate")
            cv = F.pad(cv, (0, 1), mode="replicate")
        ck = F.avg_pool1d(ck, kernel_size=2, stride=2)
        cv = F.avg_pool1d(cv, kernel_size=2, stride=2)
        k_pyr.append(ck.transpose(1, 2))
        v_pyr.append(cv.transpose(1, 2))
    return k_pyr, v_pyr


def _gather_bh(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    BH, _, dh = src.shape
    _, M       = idx.shape
    idx_exp    = idx[None, :, :].expand(BH, -1, M)
    flat_idx   = idx_exp.reshape(BH, -1)
    flat_exp   = flat_idx.unsqueeze(-1).expand(BH, -1, dh)
    return torch.gather(src, 1, flat_exp).view(BH, idx.shape[0], M, dh)


def _boundary_idx_torch(N: int, W: int, stride: int, n_clus: int, fan_out: int, device: torch.device):
    half_f      = fan_out // 2
    i_arr       = torch.arange(N, device=device)
    left_bound  = (torch.clamp(i_arr - W, min=0) - 1).div(stride, rounding_mode="floor")
    right_bound = torch.clamp(i_arr + W + 1, max=N).div(stride, rounding_mode="floor")

    left_offs   = left_bound[:, None]  - torch.arange(half_f, device=device)[None, :]
    right_offs  = right_bound[:, None] + torch.arange(half_f, device=device)[None, :]
    clus_idx    = torch.cat([left_offs, right_offs], dim=1)

    valid    = (clus_idx >= 0) & (clus_idx < n_clus)
    clus_idx = clus_idx.clamp(0, n_clus - 1)
    return clus_idx, valid


class GabrielAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        window_size: int = 64,
        fan_out: int = 4,
        q_decay: float = 0.5,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.d_head      = d_model // num_heads
        self.window_size = window_size
        self.fan_out     = fan_out
        self.q_decay     = q_decay
        self.epsilon     = epsilon

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _cutoff(self, max_lvl: int) -> int:
        if 0 < self.q_decay < 1:
            return min(int(math.ceil(math.log(self.epsilon) / math.log(self.q_decay))), max_lvl)
        return max_lvl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D  = x.shape
        dh       = self.d_head
        H        = self.num_heads
        BH       = B * H
        W        = self.window_size
        device   = x.device
        scale    = dh ** -0.5

        Q = self.q_proj(x).view(B, N, H, dh).transpose(1, 2).reshape(BH, N, dh)
        K = self.k_proj(x).view(B, N, H, dh).transpose(1, 2).reshape(BH, N, dh)
        V = self.v_proj(x).view(B, N, H, dh).transpose(1, 2).reshape(BH, N, dh)

        k_pyr, v_pyr = _build_kv_pyramid_torch(K, V)
        cutoff        = self._cutoff(len(k_pyr) - 1)

        i_arr       = torch.arange(N, device=device)
        offsets     = torch.arange(-W, W + 1, device=device)
        local_idx   = (i_arr[:, None] + offsets[None, :]).clamp(0, N - 1)
        local_valid = (i_arr[:, None] + offsets[None, :] >= 0) & (i_arr[:, None] + offsets[None, :] < N)

        K_local  = _gather_bh(K, local_idx)
        V_local  = _gather_bh(V, local_idx)

        local_s  = torch.einsum("bnd,bnwd->bnw", Q, K_local) * scale
        local_s  = local_s.masked_fill(~local_valid[None, :, :], -1e4)

        max_s    = local_s.max(dim=-1).values
        exp_l    = torch.exp(local_s - max_s.unsqueeze(-1))
        exp_l    = exp_l.masked_fill(~local_valid[None, :, :], 0.0)
        sum_exp  = exp_l.sum(dim=-1)
        out_num  = torch.einsum("bnw,bnwd->bnd", exp_l, V_local)

        for lvl in range(1, cutoff + 1):
            lvl_k  = k_pyr[lvl]
            lvl_v  = v_pyr[lvl]
            n_clus = lvl_k.shape[1]
            stride = 1 << lvl
            env    = self.q_decay ** lvl
            log_env = math.log(max(env, 1e-30))

            clus_idx, valid = _boundary_idx_torch(N, W, stride, n_clus, self.fan_out, device)

            K_lvl = _gather_bh(lvl_k, clus_idx)
            V_lvl = _gather_bh(lvl_v, clus_idx)

            lvl_s = torch.einsum("bnd,bnfd->bnf", Q, K_lvl) * scale + log_env
            lvl_s = lvl_s.masked_fill(~valid[None, :, :], -1e4)

            lvl_max  = lvl_s.max(dim=-1).values
            new_max  = torch.maximum(max_s, lvl_max)
            rescale  = torch.exp(max_s - new_max)

            exp_lvl  = torch.exp(lvl_s - new_max.unsqueeze(-1))
            exp_lvl  = exp_lvl.masked_fill(~valid[None, :, :], 0.0)

            sum_exp  = sum_exp * rescale + exp_lvl.sum(dim=-1)
            out_num  = out_num * rescale.unsqueeze(-1) + torch.einsum("bnf,bnfd->bnd", exp_lvl, V_lvl)
            max_s    = new_max

        out = out_num / sum_exp.unsqueeze(-1).clamp(min=1e-30)
        out = out.view(B, H, N, dh).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

    def count_active_pairs(self, N: int) -> int:
        W      = self.window_size
        cutoff = self._cutoff(int(math.ceil(math.log2(max(N, 2)))))
        i_arr  = np.arange(N)
        total  = 0

        local_idx = np.clip(i_arr[:, None] + np.arange(-W, W + 1)[None, :], 0, N - 1)
        local_valid = (
            (i_arr[:, None] + np.arange(-W, W + 1)[None, :] >= 0) &
            (i_arr[:, None] + np.arange(-W, W + 1)[None, :] < N)
        )
        total += int(local_valid.sum())

        for lvl in range(1, cutoff + 1):
            stride  = 1 << lvl
            n_clus  = (N + stride - 1) // stride
            half_f  = self.fan_out // 2
            left_b  = (np.maximum(0, i_arr - W).astype(int) - 1) // stride
            right_b = np.minimum(N, i_arr + W + 1) // stride
            left_o  = left_b[:, None]  - np.arange(half_f)[None, :]
            right_o = right_b[:, None] + np.arange(half_f)[None, :]
            cidx    = np.concatenate([left_o, right_o], axis=1)
            total  += int(((cidx >= 0) & (cidx < n_clus)).sum())

        return total


class DenseAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        probs  = F.softmax(scores, dim=-1)

        out = (probs @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
