import numpy as np
from typing import Tuple, List
from gtah.core import build_pyramid


def _cutoff_level(q_decay: float, epsilon: float, max_lvl: int) -> int:
    if 0 < q_decay < 1:
        return min(int(np.ceil(np.log(epsilon) / np.log(q_decay))), max_lvl)
    return max_lvl


def _boundary_clusters(i_arr: np.ndarray, window_size: int, stride: int, n_clus: int, fan_out: int) -> Tuple[np.ndarray, np.ndarray]:
    half_f      = fan_out // 2
    left_bound  = (np.maximum(0, i_arr - window_size).astype(int) - 1) // stride
    right_bound = np.minimum(len(i_arr), i_arr + window_size + 1) // stride

    left_offs   = left_bound[:, None]  - np.arange(half_f, dtype=int)[None, :]
    right_offs  = right_bound[:, None] + np.arange(half_f, dtype=int)[None, :]
    clus_idx    = np.concatenate([left_offs, right_offs], axis=1)

    valid = (clus_idx >= 0) & (clus_idx < n_clus)
    clus_idx = np.clip(clus_idx, 0, n_clus - 1)
    return clus_idx, valid


def gabriel_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    window_size: int = 64,
    fan_out: int = 4,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
) -> Tuple[np.ndarray, dict]:
    N, D   = Q.shape
    scale  = 1.0 / np.sqrt(D)
    W      = window_size
    i_arr  = np.arange(N)

    k_pyramid, v_pyramid = build_pyramid(K, V)
    n_lvls  = len(k_pyramid)
    cutoff  = _cutoff_level(q_decay, epsilon, n_lvls - 1)

    offsets     = np.arange(-W, W + 1)
    local_idx   = np.clip(i_arr[:, None] + offsets[None, :], 0, N - 1)
    local_valid = (i_arr[:, None] + offsets[None, :] >= 0) & (i_arr[:, None] + offsets[None, :] < N)

    K_local     = K[local_idx]
    V_local     = V[local_idx]
    local_s     = np.einsum("nd,nwd->nw", Q, K_local) * scale
    local_s[~local_valid] = -1e30

    max_s   = local_s.max(axis=1)
    exp_l   = np.exp(local_s - max_s[:, None])
    exp_l[~local_valid] = 0.0
    sum_exp = exp_l.sum(axis=1)
    out_num = np.einsum("nw,nwd->nd", exp_l, V_local)

    for lvl in range(1, cutoff + 1):
        lvl_k  = k_pyramid[lvl]
        lvl_v  = v_pyramid[lvl]
        n_clus = len(lvl_k)
        stride = 1 << lvl
        env    = q_decay ** lvl

        clus_idx, valid = _boundary_clusters(i_arr, W, stride, n_clus, fan_out)

        K_lvl = lvl_k[clus_idx]
        V_lvl = lvl_v[clus_idx]

        lvl_s = np.einsum("nd,nfd->nf", Q, K_lvl) * scale + np.log(max(env, 1e-30))
        lvl_s[~valid] = -1e30

        lvl_max    = lvl_s.max(axis=1)
        new_max    = np.maximum(max_s, lvl_max)
        rescale    = np.exp(max_s - new_max)
        exp_lvl    = np.exp(lvl_s - new_max[:, None])
        exp_lvl[~valid] = 0.0

        sum_exp  = sum_exp * rescale + exp_lvl.sum(axis=1)
        out_num  = out_num * rescale[:, None] + np.einsum("nf,nfd->nd", exp_lvl, V_lvl)
        max_s    = new_max

    out = out_num / np.maximum(sum_exp[:, None], 1e-30)

    M           = (2 * W + 1) + fan_out * cutoff
    total_evals = N * M
    dense_evals = N * N
    stats = {
        "N":               N,
        "dense_evals":     dense_evals,
        "gabriel_evals":   total_evals,
        "evals_per_query": M,
        "sparsity_ratio":  1.0 - total_evals / dense_evals,
    }
    return out, stats


def dense_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    N, D   = Q.shape
    scale  = 1.0 / np.sqrt(D)
    scores = Q @ K.T * scale
    exp_s  = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs  = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return probs @ V
