import numpy as np
from typing import List, Dict, Tuple
from gtah.core import gabriel_transform, gabriel_reconstruct, tail_energy, geometric_decay_fit
from gtah.attention_numpy import gabriel_attention_numpy, dense_attention_numpy


def reconstruction_error(X: np.ndarray, K: int = None) -> Dict:
    levels = gabriel_transform(X)
    if K is None:
        K = len(levels) // 2
    C, q = geometric_decay_fit(levels)

    recon_full   = gabriel_reconstruct(levels)
    recon_trunc  = gabriel_reconstruct(levels[:K] + [levels[-1]])
    err_full     = float(np.linalg.norm(X - recon_full))
    err_trunc    = float(np.linalg.norm(X - recon_trunc))
    tail_bound   = tail_energy(levels, K)

    return {
        "full_recon_error":      err_full,
        "truncated_recon_error": err_trunc,
        "tail_energy":           tail_bound,
        "C":                     C,
        "q":                     q,
        "K":                     K,
        "num_levels":            len(levels),
        "theoretical_tail":      C * (q ** (K + 1)) / max(1.0 - q, 1e-10) if 0 < q < 1 else float("inf"),
    }


def attention_output_error(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    window_size: int = 32,
    fan_out: int = 4,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
) -> Dict:
    dense_out          = dense_attention_numpy(Q, K, V)
    gabriel_out, stats = gabriel_attention_numpy(Q, K, V, window_size, fan_out, q_decay, epsilon)

    abs_err  = float(np.linalg.norm(dense_out - gabriel_out))
    rel_err  = abs_err / max(float(np.linalg.norm(dense_out)), 1e-12)
    max_err  = float(np.max(np.abs(dense_out - gabriel_out)))
    cos_sims = np.array([
        np.dot(dense_out[i], gabriel_out[i]) / (
            np.linalg.norm(dense_out[i]) * np.linalg.norm(gabriel_out[i]) + 1e-12
        )
        for i in range(Q.shape[0])
    ])

    return {
        "N":               Q.shape[0],
        "abs_error":       abs_err,
        "rel_error":       rel_err,
        "max_token_error": max_err,
        "mean_cos_sim":    float(cos_sims.mean()),
        "min_cos_sim":     float(cos_sims.min()),
        **stats,
    }


def quality_sweep(
    Ns: List[int],
    D: int = 64,
    window_size: int = 32,
    fan_out: int = 4,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
) -> List[Dict]:
    rows = []
    rng  = np.random.default_rng(0)
    for N in Ns:
        Q = rng.standard_normal((N, D)).astype(np.float32)
        K = rng.standard_normal((N, D)).astype(np.float32)
        V = rng.standard_normal((N, D)).astype(np.float32)
        row = attention_output_error(Q, K, V, window_size, fan_out, q_decay, epsilon)
        rows.append(row)
        print(
            f"N={N:6d}  rel_err={row['rel_error']:.4f}"
            f"  cos_sim={row['mean_cos_sim']:.4f}"
            f"  evals/q={row['evals_per_query']:.0f}"
            f"  sparsity={row['sparsity_ratio']:.4f}"
        )
    return rows


def fan_out_sweep(
    N: int = 512,
    D: int = 64,
    window_size: int = 32,
    fan_outs: List[int] = None,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
) -> List[Dict]:
    if fan_outs is None:
        fan_outs = [2, 4, 6, 8, 12, 16]
    rows = []
    rng  = np.random.default_rng(7)
    Q    = rng.standard_normal((N, D)).astype(np.float32)
    K    = rng.standard_normal((N, D)).astype(np.float32)
    V    = rng.standard_normal((N, D)).astype(np.float32)
    for f in fan_outs:
        row = attention_output_error(Q, K, V, window_size, f, q_decay, epsilon)
        row["fan_out"] = f
        rows.append(row)
        print(
            f"fan_out={f:3d}  rel_err={row['rel_error']:.4f}"
            f"  cos_sim={row['mean_cos_sim']:.4f}"
            f"  evals/q={row['evals_per_query']:.0f}"
        )
    return rows
