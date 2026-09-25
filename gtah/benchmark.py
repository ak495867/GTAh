import time
import tracemalloc
import numpy as np
import torch
from typing import List, Dict
from gtah.attention_numpy import gabriel_attention_numpy, dense_attention_numpy
from gtah.attention_torch import GabrielAttention, DenseAttention


def _time_numpy_fn(fn, *args, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(*args)
        times.append(time.perf_counter() - t0)
    return result, min(times)


def _time_torch_fn(model, x, reps=3):
    times = []
    with torch.no_grad():
        for _ in range(reps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return min(times)


def _peak_vram_mb(model, x):
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def _peak_ram_mb(fn, *args):
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 ** 2


def estimate_flops_dense(N: int, D: int) -> int:
    return 4 * N * N * D


def estimate_flops_gabriel(N: int, D: int, evals_per_query: int) -> int:
    return 2 * N * evals_per_query * D


def run_latency_sweep(
    Ns: List[int],
    D: int = 64,
    window_size: int = 32,
    fan_out: int = 4,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
    reps: int = 3,
) -> List[Dict]:
    rows = []
    for N in Ns:
        rng = np.random.default_rng(42)
        Q   = rng.standard_normal((N, D)).astype(np.float32)
        K   = rng.standard_normal((N, D)).astype(np.float32)
        V   = rng.standard_normal((N, D)).astype(np.float32)

        (gout, stats), g_lat = _time_numpy_fn(
            gabriel_attention_numpy, Q, K, V, window_size, fan_out, q_decay, epsilon, reps=reps
        )
        _, d_lat = _time_numpy_fn(dense_attention_numpy, Q, K, V, reps=reps)

        g_flops = estimate_flops_gabriel(N, D, stats["evals_per_query"])
        d_flops = estimate_flops_dense(N, D)

        g_ram = _peak_ram_mb(gabriel_attention_numpy, Q, K, V, window_size, fan_out, q_decay, epsilon)
        d_ram = _peak_ram_mb(dense_attention_numpy, Q, K, V)

        rows.append({
            "N":                 N,
            "dense_latency_s":   d_lat,
            "gabriel_latency_s": g_lat,
            "speedup":           d_lat / max(g_lat, 1e-12),
            "dense_flops":       d_flops,
            "gabriel_flops":     g_flops,
            "flop_ratio":        g_flops / max(d_flops, 1),
            "dense_ram_mb":      d_ram,
            "gabriel_ram_mb":    g_ram,
            "evals_per_query":   stats["evals_per_query"],
            "sparsity_ratio":    stats["sparsity_ratio"],
        })
        print(
            f"N={N:6d}  dense={d_lat*1e3:8.2f}ms  gabriel={g_lat*1e3:8.2f}ms"
            f"  speedup={d_lat/max(g_lat,1e-12):6.2f}x"
            f"  evals/q={stats['evals_per_query']:.0f}"
            f"  sparsity={stats['sparsity_ratio']:.4f}"
        )
    return rows


def run_torch_latency_sweep(
    Ns: List[int],
    D: int = 128,
    num_heads: int = 4,
    window_size: int = 32,
    fan_out: int = 4,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
    reps: int = 3,
) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows   = []
    for N in Ns:
        x = torch.randn(1, N, D, device=device)

        gm = GabrielAttention(D, num_heads, window_size, fan_out, q_decay, epsilon).to(device).eval()
        dm = DenseAttention(D, num_heads).to(device).eval()

        g_lat  = _time_torch_fn(gm, x, reps=reps)
        d_lat  = _time_torch_fn(dm, x, reps=reps)
        g_vram = _peak_vram_mb(gm, x)
        d_vram = _peak_vram_mb(dm, x)

        rows.append({
            "N":                 N,
            "dense_latency_s":   d_lat,
            "gabriel_latency_s": g_lat,
            "speedup":           d_lat / max(g_lat, 1e-12),
            "dense_vram_mb":     d_vram,
            "gabriel_vram_mb":   g_vram,
        })
        print(
            f"N={N:6d}  dense={d_lat*1e3:8.2f}ms  gabriel={g_lat*1e3:8.2f}ms"
            f"  speedup={d_lat/max(g_lat,1e-12):6.2f}x"
        )
    return rows
