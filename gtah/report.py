import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from gtah.benchmark import run_latency_sweep
from gtah.scaling   import compute_sparsity_profile, empirical_scaling_exponent, fit_models
from gtah.quality   import quality_sweep, fan_out_sweep
from gtah.plots     import (
    plot_latency,
    plot_scaling_exponent,
    plot_sparsity,
    plot_quality,
    plot_model_fit_comparison,
    plot_fan_out_sweep,
)

RESULTS_DIR = Path("results")


def save_json(data, name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"Saved: {path}")


def run_full_benchmark(
    Ns_small: List[int]  = None,
    Ns_large: List[int]  = None,
    D: int               = 64,
    window_size: int     = 32,
    fan_out: int         = 4,
    q_decay: float       = 0.5,
    epsilon: float       = 1e-4,
    reps: int            = 3,
):
    if Ns_small is None:
        Ns_small = [64, 128, 256, 512, 1024, 2048]
    if Ns_large is None:
        Ns_large = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== LATENCY SWEEP ===")
    latency_rows = run_latency_sweep(Ns_small, D, window_size, fan_out, q_decay, epsilon, reps)
    save_json(latency_rows, "latency.json")
    plot_latency(latency_rows, RESULTS_DIR / "latency.png")

    T_gabriel = [r["gabriel_latency_s"] for r in latency_rows]
    T_dense   = [r["dense_latency_s"]   for r in latency_rows]

    print("\n=== SCALING EXPONENT ===")
    alpha_rows = empirical_scaling_exponent(T_gabriel, Ns_small)
    save_json(alpha_rows, "scaling_exponent.json")
    plot_scaling_exponent(alpha_rows, RESULTS_DIR / "scaling_exponent.png")

    print("\n=== MODEL FIT ===")
    fits = fit_models(Ns_small, T_gabriel)
    save_json(fits, "model_fit.json")
    print(json.dumps(fits, indent=2, default=float))
    plot_model_fit_comparison(Ns_small, T_gabriel, T_dense, fits, RESULTS_DIR / "model_fit.png")

    print("\n=== SPARSITY PROFILE ===")
    sparsity_rows = compute_sparsity_profile(Ns_large, window_size, fan_out, q_decay, epsilon)
    save_json(sparsity_rows, "sparsity.json")
    plot_sparsity(sparsity_rows, RESULTS_DIR / "sparsity.png")

    print("\n=== QUALITY SWEEP ===")
    quality_rows = quality_sweep(Ns_small, D, window_size, fan_out, q_decay, epsilon)
    save_json(quality_rows, "quality.json")
    plot_quality(quality_rows, RESULTS_DIR / "quality.png")

    print("\n=== FAN-OUT SWEEP ===")
    fo_rows = fan_out_sweep(N=1024, D=D, window_size=window_size, fan_outs=[2, 4, 6, 8, 12, 16], q_decay=q_decay, epsilon=epsilon)
    save_json(fo_rows, "fan_out_sweep.json")
    plot_fan_out_sweep(fo_rows, RESULTS_DIR / "fan_out_sweep.png")

    print("\n=== DONE — all results in ./results/ ===")
    return {
        "latency":          latency_rows,
        "scaling_exponent": alpha_rows,
        "model_fit":        fits,
        "sparsity":         sparsity_rows,
        "quality":          quality_rows,
        "fan_out_sweep":    fo_rows,
    }


if __name__ == "__main__":
    run_full_benchmark()
