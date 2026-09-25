import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict


PALETTE = {
    "dense":   "#E63946",
    "gabriel": "#2A9D8F",
    "alpha":   "#F4A261",
    "theory":  "#457B9D",
    "grid":    "#2d2d2d",
    "bg":      "#0f0f0f",
    "fg":      "#e0e0e0",
}


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["fg"])
    ax.xaxis.label.set_color(PALETTE["fg"])
    ax.yaxis.label.set_color(PALETTE["fg"])
    ax.title.set_color(PALETTE["fg"])
    ax.spines[:].set_color(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_latency(rows: List[Dict], out_path: Path):
    Ns    = [r["N"] for r in rows]
    d_lat = [r["dense_latency_s"] * 1e3 for r in rows]
    g_lat = [r["gabriel_latency_s"] * 1e3 for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax = axes[0]
    ax.loglog(Ns, d_lat, "o-", color=PALETTE["dense"],   label="Dense O(N²)")
    ax.loglog(Ns, g_lat, "s-", color=PALETTE["gabriel"], label="Gabriel GTAH")
    _style(ax, "Latency vs N (log-log)", "N", "Latency (ms)")
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    ax = axes[1]
    speedups = [r["speedup"] for r in rows]
    ax.semilogx(Ns, speedups, "D-", color=PALETTE["alpha"])
    ax.axhline(1.0, color=PALETTE["fg"], linestyle="--", linewidth=0.8)
    _style(ax, "Speedup Gabriel / Dense", "N", "Speedup ×")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved: {out_path}")


def plot_scaling_exponent(alpha_rows: List[Dict], out_path: Path):
    midpoints = [0.5 * (r["N1"] + r["N2"]) for r in alpha_rows]
    alphas    = [r["alpha"] for r in alpha_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.semilogx(midpoints, alphas, "o-", color=PALETTE["gabriel"], label="Gabriel α(N)")
    ax.axhline(0.0, color=PALETTE["theory"], linestyle="--", linewidth=1.0, label="O(1)")
    ax.axhline(1.0, color=PALETTE["dense"],  linestyle="--", linewidth=1.0, label="O(N)")
    ax.axhline(2.0, color="#888",            linestyle=":",  linewidth=0.8, label="O(N²)")

    _style(ax, "Empirical Scaling Exponent α(N)", "N (midpoint)", "α(N)")
    ax.set_ylim(-0.5, 2.5)
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved: {out_path}")


def plot_sparsity(sparsity_rows: List[Dict], out_path: Path):
    Ns       = [r["N"] for r in sparsity_rows]
    E        = [r["E"] for r in sparsity_rows]
    E_over_N = [r["E_over_N"] for r in sparsity_rows]
    log2_N   = [r["log2_N"] for r in sparsity_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax = axes[0]
    ax.loglog(Ns, E, "o-", color=PALETTE["gabriel"], label="E(N) Gabriel")
    n_arr = np.array(Ns, dtype=float)
    ax.loglog(Ns, n_arr * np.log2(n_arr), "--", color=PALETTE["theory"], label="N log₂ N (theory)")
    ax.loglog(Ns, n_arr ** 2, ":",         color=PALETTE["dense"],  label="N² dense")
    _style(ax, "Active Pairs E(N)", "N", "E(N)")
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    ax = axes[1]
    ax.semilogx(Ns, E_over_N, "s-", color=PALETTE["gabriel"], label="E(N)/N Gabriel")
    ax.semilogx(Ns, log2_N,   "--", color=PALETTE["theory"],  label="log₂ N (theory)")
    _style(ax, "E(N)/N vs log₂ N", "N", "E(N) / N")
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved: {out_path}")


def plot_quality(quality_rows: List[Dict], out_path: Path):
    Ns       = [r["N"] for r in quality_rows]
    rel_errs = [r["rel_error"] for r in quality_rows]
    cos_sims = [r["mean_cos_sim"] for r in quality_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax = axes[0]
    ax.semilogx(Ns, rel_errs, "o-", color=PALETTE["dense"], label="Relative Error ‖Δ‖/‖Dense‖")
    _style(ax, "Output Reconstruction Error vs Dense", "N", "Relative Error")
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    ax = axes[1]
    ax.semilogx(Ns, cos_sims, "s-", color=PALETTE["gabriel"], label="Mean Cosine Similarity")
    ax.axhline(1.0, color=PALETTE["fg"], linestyle="--", linewidth=0.8)
    _style(ax, "Cosine Similarity: Gabriel vs Dense Output", "N", "Cos Sim")
    ax.set_ylim(0, 1.1)
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved: {out_path}")


def plot_model_fit_comparison(
    Ns: List[int],
    T_gabriel: List[float],
    T_dense: List[float],
    fit_results: Dict,
    out_path: Path
):
    n_arr  = np.array(Ns, dtype=float)
    n_fine = np.logspace(np.log10(n_arr[0]), np.log10(n_arr[-1]), 300)

    c_lin = fit_results["linear"]["coeffs"]
    c_log = fit_results["log"]["coeffs"]
    c_nln = fit_results["nlogn"]["coeffs"]

    fit_lin  = c_lin[0] * n_fine + c_lin[1]
    fit_log  = c_log[0] * np.log(n_fine) + c_log[1]
    fit_nln  = c_nln[0] * n_fine * np.log(n_fine) + c_nln[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax.loglog(Ns, T_gabriel, "o", color=PALETTE["gabriel"], label="Gabriel (measured)", markersize=7)
    ax.loglog(Ns, T_dense,   "^", color=PALETTE["dense"],   label="Dense (measured)",  markersize=7)
    ax.loglog(n_fine, np.clip(fit_lin,  1e-12, None), "--", color=PALETTE["alpha"],  label=f"O(N) fit  R²={fit_results['linear']['R2']:.3f}")
    ax.loglog(n_fine, np.clip(fit_log,  1e-12, None), "-.", color=PALETTE["theory"], label=f"O(logN) fit  R²={fit_results['log']['R2']:.3f}")
    ax.loglog(n_fine, np.clip(fit_nln,  1e-12, None), ":",  color="#aaa",            label=f"O(NlogN) fit  R²={fit_results['nlogn']['R2']:.3f}")

    _style(ax, "Latency Model Fit: Gabriel vs Dense", "N", "Latency (s)")
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["fg"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved: {out_path}")


def plot_fan_out_sweep(rows: List[Dict], out_path: Path):
    fan_outs  = [r["fan_out"]       for r in rows]
    cos_sims  = [r["mean_cos_sim"]  for r in rows]
    rel_errs  = [r["rel_error"]     for r in rows]
    evals     = [r["evals_per_query"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax = axes[0]
    ax.plot(fan_outs, cos_sims, "o-", color=PALETTE["gabriel"])
    ax.axhline(1.0, color=PALETTE["fg"], linestyle="--", linewidth=0.7)
    _style(ax, "Cosine Sim vs Fan-out f", "fan_out f", "Mean Cosine Sim")
    ax.set_ylim(0, 1.1)

    ax = axes[1]
    ax.plot(fan_outs, rel_errs, "s-", color=PALETTE["dense"])
    _style(ax, "Relative Error vs Fan-out f", "fan_out f", "Relative Error")

    ax = axes[2]
    ax.plot(fan_outs, evals, "D-", color=PALETTE["alpha"])
    _style(ax, "Evals per Query vs Fan-out f", "fan_out f", "Evals / Query")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"])
    plt.close()
    print(f"Saved: {out_path}")

