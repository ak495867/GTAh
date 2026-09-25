import time
import math
import json
import argparse
import sys
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models.dataset     import build_dataloaders
from models.transformer import build_model


PALETTE = {
    "gabriel": "#2A9D8F",
    "dense":   "#E63946",
    "accent":  "#F4A261",
    "bg":      "#0f0f0f",
    "fg":      "#e0e0e0",
    "grid":    "#2d2d2d",
    "bar_g":   "#2A9D8F",
    "bar_d":   "#E63946",
}


def _ax(ax, title, xlabel, ylabel, legend=False):
    ax.set_facecolor(PALETTE["bg"])
    ax.tick_params(colors=PALETTE["fg"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["fg"])
    ax.yaxis.label.set_color(PALETTE["fg"])
    ax.title.set_color(PALETTE["fg"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if legend:
        ax.legend(facecolor="#1a1a1a", labelcolor=PALETTE["fg"], fontsize=8, framealpha=0.8)


def smooth(vals, w=5):
    if len(vals) < w:
        return vals
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode="valid").tolist()


def perplexity(loss):
    return math.exp(min(float(loss), 30))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def peak_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def token_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_batches=80):
    model.eval()
    total_loss, total_acc, total_tokens = 0.0, 0.0, 0
    ctx = autocast(dtype=torch.float16) if use_amp else nullcontext()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y    = x.to(device), y.to(device)
        with ctx:
            logits  = model(x)
        B, T, V = logits.shape
        loss    = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
        acc     = token_accuracy(logits.view(B * T, V), y.view(B * T))
        total_loss   += loss.item() * B * T
        total_acc    += acc * B * T
        total_tokens += B * T
    denom = max(total_tokens, 1)
    return total_loss / denom, total_acc / denom


def measure_throughput(model, seq_len, batch_size, device, use_amp, reps=5):
    model.eval()
    x = torch.randint(0, 65, (batch_size, seq_len), device=device)
    ctx = autocast(dtype=torch.float16) if use_amp else nullcontext()
    with torch.no_grad():
        for _ in range(2):
            with ctx:
                model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(reps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ctx:
                model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    best = min(times)
    return best, (batch_size * seq_len) / best


def train_one(model, train_loader, val_loader, cfg, device, label, use_amp, accum_steps):
    scaler    = GradScaler() if use_amp else None
    max_steps = cfg.get("max_steps_per_epoch")
    steps_per_epoch = min(max_steps, len(train_loader)) if max_steps else len(train_loader)
    total_opt_steps = cfg["epochs"] * steps_per_epoch // accum_steps

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        betas        = (0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_opt_steps, 1), eta_min=cfg["lr"] / 10)

    train_losses, val_losses, val_ppls, val_accs = [], [], [], []
    step_records, lr_records, vram_records = [], [], []
    global_step = 0

    print(f"\n{'='*64}")
    print(f"  {label}  |  params={model.num_params():,}  |  device={device}  |  AMP={use_amp}")
    print(f"{'='*64}")

    reset_vram()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss, epoch_acc, epoch_tokens = 0.0, 0.0, 0
        epoch_steps = 0
        t0 = time.perf_counter()
        optimizer.zero_grad()

        for step_in_epoch, (x, y) in enumerate(train_loader):
            if max_steps and epoch_steps >= max_steps:
                break

            x, y     = x.to(device), y.to(device)
            ctx      = autocast(dtype=torch.float16) if use_amp else nullcontext()

            with ctx:
                logits   = model(x)
                B, T, V  = logits.shape
                loss     = F.cross_entropy(logits.view(B * T, V), y.view(B * T)) / accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            acc           = token_accuracy(logits.detach().view(B * T, V), y.view(B * T))
            epoch_loss   += loss.item() * accum_steps * B * T
            epoch_acc    += acc * B * T
            epoch_tokens += B * T
            epoch_steps  += 1
            global_step  += 1

            if epoch_steps % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % cfg["log_every"] == 0:
                cur_lr = scheduler.get_last_lr()[0]
                step_records.append((global_step, loss.item() * accum_steps))
                lr_records.append((global_step, cur_lr))
                vram_records.append((global_step, vram_mb()))

        avg_train = epoch_loss / max(epoch_tokens, 1)
        avg_acc   = epoch_acc  / max(epoch_tokens, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp)
        ppl       = perplexity(val_loss)

        train_losses.append(avg_train)
        val_losses.append(val_loss)
        val_ppls.append(ppl)
        val_accs.append(val_acc)

        elapsed = time.perf_counter() - t0
        print(
            f"  epoch {epoch:3d}/{cfg['epochs']}"
            f"  train={avg_train:.4f}"
            f"  val={val_loss:.4f}"
            f"  ppl={ppl:.1f}"
            f"  acc={val_acc*100:.1f}%"
            f"  vram={peak_vram_mb():.0f}MB"
            f"  {elapsed:.1f}s"
        )

    return {
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "val_ppls":       val_ppls,
        "val_accs":       val_accs,
        "step_losses":    step_records,
        "lr_records":     lr_records,
        "vram_records":   vram_records,
        "final_val_loss": val_losses[-1],
        "final_ppl":      val_ppls[-1],
        "final_acc":      val_accs[-1],
        "peak_vram_mb":   peak_vram_mb(),
        "num_params":     model.num_params(),
    }


def save_all_plots(gh, dh, cfg, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, cfg["epochs"] + 1))

    fig = plt.figure(figsize=(22, 14), facecolor=PALETTE["bg"])
    gs  = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, gh["train_losses"], "-",  color=PALETTE["gabriel"], lw=1.8, label="GTAH train")
    ax1.plot(epochs, gh["val_losses"],   "--", color=PALETTE["gabriel"], lw=1.8, label="GTAH val")
    ax1.plot(epochs, dh["train_losses"], "-",  color=PALETTE["dense"],   lw=1.8, label="Dense train")
    ax1.plot(epochs, dh["val_losses"],   "--", color=PALETTE["dense"],   lw=1.8, label="Dense val")
    _ax(ax1, "Cross-Entropy Loss", "Epoch", "Loss (nats)", legend=True)

    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.semilogy(epochs, gh["val_ppls"], "o-", color=PALETTE["gabriel"], lw=1.8, ms=4, label="GTAH")
    ax2.semilogy(epochs, dh["val_ppls"], "s-", color=PALETTE["dense"],   lw=1.8, ms=4, label="Dense")
    _ax(ax2, "Validation Perplexity (log scale)", "Epoch", "Perplexity", legend=True)

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(epochs, [a * 100 for a in gh["val_accs"]], "o-", color=PALETTE["gabriel"], lw=1.8, ms=4, label="GTAH")
    ax3.plot(epochs, [a * 100 for a in dh["val_accs"]], "s-", color=PALETTE["dense"],   lw=1.8, ms=4, label="Dense")
    _ax(ax3, "Validation Token Accuracy (%)", "Epoch", "Accuracy (%)", legend=True)

    ax4 = fig.add_subplot(gs[1, 2:])
    if gh["step_losses"] and dh["step_losses"]:
        gs_, gl_ = zip(*gh["step_losses"])
        ds_, dl_ = zip(*dh["step_losses"])
        ax4.plot(gs_, smooth(list(gl_)), alpha=0.9, color=PALETTE["gabriel"], lw=1.5, label="GTAH (smoothed)")
        ax4.plot(ds_, smooth(list(dl_)), alpha=0.9, color=PALETTE["dense"],   lw=1.5, label="Dense (smoothed)")
    _ax(ax4, "Step-Level Training Loss (smoothed)", "Global Step", "Loss", legend=True)

    metrics   = ["Val Loss", "Perplexity", "Accuracy %"]
    g_vals    = [gh["final_val_loss"], gh["final_ppl"], gh["final_acc"] * 100]
    d_vals    = [dh["final_val_loss"], dh["final_ppl"], dh["final_acc"] * 100]
    x_pos     = np.arange(len(metrics))
    w         = 0.35

    ax5 = fig.add_subplot(gs[2, :2])
    bars_g = ax5.bar(x_pos - w/2, g_vals, w, label="GTAH",  color=PALETTE["gabriel"], alpha=0.85)
    bars_d = ax5.bar(x_pos + w/2, d_vals, w, label="Dense", color=PALETTE["dense"],   alpha=0.85)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(metrics, color=PALETTE["fg"], fontsize=9)
    for bar in list(bars_g) + list(bars_d):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom",
                 color=PALETTE["fg"], fontsize=7)
    _ax(ax5, "Final Metrics Comparison", "", "Value", legend=True)

    ax6 = fig.add_subplot(gs[2, 2:])
    if gh["vram_records"] and dh["vram_records"]:
        gvs_, gvm_ = zip(*gh["vram_records"])
        dvs_, dvm_ = zip(*dh["vram_records"])
        ax6.plot(gvs_, gvm_, color=PALETTE["gabriel"], lw=1.5, label="GTAH VRAM")
        ax6.plot(dvs_, dvm_, color=PALETTE["dense"],   lw=1.5, label="Dense VRAM")
        _ax(ax6, "GPU VRAM Usage During Training", "Global Step", "VRAM (MB)", legend=True)
    else:
        ax6.text(0.5, 0.5, "VRAM tracking\nrequires CUDA", ha="center", va="center",
                 color=PALETTE["fg"], fontsize=12, transform=ax6.transAxes)
        _ax(ax6, "GPU VRAM Usage", "", "")

    plt.suptitle("GTAH vs Dense Attention — Training Results", color=PALETTE["fg"], fontsize=14, fontweight="bold", y=1.01)
    path = out_dir / "training_results.png"
    plt.savefig(path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_throughput_plot(gh_tput, dh_tput, out_dir: Path):
    labels   = ["GTAH", "Dense"]
    latency  = [gh_tput[0] * 1000, dh_tput[0] * 1000]
    tok_sec  = [gh_tput[1], dh_tput[1]]
    colors   = [PALETTE["gabriel"], PALETTE["dense"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=PALETTE["bg"])

    ax = axes[0]
    bars = ax.bar(labels, latency, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}ms", ha="center", va="bottom", color=PALETTE["fg"], fontsize=10)
    _ax(ax, "Inference Latency (lower is better)", "Model", "Latency (ms)")

    ax = axes[1]
    bars = ax.bar(labels, tok_sec, color=colors, alpha=0.85, width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.0f}", ha="center", va="bottom", color=PALETTE["fg"], fontsize=10)
    _ax(ax, "Inference Throughput (higher is better)", "Model", "Tokens / Second")

    plt.tight_layout()
    path = out_dir / "throughput.png"
    plt.savefig(path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=float)
    print(f"Saved: {path}")


def print_summary(gh, dh, gh_tput, dh_tput):
    print(f"\n{'='*64}")
    print("  FINAL COMPARISON")
    print(f"{'='*64}")
    fmt = "  {:<22} {:>16} {:>16}"
    print(fmt.format("Metric", "GTAH", "Dense"))
    print("  " + "-"*54)
    print(fmt.format("Parameters",        f"{gh['num_params']:,}",          f"{dh['num_params']:,}"))
    print(fmt.format("Val Loss",           f"{gh['final_val_loss']:.4f}",   f"{dh['final_val_loss']:.4f}"))
    print(fmt.format("Perplexity",         f"{gh['final_ppl']:.2f}",        f"{dh['final_ppl']:.2f}"))
    print(fmt.format("Token Accuracy",     f"{gh['final_acc']*100:.2f}%",   f"{dh['final_acc']*100:.2f}%"))
    print(fmt.format("Latency (ms)",       f"{gh_tput[0]*1000:.2f}",        f"{dh_tput[0]*1000:.2f}"))
    print(fmt.format("Tokens/sec",         f"{gh_tput[1]:.0f}",             f"{dh_tput[1]:.0f}"))
    print(fmt.format("Peak VRAM (MB)",     f"{gh['peak_vram_mb']:.1f}",     f"{dh['peak_vram_mb']:.1f}"))
    ppl_delta = gh["final_ppl"] - dh["final_ppl"]
    acc_delta = (gh["final_acc"] - dh["final_acc"]) * 100
    spd_delta = (gh_tput[1] / max(dh_tput[1], 1) - 1) * 100
    print("  " + "-"*54)
    print(fmt.format("PPL gap (GTAH-Dense)", f"{ppl_delta:+.2f}", ""))
    print(fmt.format("Acc gap (GTAH-Dense)", f"{acc_delta:+.2f}%", ""))
    print(fmt.format("Throughput gain",      f"{spd_delta:+.1f}%", ""))
    print(f"{'='*64}\n")


def parse_args():
    p = argparse.ArgumentParser(description="GTAH vs Dense mini-LM training comparison")
    p.add_argument("--epochs",            type=int,   default=20)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--seq-len",           type=int,   default=256)
    p.add_argument("--d-model",           type=int,   default=256)
    p.add_argument("--num-heads",         type=int,   default=8)
    p.add_argument("--num-layers",        type=int,   default=6)
    p.add_argument("--ffn-mult",          type=int,   default=4)
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--lr",                type=float, default=3e-4)
    p.add_argument("--weight-decay",      type=float, default=0.1)
    p.add_argument("--grad-clip",         type=float, default=1.0)
    p.add_argument("--accum-steps",       type=int,   default=1)
    p.add_argument("--window",            type=int,   default=64)
    p.add_argument("--fan-out",           type=int,   default=4)
    p.add_argument("--q-decay",           type=float, default=0.5)
    p.add_argument("--epsilon",           type=float, default=1e-4)
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--log-every",         type=int,   default=50)
    p.add_argument("--no-amp",            action="store_true")
    p.add_argument("--compile",           action="store_true")
    p.add_argument("--data-path",         type=str,   default="data/shakespeare.txt")
    p.add_argument("--out-dir",           type=str,   default="results/training")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    device  = get_device()
    use_amp = not args.no_amp and device.type == "cuda"

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device : {device}")
    print(f"AMP    : {use_amp}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_loader, val_loader, vocab_size, _ = build_dataloaders(
        seq_len    = args.seq_len,
        batch_size = args.batch_size,
        data_path  = args.data_path,
    )
    print(f"Vocab  : {vocab_size}  |  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    cfg = {
        "vocab_size":          vocab_size,
        "seq_len":             args.seq_len,
        "d_model":             args.d_model,
        "num_heads":           args.num_heads,
        "num_layers":          args.num_layers,
        "ffn_mult":            args.ffn_mult,
        "dropout":             args.dropout,
        "lr":                  args.lr,
        "weight_decay":        args.weight_decay,
        "grad_clip":           args.grad_clip,
        "epochs":              args.epochs,
        "log_every":           args.log_every,
        "window_size":         args.window,
        "fan_out":             args.fan_out,
        "q_decay":             args.q_decay,
        "epsilon":             args.epsilon,
        "max_steps_per_epoch": args.max_steps_per_epoch,
    }

    torch.manual_seed(args.seed)
    gabriel_model = build_model(cfg, attn_type="gabriel").to(device)
    if args.compile and hasattr(torch, "compile"):
        gabriel_model = torch.compile(gabriel_model)
    gabriel_hist = train_one(gabriel_model, train_loader, val_loader, cfg, device,
                             "GTAH Gabriel", use_amp, args.accum_steps)

    reset_vram()

    torch.manual_seed(args.seed)
    dense_model = build_model(cfg, attn_type="dense").to(device)
    if args.compile and hasattr(torch, "compile"):
        dense_model = torch.compile(dense_model)
    dense_hist = train_one(dense_model, train_loader, val_loader, cfg, device,
                           "Dense Baseline", use_amp, args.accum_steps)

    gabriel_model.eval()
    dense_model.eval()
    gabriel_tput = measure_throughput(gabriel_model, args.seq_len, args.batch_size, device, use_amp)
    dense_tput   = measure_throughput(dense_model,   args.seq_len, args.batch_size, device, use_amp)

    out_dir = Path(args.out_dir)
    save_all_plots(gabriel_hist, dense_hist, cfg, out_dir)
    save_throughput_plot(gabriel_tput, dense_tput, out_dir)

    comparison = {
        "gabriel": {**gabriel_hist, "latency_s": gabriel_tput[0], "tokens_per_sec": gabriel_tput[1]},
        "dense":   {**dense_hist,   "latency_s": dense_tput[0],   "tokens_per_sec": dense_tput[1]},
        "config":  cfg,
        "device":  str(device),
    }
    save_json(comparison, out_dir / "comparison.json")
    save_json(cfg,        out_dir / "config.json")

    print_summary(gabriel_hist, dense_hist, gabriel_tput, dense_tput)
    print(f"All results saved to: {out_dir.resolve()}")
