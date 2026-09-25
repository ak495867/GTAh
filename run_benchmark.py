import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gtah.report import run_full_benchmark


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--Ns",        nargs="+", type=int,   default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--Ns-large",  nargs="+", type=int,   default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--D",         type=int,   default=64)
    p.add_argument("--window",    type=int,   default=32)
    p.add_argument("--fan-out",   type=int,   default=4)
    p.add_argument("--q-decay",   type=float, default=0.5)
    p.add_argument("--epsilon",   type=float, default=1e-4)
    p.add_argument("--reps",      type=int,   default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_benchmark(
        Ns_small    = args.Ns,
        Ns_large    = getattr(args, "Ns_large"),
        D           = args.D,
        window_size = args.window,
        fan_out     = getattr(args, "fan_out"),
        q_decay     = args.q_decay,
        epsilon     = args.epsilon,
        reps        = args.reps,
    )
