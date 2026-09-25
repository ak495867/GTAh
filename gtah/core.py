import numpy as np
from typing import List, Tuple


def build_pyramid(K: np.ndarray, V: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    k_pyr = [K]
    v_pyr = [V]
    ck, cv = K.copy(), V.copy()
    while ck.shape[0] > 1:
        n = ck.shape[0]
        if n % 2 != 0:
            ck = np.pad(ck, ((0, 1), (0, 0)), mode="edge")
            cv = np.pad(cv, ((0, 1), (0, 0)), mode="edge")
        ck = 0.5 * (ck[0::2] + ck[1::2])
        cv = 0.5 * (cv[0::2] + cv[1::2])
        k_pyr.append(ck)
        v_pyr.append(cv)
    return k_pyr, v_pyr


def gabriel_transform(X: np.ndarray, num_levels: int = None) -> List[np.ndarray]:
    levels = []
    cur = X.copy()
    max_levels = int(np.ceil(np.log2(X.shape[0]))) if num_levels is None else num_levels
    for _ in range(max_levels):
        n = cur.shape[0]
        if n <= 1:
            break
        if n % 2 != 0:
            cur = np.pad(cur, ((0, 1), (0, 0)), mode="edge")
        coarse = 0.5 * (cur[0::2] + cur[1::2])
        detail = cur - np.repeat(coarse, 2, axis=0)[:n]
        levels.append(detail)
        cur = coarse
    levels.append(cur)
    return levels


def gabriel_reconstruct(levels: List[np.ndarray]) -> np.ndarray:
    recon = levels[-1].copy()
    for detail in reversed(levels[:-1]):
        n_out = detail.shape[0]
        expanded = np.repeat(recon, 2, axis=0)[:n_out]
        recon = expanded + detail
    return recon


def tail_energy(levels: List[np.ndarray], K: int) -> float:
    return float(sum(np.linalg.norm(lvl) for lvl in levels[K:]))


def geometric_decay_fit(levels: List[np.ndarray]) -> Tuple[float, float]:
    norms = np.array([np.linalg.norm(lvl) for lvl in levels])
    norms = norms[norms > 0]
    if len(norms) < 2:
        return float("nan"), float("nan")
    log_norms = np.log(norms)
    xs = np.arange(len(log_norms), dtype=float)
    slope, intercept = np.polyfit(xs, log_norms, 1)
    return float(np.exp(intercept)), float(np.exp(slope))
