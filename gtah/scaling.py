import numpy as np
from typing import List, Dict
from gtah.attention_torch import GabrielAttention


def compute_sparsity_profile(
    Ns: List[int],
    window_size: int = 32,
    fan_out: int = 4,
    q_decay: float = 0.5,
    epsilon: float = 1e-4,
) -> List[Dict]:
    rows = []
    for N in Ns:
        model  = GabrielAttention(64, 1, window_size, fan_out, q_decay, epsilon)
        E      = model.count_active_pairs(N)
        E_over_N     = E / N
        log2_N       = np.log2(N)
        rows.append({
            "N":            N,
            "E":            E,
            "E_over_N":     E_over_N,
            "log2_N":       log2_N,
            "E_over_NlogN": E / max(N * log2_N, 1),
        })
        print(f"N={N:6d}  E={E:10d}  E/N={E_over_N:8.2f}  log2(N)={log2_N:.2f}  E/(N log N)={E/max(N*log2_N,1):.3f}")
    return rows


def empirical_scaling_exponent(T: List[float], Ns: List[int]) -> List[Dict]:
    rows = []
    for i in range(len(Ns) - 1):
        N1, N2 = Ns[i], Ns[i + 1]
        T1, T2 = T[i], T[i + 1]
        if T1 <= 0 or T2 <= 0:
            continue
        alpha = (np.log(T2) - np.log(T1)) / (np.log(N2) - np.log(N1))
        rows.append({"N1": N1, "N2": N2, "T1": T1, "T2": T2, "alpha": alpha})
        print(f"alpha({N1}->{N2}) = {alpha:.4f}")
    return rows


def fit_models(Ns: List[int], T: List[float]) -> Dict:
    Ns_arr = np.array(Ns, dtype=float)
    T_arr  = np.array(T, dtype=float)

    def fit_r2(A):
        coeffs, _, _, _ = np.linalg.lstsq(A, T_arr, rcond=None)
        pred   = A @ coeffs
        ss_res = np.sum((T_arr - pred) ** 2)
        ss_tot = np.sum((T_arr - T_arr.mean()) ** 2)
        return coeffs, 1.0 - ss_res / max(ss_tot, 1e-30)

    c_lin, r2_lin = fit_r2(np.column_stack([Ns_arr, np.ones_like(Ns_arr)]))
    c_log, r2_log = fit_r2(np.column_stack([np.log(Ns_arr), np.ones_like(Ns_arr)]))
    c_nln, r2_nln = fit_r2(np.column_stack([Ns_arr * np.log(Ns_arr), np.ones_like(Ns_arr)]))

    return {
        "linear": {"coeffs": c_lin.tolist(), "R2": r2_lin},
        "log":    {"coeffs": c_log.tolist(), "R2": r2_log},
        "nlogn":  {"coeffs": c_nln.tolist(), "R2": r2_nln},
    }
