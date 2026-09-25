import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from gtah.core              import gabriel_transform, gabriel_reconstruct, geometric_decay_fit
from gtah.attention_numpy   import gabriel_attention_numpy, dense_attention_numpy
from gtah.attention_torch   import GabrielAttention, DenseAttention
from gtah.scaling           import compute_sparsity_profile, empirical_scaling_exponent, fit_models
from gtah.quality           import reconstruction_error, attention_output_error, fan_out_sweep


def test_pyramid_roundtrip():
    rng = np.random.default_rng(1)
    X   = rng.standard_normal((128, 32)).astype(np.float32)
    levels = gabriel_transform(X)
    recon  = gabriel_reconstruct(levels)
    err    = np.max(np.abs(X - recon))
    assert err < 1e-4, f"Roundtrip error too large: {err}"
    print(f"[PASS] pyramid roundtrip  max_err={err:.2e}")


def test_geometric_decay():
    levels = gabriel_transform(np.random.default_rng(2).standard_normal((256, 16)).astype(np.float32))
    C, q   = geometric_decay_fit(levels)
    assert 0 < q < 1.5, f"Unexpected q={q}"
    print(f"[PASS] geometric decay  C={C:.4f}  q={q:.4f}")


def test_attention_shapes():
    N, D = 64, 32
    rng  = np.random.default_rng(3)
    Q    = rng.standard_normal((N, D)).astype(np.float32)
    K    = rng.standard_normal((N, D)).astype(np.float32)
    V    = rng.standard_normal((N, D)).astype(np.float32)
    out, stats = gabriel_attention_numpy(Q, K, V, window_size=16, fan_out=4)
    assert out.shape == (N, D), f"Wrong shape: {out.shape}"
    assert stats["gabriel_evals"] <= N * N
    print(f"[PASS] numpy attention shapes  evals_per_q={stats['evals_per_query']}")


def test_vectorised_no_nan():
    N, D = 256, 64
    rng  = np.random.default_rng(10)
    Q    = rng.standard_normal((N, D)).astype(np.float32)
    K    = rng.standard_normal((N, D)).astype(np.float32)
    V    = rng.standard_normal((N, D)).astype(np.float32)
    out, _ = gabriel_attention_numpy(Q, K, V, window_size=32, fan_out=4)
    assert not np.any(np.isnan(out)), "NaN in output"
    assert not np.any(np.isinf(out)), "Inf in output"
    print(f"[PASS] vectorised no NaN/Inf  shape={out.shape}")


def test_torch_forward():
    B, N, D = 1, 64, 32
    model = GabrielAttention(D, num_heads=4, window_size=8, fan_out=4)
    x     = torch.randn(B, N, D)
    out   = model(x)
    assert out.shape == (B, N, D), f"Wrong shape: {out.shape}"
    assert not torch.any(torch.isnan(out)), "NaN in torch output"
    print(f"[PASS] torch forward  shape={out.shape}")


def test_torch_gradient_flows():
    B, N, D = 1, 32, 16
    model = GabrielAttention(D, num_heads=2, window_size=4, fan_out=4)
    x     = torch.randn(B, N, D, requires_grad=True)
    out   = model(x)
    loss  = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient"
    assert not torch.any(torch.isnan(x.grad)), "NaN gradient"
    print(f"[PASS] torch gradient flows  grad_norm={x.grad.norm().item():.4f}")


def test_sparsity_structure():
    rows = compute_sparsity_profile([64, 128, 256], window_size=16, fan_out=4)
    for r in rows:
        assert r["E"] <= r["N"] ** 2
    E_per_N = [r["E_over_N"] for r in rows]
    assert E_per_N[-1] < rows[-1]["N"], "E/N should be < N (not O(N^2))"
    print(f"[PASS] sparsity bounds ok  E/N={[round(r['E_over_N'],1) for r in rows]}")


def test_fan_out_improves_quality():
    rows = fan_out_sweep(N=256, D=32, window_size=16, fan_outs=[2, 4, 8])
    cos_2 = rows[0]["mean_cos_sim"]
    cos_8 = rows[2]["mean_cos_sim"]
    assert cos_8 >= cos_2 - 0.05, f"Higher fan_out should improve or maintain quality: {cos_2:.3f} vs {cos_8:.3f}"
    print(f"[PASS] fan_out quality  cos@f=2={cos_2:.4f}  cos@f=8={cos_8:.4f}")


def test_quality_metrics():
    N, D = 128, 32
    rng  = np.random.default_rng(4)
    Q    = rng.standard_normal((N, D)).astype(np.float32)
    K    = rng.standard_normal((N, D)).astype(np.float32)
    V    = rng.standard_normal((N, D)).astype(np.float32)
    result = attention_output_error(Q, K, V, window_size=16, fan_out=8)
    assert 0 <= result["mean_cos_sim"] <= 1.0 + 1e-6
    print(f"[PASS] quality metrics  cos_sim={result['mean_cos_sim']:.4f}  rel_err={result['rel_error']:.4f}")


def test_scaling_exponent():
    Ns = [64, 128, 256, 512]
    T  = [n ** 1.1 * 1e-6 for n in Ns]
    rows = empirical_scaling_exponent(T, Ns)
    for r in rows:
        assert 0.5 < r["alpha"] < 2.0, f"Alpha out of range: {r['alpha']}"
    print(f"[PASS] scaling exponent  alphas={[round(r['alpha'],3) for r in rows]}")


def test_model_fit():
    Ns = [64, 128, 256, 512, 1024]
    T  = [n * np.log2(n) * 1e-7 for n in Ns]
    fits = fit_models(Ns, T)
    assert fits["nlogn"]["R2"] > 0.99
    print(f"[PASS] model fit  nlogn_R2={fits['nlogn']['R2']:.4f}")


if __name__ == "__main__":
    tests = [
        test_pyramid_roundtrip,
        test_geometric_decay,
        test_attention_shapes,
        test_vectorised_no_nan,
        test_torch_forward,
        test_torch_gradient_flows,
        test_sparsity_structure,
        test_fan_out_improves_quality,
        test_quality_metrics,
        test_scaling_exponent,
        test_model_fit,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'All tests passed!' if failed == 0 else f'{failed} test(s) FAILED'}")
    sys.exit(failed)
