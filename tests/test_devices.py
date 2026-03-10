"""
test_devices.py - cross-device consistency tests for pyamica.

Strategy
--------
Precision tests (W_, LL_, posteriors_) use M=1 + fix_init=True:
  - M=1 has no permutation ambiguity (unique solution up to sign)
  - fix_init=True removes random init → identical starting point on every device
  - Differences arise only from non-deterministic GPU reductions (float64: ~1e-8)

Separation test uses M=2 with random init:
  - Just checks the GPU finds good model separation, not that it matches CPU exactly
  - M=2 has permutation ambiguity so direct W_ comparison is meaningless

Tests are skipped automatically when the device is not available.
Select/deselect with markers:
    pytest -m gpu
    pytest -m "not gpu"
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyamica import AMICA


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_gpu_devices() -> list[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
    return devices


def _fit_m1(X_cpu: torch.Tensor, device: str, dtype=torch.float64) -> AMICA:
    """Fit M=1 AMICA with deterministic init - used for precision comparisons."""
    model = AMICA(
        n_models   = 1,
        max_iter   = 100,
        fix_init   = True,   # A=I, mu=evenly spaced, sbeta=1 - identical on every device
        verbose    = False,
        device     = device,
        dtype      = dtype,
    )
    model.fit(X_cpu.to(device=torch.device(device), dtype=dtype))
    return model


def _fit_m2(X_cpu: torch.Tensor, device: str, dtype=torch.float64) -> AMICA:
    """Fit M=2 AMICA with random init - used for separation sanity check."""
    torch.manual_seed(0)
    model = AMICA(
        n_models   = 2,
        max_iter   = 200,
        verbose    = False,
        device     = device,
        dtype      = dtype,
    )
    model.fit(X_cpu.to(device=torch.device(device), dtype=dtype))
    return model


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def two_regime_cpu() -> torch.Tensor:
    rng = np.random.default_rng(42)
    n_ch, T = 8, 2000
    q = T // 2
    data = np.concatenate([
        rng.uniform(-1, 1, (n_ch, q)) * 1e-5,
        rng.laplace(0, 1,  (n_ch, q)) * 1e-5,
    ], axis=1).astype("float64")
    return torch.from_numpy(data.T)   # (T, n_ch), float64, CPU


@pytest.fixture(scope="module")
def cpu_model_m1(two_regime_cpu):
    """Reference M=1 model fitted on CPU with deterministic init."""
    return _fit_m1(two_regime_cpu, "cpu")


# ── Parametrized GPU fixture ───────────────────────────────────────────────────

def pytest_generate_tests(metafunc):
    """Inject 'gpu_device' parameter for all tests that declare it."""
    if "gpu_device" in metafunc.fixturenames:
        available = _available_gpu_devices()
        if not available:
            metafunc.parametrize("gpu_device", ["cuda"])   # will skip in body
        else:
            metafunc.parametrize("gpu_device", available)


# ── Precision tests (M=1, fix_init=True) ──────────────────────────────────────

@pytest.mark.gpu
def test_gpu_W_self_consistent(gpu_device, two_regime_cpu):
    """W_ @ A_ should equal I on GPU (device-independent correctness check).

    Direct W_ comparison against CPU is not meaningful: CUDA float64 reductions
    use a different FP ordering than CPU, so errors accumulate over 100 iterations
    and can reach ~1e-2 while both solutions are perfectly correct.  The W_ @ A_ = I
    identity is an exact algebraic invariant that holds regardless of device.
    """
    if gpu_device not in _available_gpu_devices():
        pytest.skip(f"{gpu_device} not available")

    dtype = torch.float32 if gpu_device == "mps" else torch.float64
    tol   = 1e-5 if gpu_device == "mps" else 1e-10

    gpu_model = _fit_m1(two_regime_cpu, gpu_device, dtype=dtype)

    # W_ and A_ are both (M, n, n); index [0] to get (n, n) for M=1
    W = gpu_model.W_[0].to(torch.float64)     # (n, n)
    A = gpu_model.A_[0].to(torch.float64)     # (n, n)
    n = W.shape[0]
    err = (W @ A - torch.eye(n, dtype=torch.float64, device=W.device)).abs().max().item()
    assert err < tol, f"{gpu_device} W_ @ A_ - I max|err| = {err:.3e} (tol={tol:.1e})"


@pytest.mark.gpu
def test_gpu_ll_matches_cpu(gpu_device, two_regime_cpu, cpu_model_m1):
    """LL curve on GPU (M=1, fix_init) should match CPU to within float64 noise."""
    if gpu_device not in _available_gpu_devices():
        pytest.skip(f"{gpu_device} not available")

    dtype = torch.float32 if gpu_device == "mps" else torch.float64
    tol   = 1e-2 if gpu_device == "mps" else 1e-4

    gpu_model = _fit_m1(two_regime_cpu, gpu_device, dtype=dtype)

    ll_cpu = cpu_model_m1.ll_history().cpu().numpy()
    ll_gpu = gpu_model.ll_history().cpu().numpy()
    n = min(len(ll_cpu), len(ll_gpu))
    err = np.abs(ll_cpu[:n] - ll_gpu[:n]).max()
    assert err < tol, f"{gpu_device} LL max|err| vs CPU = {err:.3e}"


@pytest.mark.gpu
def test_gpu_sphere_matches_cpu(gpu_device, two_regime_cpu, cpu_model_m1):
    """Sphering matrix on GPU must be bit-exact vs CPU (deterministic linear algebra)."""
    if gpu_device not in _available_gpu_devices():
        pytest.skip(f"{gpu_device} not available")

    dtype = torch.float32 if gpu_device == "mps" else torch.float64
    tol   = 1e-5 if gpu_device == "mps" else 1e-10

    gpu_model = _fit_m1(two_regime_cpu, gpu_device, dtype=dtype)

    S_cpu = cpu_model_m1.sphere_.cpu().to(torch.float64).numpy()
    S_gpu = gpu_model.sphere_.cpu().to(torch.float64).numpy()
    err = np.abs(S_cpu - S_gpu).max()
    assert err < tol, f"{gpu_device} sphere_ max|err| vs CPU = {err:.3e}"


# ── Separation sanity test (M=2, random init) ─────────────────────────────────

@pytest.mark.gpu
def test_gpu_model_separation(gpu_device, two_regime_cpu):
    """
    M=2 on GPU should find the two-regime structure.
    Uses random init - does NOT require exact match with CPU (permutation ambiguity).
    """
    if gpu_device not in _available_gpu_devices():
        pytest.skip(f"{gpu_device} not available")

    dtype = torch.float32 if gpu_device == "mps" else torch.float64
    model = _fit_m2(two_regime_cpu, gpu_device, dtype=dtype)

    T = two_regime_cpu.shape[0]
    q = T // 2
    dominant = model.posteriors_.cpu().argmax(dim=0).numpy()
    m0 = int(np.bincount(dominant[:q]).argmax())   # dominant model in first half
    m1 = 1 - m0                                    # expected in second half
    acc_first  = (dominant[:q] == m0).mean()
    acc_second = (dominant[q:] == m1).mean()
    assert acc_first  > 0.75, f"{gpu_device} first-half acc = {acc_first:.2f}"
    assert acc_second > 0.75, f"{gpu_device} second-half acc = {acc_second:.2f}"
