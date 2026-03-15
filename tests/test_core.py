"""
test_core.py - tests for the AMICA PyTorch estimator (pyamica._core).

All tests use synthetic data and run on CPU with few iterations to stay fast.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pyamica import AMICA


N_CH = 8
T    = 2000
SFREQ = 250.0
_rng  = np.random.default_rng(0)


def _two_regime_tensor(seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    q = T // 2
    data = np.concatenate([
        rng.uniform(-1, 1, (N_CH, q)) * 1e-5,
        rng.laplace(0, 1,  (N_CH, q)) * 1e-5,
    ], axis=1).astype("float64")
    return torch.from_numpy(data.T)   # (T, n_ch)


def _three_regime_tensor(seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    q = T // 3
    data = np.concatenate([
        rng.uniform(-1, 1, (N_CH, q)) * 1e-5,
        rng.laplace(0, 1,  (N_CH, q)) * 1e-5,
        rng.normal(0, 1,   (N_CH, q)) * 1e-6,
    ], axis=1).astype("float64")
    return torch.from_numpy(data.T)   # (T, n_ch)


# ── Input validation ──────────────────────────────────────────────────────────

def test_fit_raises_on_nan():
    X = _two_regime_tensor()
    X[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        AMICA(max_iter=5, verbose=False).fit(X)


def test_fit_raises_on_inf():
    X = _two_regime_tensor()
    X[10, 3] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        AMICA(max_iter=5, verbose=False).fit(X)


# ── Basic fit ─────────────────────────────────────────────────────────────────

def test_fit_m1_runs():
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=30, verbose=False)
    model.fit(X)
    assert model.n_iter_ > 0
    assert model.ll_history().shape[0] == model.n_iter_


def test_fit_returns_self():
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=10, verbose=False)
    ret = model.fit(X)
    assert ret is model


def test_ll_increases_overall():
    """Final LL should be higher than initial LL (AMICA is a maximisation)."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=50, verbose=False)
    model.fit(X)
    ll = model.ll_history().cpu().numpy()
    assert ll[-1] > ll[0], f"LL did not increase: {ll[0]:.6f} → {ll[-1]:.6f}"


# ── Model separation ──────────────────────────────────────────────────────────

def test_model_separation_m2():
    """
    With two-segment data (uniform | Laplace), one model should dominate the
    first half and the other the second half.
    """
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=200, verbose=False)
    model.fit(X)

    p = model.posteriors_.cpu().numpy()   # (2, T)
    q = T // 2
    dominant = p.argmax(axis=0)           # (T,) - which model wins at each t

    # The model that wins in the first half should lose in the second
    m0 = int(dominant[:q].mean().round())   # dominant model in first half
    m1 = 1 - m0                             # expected dominant in second half

    acc_first  = (dominant[:q] == m0).mean()
    acc_second = (dominant[q:] == m1).mean()
    assert acc_first  > 0.80, f"First half accuracy: {acc_first:.2f}"
    assert acc_second > 0.80, f"Second half accuracy: {acc_second:.2f}"


def test_model_separation_m3():
    """Three-segment data → each third dominated by a distinct model."""
    X = _three_regime_tensor(seed=3)
    torch.manual_seed(0)
    model = AMICA(n_models=3, max_iter=300, verbose=False)
    model.fit(X)

    p = model.posteriors_.cpu().numpy()   # (3, T)
    q = T // 3
    dominant = p.argmax(axis=0)

    thirds = [dominant[:q], dominant[q:2*q], dominant[2*q:]]
    dominant_per_third = [int(np.bincount(s, minlength=3).argmax()) for s in thirds]
    # All three thirds should have different dominant models
    assert len(set(dominant_per_third)) == 3, (
        f"Expected 3 distinct dominant models, got {dominant_per_third}"
    )


# ── Transform ────────────────────────────────────────────────────────────────

def test_transform_shape():
    X = _two_regime_tensor()
    model = AMICA(n_models=2, max_iter=20, verbose=False)
    model.fit(X)
    S = model.transform(X)
    assert S.shape == (T, 2, N_CH), f"Expected (T,2,n), got {S.shape}"


def test_fit_transform_consistent():
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=20, verbose=False)
    S1 = model.fit_transform(X)
    S2 = model.transform(X)
    assert torch.allclose(S1, S2), "fit_transform and transform disagree"


def test_transform_shape_rank_deficient():
    """transform() returns correct shape when sphere_ is rectangular (rank-deficient)."""
    X = _two_regime_tensor()
    X_def = X.clone()
    X_def[:, -1] = X_def[:, 0]   # duplicate channel → rank N_CH - 1

    model = AMICA(n_models=2, max_iter=5, verbose=False)
    model.fit(X_def)

    n_keep = N_CH - 1
    S = model.transform(X_def)
    assert S.shape == (T, 2, n_keep), f"Expected (T, 2, {n_keep}), got {S.shape}"


# ── Sphering ─────────────────────────────────────────────────────────────────

def test_sphere_is_zca():
    """
    ZCA sphere S satisfies  S.T @ Cov @ S = I  (whitening property).
    Equivalently, the sphered data has identity covariance.
    """
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=5, verbose=False)
    model.fit(X)

    S = model.sphere_.cpu().numpy()          # (n_ch, n_keep)
    X_np = X.numpy()
    X_c = X_np - X_np.mean(axis=0, keepdims=True)
    Cov = (X_c.T @ X_c) / X_np.shape[0]    # (n_ch, n_ch)

    Cov_sph = S.T @ Cov @ S                 # should ≈ I
    err = np.max(np.abs(Cov_sph - np.eye(S.shape[1])))
    assert err < 1e-8, f"Sphere not whitening: max|Cov_sph - I| = {err:.3e}"


def test_pca_vals_descending():
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=5, verbose=False)
    model.fit(X)
    vals = model.pca_vals_.cpu().numpy()
    assert np.all(np.diff(vals) <= 0), "PCA eigenvalues not in descending order"


# ── Convergence controls ──────────────────────────────────────────────────────

def test_early_stop_minlrate():
    """Setting minlrate very high should force early termination."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=2000, minlrate=0.5, verbose=False)
    model.fit(X)
    assert model.n_iter_ < 2000, "Expected early termination via minlrate"


def test_no_sphere():
    """do_sphere=False should still produce a valid (increasing) LL."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=30, do_sphere=False, verbose=False)
    model.fit(X)
    ll = model.ll_history().cpu().numpy()
    assert ll[-1] > ll[0]


# ── Posteriors ────────────────────────────────────────────────────────────────

def test_posteriors_sum_to_one():
    X = _two_regime_tensor()
    model = AMICA(n_models=2, max_iter=20, verbose=False)
    model.fit(X)
    col_sums = model.posteriors_.sum(dim=0)   # (T,) - each time point
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-6), \
        f"Posteriors don't sum to 1: max|sum-1| = {(col_sums - 1).abs().max():.3e}"


def test_posteriors_shape():
    X = _two_regime_tensor()
    model = AMICA(n_models=3, max_iter=10, verbose=False)
    model.fit(X)
    assert model.posteriors_.shape == (3, T)


# ── Rank-deficient handling ───────────────────────────────────────────────────

def test_sphere_full_rank():
    """Full-rank data: sphering matrix is square (ZCA) and W has n_ch components."""
    X = _two_regime_tensor()
    model = AMICA(max_iter=5, verbose=False)
    model.fit(X)
    assert model.sphere_.shape == (N_CH, N_CH), "Expected square ZCA sphere for full-rank data"
    assert model.W_.shape == (1, N_CH, N_CH)


def test_sphere_rank_deficient_duplicate_channel():
    """One channel being a copy of another reduces effective rank by 1."""
    X = _two_regime_tensor()
    X_def = X.clone()
    X_def[:, -1] = X_def[:, 0]   # last channel = first → rank N_CH-1

    model = AMICA(max_iter=5, verbose=False)
    model.fit(X_def)

    n_keep = N_CH - 1
    assert model.sphere_.shape == (N_CH, n_keep), \
        f"Expected rectangular sphere {(N_CH, n_keep)}, got {model.sphere_.shape}"
    assert model.W_.shape == (1, n_keep, n_keep)


def test_sphere_average_reference():
    """Average-referenced data (rank n-1) is handled correctly."""
    X = _two_regime_tensor()
    X_avg = X - X.mean(dim=1, keepdim=True)   # subtract channel mean → rank N_CH-1

    model = AMICA(max_iter=5, verbose=False)
    model.fit(X_avg)

    n_keep = N_CH - 1
    assert model.sphere_.shape == (N_CH, n_keep), \
        f"Expected rectangular sphere {(N_CH, n_keep)}, got {model.sphere_.shape}"
    assert model.W_.shape == (1, n_keep, n_keep)


def test_sphere_rank_deficient_n_components_explicit():
    """Explicit n_components below full rank: shape reflects user request."""
    X = _two_regime_tensor()
    n_req = N_CH - 2

    model = AMICA(n_components=n_req, max_iter=5, verbose=False)
    model.fit(X)

    assert model.sphere_.shape == (N_CH, n_req)
    assert model.W_.shape == (1, n_req, n_req)
