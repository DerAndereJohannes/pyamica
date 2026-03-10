"""
test_sort.py - Tests for _sort_outputs() and _compute_svar() in AMICA._core.

After fit(), AMICA guarantees:
  - Models are in descending gm_ order (model 0 = most probable).
  - Components within each model are in descending variance-explained order.
  - All parameter tensors (W_, A_, c_, alpha_, mu_, sbeta_, rho_, posteriors_)
    are permuted consistently: W_[m] @ A_[m] = I still holds after the sort.
"""
from __future__ import annotations

import numpy as np
import torch

from pyamica import AMICA


N_CH = 8
T    = 2000


def _two_regime_tensor(seed=0):
    rng = np.random.default_rng(seed)
    q   = T // 2
    data = np.concatenate([
        rng.uniform(-1, 1, (N_CH, q)) * 1e-5,
        rng.laplace(0, 1,  (N_CH, q)) * 1e-5,
    ], axis=1).astype("float64")
    return torch.from_numpy(data.T)   # (T, N_CH)


def _three_regime_tensor(seed=1):
    rng = np.random.default_rng(seed)
    q   = T // 3
    data = np.concatenate([
        rng.uniform(-1, 1, (N_CH, q)) * 1e-5,
        rng.laplace(0, 1,  (N_CH, q)) * 1e-5,
        rng.normal(0, 1,   (N_CH, q)) * 1e-6,
    ], axis=1).astype("float64")
    return torch.from_numpy(data.T)   # (T, N_CH)


# ── Model order ────────────────────────────────────────────────────────────────

def test_gm_descending_m2():
    """gm_ is in descending order for M=2."""
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=150, verbose=False)
    model.fit(X)
    gm = model.gm_.cpu().numpy()
    assert gm[0] >= gm[1], f"gm_ not descending: {gm}"


def test_gm_descending_m3():
    """gm_ is in strictly non-increasing order for M=3."""
    X = _three_regime_tensor(seed=3)
    torch.manual_seed(0)
    model = AMICA(n_models=3, max_iter=200, verbose=False)
    model.fit(X)
    gm = model.gm_.cpu().numpy()
    assert np.all(np.diff(gm) <= 0), f"gm_ not descending: {gm}"


def test_gm_sums_to_one():
    """gm_ still sums to 1.0 after sorting (no mass lost)."""
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=3, max_iter=100, verbose=False)
    model.fit(X)
    total = model.gm_.sum().item()
    assert abs(total - 1.0) < 1e-6, f"gm_ sum = {total}"


# ── Component order ────────────────────────────────────────────────────────────

def test_svar_descending_m1():
    """Components are in descending svar order for M=1."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=50, verbose=False)
    model.fit(X)
    svar = model._compute_svar(0).cpu().numpy()
    assert np.all(np.diff(svar) <= 0), \
        f"svar not descending for M=1: {np.round(svar, 4)}"


def test_svar_descending_m2():
    """Components are in descending svar order for both models in M=2."""
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=100, verbose=False)
    model.fit(X)
    for m in range(2):
        svar = model._compute_svar(m).cpu().numpy()
        assert np.all(np.diff(svar) <= 0), \
            f"svar not descending for model {m}: {np.round(svar, 4)}"


# ── _compute_svar formula ──────────────────────────────────────────────────────

def test_svar_positive():
    """_compute_svar returns strictly positive values (well-defined variances)."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=20, verbose=False)
    model.fit(X)
    svar = model._compute_svar(0).cpu().numpy()
    assert np.all(svar > 0), f"svar has non-positive entries: {svar}"


def test_svar_finite():
    """_compute_svar returns finite values (no NaN or Inf)."""
    X = _two_regime_tensor()
    model = AMICA(n_models=2, max_iter=30, verbose=False)
    model.fit(X)
    for m in range(2):
        svar = model._compute_svar(m).cpu().numpy()
        assert np.all(np.isfinite(svar)), \
            f"svar has non-finite entries for model {m}: {svar}"


def test_svar_shape():
    """_compute_svar returns a vector of length n_components."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=10, verbose=False)
    model.fit(X)
    svar = model._compute_svar(0)
    assert svar.shape == (N_CH,), f"Expected ({N_CH},), got {svar.shape}"


# ── Tensor consistency after sort ──────────────────────────────────────────────

def test_wa_identity_m1():
    """W_[0] @ A_[0] == I after sorting (rows of W and cols of A permuted consistently)."""
    X = _two_regime_tensor()
    model = AMICA(n_models=1, max_iter=30, verbose=False)
    model.fit(X)
    WA  = (model.W_[0] @ model.A_[0]).cpu().numpy()
    err = np.max(np.abs(WA - np.eye(WA.shape[0])))
    assert err < 1e-10, f"W@A not identity after sort: max|err|={err:.3e}"


def test_wa_identity_m2():
    """W_[m] @ A_[m] == I for both models after sorting."""
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=100, verbose=False)
    model.fit(X)
    for m in range(2):
        WA  = (model.W_[m] @ model.A_[m]).cpu().numpy()
        err = np.max(np.abs(WA - np.eye(WA.shape[0])))
        assert err < 1e-10, f"W@A not identity for model {m}: max|err|={err:.3e}"


def test_alpha_sums_to_one_after_sort():
    """alpha_[m] still sums to 1 over mixture components for each source after sort."""
    X = _two_regime_tensor()
    model = AMICA(n_models=2, max_iter=30, verbose=False)
    model.fit(X)
    for m in range(2):
        row_sums = model.alpha_[m].sum(dim=-1).cpu().numpy()   # (n,)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"alpha rows don't sum to 1 for model {m}: {row_sums}"


def test_sbeta_positive_after_sort():
    """sbeta_ (inverse scale) remains positive for all components after sort."""
    X = _two_regime_tensor()
    model = AMICA(n_models=2, max_iter=30, verbose=False)
    model.fit(X)
    for m in range(2):
        assert (model.sbeta_[m] > 0).all(), \
            f"sbeta_ has non-positive entries for model {m}"


def test_rho_in_bounds_after_sort():
    """rho_ stays within [minrho, maxrho] after sort."""
    X = _two_regime_tensor()
    model = AMICA(n_models=2, max_iter=30, verbose=False)
    model.fit(X)
    for m in range(2):
        rho = model.rho_[m].cpu().numpy()
        assert np.all(rho >= model.minrho - 1e-8) and np.all(rho <= model.maxrho + 1e-8), \
            f"rho_ out of bounds for model {m}: min={rho.min():.3f}, max={rho.max():.3f}"


# ── Posterior consistency ──────────────────────────────────────────────────────

def test_posteriors_mean_matches_gm():
    """
    Mean of posteriors_[m] over time ≈ gm_[m].

    This is an EM fixed-point property: the model weights equal the expected
    fraction of time points assigned to each model.  Also serves as a cross-check
    that posteriors_ and gm_ were sorted by the same permutation.
    """
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=200, verbose=False)
    model.fit(X)
    post_mean = model.posteriors_.mean(dim=1).cpu().numpy()   # (M,)
    gm        = model.gm_.cpu().numpy()
    assert np.max(np.abs(post_mean - gm)) < 0.05, \
        f"post_mean={np.round(post_mean,4)} vs gm_={np.round(gm,4)}"


def test_model_permutation_correctness(monkeypatch):
    """
    Directly verify the model-level permutation.

    Captures the pre-sort gm_ and posteriors_, then confirms that after
    sorting, model 0 holds exactly the values that belonged to whichever
    model had the highest gm_ before sorting, and model 1 holds the other.

    This catches bugs where gm_ is reordered but other tensors are not
    (or are reordered by a different permutation).
    """
    pre: dict = {}
    _orig = AMICA._sort_outputs

    def _capture_then_sort(self):
        pre['gm']        = self.gm_.clone()
        pre['posteriors'] = self.posteriors_.clone()
        pre['W']         = self.W_.clone()
        _orig(self)

    monkeypatch.setattr(AMICA, '_sort_outputs', _capture_then_sort)

    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=150, verbose=False)
    model.fit(X)

    gm_pre   = pre['gm'].cpu().numpy()
    gm_order = np.argsort(-gm_pre)          # e.g. [1, 0] if pre-sort model 1 was dominant

    gm_post  = model.gm_.cpu().numpy()
    post_arr = model.posteriors_.cpu().numpy()
    pre_arr  = pre['posteriors'].cpu().numpy()

    for new_idx, old_idx in enumerate(gm_order):
        assert np.isclose(gm_post[new_idx], gm_pre[old_idx], atol=1e-12), (
            f"gm_[{new_idx}] after sort = {gm_post[new_idx]:.6f}, "
            f"expected pre-sort gm_[{old_idx}] = {gm_pre[old_idx]:.6f}"
        )
        assert np.allclose(post_arr[new_idx], pre_arr[old_idx], atol=1e-12), (
            f"posteriors_[{new_idx}] after sort does not match "
            f"pre-sort posteriors_[{old_idx}]"
        )


def test_component_permutation_correctness(monkeypatch):
    """
    Directly verify the component-level permutation within each model.

    Captures the pre-sort gm_, W_, and svar for each model, then confirms
    that after sorting:
      - the rows of W_ for each post-sort model match the svar-ordered rows
        of the corresponding pre-sort model (accounting for the model-level
        permutation that was also applied).

    This catches bugs where svar is computed correctly but the permutation
    is applied to the wrong tensor axis, or applied to W_ but not A_.
    """
    pre: dict = {}
    _orig = AMICA._sort_outputs

    def _capture_then_sort(self):
        pre['gm']   = self.gm_.clone()
        pre['W']    = self.W_.clone()
        pre['svar'] = [self._compute_svar(m).clone() for m in range(self.n_models)]
        _orig(self)

    monkeypatch.setattr(AMICA, '_sort_outputs', _capture_then_sort)

    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=150, verbose=False)
    model.fit(X)

    gm_pre   = pre['gm'].cpu().numpy()
    gm_order = np.argsort(-gm_pre)          # e.g. [1, 0] if model 1 was dominant

    # For post-sort model new_m:
    #   - it came from pre-sort model gm_order[new_m]
    #   - its rows were then sorted by svar of that pre-sort model
    for new_m, old_m in enumerate(gm_order):
        svar_pre   = pre['svar'][old_m].cpu().numpy()
        W_pre_m    = pre['W'][old_m].cpu().numpy()   # rows before component sort
        W_post_m   = model.W_[new_m].cpu().numpy()   # rows after both sorts

        comp_order = np.argsort(-svar_pre)            # expected row order

        for new_comp, old_comp in enumerate(comp_order):
            assert np.allclose(W_post_m[new_comp], W_pre_m[old_comp], atol=1e-12), (
                f"Model {old_m}→{new_m}: W_[{new_comp}] after sort does not match "
                f"pre-sort W_[{old_comp}] (svar rank {new_comp})"
            )


def test_posteriors_consistent_with_sorted_params():
    """
    Re-computing posteriors from the sorted parameters reproduces posteriors_.

    Catches any case where the posteriors_ array was sorted by a different
    permutation than the model parameters (W_, A_, gm_, etc.).
    """
    X = _two_regime_tensor(seed=7)
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=100, verbose=False)
    model.fit(X)

    stored = model.posteriors_.clone()

    # Reconstruct the sphered data that the model was trained on
    Xs = (X.to(device=model.device, dtype=torch.float64) - model.mean_) @ model.sphere_

    with torch.inference_mode():
        model._compute_posteriors(Xs)

    err = (model.posteriors_ - stored).abs().max().item()
    assert err < 1e-6, \
        f"Recomputed posteriors differ from stored after sort: max|err|={err:.3e}"
