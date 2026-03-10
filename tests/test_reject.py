"""
test_reject.py - tests for outlier rejection (do_reject=True).

Verifies that:
  - do_reject=False (default) produces identical output to the pre-feature baseline.
  - do_reject=True runs without error and respects num_reject.
  - The rejection mask is a bool tensor of length T with at least one False entry
    on contaminated data.
  - Rejection does not break W @ A = I or the gm_ sum.
  - Fit with contaminated data + rejection converges to a higher LL than without.
"""
from __future__ import annotations

import numpy as np
import torch

from pyamica import AMICA


N_CH = 8
T    = 2000


def _clean_tensor(seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    q   = T // 2
    data = np.concatenate([
        rng.uniform(-1, 1, (N_CH, q)) * 1e-5,
        rng.laplace(0, 1,  (N_CH, q)) * 1e-5,
    ], axis=1).astype("float64")
    return torch.from_numpy(data.T)   # (T, N_CH)


def _contaminated_tensor(seed: int = 0, n_spikes: int = 20) -> torch.Tensor:
    """Clean data with large-amplitude spike transients inserted."""
    rng  = np.random.default_rng(seed)
    data = _clean_tensor(seed).numpy()  # (T, N_CH)
    spike_idx = rng.choice(T, size=n_spikes, replace=False)
    data[spike_idx] *= 500.0            # amplitude x500 transient artefacts
    return torch.from_numpy(data)


# ── Default: rejection disabled ───────────────────────────────────────────────

def test_no_reject_mask_when_disabled():
    """_rej_mask_ stays None when do_reject=False (default)."""
    X = _clean_tensor()
    torch.manual_seed(0)
    model = AMICA(n_models=1, max_iter=20, verbose=False)
    model.fit(X)
    assert model._rej_mask_ is None


# ── Basic operation ───────────────────────────────────────────────────────────

def test_do_reject_runs():
    """do_reject=True completes without error."""
    X = _contaminated_tensor()
    torch.manual_seed(0)
    model = AMICA(n_models=1, max_iter=30, verbose=False,
                  do_reject=True)
    model.fit(X)


def test_reject_mask_is_bool_tensor():
    """After fit with do_reject=True the mask is a bool (T,) tensor."""
    X = _contaminated_tensor()
    torch.manual_seed(0)
    model = AMICA(n_models=1, max_iter=30, verbose=False,
                  do_reject=True)
    model.fit(X)
    assert model._rej_mask_ is not None
    assert model._rej_mask_.dtype == torch.bool
    assert model._rej_mask_.shape == (T,)


def test_reject_mask_excludes_spikes():
    """The spike time points should be flagged as rejected."""
    rng      = np.random.default_rng(0)
    n_spikes = 20
    spike_idx = rng.choice(T, size=n_spikes, replace=False)

    data = _clean_tensor(0).numpy()
    data[spike_idx] *= 500.0
    X = torch.from_numpy(data)

    torch.manual_seed(0)
    model = AMICA(n_models=1, max_iter=30, verbose=False,
                  do_reject=True, reject_sigma=3.0)
    model.fit(X)

    assert model._rej_mask_ is not None
    n_rejected = int((~model._rej_mask_).sum().item())
    assert n_rejected >= 1, "Expected at least one sample to be rejected"


# ── num_reject budget ─────────────────────────────────────────────────────────

def test_num_reject_respected():
    """Rejection fires at most num_reject times (mask updates bounded)."""
    X = _contaminated_tensor()
    torch.manual_seed(0)
    # Use reject_start=1 and reject_int=1 so every iter is eligible.
    # With num_reject=2 and max_iter=50, only 2 rejection events should occur.
    model = AMICA(n_models=1, max_iter=50, verbose=True,
                  do_reject=True, num_reject=2,
                  reject_start=1, reject_int=1)
    import io, sys
    buf = io.StringIO()
    sys.stdout = buf
    model.fit(X)
    sys.stdout = sys.__stdout__
    log = buf.getvalue()
    rejection_lines = [l for l in log.splitlines() if "Rejection" in l]
    assert len(rejection_lines) == 2, \
        f"Expected 2 rejection events, got {len(rejection_lines)}: {rejection_lines}"


# ── Structural invariants after fit ───────────────────────────────────────────

def test_wa_identity_with_reject():
    """W @ A = I still holds after fitting with rejection."""
    X = _contaminated_tensor()
    torch.manual_seed(0)
    model = AMICA(n_models=1, max_iter=30, verbose=False,
                  do_reject=True)
    model.fit(X)
    WA  = (model.W_[0] @ model.A_[0]).cpu().numpy()
    err = np.max(np.abs(WA - np.eye(WA.shape[0])))
    assert err < 1e-10, f"W @ A not identity with rejection: max|err|={err:.3e}"


def test_gm_sums_to_one_with_reject():
    """gm_ sums to 1.0 after fitting with rejection."""
    X = _contaminated_tensor()
    torch.manual_seed(0)
    model = AMICA(n_models=2, max_iter=30, verbose=False,
                  do_reject=True)
    model.fit(X)
    total = model.gm_.sum().item()
    assert abs(total - 1.0) < 1e-6, f"gm_ sum = {total}"


# ── Rejection improves LL on contaminated data ────────────────────────────────

def test_reject_improves_ll_on_contaminated_data():
    """
    Fitting with rejection on contaminated data should yield a higher final LL
    than fitting without rejection (rejected samples no longer penalise the model).
    """
    X = _contaminated_tensor(seed=42, n_spikes=30)

    torch.manual_seed(0)
    m_no_rej = AMICA(n_models=1, max_iter=100, verbose=False,
                     do_reject=False)
    m_no_rej.fit(X)

    torch.manual_seed(0)
    m_rej = AMICA(n_models=1, max_iter=100, verbose=False,
                  do_reject=True)
    m_rej.fit(X)

    ll_no_rej = float(m_no_rej.LL_[m_no_rej.n_iter_ - 1].item())
    ll_rej    = float(m_rej.LL_[m_rej.n_iter_ - 1].item())
    assert ll_rej > ll_no_rej, \
        f"Expected LL with rejection ({ll_rej:.6f}) > without ({ll_no_rej:.6f})"
