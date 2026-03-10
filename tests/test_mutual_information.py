"""
test_mutual_information.py - tests for score_mutual_information().

Key properties verified:
1. Output shape and symmetry.
2. Diagonal is zero (a signal has zero MI with itself when treated as two
   identical channels -- actually diagonal is undefined/zero by convention).
3. All values are non-negative.
4. MI between truly independent sources is lower than MI between their
   linear mixtures -- the core ICA claim.
5. Standalone function works with extended Infomax from MNE.
"""
from __future__ import annotations

import numpy as np
import pytest
import mne
import matplotlib
matplotlib.use("Agg")

from pyamica import AmicaICA, score_mutual_information

mne.set_log_level("WARNING")

CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
SFREQ    = 250.0
N_CH     = len(CH_NAMES)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _raw(data: np.ndarray) -> mne.io.RawArray:
    info = mne.create_info(CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020")
    return raw


def _independent_mix(rng, n_ch=N_CH, T=2000):
    """Return (raw_mixed, raw_sources) where sources are truly independent."""
    S = rng.laplace(0, 1, (n_ch, T))
    A = rng.standard_normal((n_ch, n_ch))
    X = A @ S
    X = X / X.std() * 1e-5
    return _raw(X), S


def _fitted_amica(raw):
    ica = AmicaICA(max_iter=50, verbose=False)
    ica.fit(raw, picks="eeg")
    return ica


# ── Output properties ──────────────────────────────────────────────────────────

def test_returns_square_symmetric_matrix():
    rng = np.random.default_rng(0)
    raw, _ = _independent_mix(rng)
    ica    = _fitted_amica(raw)
    mi     = ica.score_mutual_information(raw)
    assert mi.shape == (N_CH, N_CH), f"Expected ({N_CH},{N_CH}), got {mi.shape}"
    assert np.allclose(mi, mi.T), "MI matrix is not symmetric"


def test_diagonal_is_zero():
    rng = np.random.default_rng(1)
    raw, _ = _independent_mix(rng)
    ica    = _fitted_amica(raw)
    mi     = ica.score_mutual_information(raw)
    assert np.all(mi.diagonal() == 0.0), f"Diagonal not zero: {mi.diagonal()}"


def test_values_are_non_negative():
    rng = np.random.default_rng(2)
    raw, _ = _independent_mix(rng)
    ica    = _fitted_amica(raw)
    mi     = ica.score_mutual_information(raw)
    assert np.all(mi >= 0.0), f"Negative MI values found: {mi.min():.4f}"


def test_raises_if_not_fitted():
    rng = np.random.default_rng(3)
    raw, _ = _independent_mix(rng)
    ica    = AmicaICA()
    with pytest.raises(RuntimeError, match="fit()"):
        ica.score_mutual_information(raw)


# ── Core ICA claim ─────────────────────────────────────────────────────────────

def test_mi_lower_after_ica_than_in_mixed_data():
    """
    The mean pairwise MI between AMICA components must be lower than the mean
    pairwise MI between the original mixed channels.  This is the fundamental
    statistical independence claim of ICA.
    """
    from pyamica._mne import _mi_matrix

    rng        = np.random.default_rng(4)
    raw, _     = _independent_mix(rng, T=3000)
    ica        = _fitted_amica(raw)

    mi_sources = ica.score_mutual_information(raw)
    mi_mixed   = _mi_matrix(raw.get_data(), n_bins=30)

    # Take mean of upper triangle (excluding diagonal)
    idx        = np.triu_indices(N_CH, k=1)
    mean_src   = mi_sources[idx].mean()
    mean_mix   = mi_mixed[idx].mean()

    assert mean_src < mean_mix, (
        f"ICA should reduce MI: sources {mean_src:.4f} >= mixed {mean_mix:.4f}"
    )


def test_multi_model_idx():
    """score_mutual_information works for model_idx=1."""
    rng = np.random.default_rng(5)
    raw, _ = _independent_mix(rng)
    ica    = AmicaICA(n_models=2, max_iter=30, verbose=False)
    ica.fit(raw, picks="eeg")
    mi = ica.score_mutual_information(raw, model_idx=1)
    assert mi.shape == (N_CH, N_CH)
    assert np.all(mi >= 0.0)


# ── Standalone function ────────────────────────────────────────────────────────

def test_standalone_works_with_extended_infomax():
    """Standalone score_mutual_information() works with extended Infomax."""
    rng = np.random.default_rng(6)
    raw, _ = _independent_mix(rng)
    mne_ica = mne.preprocessing.ICA(
        n_components=N_CH, method="infomax", random_state=0,
        fit_params=dict(extended=True),
    )
    mne_ica.fit(raw, picks="eeg")

    mi = score_mutual_information(mne_ica, raw)
    assert mi.shape == (N_CH, N_CH)
    assert np.allclose(mi, mi.T)
    assert np.all(mi >= 0.0)
