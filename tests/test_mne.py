"""
test_mne.py - tests for AmicaICA (MNE-Python wrapper).

Covers fitting, unmixing/mixing identity, round-trip reconstruction,
multi-model apply, dominant model lookup, and plotting method return types.
"""
from __future__ import annotations

import numpy as np
import pytest
import mne
import mne.preprocessing

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI

from pyamica import AmicaICA

mne.set_log_level("WARNING")


# ── Fit ───────────────────────────────────────────────────────────────────────

def test_fit_returns_self(synthetic_raw):
    ica = AmicaICA(max_iter=10)
    ret = ica.fit(synthetic_raw, picks="eeg")
    assert ret is ica


def test_get_mne_ica_type(synthetic_raw):
    ica = AmicaICA(max_iter=10)
    ica.fit(synthetic_raw, picks="eeg")
    mne_ica = ica.get_mne_ica()
    assert isinstance(mne_ica, mne.preprocessing.ICA)


def test_get_mne_ica_cached(synthetic_raw):
    """Repeated calls return the same object (cache hit)."""
    ica = AmicaICA(max_iter=10)
    ica.fit(synthetic_raw, picks="eeg")
    a = ica.get_mne_ica(0)
    b = ica.get_mne_ica(0)
    assert a is b


def test_model0_is_dominant(synthetic_raw):
    """After fit(), model 0 has the highest gm_ (most probable model)."""
    ica = AmicaICA(n_models=2, max_iter=30)
    ica.fit(synthetic_raw, picks="eeg")
    gm = ica._model.gm_.cpu().numpy()
    assert gm[0] >= gm[1], f"Model 0 should be dominant, got gm={gm}"


# ── Unmixing / mixing identity ────────────────────────────────────────────────

def test_unmixing_mixing_identity(synthetic_raw):
    ica = AmicaICA(max_iter=50)
    ica.fit(synthetic_raw, picks="eeg")
    mne_ica = ica.get_mne_ica()
    eye_approx = mne_ica.unmixing_matrix_ @ mne_ica.mixing_matrix_
    n = eye_approx.shape[0]
    err = np.max(np.abs(eye_approx - np.eye(n)))
    assert err < 1e-10, f"unmixing @ mixing ≠ I: max|err| = {err:.3e}"


# ── Round-trip reconstruction ─────────────────────────────────────────────────

def test_roundtrip_no_exclusions(synthetic_raw):
    """apply() with no exclusions should reconstruct the original signal."""
    ica = AmicaICA(max_iter=50)
    ica.fit(synthetic_raw, picks="eeg")

    raw_copy = synthetic_raw.copy()
    ica.apply(raw_copy)

    orig  = synthetic_raw.get_data(picks="eeg")
    recon = raw_copy.get_data(picks="eeg")
    rms_err    = np.sqrt(np.mean((orig - recon) ** 2))
    rms_signal = np.sqrt(np.mean(orig ** 2))
    rel_err = rms_err / rms_signal
    assert rel_err < 1e-10, f"Round-trip rel error = {rel_err:.3e}"


# ── PCA ───────────────────────────────────────────────────────────────────────

def test_pca_variance_descending(synthetic_raw):
    ica = AmicaICA(max_iter=10)
    ica.fit(synthetic_raw, picks="eeg")
    ev = ica.get_mne_ica().pca_explained_variance_
    assert np.all(np.diff(ev) <= 0), "pca_explained_variance_ not descending"


# ── apply() ───────────────────────────────────────────────────────────────────

def test_apply_m1_inplace(synthetic_raw):
    ica = AmicaICA(max_iter=20)
    ica.fit(synthetic_raw, picks="eeg")
    raw_copy = synthetic_raw.copy()
    original_shape = raw_copy.get_data().shape
    ica.apply(raw_copy)
    assert raw_copy.get_data().shape == original_shape


def test_apply_m2(synthetic_raw):
    """Multi-model apply should complete without error."""
    ica = AmicaICA(n_models=2, max_iter=30)
    ica.fit(synthetic_raw, picks="eeg")
    ica.get_mne_ica(0).exclude = [0]
    ica.get_mne_ica(1).exclude = [1]
    raw_copy = synthetic_raw.copy()
    ica.apply(raw_copy)  # should not raise
    assert raw_copy.get_data().shape == synthetic_raw.get_data().shape


# ── find_bads_* ───────────────────────────────────────────────────────────────

def _make_raw_with_eog(rng=None):
    """Synthetic Raw with one dedicated EOG channel."""
    import mne
    if rng is None:
        rng = np.random.default_rng(99)
    n_eeg, T, sfreq = 8, 2000, 250.0
    eeg = rng.uniform(-1, 1, (n_eeg, T)) * 1e-5
    # VEOG: slow sinusoidal blink artefact at ~0.3 Hz
    t = np.arange(T) / sfreq
    veog = (np.sin(2 * np.pi * 0.3 * t) * 1e-3).reshape(1, T)
    data = np.concatenate([eeg, veog], axis=0)
    info = mne.create_info(
        [f"EEG{i:03d}" for i in range(n_eeg)] + ["VEOG"],
        sfreq=sfreq,
        ch_types=["eeg"] * n_eeg + ["eog"],
    )
    return mne.io.RawArray(data, info, verbose=False)


def test_find_bads_eog_returns_list():
    raw = _make_raw_with_eog()
    ica = AmicaICA(max_iter=50)
    ica.fit(raw, picks="eeg")
    bads, scores = ica.find_bads_eog(raw, ch_name="VEOG")
    assert isinstance(bads, list)
    assert hasattr(scores, "__len__")


# ── Plotting (non-interactive, return type) ────────────────────────────────────

def test_plot_model_posteriors_returns_axes(synthetic_raw):
    import matplotlib.pyplot as plt
    ica = AmicaICA(n_models=2, max_iter=20)
    ica.fit(synthetic_raw, picks="eeg")
    ax = ica.plot_model_posteriors()
    assert hasattr(ax, "set_xlabel")
    plt.close("all")


def test_plot_model_dominance_returns_axes(synthetic_raw):
    import matplotlib.pyplot as plt
    ica = AmicaICA(n_models=2, max_iter=20)
    ica.fit(synthetic_raw, picks="eeg")
    ax = ica.plot_model_dominance(smooth_s=0.5)
    assert hasattr(ax, "set_xlabel")
    plt.close("all")


def test_plot_model_dominance_no_smooth(synthetic_raw):
    import matplotlib.pyplot as plt
    ica = AmicaICA(n_models=2, max_iter=20)
    ica.fit(synthetic_raw, picks="eeg")
    ax = ica.plot_model_dominance(smooth_s=0.0)
    assert ax is not None
    plt.close("all")


# ── Rank-deficient handling ───────────────────────────────────────────────────

def _make_raw(n_ch: int = 8, T: int = 2000, seed: int = 0):
    """Helper: full-rank synthetic EEG Raw (Volts)."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1e-5, (n_ch, T))
    info = mne.create_info(
        [f"EEG{i:03d}" for i in range(n_ch)], sfreq=250.0, ch_types="eeg"
    )
    return mne.io.RawArray(data, info, verbose=False)


def test_average_reference_reduces_rank():
    """Average-referenced EEG (rank n_ch - 1) results in n_components = n_ch - 1."""
    n_ch = 8
    raw = _make_raw(n_ch)
    raw.set_eeg_reference("average", verbose=False)   # applies directly to data

    ica = AmicaICA(max_iter=5, verbose=False)
    ica.fit(raw, picks="eeg")

    assert ica._model.W_.shape[1] == n_ch - 1, (
        f"Expected {n_ch - 1} components after average ref, "
        f"got {ica._model.W_.shape[1]}"
    )


def test_bad_channel_excluded_from_fit():
    """Bad channel is excluded from picks; remaining good channels fit at full rank."""
    n_ch = 8
    raw = _make_raw(n_ch)
    raw.info["bads"] = ["EEG000"]

    ica = AmicaICA(max_iter=5, verbose=False)
    ica.fit(raw, picks="eeg")

    n_good = n_ch - 1
    assert ica._model.W_.shape[1] == n_good, (
        f"Expected {n_good} components (1 bad channel excluded), "
        f"got {ica._model.W_.shape[1]}"
    )
    assert "EEG000" not in ica._ch_names


def test_full_rank_no_reduction():
    """Full-rank data with no projectors: n_components equals n_ch."""
    n_ch = 8
    raw = _make_raw(n_ch)

    ica = AmicaICA(max_iter=5, verbose=False)
    ica.fit(raw, picks="eeg")

    assert ica._model.W_.shape[1] == n_ch
    assert ica._model.sphere_.shape == (n_ch, n_ch)


def test_explicit_n_components_respected():
    """Explicit n_components is passed through; compute_rank is skipped."""
    n_ch, n_req = 8, 5
    raw = _make_raw(n_ch)

    ica = AmicaICA(n_components=n_req, max_iter=5, verbose=False)
    ica.fit(raw, picks="eeg")

    assert ica._model.W_.shape[1] == n_req


def test_unmixing_mixing_identity_rank_deficient():
    """unmixing @ mixing ≈ I for rank-deficient (average-referenced) data."""
    n_ch = 8
    raw = _make_raw(n_ch)
    raw.set_eeg_reference("average", verbose=False)

    ica = AmicaICA(max_iter=50, verbose=False)
    ica.fit(raw, picks="eeg")
    mne_ica = ica.get_mne_ica()

    n = mne_ica.n_components_
    assert n == n_ch - 1
    eye_approx = mne_ica.unmixing_matrix_ @ mne_ica.mixing_matrix_
    err = np.max(np.abs(eye_approx - np.eye(n)))
    assert err < 1e-10, f"unmixing @ mixing ≠ I for rank-deficient: max|err| = {err:.3e}"


def test_roundtrip_rank_deficient():
    """apply() with no exclusions reconstructs signal exactly for rank-deficient data.

    With average reference applied directly to the data, the signal lives entirely
    in the (n_ch - 1)-dimensional subspace spanned by V_k, so V_k V_k^T @ x = x
    and the round-trip is exact (machine precision).
    """
    n_ch = 8
    raw = _make_raw(n_ch)
    raw.set_eeg_reference("average", verbose=False)   # data now genuinely rank n_ch - 1

    ica = AmicaICA(max_iter=50, verbose=False)
    ica.fit(raw, picks="eeg")

    raw_copy = raw.copy()
    ica.apply(raw_copy)

    orig  = raw.get_data(picks="eeg")
    recon = raw_copy.get_data(picks="eeg")
    rms_err    = np.sqrt(np.mean((orig - recon) ** 2))
    rms_signal = np.sqrt(np.mean(orig ** 2))
    rel_err = rms_err / rms_signal
    assert rel_err < 1e-10, f"Round-trip rel error (rank-deficient) = {rel_err:.3e}"
