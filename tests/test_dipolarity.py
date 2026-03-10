"""
test_dipolarity.py - tests for AmicaICA.score_dipolarity().

The credibility of the metric rests on two complementary tests:

1. A mixing matrix column that is *exactly* linear in x/y channel coordinates
   (i.e. a mathematically perfect dipole) must return R2 == 1.0.  This is
   verified directly by injecting a known linear column into the MNE ICA
   mixing matrix without running AMICA at all - isolating the scoring logic
   from the ICA fitting.

2. A mixing matrix built from random (non-dipolar) columns must return scores
   clearly below 1.0, confirming that the metric is not trivially saturated.
"""
from __future__ import annotations

import numpy as np
import pytest
import mne
import matplotlib
matplotlib.use("Agg")

from pyamica import AmicaICA, score_dipolarity

mne.set_log_level("WARNING")

CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
SFREQ    = 250.0


# ── Shared helpers ────────────────────────────────────────────────────────────

def _raw_with_montage(data: np.ndarray) -> mne.io.RawArray:
    """Create a Raw with standard_1020 montage from (n_ch, T) data."""
    info = mne.create_info(CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020")
    return raw



def _fitted_ica(raw: mne.io.BaseRaw) -> AmicaICA:
    ica = AmicaICA(max_iter=30, verbose=False)
    ica.fit(raw, picks="eeg")
    return ica


# ── Error cases ───────────────────────────────────────────────────────────────

def test_raises_if_not_fitted():
    rng  = np.random.default_rng(0)
    raw  = _raw_with_montage(rng.standard_normal((8, 500)))
    ica  = AmicaICA()
    with pytest.raises(RuntimeError, match="fit()"):
        ica.score_dipolarity(raw)


def test_raises_if_no_montage():
    rng  = np.random.default_rng(0)
    data = rng.laplace(0, 1, (8, 1000)) * 1e-5
    info = mne.create_info(CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    raw_no_montage = mne.io.RawArray(data, info, verbose=False)

    ica = _fitted_ica(_raw_with_montage(data))  # fit on montaged copy
    with pytest.raises(ValueError, match="montage"):
        ica.score_dipolarity(raw_no_montage)     # score on the unset one


# ── Output properties ─────────────────────────────────────────────────────────

def test_returns_ndarray_of_correct_shape():
    rng = np.random.default_rng(1)
    raw = _raw_with_montage(rng.laplace(0, 1, (8, 1000)) * 1e-5)
    ica = _fitted_ica(raw)
    scores = ica.score_dipolarity(raw)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(CH_NAMES),)


def test_scores_bounded_between_zero_and_one():
    rng = np.random.default_rng(2)
    raw = _raw_with_montage(rng.laplace(0, 1, (8, 1000)) * 1e-5)
    ica = _fitted_ica(raw)
    scores = ica.score_dipolarity(raw)
    assert np.all(scores >= 0.0), f"Scores below 0: {scores}"
    assert np.all(scores <= 1.0), f"Scores above 1: {scores}"


# ── Metric correctness ────────────────────────────────────────────────────────

def test_perfect_dipole_scores_one():
    """
    A topomap that is exactly linear in x/y is a mathematically perfect dipole
    and must return R2 == 1.0.

    We inject this directly into the MNE ICA mixing matrix so the test is
    independent of whether AMICA actually recovers the source.  This isolates
    the scoring logic from ICA fitting quality.
    """
    rng = np.random.default_rng(3)

    data = rng.laplace(0, 1, (8, 2000)).astype("float64")
    raw  = _raw_with_montage(data)
    ica  = _fitted_ica(raw)

    # Use the same positions score_dipolarity will read from inst.info so that
    # dipole_col is exactly linear in the xy values it sees (R2 = 1.0 exactly).
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    pos3d = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    norms = np.linalg.norm(pos3d, axis=1, keepdims=True)
    xy    = pos3d[:, :2] / np.where(norms > 0, norms, 1.0)
    dipole_col = 2.5 * xy[:, 0] + 1.2 * xy[:, 1] + 0.4   # exact linear → R2 = 1

    # Inject dipole_col into sensor-space column 0 of get_components().
    # get_components()[:, j] = mixing_matrix_[:, j] @ pca_components_
    # => mixing_matrix_[:, 0] = dipole_col @ pca_components_.T  (P is orthogonal)
    mne_ica = ica.get_mne_ica(0)
    new_M    = mne_ica.mixing_matrix_.copy()
    new_M[:, 0] = dipole_col @ mne_ica.pca_components_.T
    mne_ica.mixing_matrix_ = new_M

    scores = ica.score_dipolarity(raw)
    assert scores[0] > 0.999, (
        f"Perfect linear topomap should score ~1.0, got {scores[0]:.4f}")


def test_random_topomaps_score_below_perfect_dipole():
    """
    Random mixing matrix columns have no particular spatial structure, so
    their dipolarity scores should be well below 1.0 on average.
    """
    rng = np.random.default_rng(4)

    data = rng.laplace(0, 1, (8, 2000)).astype("float64")
    raw  = _raw_with_montage(data)
    ica  = _fitted_ica(raw)

    # Use the same positions score_dipolarity will read from inst.info.
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    pos3d = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    norms = np.linalg.norm(pos3d, axis=1, keepdims=True)
    xy    = pos3d[:, :2] / np.where(norms > 0, norms, 1.0)
    dipole_col = 2.5 * xy[:, 0] + 1.2 * xy[:, 1] + 0.4

    mne_ica = ica.get_mne_ica(0)
    new_M    = mne_ica.mixing_matrix_.copy()
    new_M[:, 0] = dipole_col @ mne_ica.pca_components_.T
    mne_ica.mixing_matrix_ = new_M

    scores = ica.score_dipolarity(raw)
    assert scores[0] > 0.999, "Column 0 is a perfect dipole, should score ~1.0"
    assert scores[1:].mean() < 0.9, (
        f"Random columns should score clearly below 1.0 on average, "
        f"got mean={scores[1:].mean():.3f}")


def test_multi_model_idx():
    """score_dipolarity works for model_idx=1 in a two-model fit."""
    rng = np.random.default_rng(5)
    raw = _raw_with_montage(rng.laplace(0, 1, (8, 1000)) * 1e-5)
    ica = AmicaICA(n_models=2, max_iter=20, verbose=False)
    ica.fit(raw, picks="eeg")
    scores = ica.score_dipolarity(raw, model_idx=1)
    assert scores.shape == (len(CH_NAMES),)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ── Standalone function with other MNE ICA methods ────────────────────────────

@pytest.mark.parametrize("extended", [False, True], ids=["infomax", "extended_infomax"])
def test_standalone_score_dipolarity_mne_methods(extended):
    """score_dipolarity() works with infomax and extended infomax."""
    rng  = np.random.default_rng(6)
    raw  = _raw_with_montage(rng.laplace(0, 1, (8, 1000)) * 1e-5)
    mne_ica = mne.preprocessing.ICA(
        n_components=8, method="infomax", random_state=0,
        fit_params=dict(extended=extended),
    )
    mne_ica.fit(raw, picks="eeg")

    scores = score_dipolarity(mne_ica, raw)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (8,)
    assert np.all(scores >= 0.0), f"Scores below 0: {scores}"
    assert np.all(scores <= 1.0), f"Scores above 1: {scores}"


def test_standalone_perfect_dipole_scores_one():
    """Standalone score_dipolarity() gives R2=1.0 for a linear topomap (infomax)."""
    rng = np.random.default_rng(7)
    raw = _raw_with_montage(rng.laplace(0, 1, (8, 2000)) * 1e-5)

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    pos3d = np.array([raw.info["chs"][i]["loc"][:3] for i in picks])
    norms = np.linalg.norm(pos3d, axis=1, keepdims=True)
    xy    = pos3d[:, :2] / np.where(norms > 0, norms, 1.0)
    dipole_col = 2.5 * xy[:, 0] + 1.2 * xy[:, 1] + 0.4

    mne_ica = mne.preprocessing.ICA(n_components=8, method="infomax", random_state=0)
    mne_ica.fit(raw, picks="eeg")

    new_M = mne_ica.mixing_matrix_.copy()
    new_M[:, 0] = dipole_col @ mne_ica.pca_components_.T
    mne_ica.mixing_matrix_ = new_M

    scores = score_dipolarity(mne_ica, raw)
    assert scores[0] > 0.999, f"Perfect dipole should score ~1.0, got {scores[0]:.4f}"


def test_standalone_raises_if_no_montage():
    """Standalone score_dipolarity() raises ValueError when no montage is set."""
    rng  = np.random.default_rng(8)
    data = rng.laplace(0, 1, (8, 1000)) * 1e-5
    info = mne.create_info(CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    raw_no_montage = mne.io.RawArray(data, info, verbose=False)

    raw_with = _raw_with_montage(data)
    mne_ica  = mne.preprocessing.ICA(n_components=8, method="infomax", random_state=0)
    mne_ica.fit(raw_with, picks="eeg")

    with pytest.raises(ValueError, match="montage"):
        score_dipolarity(mne_ica, raw_no_montage)
