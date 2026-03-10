"""
test_save_load.py - tests for AmicaICA.save() / load() round-trip.

Verifies that model weights, posteriors, exclusion lists, metadata, and
apply() output are all bit-exact after a save/load cycle.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import mne

from pyamica import AmicaICA

mne.set_log_level("WARNING")


# ── Fixture: fitted M=2 model with exclusions ─────────────────────────────────

@pytest.fixture(scope="module")
def fitted_pair(synthetic_raw):
    """Returns (ica_orig, raw) - M=2, 50 iters, exclusions set manually."""
    ica = AmicaICA(n_models=2, max_iter=50)
    ica.fit(synthetic_raw, picks="eeg")
    ica.get_mne_ica(0).exclude = [0, 2]
    ica.get_mne_ica(1).exclude = [1]
    return ica, synthetic_raw


@pytest.fixture(scope="module")
def roundtrip(fitted_pair, tmp_path_factory):
    """Save and reload the fitted model; return (orig, loaded, raw)."""
    ica, raw = fitted_pair
    tmp = tmp_path_factory.mktemp("save_load")
    save_path = tmp / "model"
    ica.save(save_path)
    ica2 = AmicaICA.load(save_path.with_suffix("").with_suffix(".amica.npz"))
    return ica, ica2, raw


# ── Weight checks ─────────────────────────────────────────────────────────────

def test_W_bitexact(roundtrip):
    ica, ica2, _ = roundtrip
    err = (ica2._model.W_ - ica._model.W_).abs().max().item()
    assert err == 0.0, f"W_ max|err| = {err:.3e}"


def test_A_bitexact(roundtrip):
    ica, ica2, _ = roundtrip
    err = (ica2._model.A_ - ica._model.A_).abs().max().item()
    assert err == 0.0, f"A_ max|err| = {err:.3e}"


def test_gm_bitexact(roundtrip):
    ica, ica2, _ = roundtrip
    err = (ica2._model.gm_ - ica._model.gm_).abs().max().item()
    assert err == 0.0, f"gm_ max|err| = {err:.3e}"


def test_posteriors_bitexact(roundtrip):
    ica, ica2, _ = roundtrip
    err = (ica2._model.posteriors_ - ica._model.posteriors_).abs().max().item()
    assert err == 0.0, f"posteriors_ max|err| = {err:.3e}"


# ── Exclusion lists ───────────────────────────────────────────────────────────

def test_exclusions_model0(roundtrip):
    _, ica2, _ = roundtrip
    assert ica2.get_mne_ica(0).exclude == [0, 2]


def test_exclusions_model1(roundtrip):
    _, ica2, _ = roundtrip
    assert ica2.get_mne_ica(1).exclude == [1]


# ── Metadata ─────────────────────────────────────────────────────────────────

def test_ch_names_preserved(roundtrip):
    ica, ica2, _ = roundtrip
    assert ica2._ch_names == ica._ch_names


def test_n_samples_preserved(roundtrip):
    ica, ica2, _ = roundtrip
    assert ica2._n_samples == ica._n_samples


def test_fit_type_preserved(roundtrip):
    ica, ica2, _ = roundtrip
    assert ica2._fit_type == ica._fit_type


def test_sldet_preserved(roundtrip):
    ica, ica2, _ = roundtrip
    assert ica2._model.sldet_ == ica._model.sldet_


# ── apply() output ────────────────────────────────────────────────────────────

def test_apply_bitexact(roundtrip):
    ica, ica2, raw = roundtrip
    # apply with no exclusions to test reconstruction identity
    raw_ref  = raw.copy()
    raw_load = raw.copy()
    ica.get_mne_ica(0).exclude  = []
    ica.get_mne_ica(1).exclude  = []
    ica2.get_mne_ica(0).exclude = []
    ica2.get_mne_ica(1).exclude = []
    ica.apply(raw_ref)
    ica2.apply(raw_load)
    err = np.max(np.abs(
        raw_ref.get_data(picks="eeg") - raw_load.get_data(picks="eeg")
    ))
    assert err == 0.0, f"apply() max|err| = {err:.3e}"


# ── Path / extension handling ─────────────────────────────────────────────────

def test_path_gets_extension(synthetic_raw, tmp_path):
    ica = AmicaICA(max_iter=10)
    ica.fit(synthetic_raw, picks="eeg")
    # Pass path WITHOUT .amica.npz extension
    ica.save(tmp_path / "model_noext")
    assert (tmp_path / "model_noext.amica.npz").exists()


def test_path_already_has_extension(synthetic_raw, tmp_path):
    ica = AmicaICA(max_iter=10)
    ica.fit(synthetic_raw, picks="eeg")
    ica.save(tmp_path / "model_full.amica.npz")
    # Should not create model_full.amica.npz.amica.npz
    assert (tmp_path / "model_full.amica.npz").exists()
    assert not (tmp_path / "model_full.amica.npz.amica.npz").exists()
