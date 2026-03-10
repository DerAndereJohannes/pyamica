"""
test_checkpoint.py - tests for AMICA checkpointing in AmicaICA.fit().

Covers:
  - Checkpoint file is written after fit
  - Resume: 50 + 50 iters bit-exact with 100 iters from the same seed
  - Already-complete checkpoint skips the EM loop (n_iter_ == max_iter)
"""
from __future__ import annotations

import mne
import pytest
import torch

from pyamica import AmicaICA

mne.set_log_level("WARNING")


def test_checkpoint_file_created(synthetic_raw, tmp_path):
    ckpt = tmp_path / "ckpt.npz"
    ica = AmicaICA(max_iter=20, checkpoint_every=20, checkpoint_path=str(ckpt))
    ica.fit(synthetic_raw, picks="eeg")
    assert ckpt.exists(), "Checkpoint file was not created"


def test_resume_bitexact(synthetic_raw, tmp_path):
    """
    50 iters + resume 50 iters should give the same W_ as 100 iters,
    provided the random seed is identical at the start.
    """
    ckpt = tmp_path / "ckpt.npz"

    # Reference: 100 iters straight
    torch.manual_seed(0)
    ica_ref = AmicaICA(n_models=1, max_iter=100)
    ica_ref.fit(synthetic_raw, picks="eeg")

    # Phase 1: 50 iters, save checkpoint
    torch.manual_seed(0)
    ica_p1 = AmicaICA(n_models=1, max_iter=50,
                      checkpoint_every=50, checkpoint_path=str(ckpt))
    ica_p1.fit(synthetic_raw, picks="eeg")
    assert ckpt.exists()

    # Phase 2: resume and run to 100 iters
    ica_p2 = AmicaICA(n_models=1, max_iter=100,
                      checkpoint_every=100, checkpoint_path=str(ckpt))
    ica_p2.fit(synthetic_raw, picks="eeg")

    err = (ica_p2._model.W_ - ica_ref._model.W_).abs().max().item()
    assert err == 0.0, f"Resumed W_ max|err| = {err:.3e} (want 0)"


def test_already_complete_skips_loop(synthetic_raw, tmp_path):
    """
    If a checkpoint exists at iter == max_iter, fit() should skip the EM
    loop entirely and return immediately with n_iter_ == max_iter.
    """
    ckpt = tmp_path / "ckpt.npz"
    max_iter = 30

    # First run: fit fully, save at max_iter
    ica1 = AmicaICA(n_models=1, max_iter=max_iter,
                    checkpoint_every=max_iter, checkpoint_path=str(ckpt))
    ica1.fit(synthetic_raw, picks="eeg")
    assert ica1._model.n_iter_ == max_iter

    # Second run: same params + same checkpoint → should skip
    ica2 = AmicaICA(n_models=1, max_iter=max_iter,
                    checkpoint_every=max_iter, checkpoint_path=str(ckpt))
    ica2.fit(synthetic_raw, picks="eeg")

    assert ica2._model.n_iter_ == max_iter, (
        f"Expected n_iter_={max_iter}, got {ica2._model.n_iter_}"
    )
    # Weights should be identical (same checkpoint)
    err = (ica2._model.W_ - ica1._model.W_).abs().max().item()
    assert err == 0.0, f"Already-complete W_ max|err| = {err:.3e}"


def test_posteriors_recomputed_after_skip(synthetic_raw, tmp_path):
    """posteriors_ should be available even when the EM loop is skipped."""
    ckpt = tmp_path / "ckpt.npz"
    max_iter = 20

    ica1 = AmicaICA(n_models=2, max_iter=max_iter,
                    checkpoint_every=max_iter, checkpoint_path=str(ckpt))
    ica1.fit(synthetic_raw, picks="eeg")

    ica2 = AmicaICA(n_models=2, max_iter=max_iter,
                    checkpoint_every=max_iter, checkpoint_path=str(ckpt))
    ica2.fit(synthetic_raw, picks="eeg")

    assert ica2._model.posteriors_ is not None
    assert ica2._model.posteriors_.shape[0] == 2
