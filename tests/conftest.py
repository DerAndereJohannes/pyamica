"""
conftest.py - shared pytest fixtures for pyamica tests.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_mne_raw(data: np.ndarray, sfreq: float = 250.0):
    import mne
    n_ch = data.shape[0]
    info = mne.create_info(
        [f"EEG{i:03d}" for i in range(n_ch)], sfreq=sfreq, ch_types="eeg"
    )
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_raw(rng):
    """8-channel, 2000-sample, 250 Hz Raw with two data segments (uniform | Laplace)."""
    n_ch, T = 8, 2000
    q = T // 2
    data = np.concatenate([
        rng.uniform(-1, 1, (n_ch, q)) * 1e-5,
        rng.laplace(0, 1,  (n_ch, q)) * 1e-5,
    ], axis=1).astype("float64")
    return _make_mne_raw(data)


@pytest.fixture(scope="session")
def synthetic_raw_3reg(rng):
    """8-channel, 3000-sample Raw with three data segments (uniform | Laplace | Gauss)."""
    n_ch, T = 8, 3000
    q = T // 3
    data = np.concatenate([
        rng.uniform(-1, 1, (n_ch, q)) * 1e-5,
        rng.laplace(0, 1,  (n_ch, q)) * 1e-5,
        rng.normal(0, 1,   (n_ch, q)) * 1e-6,   # much smaller → distinct statistics
    ], axis=1).astype("float64")
    return _make_mne_raw(data)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def memorize_data():
    """
    Load Memorize.fdt as a (T, N_CH) float64 torch tensor.
    Skips if the file is absent.
    """
    fdt = DATA_DIR / "Memorize.fdt"
    if not fdt.exists():
        pytest.skip(f"Memorize.fdt not found at {fdt}")
    N_CH, T = 71, 319_500
    X = np.fromfile(fdt, dtype="float32").reshape(T, N_CH)
    return torch.from_numpy(X.astype("float64"))


@pytest.fixture(scope="session")
def fortran_binary() -> Path:
    """Path to amica15ub; skips if absent."""
    bin_path = DATA_DIR / "amica15ub"
    if not bin_path.exists():
        pytest.skip(f"amica15ub not found at {bin_path}")
    return bin_path
