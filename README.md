# pyamica

[![PyPI versions](https://img.shields.io/pypi/pyversions/pyamica.svg?logo=python&logoColor=FFE873)](https://pypi.python.org/pypi/pyamica)
[![PyPI version](https://img.shields.io/pypi/v/pyamica.svg?logo=pypi&logoColor=FFE873)](https://pypi.python.org/pypi/pyamica)
[![Tests](https://github.com/DerAndereJohannes/pyamica/actions/workflows/test.yml/badge.svg)](https://github.com/DerAndereJohannes/pyamica/actions/workflows/test.yml)

PyTorch implementation of **AMICA** (Adaptive Mixture Independent Component Analysis).

AMICA fits a mixture of ICA models to multi-channel time-series data. Each model has its own
unmixing matrix and source densities; a per-sample posterior indicates which model most likely
generated each data point. This makes it well-suited for EEG recordings where different
experimental conditions have different source statistics.

## Install

```bash
pip install pyamica            # core (no MNE)
pip install "pyamica[mne]"     # with MNE-Python integration
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add pyamica
uv add "pyamica[mne]"
```

## Quick start

```python
import torch
from pyamica import AMICA, AmicaICA

# Low-level PyTorch API - X shape: (T, n_channels), float64
model = AMICA(n_models=2, max_iter=200, device="cuda")
model.fit(X)
print(model.gm_)           # global model weights
print(model.posteriors_)   # (n_models, T) per-sample posteriors

# MNE-Python wrapper
ica = AmicaICA(n_models=2, max_iter=200, device="cuda")
ica.fit(raw, picks="eeg")
ica.plot_model_dominance(smooth_s=1.0)
for m in range(ica.n_models):
    ica.review(raw, model_idx=m, eog_ch=["HEOG", "VEOG"], ecg_ch="ECG")
ica.apply(raw)
```

## Background

pyamica started as a weekend project: a translation of the original Fortran
AMICA implementation ([sccn/amica](https://github.com/sccn/amica)) into
PyTorch, with an MNE-Python wrapper to make it usable in modern EEG pipelines.
It was also used as a practical test of LLM-assisted code translation. The
translation was supported by Claude Sonnet 4.6 with a lot of back-and-forth,
and turned out good enough (I think) to publish.

All credit for the algorithm, theory, and original implementation goes to
Jason A. Palmer and the SCCN team. The canonical reference is:

> Palmer, J. A., Kreutz-Delgado, K., & Makeig, S. (2012).
> *AMICA: An Adaptive Mixture of Independent Component Analyzers with Shared Components*.
> Technical Report, UC San Diego.
> [[PDF]](https://sccn.ucsd.edu/~jason/amica_a.pdf)

The [AMICA introduction](https://github.com/sccn/amica/wiki/AMICA-Introduction)
on their wiki covers what it does, why it works, and how to interpret multi-model fits.

## Development install

```bash
git clone https://github.com/DerAndereJohannes/pyamica
cd pyamica/python-package
pip install -e ".[mne,dev]"
```

With uv:

```bash
git clone https://github.com/DerAndereJohannes/pyamica
cd pyamica/python-package
uv sync --extra mne --extra dev
```

## Tests

```bash
# fast tests (synthetic data only)
pytest tests/ -v -m "not slow and not gpu"

# with coverage
pytest tests/ -v -m "not slow and not gpu" --cov=pyamica

# Fortran comparison (requires data/ directory)
pytest tests/ -v -m slow
```

With uv:

```bash
uv run pytest tests/ -v -m "not slow and not gpu"
```
