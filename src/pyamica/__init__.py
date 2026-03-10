"""
pyamica - PyTorch AMICA: Adaptive Mixture Independent Component Analysis

Public API
----------
AMICA    : core PyTorch estimator (no MNE dependency)
AmicaICA : MNE-Python wrapper with plot/review/apply helpers (requires mne)

Example
-------
    from pyamica import AMICA, AmicaICA
"""
from pyamica._core import AMICA
from pyamica._mne import AmicaICA

__all__ = ["AMICA", "AmicaICA"]
from pyamica._version import __version__
