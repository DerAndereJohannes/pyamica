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
from pyamica._mne import AmicaICA, score_dipolarity, score_mutual_information

__all__ = ["AMICA", "AmicaICA", "score_dipolarity", "score_mutual_information"]
from pyamica._version import __version__
