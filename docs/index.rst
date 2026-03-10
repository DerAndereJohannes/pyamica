pyamica
=======

PyTorch implementation of **AMICA** (Adaptive Mixture Independent Component Analysis).

AMICA fits a mixture of ICA models to multi-channel time-series data.
Each model has its own unmixing matrix and source densities; a per-sample
posterior probability indicates which model most likely generated each
data point.
This makes it especially useful for EEG data where recording conditions
(resting state, task, artefact epochs) differ in their source statistics.

.. code-block:: python

   from pyamica import AMICA, AmicaICA

   # Low-level PyTorch API
   model = AMICA(n_models=3, device="cuda")
   model.fit(X)   # X: (T, n_channels) float64 tensor

   # MNE-Python wrapper
   ica = AmicaICA(n_models=3, device="cuda")
   ica.fit(raw, picks="eeg")
   ica.plot_model_dominance(smooth_s=1.0)
   ica.apply(raw)

Background
----------

pyamica started as a weekend project: a translation of the original Fortran
AMICA implementation (`sccn/amica <https://github.com/sccn/amica>`_) into
PyTorch, with an MNE-Python wrapper to make it usable in modern EEG pipelines.
It was also used as a practical test of LLM-assisted code translation. The
translation was supported by `Claude Sonnet 4.6 <https://www.anthropic.com>`_
with a lot of back-and-forth, and turned out good enough (I think) to publish.

All credit for the algorithm, theory, and original implementation goes to
Jason A. Palmer and the SCCN team. The canonical reference is:

   Palmer, J. A., Kreutz-Delgado, K., & Makeig, S. (2012).
   *AMICA: An Adaptive Mixture of Independent Component Analyzers with
   Shared Components*. Technical Report, UC San Diego.
   `[PDF] <https://sccn.ucsd.edu/~jason/amica_a.pdf>`_

The `AMICA introduction <https://github.com/sccn/amica/wiki/AMICA-Introduction>`_
on their wiki covers what it does, why it works, and how to interpret
multi-model fits.

Performance
-----------

The uncompiled CPU backend runs roughly 2x slower than the original Fortran
binary. With ``torch.compile``, CPU reaches near-Fortran speed in the Newton
phase (1.2x faster). CUDA is 5-7x faster than Fortran, compiled or not.
All backends converge to the same log-likelihood.

See the :doc:`benchmark` page for full results and methodology.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   quickstart
   benchmark
   _generated/examples/index
   api
