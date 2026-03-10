Quick Start
===========

Low-Level PyTorch API
---------------------

:class:`~pyamica.AMICA` operates directly on PyTorch tensors.
Input shape is ``(T, n_channels)``: samples x channels.

.. code-block:: python

   import torch
   from pyamica import AMICA

   # Synthetic data: 8 channels, 5000 samples
   X = torch.randn(5000, 8, dtype=torch.float64)

   model = AMICA(
       n_models=3,      # number of mixture models (M)
       max_iter=200,
       device="cpu",    # or "cuda" / "mps"
   )
   model.fit(X)

   # Log-likelihood curve
   ll = model.ll_history()       # (n_iter,)

   # Posterior probabilities: which model is dominant at each sample
   post = model.posteriors_      # (n_models, T)

   # Source activations for each model
   sources = model.transform(X)  # (T, n_models, n_components)

MNE-Python Wrapper
------------------

:class:`~pyamica.AmicaICA` wraps :class:`~pyamica.AMICA` with an MNE-compatible
interface.  It accepts :class:`mne.io.Raw` and :class:`mne.Epochs` objects.

.. code-block:: python

   import mne
   from pyamica import AmicaICA

   raw = mne.io.read_raw_fif("recording.fif", preload=True)
   raw.filter(1, 40)

   ica = AmicaICA(
       n_models=3,
       n_components=30,
       device="cuda",
   )
   ica.fit(raw, picks="eeg")

Visualising Model Dominance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Stacked area chart: which model dominates each moment
   ica.plot_model_dominance(smooth_s=1.0)

   # Line plot of raw posteriors
   ica.plot_model_posteriors()

Inspecting and Removing Artefacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Open the MNE component browser for model 0
   ica.review(raw, model_idx=0, eog_ch=["HEOG", "VEOG"], ecg_ch="ECG")

   # Automatic EOG detection
   eog_bads, scores = ica.find_bads_eog(raw, model_idx=0)
   ica.get_mne_ica(0).exclude = eog_bads

   # Remove artefacts and reconstruct (uses the dominant model per sample)
   ica.apply(raw)

Saving and Loading
------------------

.. code-block:: python

   ica.save("my_ica")           # writes my_ica.amica.npz

   from pyamica import AmicaICA
   ica2 = AmicaICA.load("my_ica.amica.npz")

Multi-Model vs Single-Model
---------------------------

With ``n_models=1``, pyamica behaves like standard ICA (equivalent to
Infomax / extended Infomax).  Increasing ``n_models`` adds mixture
models that compete to explain each time point.

.. code-block:: python

   # Standard ICA (M=1)
   model = AMICA(n_models=1, max_iter=100)

   # Mixture of three models (M=3)
   model = AMICA(n_models=3, max_iter=200)
