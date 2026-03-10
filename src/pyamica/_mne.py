"""
pyamica._mne - MNE-Python integration for PyTorch AMICA.

Provides AmicaICA, a wrapper that fits AMICA on MNE Raw / Epochs data and
integrates with the full MNE artifact-rejection workflow.

Single-model usage (M=1)
------------------------
    ica = AmicaICA(device='cuda')
    ica.fit(raw, picks='eeg')

    ica.plot_components(inst=raw)
    ica.plot_sources(raw, block=True)   # click to mark bad components
    ica.apply(raw)                      # removes marked components in-place

Multi-model usage (M>1)
-----------------------
    ica = AmicaICA(n_models=3, device='cuda', chunk_t=32768)
    ica.fit(raw, picks='eeg')

    ica.plot_model_posteriors()         # inspect which model dominates when

    for m in range(3):
        ica.plot_sources(raw, model_idx=m, block=True)  # mark bad ICs per model

    ica.apply(raw)   # applies per-model exclusions using hard posteriors assignment
"""
from __future__ import annotations

import numpy as np
import torch

from pyamica._core import AMICA


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_picks(info, picks):
    """Return an int array of channel indices for *picks*."""
    import mne
    if isinstance(picks, str):
        _type_map = {
            'eeg':  dict(eeg=True,      meg=False),
            'meg':  dict(meg=True,      eeg=False),
            'grad': dict(meg='grad',    eeg=False),
            'mag':  dict(meg='mag',     eeg=False),
        }
        kw = _type_map.get(picks, {picks: True})
        return mne.pick_types(info, exclude='bads', **kw)
    elif isinstance(picks, (list, np.ndarray)):
        first = picks[0] if len(picks) else None
        if isinstance(first, str):
            return mne.pick_channels(info['ch_names'], picks,
                                     exclude=info['bads'])
        return np.asarray(picks, dtype=int)
    raise TypeError(f"picks must be str or list, got {type(picks).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class AmicaICA:
    """
    AMICA ICA fitted to MNE Raw or Epochs data.

    Parameters
    ----------
    n_components : int or None
        Number of ICA components. Default None (equal to the number of
        selected channels).
    n_models : int
        Number of AMICA mixture models (M). 1 = standard ICA. M > 1 learns
        separate unmixing matrices and source densities per mixture model. Default 1.
    device : str
        PyTorch device string: ``'cpu'``, ``'cuda'``, or ``'mps'``.
        Default ``'cpu'``.
    **amica_kwargs
        Passed verbatim to :class:`~pyamica.AMICA`, e.g. ``max_iter``,
        ``lrate``, ``do_newton``, ``compile``, ``time_iters``, ``chunk_t``,
        ``do_reject``, ``reject_sigma``, ``num_reject``, ``reject_start``,
        ``reject_int``.

    Notes
    -----
    Data are scaled from Volts (MNE convention) to microvolts before fitting.
    This improves numerical conditioning of the PCA/sphering step and matches
    the scale on which AMICA's default hyperparameters were designed.
    The MNE ICA object returned by :meth:`to_mne_ica` operates in Volts so it
    integrates transparently with the rest of MNE.
    """

    _SCALE: float = 1e6   # V → µV

    def __init__(self,
                 n_components: int | None = None,
                 n_models: int = 1,
                 device: str = 'cpu',
                 **amica_kwargs):
        self.n_components = n_components
        self.n_models     = n_models
        self.device       = device
        self.amica_kwargs = amica_kwargs

        # populated by fit()
        self._model:      AMICA | None      = None
        self._picks_idx:  np.ndarray | None = None
        self._ch_names:   list[str] | None  = None
        self._inst_info                     = None   # mne.Info for picked channels
        self._n_samples:  int               = 0
        self._fit_type:   str               = 'raw'  # 'raw' | 'epc'
        self._times:      np.ndarray | None = None   # (T,) timestamps of good samples (s)
        self._mne_icas:   dict              = {}     # cache: model_idx → MNE ICA

    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, inst, picks: str | list = 'eeg') -> 'AmicaICA':
        """
        Fit AMICA on the good data in *inst*.

        Bad channels (``inst.info['bads']``) and bad time segments
        (BAD annotations in Raw; rejected Epochs) are automatically excluded
        via MNE's built-in data extraction.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
        picks : str or list of str
            Channel type string ('eeg', 'meg', 'grad', 'mag') or explicit
            list of channel names.

        Returns
        -------
        self
        """
        import mne

        # ── channel selection ──────────────────────────────────────────────
        picks_idx = _resolve_picks(inst.info, picks)
        if len(picks_idx) == 0:
            raise ValueError(f"No channels selected by picks={picks!r}.")
        self._picks_idx = picks_idx
        self._ch_names  = [inst.info['ch_names'][i] for i in picks_idx]
        self._inst_info = mne.pick_info(inst.info, picks_idx)

        # ── extract good data (MNE handles bad segments / epochs) ──────────
        if isinstance(inst, mne.io.BaseRaw):
            # reject_by_annotation='omit' concatenates only the good segments
            data, times = inst.get_data(picks=picks_idx,
                                        reject_by_annotation='omit',
                                        return_times=True)          # (n_ch, T), (T,)
            self._times    = times
            self._fit_type = 'raw'
        elif isinstance(inst, mne.BaseEpochs):
            # get_data() returns only non-rejected epochs
            ep = inst.get_data(picks=picks_idx)                    # (n_ep, n_ch, n_t)
            n_ep, n_ch, n_t = ep.shape
            data = ep.transpose(0, 2, 1).reshape(n_ep * n_t, n_ch).T  # (n_ch, T)
            # unroll epoch-relative times across all epochs
            self._times    = np.tile(inst.times, n_ep)
            self._fit_type = 'epc'
        else:
            raise TypeError(
                f"inst must be mne.io.BaseRaw or mne.BaseEpochs, "
                f"got {type(inst).__name__}"
            )

        # ── V → µV (better numerical conditioning for covariance / sphere) ─
        data = data * self._SCALE                                   # (n_ch, T) µV

        # ── (n_ch, T) → (T, n_ch)  (AMICA expects time-first) ────────────
        X = torch.from_numpy(data.T.astype('float64'))             # (T, n_ch)
        self._n_samples = X.shape[0]

        # ── fit AMICA ──────────────────────────────────────────────────────
        self._mne_icas = {}   # invalidate any previously cached MNE ICA objects
        n_comp = self.n_components or data.shape[0]
        self._model = AMICA(
            n_components = n_comp,
            n_models     = self.n_models,
            device       = self.device,
            **self.amica_kwargs,
        )
        self._model.fit(X)
        return self

    # ─────────────────────────────────────────────────────────────────────────

    def plot_model_posteriors(self, ax=None, figsize=None):
        """
        Plot per-time-point model posteriors p(m|t) against recording time.

        Each line shows the probability that model m is active at each sample.
        For well-separated conditions the lines should approach 0/1 in distinct
        time segments, making it easy to verify that AMICA's learned models
        align with the known experimental conditions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot into.  If None, a new figure is created.
        figsize : tuple or None
            Figure size passed to ``plt.figure()``.  Ignored if *ax* is given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self._model is None or self._model.posteriors_ is None:
            raise RuntimeError("Call fit() before plot_model_posteriors().")
        if self._times is None:
            raise RuntimeError("No time information available. fit() may not have stored it.")

        assert self._model.gm_ is not None
        posteriors = self._model.posteriors_.cpu().numpy()   # (M, T)
        gm         = self._model.gm_.cpu().numpy()           # (M,)
        times      = self._times                             # (T,)
        M          = posteriors.shape[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (12, 3))

        colors = plt.cm.tab10.colors
        for m in range(M):
            ax.plot(times, posteriors[m],
                    color=colors[m % len(colors)],
                    lw=0.5, alpha=0.8,
                    label=f'Model {m}  (gm={gm[m]:.3f})')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('p(model | t)')
        ax.set_title('AMICA model posteriors')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')
        ax.figure.tight_layout()
        return ax

    # ─────────────────────────────────────────────────────────────────────────

    def plot_model_dominance(self, smooth_s: float = 0.0,
                             figsize=None, ax=None):
        """
        Stacked area chart of model posteriors, easier to read than
        overlapping lines when models switch rapidly.

        Because posteriors sum to 1, each model's share is shown as a filled
        band.  Rapid within-second switches appear as mixed colours rather
        than unreadable flickering.

        An optional Gaussian smooth (``smooth_s`` seconds) can be applied
        before plotting to suppress fast transients and reveal the broader
        structure of model dominance.

        Parameters
        ----------
        smooth_s : float
            Standard deviation of the Gaussian kernel (seconds).
            ``0`` (default) means no smoothing.
        ax : matplotlib.axes.Axes or None
            Axes to plot into.  If None, a new figure is created.
        figsize : tuple or None
            Figure size.  Ignored when *ax* is given.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d

        if self._model is None or self._model.posteriors_ is None:
            raise RuntimeError("Call fit() before plot_model_dominance().")
        if self._times is None:
            raise RuntimeError("No time information available.")

        assert self._model.gm_ is not None
        posteriors = self._model.posteriors_.cpu().numpy().copy()   # (M, T)
        gm         = self._model.gm_.cpu().numpy()
        times      = self._times
        M          = posteriors.shape[0]

        if smooth_s > 0:
            sfreq  = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1.0
            sigma  = smooth_s * sfreq              # samples
            posteriors = gaussian_filter1d(posteriors, sigma=sigma, axis=1)
            # re-normalise after smoothing (posteriors should already sum to ~1)
            posteriors /= posteriors.sum(axis=0, keepdims=True).clip(min=1e-30)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (12, 3))

        colors = [plt.cm.tab10.colors[m % 10] for m in range(M)]
        labels = [f'Model {m}  (gm={gm[m]:.3f})' for m in range(M)]

        ax.stackplot(times, posteriors, labels=labels, colors=colors, alpha=0.85)

        smooth_str = f', smoothed {smooth_s}s' if smooth_s > 0 else ''
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('p(model | t)')
        ax.set_title(f'AMICA model dominance{smooth_str}')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right')
        ax.figure.tight_layout()
        return ax

    # ─────────────────────────────────────────────────────────────────────────

    def get_mne_ica(self, model_idx: int = 0):
        """
        Return the cached ``mne.preprocessing.ICA`` for *model_idx*.

        The object is created on first access and cached; subsequent calls
        return the same object so that interactive exclusion selections
        (``ica.exclude``) made in ``plot_sources`` / ``plot_components``
        are preserved and picked up by ``apply()``.

        Parameters
        ----------
        model_idx : int
            0-indexed model.  Model 0 is always the most probable model after fit().
        """
        if self._model is None:
            raise RuntimeError("Call fit() before get_mne_ica().")
        if model_idx not in self._mne_icas:
            self._mne_icas[model_idx] = self.to_mne_ica(model_idx)
        return self._mne_icas[model_idx]

    # ─────────────────────────────────────────────────────────────────────────

    def plot_sources(self, inst, model_idx: int = 0, **kwargs):
        """
        Plot ICA source time-courses for *model_idx* (delegates to MNE).

        Clicking on a source in the interactive window marks it for removal;
        selections are stored on the cached MNE ICA object and read by
        ``apply()``.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
        model_idx : int
            Which AMICA model to inspect.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.plot_sources()``.
        """
        return self.get_mne_ica(model_idx).plot_sources(inst, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def plot_components(self, inst=None, model_idx: int = 0, **kwargs):
        """
        Plot ICA component topographies for *model_idx* (delegates to MNE).

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs or None
        model_idx : int
            Which AMICA model to inspect.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.plot_components()``.
        """
        return self.get_mne_ica(model_idx).plot_components(inst=inst, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def plot_properties(self, inst, picks=None, model_idx: int = 0, **kwargs):
        """
        Plot detailed component properties for *model_idx* (delegates to MNE).

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
        picks : int or list of int or None
            Component indices to plot.  None = all.
        model_idx : int
            Which AMICA model to inspect.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.plot_properties()``.
        """
        return self.get_mne_ica(model_idx).plot_properties(inst, picks=picks, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def plot_overlay(self, inst, model_idx: int = 0, **kwargs):
        """
        Plot original vs. ICA-cleaned signal for *model_idx* (delegates to MNE).

        Uses the excluded components stored on the cached MNE ICA object, so
        call this after marking components via ``plot_sources()`` to preview
        what will be removed.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.Evoked
        model_idx : int
            Which AMICA model to inspect.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.plot_overlay()``.
        """
        return self.get_mne_ica(model_idx).plot_overlay(inst, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def plot_scores(self, scores, model_idx: int = 0, **kwargs):
        """
        Plot component scores (e.g. from ``find_bads_eog``) for *model_idx*.

        Typical usage::

            eog_idx, scores = ica.find_bads_eog(raw, model_idx=m, ch_name='HEOG')
            ica.plot_scores(scores, model_idx=m)

        Parameters
        ----------
        scores : array-like
            Scores returned by ``find_bads_eog()`` or ``find_bads_ecg()``.
        model_idx : int
            Which AMICA model the scores belong to.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.plot_scores()``.
        """
        return self.get_mne_ica(model_idx).plot_scores(scores, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def find_bads_eog(self, inst, model_idx: int = 0, **kwargs):
        """
        Identify EOG-correlated components for *model_idx* (delegates to MNE).

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
        model_idx : int
            Which AMICA model to inspect.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.find_bads_eog()``
            (e.g. ``ch_name``, ``threshold``, ``measure``).

        Returns
        -------
        eog_indices : list of int
        scores : ndarray
        """
        return self.get_mne_ica(model_idx).find_bads_eog(inst, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def find_bads_ecg(self, inst, model_idx: int = 0, **kwargs):
        """
        Identify ECG-correlated components for *model_idx* (delegates to MNE).

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
        model_idx : int
            Which AMICA model to inspect.  Default 0.
        **kwargs
            Forwarded to ``mne.preprocessing.ICA.find_bads_ecg()``
            (e.g. ``ch_name``, ``threshold``, ``method``).

        Returns
        -------
        ecg_indices : list of int
        scores : ndarray
        """
        return self.get_mne_ica(model_idx).find_bads_ecg(inst, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────

    def review(self, inst, model_idx: int = 0,
               eog_ch=None, ecg_ch=None,
               eog_kwargs: dict | None = None,
               ecg_kwargs: dict | None = None) -> list[int]:
        """
        Semi-automatic component review for one model.

        Runs automated bad-component detection, prints a terminal summary,
        pre-selects flagged components, then opens ``plot_sources()`` so you
        can confirm or adjust the selection interactively before closing the
        window.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
        model_idx : int
            Which AMICA model to review.  Default 0.
        eog_ch : str or list of str or None
            EOG channel name(s) to pass to ``find_bads_eog()``.
            Pass a list to run detection separately for each channel
            (e.g. ``['HEOG', 'VEOG']``).
        ecg_ch : str or None
            ECG channel name for ``find_bads_ecg()``.
        eog_kwargs : dict or None
            Extra keyword arguments for ``find_bads_eog()``.
        ecg_kwargs : dict or None
            Extra keyword arguments for ``find_bads_ecg()``.

        Returns
        -------
        list of int
            Final component exclusion list for this model (same object as
            ``get_mne_ica(model_idx).exclude``).

        Example
        -------
        ::

            for m in range(ica.n_models):
                ica.review(raw, model_idx=m,
                           eog_ch=['HEOG', 'VEOG'], ecg_ch='ECG')
            ica.apply(raw)
        """
        mne_ica = self.get_mne_ica(model_idx)
        # flagged: label → (indices, scores_array of shape (n_comp,))
        flagged: dict[str, tuple[list[int], np.ndarray]] = {}

        # ── automated detection ───────────────────────────────────────────────
        if eog_ch is not None:
            channels = [eog_ch] if isinstance(eog_ch, str) else list(eog_ch)
            for ch in channels:
                idx, scores = mne_ica.find_bads_eog(inst, ch_name=ch,
                                                     **(eog_kwargs or {}))
                flagged[f'EOG ({ch})'] = (idx, scores)

        if ecg_ch is not None:
            idx, scores = mne_ica.find_bads_ecg(inst, ch_name=ecg_ch,
                                                 **(ecg_kwargs or {}))
            flagged[f'ECG ({ecg_ch})'] = (idx, scores)

        # ── terminal summary ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Model {model_idx} - automated detection")
        print(f"{'='*60}")

        all_flagged: set[int] = set()
        if flagged:
            for label, (indices, scores) in flagged.items():
                all_flagged.update(indices)

                # scores for flagged components
                flagged_str = '  '.join(
                    f'IC{i:03d}={scores[i]:+.2f}' for i in sorted(indices)
                )

                # highest-scoring non-flagged component as a clean reference
                ranked = np.argsort(np.abs(scores))[::-1]
                clean  = next((i for i in ranked if i not in indices), None)
                ref_str = (f'  |  best clean: IC{clean:03d}={scores[clean]:+.2f}'
                           if clean is not None else '')

                print(f"  {label:<22s}: {flagged_str or '(none)'}{ref_str}")
        else:
            print("  (no automated detection requested)")

        mne_ica.exclude = sorted(all_flagged)
        print(f"\n  Pre-selected : {mne_ica.exclude}")
        print("  Close the sources window to confirm (add/remove by clicking).")

        # ── interactive review ────────────────────────────────────────────────
        self.plot_sources(inst, model_idx=model_idx, block=True)

        print(f"  Final exclusions for model {model_idx}: {mne_ica.exclude}")
        return list(mne_ica.exclude)

    # ─────────────────────────────────────────────────────────────────────────

    def apply(self, inst) -> 'AmicaICA':
        """
        Apply ICA artifact removal in-place.

        Component exclusions are read from the cached MNE ICA objects. Set
        them interactively via ``plot_sources()`` / ``plot_components()``, or
        directly via ``get_mne_ica(m).exclude = [0, 2]``.

        For ``n_models=1`` this delegates directly to MNE's ``ICA.apply()``.
        For ``n_models>1`` a hard per-sample model assignment (argmax of
        posteriors) selects which model's unmixing and exclusion list is used
        at each time point.

        Parameters
        ----------
        inst : mne.io.BaseRaw or mne.BaseEpochs
            Modified **in-place**.

        Returns
        -------
        inst
        """
        if self._model is None:
            raise RuntimeError("Call fit() before apply().")

        M = self._model.n_models

        if M == 1:
            self.get_mne_ica(0).apply(inst)
        else:
            exclude_per_model = [list(self.get_mne_ica(m).exclude) for m in range(M)]
            self._apply_multi(inst, exclude_per_model)

        return inst

    # ─────────────────────────────────────────────────────────────────────────

    def _apply_multi(self, inst, exclude_per_model: list[list[int]]):
        """Hard-assignment multi-model apply (internal)."""
        import mne

        assert self._model is not None and self._model.posteriors_ is not None
        assert self._picks_idx is not None

        M          = self._model.n_models
        mne_icas   = [self.get_mne_ica(m) for m in range(M)]
        assignment = self._model.posteriors_.cpu().numpy().argmax(axis=0)
        picks_idx  = self._picks_idx

        if isinstance(inst, mne.io.BaseRaw):
            data = inst.get_data(picks=picks_idx)               # (n_ch, T_full)
            _, T_full = data.shape

            _, good_times = inst.get_data(picks=picks_idx[:1],
                                          reject_by_annotation='omit',
                                          return_times=True)
            good_idx = np.searchsorted(inst.times, good_times).clip(0, T_full - 1)

            full_assignment = np.zeros(T_full, dtype=int)
            full_assignment[good_idx] = assignment

            for m, mne_ica in enumerate(mne_icas):
                mask = full_assignment == m
                if not np.any(mask) or not exclude_per_model[m]:
                    continue
                mne_ica.exclude = exclude_per_model[m]
                seg_raw = mne.io.RawArray(
                    data[:, mask], mne.pick_info(inst.info, picks_idx), verbose=False
                )
                mne_ica.apply(seg_raw, verbose=False)
                data[:, mask] = seg_raw.get_data()

            inst._data[np.ix_(picks_idx, np.arange(T_full))] = data

        elif isinstance(inst, mne.BaseEpochs):
            ep_data = inst.get_data(picks=picks_idx)            # (n_ep, n_ch, n_t)
            n_ep, n_ch, n_t = ep_data.shape
            flat = ep_data.transpose(0, 2, 1).reshape(n_ep * n_t, n_ch).T

            for m, mne_ica in enumerate(mne_icas):
                mask = assignment == m
                if not np.any(mask) or not exclude_per_model[m]:
                    continue
                mne_ica.exclude = exclude_per_model[m]
                seg_raw = mne.io.RawArray(
                    flat[:, mask], mne.pick_info(inst.info, picks_idx), verbose=False
                )
                mne_ica.apply(seg_raw, verbose=False)
                flat[:, mask] = seg_raw.get_data()

            inst._data[:, picks_idx, :] = flat.T.reshape(n_ep, n_t, n_ch).transpose(0, 2, 1)

        else:
            raise TypeError(f"inst must be Raw or Epochs, got {type(inst).__name__}")

    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """
        Save the fitted model and component selections to a ``.amica`` file.

        The file is a compressed NumPy archive containing all fitted tensors,
        recording metadata, and per-model exclusion lists.  Reload with
        ``AmicaICA.load(path)``.

        Parameters
        ----------
        path : str or Path
            Output path.  ``.amica.npz`` is appended if not already present.
        """
        import json
        import pickle
        from pathlib import Path as _Path

        if self._model is None:
            raise RuntimeError("Call fit() before save().")

        path = _Path(path)
        if not str(path).endswith('.amica.npz'):
            path = _Path(str(path) + '.amica.npz')

        m      = self._model
        arrays: dict[str, np.ndarray] = {}

        # ── AMICA tensors ─────────────────────────────────────────────────
        for attr in ['W_', 'A_', 'gm_', 'mu_', 'alpha_', 'beta_', 'rho_',
                     'sbeta_', 'mean_', 'sphere_', 'pca_vecs_', 'pca_vals_',
                     'posteriors_']:
            val = getattr(m, attr, None)
            if val is not None:
                arrays[f'model_{attr}'] = val.cpu().numpy()

        if self._times is not None:
            arrays['_times'] = self._times
        if self._picks_idx is not None:
            arrays['_picks_idx'] = self._picks_idx

        # ── metadata ──────────────────────────────────────────────────────
        meta = {
            'n_components': self.n_components,
            'n_models':     self.n_models,
            'device':       str(self.device),
            'ch_names':     self._ch_names,
            'n_samples':    self._n_samples,
            'fit_type':     self._fit_type,
            'sldet':        float(m.sldet_),
            'max_iter':     int(m.max_iter),
            'exclude_per_model': {
                str(mi): list(self._mne_icas[mi].exclude)
                for mi in self._mne_icas
            },
        }
        arrays['_meta'] = np.array(json.dumps(meta))

        # ── MNE info (pickled into a uint8 array) ─────────────────────────
        if self._inst_info is not None:
            arrays['_info_pkl'] = np.frombuffer(
                pickle.dumps(self._inst_info), dtype=np.uint8
            )

        np.savez_compressed(path, **arrays)   # path already ends in .npz, no suffix added
        print(f"Saved to {path}")

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path) -> 'AmicaICA':
        """
        Load a fitted AmicaICA from a ``.amica`` file saved with ``save()``.

        All fitted tensors and per-model exclusion lists are restored.  Call
        ``apply(inst)`` afterwards to clean the original recording.

        Parameters
        ----------
        path : str or Path
            Path to the ``.amica.npz`` file.

        Returns
        -------
        AmicaICA
        """
        import json
        import pickle
        from pathlib import Path as _Path

        path = _Path(path)
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data['_meta']))

        # ── reconstruct AmicaICA ──────────────────────────────────────────
        obj              = cls.__new__(cls)
        obj.n_components = meta['n_components']
        obj.n_models     = meta['n_models']
        obj.device       = meta['device']
        obj.amica_kwargs = {}
        obj._ch_names    = meta['ch_names']
        obj._n_samples   = meta['n_samples']
        obj._fit_type    = meta['fit_type']
        obj._times       = data['_times']    if '_times'    in data else None
        obj._picks_idx   = data['_picks_idx'] if '_picks_idx' in data else None
        obj._mne_icas    = {}
        obj._inst_info   = (pickle.loads(data['_info_pkl'].tobytes())
                            if '_info_pkl' in data else None)

        # ── reconstruct minimal AMICA stub ────────────────────────────────
        # Only the attributes accessed by to_mne_ica() and apply() are needed.
        amica          = AMICA.__new__(AMICA)
        amica.n_models = meta['n_models']
        amica.max_iter = meta['max_iter']
        amica.sldet_   = float(meta['sldet'])

        device = torch.device('cpu')
        for attr in ['W_', 'A_', 'gm_', 'mu_', 'alpha_', 'beta_', 'rho_',
                     'sbeta_', 'mean_', 'sphere_', 'pca_vecs_', 'pca_vals_',
                     'posteriors_']:
            key = f'model_{attr}'
            setattr(amica, attr,
                    torch.from_numpy(data[key]).to(device) if key in data else None)

        obj._model = amica

        # ── restore exclusion lists ────────────────────────────────────────
        for mi_str, exclude in meta['exclude_per_model'].items():
            mi = int(mi_str)
            mne_ica         = obj.to_mne_ica(mi)
            mne_ica.exclude = list(exclude)
            obj._mne_icas[mi] = mne_ica

        n_comp = amica.W_.shape[-1] if amica.W_ is not None else '?'
        print(f"Loaded from {path}  "
              f"(n_models={meta['n_models']}, n_components={n_comp})")
        return obj

    # ─────────────────────────────────────────────────────────────────────────

    def to_mne_ica(self, model_idx: int = 0):
        """
        Build and return a fitted ``mne.preprocessing.ICA`` object.

        After fit(), AMICA models are sorted so that model 0 is always the most
        probable model (highest ``gm_``) and components within each model are
        ordered by decreasing variance explained.  For multi-model fits, call
        once per model:

            ica0 = amica_ica.to_mne_ica(model_idx=0)   # most probable model
            ica1 = amica_ica.to_mne_ica(model_idx=1)

        AMICA's ZCA sphere and W matrix are decomposed into MNE's internal
        PCA + ICA representation:

        * ``pca_components_``  = V.T  (orthonormal PCA eigenvectors)
        * ``unmixing_matrix_`` = W × V × diag(D⁻½)
        * ``mixing_matrix_``   = diag(D^½) × V.T × A

        where V, D come from the ZCA sphere (shared across all models) and
        A = inv(W) is AMICA's mixing matrix for the selected model.

        Parameters
        ----------
        model_idx : int
            Which AMICA model to export (0-indexed).  Default 0 (most probable
            model after fit()).

        Returns
        -------
        mne.preprocessing.ICA
            Fully populated, ready for ``plot_components()``,
            ``find_bads_eog()``, ``apply()``, ``save()`` / ``load()``.
        """
        import mne
        from mne.preprocessing import ICA as MNE_ICA

        if self._model is None or self._ch_names is None:
            raise RuntimeError("Call fit() before to_mne_ica().")

        m = self._model
        assert m.W_ is not None and m.pca_vecs_ is not None \
               and m.pca_vals_ is not None and m.mean_ is not None, \
               "AMICA model is missing fitted attributes. fit() may not have completed."

        if not (0 <= model_idx < m.n_models):
            raise ValueError(f"model_idx={model_idx} out of range for n_models={m.n_models}.")

        n_comp = int(m.W_.shape[1])    # W_ is (M, n_comp, n_comp)
        scale  = self._SCALE

        # ── Recover PCA eigenvectors / eigenvalues from stored attributes ──
        # pca_vecs_: (n_ch, n_comp) eigenvectors V  (set by _compute_sphere)
        # pca_vals_: (n_comp,) eigenvalues of cov in µV²  (descending)
        V        = m.pca_vecs_.cpu().numpy().astype('float64')     # (n_ch, n_comp)
        d_vals   = m.pca_vals_.cpu().numpy().astype('float64')     # µV²  descending

        d_invsqrt = 1.0 / np.sqrt(d_vals)                         # D^{-½}

        # ── AMICA matrices ─────────────────────────────────────────────────
        W = m.W_[model_idx].cpu().numpy().astype('float64')        # (n_comp, n_comp)
        A = m.A_[model_idx].cpu().numpy().astype('float64')        # (n_comp, n_comp) = inv(W)

        # ── MNE attribute mapping ──────────────────────────────────────────
        #
        # MNE's apply() always does:
        #   1. x_pre  = x_V / pre_whitener_          (pre-whiten: V → µV if pre_whitener_=1/scale)
        #   2. x_pca  = pca_components_ @ (x_pre - pca_mean_)
        #   3. sources = unmixing_matrix_ @ x_pca
        #   [reconstruction reverses in opposite order, multiplying by pre_whitener_ at the end]
        #
        # AMICA forward pass (µV-scale):
        #   sources = W @ sphere @ (x_µV - mean_µV)
        #           = W @ V @ diag(D⁻½) @ V.T @ (x_µV - mean_µV)
        #
        # Setting pre_whitener_ = 1/scale means x_pre = x_V * scale = x_µV.
        # Then with pca_components_ = V.T and pca_mean_ = mean_µV:
        #   x_pca  = V.T @ (x_µV - mean_µV)                             ✓
        #   unmixing_matrix_ = W @ V @ diag(D⁻½)  (no scale factor)     ✓
        #   mixing_matrix_   = diag(D^½) @ V.T @ A = (V*d_sqrt).T @ A   ✓
        #
        # Round-trip (no exclusions, full rank):
        #   pre_whitener_ * (V @ (V.T@(x_µV-mean_µV)) + mean_µV) = (1/scale)*x_µV = x_V  ✓

        n_ch             = V.shape[0]
        d_sqrt           = np.sqrt(d_vals)
        pre_whitener_    = np.full((n_ch, 1), 1.0 / scale)         # (n_ch, 1): V → µV
        pca_components_         = V.T.copy()                        # (n_comp, n_ch)
        pca_explained_variance_ = d_vals                            # (n_comp,) in µV²
        pca_mean_               = m.mean_.cpu().numpy()             # (n_ch,)   in µV

        # V * d_invsqrt: col j of V scaled by d_invsqrt[j]  =  V @ diag(D⁻½)
        unmixing_matrix_ = W @ (V * d_invsqrt)                     # (n_comp, n_comp)
        # inv(unmixing) = diag(D^½) @ V.T @ inv(W) = (V * d_sqrt).T @ A
        mixing_matrix_   = (V * d_sqrt).T @ A                      # (n_comp, n_comp)

        # ── Construct MNE ICA and populate fitted state ────────────────────
        ica = MNE_ICA(
            n_components = n_comp,
            method       = 'fastica',   # must pass validation; overwritten below
            max_iter     = m.max_iter,
        )
        ica.method = 'amica'            # honest label for repr / save

        ica.pre_whitener_           = pre_whitener_
        ica.pca_components_         = pca_components_
        ica.pca_explained_variance_ = pca_explained_variance_
        ica.pca_mean_               = pca_mean_
        ica.unmixing_matrix_        = unmixing_matrix_
        ica.mixing_matrix_          = mixing_matrix_
        ica.n_components_           = n_comp
        ica.n_pca_components_       = n_comp
        ica.n_samples_              = self._n_samples
        ica.ch_names                = list(self._ch_names)
        ica.info                    = self._inst_info
        ica.exclude                 = []
        ica.current_fit             = self._fit_type
        ica._ica_names              = [f'ICA{i:03d}' for i in range(n_comp)]
        ica.reject_                 = None
        ica.labels_                 = {}

        return ica
