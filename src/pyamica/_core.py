"""
pyamica._core
=============
Pure-PyTorch translation of AMICA 1.7
(Adaptive Mixture Independent Component Analysis).

Reference
---------
Palmer, J.A., Makeig, S., Kreutz-Delgado, K. & Rao, B.D. (2008).
Newton method for the ICA mixture model. ICASSP 2008.

Palmer, J.A., Kreutz-Delgado, K. & Makeig, S. (2012).
AMICA: An Adaptive Mixture of Independent Component Analyzers
with Shared Components.  Tech Report, UCSD.

Original Fortran source: https://github.com/japalmer29/amica

Architectural mapping
---------------------
MPI/OpenMP loops  → vectorised ops over the T (time) dimension on GPU
Global Fortran arrays → self.A_, self.W_, self.gm_, self.alpha_, ...
Block accumulation → full-batch einsum / sum, no Python loops over dims
Natural gradient   → explicit dA = A @ (I − dWtmp/Nv), A -= lr*dA, W=inv(A)
Numerical hacks    → torch.clamp(), torch.logsumexp(), safe_log()
"""

from __future__ import annotations

import math
import time

import torch
from torch import Tensor


# ────────────────────────── tiny helpers ────────────────────────────────────

def _physical_core_count() -> int:
    """Physical (non-hyperthreaded) CPU core count via psutil."""
    import psutil
    return psutil.cpu_count(logical=False) or 1


def _safe_log(x: Tensor, eps: float = 1e-30) -> Tensor:
    """Clamp then log to avoid log(0) without touching the autograd graph."""
    return torch.log(x.clamp(min=eps))


def _slogdet(M: Tensor) -> Tensor:
    """Batched log|det| for a (B, n, n) stack of square matrices."""
    return torch.linalg.slogdet(M).logabsdet          # (B,)


# ────────────────────────── AMICA ───────────────────────────────────────────

class AMICA:
    """
    Adaptive Mixture ICA estimator (scikit-learn-style interface).

    Parameters
    ----------
    n_components : int or None
        Number of ICA components. Default None (equal to number of channels).
    n_models : int
        Number of ICA models fitted simultaneously (M). Default 1.
    n_mix : int
        Number of generalised-Gaussian mixture components per source (J).
        Default 3.
    max_iter : int
        Hard ceiling on EM iterations. Default 2000.
    lrate : float
        Initial learning rate for the natural-gradient A update. Default 0.1.
    lrate0 : float
        Maximum learning rate after the ramp-up phase. Default 0.1.
    lratefact : float
        Multiplicative reduction applied to lrate when LL decreases. Default 0.5.
    rho0 : float
        Initial GGD shape parameter (1 = Laplacian, 2 = Gaussian). Default 1.5.
    minrho : float
        Lower clamp on rho. Default 1.0.
    maxrho : float
        Upper clamp on rho. Default 2.0.
    rholrate : float
        Step size for rho gradient updates. Default 0.05.
    rholratefact : float
        Reduction factor applied to rholrate when LL decreases. Default 0.5.
    do_sphere : bool
        PCA-whiten the data before ICA. Default True.
    do_newton : bool
        Use the Newton correction for W (see Palmer 2008). Default True.
    newt_start : int
        Iteration at which to begin Newton steps. Default 50.
    newtrate : float
        Newton learning-rate ceiling. Default 1.0.
    newt_ramp : int
        Number of ramp-up iterations before reaching newtrate. Default 10.
    doscaling : bool
        Rescale A columns to unit norm after each update. Default True.
    min_dll : float
        Stop when the LL improvement is below this for ``maxincs`` consecutive
        iterations. Default 1e-9.
    min_nd : float
        Stop when gradient norm falls below this. Default 1e-6.
    use_grad_norm : bool
        Enable the gradient-norm stopping criterion. Default True.
    use_min_dll : bool
        Enable the LL-improvement stopping criterion. Default True.
    maxdecs : int
        Number of consecutive LL decreases before halving lrate. Default 3.
    maxincs : int
        Number of consecutive iterations below min_dll before stopping. Default 5.
    minlrate : float
        Stop if lrate falls below this value. Default 1e-8.
    invsigmax : float
        Upper clamp on the inverse scale parameter sbeta. Default 100.0.
    invsigmin : float
        Lower clamp on the inverse scale parameter sbeta. Default 1e-8.
    writestep : int
        Print a progress line every this many iterations. Default 100.
    verbose : bool
        Print training progress. Default True.
    dtype : torch.dtype
        Floating-point type. Default torch.float64 (matches Fortran double
        precision and is required for numerical agreement with the reference).
    device : str or torch.device
        Compute device, e.g. ``'cpu'``, ``'cuda'``, ``'mps'``. Default ``'cpu'``.
    chunk_t : int or None
        Number of time samples per E-step chunk. Limits peak memory to
        O(chunk_t * M * n * J) instead of O(T * M * n * J), and improves
        CPU cache utilisation by keeping intermediates in L3.
        ``None`` (default): auto-selected on CPU (targeting ~32 MB); on GPU
        the full dataset is processed in one pass (maximum parallelism).
        Set explicitly on GPU if peak VRAM usage is a concern - each
        ``(T, M, n, J)`` intermediate costs roughly
        ``T * M * n * J * 8`` bytes, and several are live simultaneously.
        A value of 8192 is a conservative starting point for laptops.
        or limited VRAM.
    compile : bool
        Apply ``torch.compile`` to the E-step and M-step before the EM loop.
        The first iteration incurs a JIT compilation overhead; subsequent
        iterations benefit from kernel fusion. Default False.
    time_iters : bool
        Record wall time for each EM iteration in ``iter_times_`` (list of
        floats, seconds). For CUDA, a ``synchronize()`` call is inserted so
        times reflect actual GPU work. Default False.
    checkpoint_every : int
        Save a checkpoint every this many iterations. 0 disables checkpointing.
        Default 0.
    checkpoint_path : str or None
        File path for the checkpoint. Required if ``checkpoint_every > 0``.
        If the file exists at the start of ``fit()``, training resumes from
        that checkpoint. Default None.
    fix_init : bool
        If True, skip parameter initialisation and use whatever values are
        already stored on the instance. Intended for checkpoint resumption;
        not normally needed directly. Default False.
    do_reject : bool
        Enable per-sample outlier rejection during training. When True, time
        points whose per-sample log-likelihood falls more than
        ``reject_sigma`` standard deviations below the mean are excluded from
        the EM parameter updates for the remainder of training. The rejection
        mask is recomputed up to ``num_reject`` times. Disabled by default,
        matching the Fortran default (``do_reject 0``). Useful for data with
        large transient artefacts that have not been removed beforehand.
    reject_sigma : float
        Number of standard deviations below the mean log-likelihood used as
        the rejection threshold. A sample at iteration t is excluded when
        LL_t < mean(LL) - reject_sigma * std(LL). Default 3.0.
    num_reject : int
        Maximum number of rejection events. After this many mask updates the
        rejection mask is frozen for the rest of training. Default 5.
    reject_start : int
        Iteration at which the first rejection event may occur. Default 1.
    reject_int : int
        Minimum number of iterations between consecutive rejection events.
        Default 1 (evaluate every iteration until ``num_reject`` events have
        occurred).
    """

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        n_components: int | None = None,
        n_models:     int   = 1,
        n_mix:        int   = 3,
        max_iter:     int   = 2000,
        lrate:        float = 0.1,
        lrate0:       float = 0.1,
        lratefact:    float = 0.5,
        rho0:         float = 1.5,
        minrho:       float = 1.0,
        maxrho:       float = 2.0,
        rholrate:     float = 0.05,
        rholratefact: float = 0.5,
        do_sphere:    bool  = True,
        do_newton:    bool  = True,
        newt_start:   int   = 50,
        newtrate:     float = 1.0,
        newt_ramp:    int   = 10,
        doscaling:    bool  = True,
        min_dll:      float = 1e-9,
        min_nd:       float = 1e-6,
        use_grad_norm:bool  = True,
        use_min_dll:  bool  = True,
        maxdecs:      int   = 3,
        maxincs:      int   = 5,
        minlrate:     float = 1e-8,
        invsigmax:    float = 100.0,
        invsigmin:    float = 1e-8,
        writestep:    int   = 100,
        verbose:      bool  = True,
        dtype:        torch.dtype = torch.float64,
        device:       str   = "cpu",
        chunk_t:          int | None  = None,
        compile:          bool           = False,
        time_iters:       bool           = False,
        checkpoint_every: int            = 0,
        checkpoint_path:  str | None  = None,
        fix_init:         bool           = False,
        do_reject:        bool           = False,
        reject_sigma:     float          = 3.0,
        num_reject:       int            = 5,
        reject_start:     int            = 1,
        reject_int:       int            = 1,
    ):
        self.n_components  = n_components
        self.n_models      = n_models
        self.n_mix         = n_mix
        self.max_iter      = max_iter
        self.lrate         = lrate
        self.lrate0        = lrate0
        self.lratefact     = lratefact
        self.rho0          = rho0
        self.minrho        = minrho
        self.maxrho        = maxrho
        self.rholrate      = rholrate
        self.rholratefact  = rholratefact
        self.do_sphere     = do_sphere
        self.do_newton     = do_newton
        self.newt_start    = newt_start
        self.newtrate      = newtrate
        self.newt_ramp     = newt_ramp
        self.doscaling     = doscaling
        self.min_dll       = min_dll
        self.min_nd        = min_nd
        self.use_grad_norm = use_grad_norm
        self.use_min_dll   = use_min_dll
        self.maxdecs       = maxdecs
        self.maxincs       = maxincs
        self.minlrate      = minlrate
        self.invsigmax     = invsigmax
        self.invsigmin     = invsigmin
        self.writestep     = writestep
        self.verbose       = verbose
        self.dtype         = dtype
        self.device        = torch.device(device)
        self.chunk_t           = chunk_t
        self.compile           = compile
        self.time_iters        = time_iters
        self.checkpoint_every  = checkpoint_every
        self.checkpoint_path   = checkpoint_path
        self.fix_init          = fix_init
        self.do_reject         = do_reject
        self.reject_sigma      = reject_sigma
        self.num_reject        = num_reject
        self.reject_start      = reject_start
        self.reject_int        = reject_int

        # Fitted attributes (populated by fit())
        self.mean_:   Tensor | None = None   # (n_orig,)
        self.sphere_: Tensor | None = None   # (n_orig, n)  sphering matrix S
        self.pca_vecs_: Tensor | None = None  # (n_orig, n_keep) eigenvectors V
        self.pca_vals_: Tensor | None = None  # (n_keep,) eigenvalues of cov (descending)
        self.sldet_:  float = 0.0               # log|det S|  (LL contribution)
        self.A_:      Tensor | None = None   # (M, n, n)  mixing matrices
        self.W_:      Tensor | None = None   # (M, n, n)  unmixing matrices
        self.c_:      Tensor | None = None   # (M, n)     DC bias (source space)
        self.gm_:     Tensor | None = None   # (M,)       model weights
        self.alpha_:  Tensor | None = None   # (M, n, J)  mixture weights
        self.mu_:     Tensor | None = None   # (M, n, J)  mixture means
        self.sbeta_:  Tensor | None = None   # (M, n, J)  inverse scales (1/σ)
        self.rho_:    Tensor | None = None   # (M, n, J)  shape parameters
        self.LL_:           Tensor | None = None   # (max_iter,) LL history
        self.nd_:           Tensor | None = None   # (max_iter,) gradient-norm history
        self.n_iter_:       int = 0
        self.iter_times_:   list[float] = []          # wall-time per iter (if time_iters)
        self.posteriors_:   Tensor | None = None   # (M, T) model posteriors p(m|t)
        self._rej_mask_:    Tensor | None = None   # (T,) bool; kept samples during fit

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, base_path: str, it: int,
                         lrate: float, lrate0: float,
                         newtrate: float, rholrate0: float,
                         numdecs: int, numincs: int,
                         newton_active: bool,
                         n_rej_done: int = 0) -> None:
        """Save current model state and loop variables to a .npz checkpoint.

        The file is written to ``{base_path[:-4]}_{it:06d}.npz`` so that each
        checkpoint has a unique name that encodes the iteration number.
        """
        import numpy as np
        stem = base_path[:-4] if base_path.endswith('.npz') else base_path
        path = f"{stem}_{it:06d}.npz"
        arrays: dict = {}
        for attr in ['W_', 'A_', 'gm_', 'alpha_', 'mu_', 'sbeta_', 'rho_', 'c_']:
            val = getattr(self, attr, None)
            if val is not None:
                arrays[attr] = val.cpu().numpy()
        if self._rej_mask_ is not None:
            arrays['_rej_mask_'] = self._rej_mask_.cpu().numpy()
        # LL/nd history up to current iter
        if self.LL_ is not None:
            arrays['LL_'] = self.LL_[:it].cpu().numpy()
        if self.nd_ is not None:
            arrays['nd_'] = self.nd_[:it].cpu().numpy()
        # Loop scalars as 0-d arrays
        arrays['_it']           = np.array(it)
        arrays['_lrate']        = np.array(lrate)
        arrays['_lrate0']       = np.array(lrate0)
        arrays['_newtrate']     = np.array(newtrate)
        arrays['_rholrate0']    = np.array(rholrate0)
        arrays['_numdecs']      = np.array(numdecs)
        arrays['_numincs']      = np.array(numincs)
        arrays['_newton_active']= np.array(newton_active)
        arrays['_n_rej_done']   = np.array(n_rej_done)
        np.savez_compressed(path, **arrays)

    def _load_checkpoint(self, base_path: str) -> dict | None:
        """Load the latest checkpoint matching ``base_path``.

        Scans for files named ``{base_path[:-4]}_{it:06d}.npz`` and loads the
        one with the highest iteration number.  Returns a loop state dict or
        None if no matching file is found.
        """
        import numpy as np
        import glob
        stem    = base_path[:-4] if base_path.endswith('.npz') else base_path
        matches = sorted(glob.glob(f"{stem}_*.npz"))
        if not matches:
            return None
        path = matches[-1]   # lexicographic sort puts highest iter last
        data = np.load(path)
        state: dict = {
            'it':           int(data['_it']),
            'lrate':        float(data['_lrate']),
            'lrate0':       float(data['_lrate0']),
            'newtrate':     float(data['_newtrate']),
            'rholrate0':    float(data['_rholrate0']),
            'numdecs':      int(data['_numdecs']),
            'numincs':      int(data['_numincs']),
            'newton_active':bool(data['_newton_active']),
            'n_rej_done':   int(data['_n_rej_done']) if '_n_rej_done' in data else 0,
        }
        # Store tensors as numpy arrays; applied after _init_params allocates buffers
        tensors = {}
        for attr in ['W_', 'A_', 'gm_', 'alpha_', 'mu_', 'sbeta_', 'rho_', 'c_']:
            if attr in data:
                tensors[attr] = data[attr]
        state['_tensors']   = tensors
        state['_LL']        = data['LL_']       if 'LL_'       in data else None
        state['_nd']        = data['nd_']       if 'nd_'       in data else None
        state['_rej_mask']  = data['_rej_mask_'] if '_rej_mask_' in data else None
        return state

    # ── device / dtype helper ─────────────────────────────────────────────────

    def _t(self, x: Tensor) -> Tensor:
        """Cast tensor to configured device and dtype."""
        return x.to(device=self.device, dtype=self.dtype)

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_sphere(self, X: Tensor) -> tuple[Tensor, Tensor, float, int]:
        """
        PCA-whiten the data.

        Computes sphering matrix S from the eigenvectors/eigenvalues of the
        sample covariance.  Two cases:

        Full rank (n_keep == n_orig):
            ZCA (symmetric) sphere  S = V D^{-1/2} V^T   shape (n_orig, n_orig)
            Matches Fortran do_approx_sphere=1.  Data stays in sensor space.

        Rank-deficient (n_keep < n_orig):
            PCA whitening  S = V_k D_k^{-1/2}             shape (n_orig, n_keep)
            Projects data to the n_keep-dimensional subspace, discarding the
            directions with zero (or near-zero) variance.  Matches Fortran's
            rank-deficient branch (numeigs < nx).

        Returns
        -------
        X_sph  : (T, n_keep)       sphered data
        S      : (n_orig, n_keep)  sphering matrix
        sldet  : float             log|det S|  (sum over kept eigenvalues)
        n_keep : int               number of retained components
        """
        T, n_orig = X.shape
        cov       = (X.T @ X) / T                     # (n_orig, n_orig)

        # torch.linalg.eigh returns eigenvalues in ascending order
        vals, vecs = torch.linalg.eigh(cov)
        vals = vals.flip(0);  vecs = vecs.flip(1)      # descending

        n_keep = self.n_components if self.n_components is not None else n_orig
        n_keep = int(min(n_keep, int((vals > 1e-15).sum().item())))

        if n_keep < n_orig and self.verbose:
            print(f"  Rank deficiency detected: reducing from {n_orig} to "
                  f"{n_keep} components.")

        vals_k = vals[:n_keep]                         # (n_keep,)
        vecs_k = vecs[:, :n_keep]                      # (n_orig, n_keep)

        S_pca  = vecs_k / vals_k.sqrt().unsqueeze(0)   # (n_orig, n_keep)  V_k D_k^{-1/2}
        if n_keep == n_orig:
            # ZCA: map back to sensor space — square, invertible
            S = S_pca @ vecs_k.T                       # (n_orig, n_orig)
        else:
            # PCA whitening: project into reduced subspace — rectangular
            S = S_pca                                  # (n_orig, n_keep)

        sldet  = float(-0.5 * vals_k.log().sum().item())
        X_sph  = X @ S                                 # (T, n_keep)
        self.pca_vecs_ = vecs_k   # store for MNE mapping
        self.pca_vals_ = vals_k   # store for MNE mapping
        return X_sph, S, sldet, n_keep

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_params(self, n: int) -> None:
        """
        Initialise all learnable parameters for n components.

        Shapes
        ------
        gm    : (M,)
        alpha : (M, n, J)
        mu    : (M, n, J)
        sbeta : (M, n, J)
        rho   : (M, n, J)
        c     : (M, n)
        A, W  : (M, n, n)
        """
        M, J = self.n_models, self.n_mix
        kw   = dict(device=self.device, dtype=self.dtype)

        # Model weights - uniform
        self.gm_ = torch.ones(M, **kw) / M

        # Mixture weights - uniform
        self.alpha_ = torch.ones(M, n, J, **kw) / J

        # Means: evenly spaced across components (+ random jitter unless fix_init)
        offsets    = torch.arange(J, **kw) - (J - 1) / 2.0   # (J,)
        mu_init    = offsets.view(1, 1, J).expand(M, n, J).clone()
        if not self.fix_init:
            mu_init += 0.05 * (1.0 - 2.0 * torch.rand(M, n, J, **kw))
        self.mu_   = mu_init

        # Inverse scales: 1.0 exactly (fix_init) or 1 + tiny jitter
        if self.fix_init:
            self.sbeta_ = torch.ones(M, n, J, **kw)
        else:
            self.sbeta_ = (
                1.0 + 0.1 * (0.5 - torch.rand(M, n, J, **kw))
            ).clamp(min=self.invsigmin, max=self.invsigmax)

        # Shape parameters
        self.rho_ = torch.full((M, n, J), self.rho0, **kw)

        # DC bias - zero
        self.c_ = torch.zeros(M, n, **kw)

        # Mixing matrices: identity exactly (fix_init) or identity + tiny noise
        A = torch.zeros(M, n, n, **kw)
        for m in range(M):
            if self.fix_init:
                A[m] = torch.eye(n, **kw)
            else:
                noise   = 0.01 * (0.5 - torch.rand(n, n, **kw))
                A[m]    = noise
                A[m].diagonal().fill_(1.0)
                col_nrm = A[m].norm(dim=0, keepdim=True).clamp(min=1e-30)
                A[m]   /= col_nrm
        self.A_ = A
        self.W_ = torch.linalg.inv(self.A_)            # (M, n, n)

        # History buffers
        self.LL_ = torch.zeros(self.max_iter, **kw)
        self.nd_ = torch.zeros(self.max_iter, **kw)

    # ─────────────────────────────────────────────────────────────────────────
    # Score function  fp = ∂/∂y |y|^ρ
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _score(y: Tensor, rho: Tensor) -> Tensor:
        """
        Compute the score function of the GGD exponent.

        The generalised Gaussian density is proportional to exp(−|y|^ρ),
        so the score is fp = ∂/∂y |y|^ρ.

        Special-cased for rho=1 (Laplacian) and rho=2 (Gaussian) to avoid
        log(0) in the general branch.

            rho=1 : fp = sign(y)
            rho=2 : fp = 2 y
            else  : fp = ρ · sign(y) · |y|^(ρ−1)

        All tensors must share the same shape or be broadcast-compatible.
        """
        sign_y  = torch.sign(y)
        abs_y   = y.abs().clamp(min=1e-30)
        log_abs = abs_y.log()                            # log|y|

        # General case - valid for rho ∈ (0, ∞) \ {0}
        fp = rho * sign_y * torch.exp((rho - 1.0) * log_abs)

        # Override exact integer cases for numerical cleanliness
        fp = torch.where(rho.eq(1.0), sign_y,   fp)
        fp = torch.where(rho.eq(2.0), 2.0 * y,  fp)
        return fp

    # ─────────────────────────────────────────────────────────────────────────
    # E-step
    # ─────────────────────────────────────────────────────────────────────────

    def _e_step(self, X: Tensor, sldet: float) -> dict:
        """
        Compute model / mixture posteriors and all gradient accumulators.

        Processes X in time chunks of size self.chunk_t (or all at once if
        chunk_t is None) and accumulates sufficient statistics, so peak memory
        scales with chunk_t·M·n·J rather than T·M·n·J.

        Notation
        --------
        T = number of time points,  M = number of models
        n = number of ICA components,  J = mixture components per source

        Returned stats contain only reduced (M, n, J) / (M, n) / (M,) tensors -
        no full (T, …) tensors are kept after the chunk loop.
        """
        T, n  = X.shape
        M, J  = self.n_models, self.n_mix
        chunk = self.chunk_t if self.chunk_t is not None else T

        # ── Model-level constants (no T axis, computed once) ──────────────
        wc        = torch.einsum("mij,mj->mi", self.W_, self.c_)    # (M, n)
        log_det_W = _slogdet(self.W_)                               # (M,)
        log_gm    = _safe_log(self.gm_)                             # (M,)
        rho_b     = self.rho_[None]                                 # (1,M,n,J)
        log_part  = math.log(2.0) + torch.lgamma(
            1.0 + 1.0 / self.rho_)[None]                           # (1,M,n,J)

        # ── Accumulator tensors ───────────────────────────────────────────
        LL_acc          = X.new_zeros(())
        Nv_acc          = X.new_zeros(M)
        dWtmp_acc       = X.new_zeros(M, n, n)
        usum_acc        = X.new_zeros(M, n, J)   # = dalpha/dbeta_numer/drho_denom
        dmu_numer_acc   = X.new_zeros(M, n, J)
        dmu_denom_acc   = X.new_zeros(M, n, J)
        dbeta_denom_acc = X.new_zeros(M, n, J)
        drho_numer_acc  = X.new_zeros(M, n, J)
        # Newton accumulators (small; always computed)
        ufp2_acc  = X.new_zeros(M, n, J)
        ufpy2_acc = X.new_zeros(M, n, J)
        vbb_acc   = X.new_zeros(M, n)

        # ── Chunk loop over T ─────────────────────────────────────────────
        ll_chunks: list[Tensor] = []   # populated when do_reject=True
        for start in range(0, T, chunk):
            Xc  = X[start : start + chunk]                          # (Tc, n)

            # source activations
            b   = torch.einsum("mij,tj->tmi", self.W_, Xc) - wc[None]  # (Tc,M,n)

            # GGD standardised values
            y   = self.sbeta_[None] * (b[..., None] - self.mu_[None])  # (Tc,M,n,J)

            # |y|^rho
            abs_y    = y.abs().clamp(min=1e-30)
            log_abs  = abs_y.log()
            exponent = torch.exp(rho_b * log_abs)
            exponent = torch.where(rho_b.eq(1.0), abs_y,  exponent)
            exponent = torch.where(rho_b.eq(2.0), y * y,  exponent)

            # GGD log-probability of each mixture component
            z0 = (_safe_log(self.alpha_[None]) +
                  _safe_log(self.sbeta_[None])  -
                  exponent - log_part)                              # (Tc,M,n,J)

            # per-component mixture log-probability + mixture posteriors.
            log_p_comp = torch.logsumexp(z0, dim=-1)               # (Tc, M, n)
            z   = torch.softmax(z0, dim=-1).clamp(min=1e-15)       # (Tc,M,n,J)
            # z0 no longer needed after this point

            # model log-probability and LL contribution
            P    = ((log_det_W + log_gm + sldet)[None] +
                    log_p_comp.sum(dim=-1))                         # (Tc, M)
            LL_t = torch.logsumexp(P, dim=-1)                      # (Tc,)
            if self.do_reject:
                # Store full-T LL for computing the rejection threshold; the
                # threshold uses unmasked likelihoods so previously rejected
                # samples (which are far from the model) stay rejected.
                ll_chunks.append(LL_t)

            # posteriors
            v   = torch.softmax(P,  dim=-1)                        # (Tc, M)

            # Rejection mask: zero out contributions from excluded time points.
            # LL is also accumulated only over kept samples so that the
            # reported LL is per-kept-sample, matching Fortran behaviour where
            # rejected samples are removed from the dataset entirely.
            if self._rej_mask_ is not None:
                mk     = self._rej_mask_[start : start + LL_t.shape[0]].to(dtype=X.dtype)
                v      = v * mk[:, None]                           # (Tc, M)
                LL_acc = LL_acc + (LL_t * mk).sum()
            else:
                LL_acc = LL_acc + LL_t.sum()

            # score function  fp = d/dy |y|^rho
            fp  = self._score(y, rho_b)                            # (Tc,M,n,J)

            # joint responsibility  u = v * z
            u   = v[..., None, None] * z                           # (Tc,M,n,J)

            # model-posterior-weighted score
            g   = (self.sbeta_[None] * u * fp).sum(dim=-1)        # (Tc, M, n)

            # ── Accumulate ────────────────────────────────────────────────
            Nv_acc    = Nv_acc    + v.sum(dim=0)                   # (M,)
            dWtmp_acc = dWtmp_acc + torch.einsum("tmi,tmj->mij", g, b)
            usum_acc  = usum_acc  + u.sum(dim=0)                   # (M,n,J)

            dmu_numer_acc = dmu_numer_acc + (u * fp).sum(dim=0)

            safe_y    = torch.where(y.abs() < 1e-30,
                            torch.full_like(y, 1e-30) * y.sign().clamp(min=1),
                            y)
            fp_over_y = fp / safe_y
            ufp       = u * fp
            dmu_denom_acc = dmu_denom_acc + torch.where(
                rho_b.expand_as(ufp) <= 2.0,
                self.sbeta_[None] * u * fp_over_y,
                self.sbeta_[None] * u * fp * fp,
            ).sum(dim=0)

            dbeta_denom_acc = dbeta_denom_acc + (u * fp * y).sum(dim=0)

            log_exp = rho_b * log_abs
            logab   = torch.where(exponent < 1e-16,
                          torch.zeros_like(log_exp), log_exp)
            drho_numer_acc = drho_numer_acc + (u * exponent * logab).sum(dim=0)

            # Newton accumulators
            ufp2_acc  = ufp2_acc  + (ufp * fp).sum(dim=0)
            fpy_m1    = fp * y - 1.0
            ufpy2_acc = ufpy2_acc + (u * fpy_m1 * fpy_m1).sum(dim=0)
            vbb_acc   = vbb_acc   + (v[..., None] * b * b).sum(dim=0)

        # ── Finalise ──────────────────────────────────────────────────────
        # T_eff = number of kept (non-rejected) samples.  Nv_acc.sum() equals
        # T_eff because v sums to 1 over models for each kept time point and
        # 0 for rejected ones.  Without rejection T_eff == T exactly.
        T_eff   = Nv_acc.sum().clamp(min=1.0)
        LL      = LL_acc / (T_eff * n)
        safe_Nv = Nv_acc.clamp(min=1.0)
        I_      = torch.eye(n, dtype=X.dtype, device=X.device)[None]
        dA_dir_ = I_ - dWtmp_acc / safe_Nv[:, None, None]
        dA_     = torch.bmm(self.A_, dA_dir_)                      # (M, n, n)
        nd_     = (dA_ * dA_).sum() / (n * self.n_models)   # scalar tensor; sqrt in fit()

        result = dict(
            LL=LL, Nv=Nv_acc,
            dgm_numer=Nv_acc,
            dalpha_numer=usum_acc,      dalpha_denom=Nv_acc,
            dmu_numer=dmu_numer_acc,    dmu_denom=dmu_denom_acc,
            dbeta_numer=usum_acc,       dbeta_denom=dbeta_denom_acc,
            drho_numer=drho_numer_acc,  drho_denom=usum_acc,
            dA=dA_, dA_dir=dA_dir_, nd=nd_,
            # Pre-accumulated Newton stats (no full-T tensors)
            _usum=usum_acc, _ufp2=ufp2_acc,
            _ufpy2=ufpy2_acc, _vbb=vbb_acc,
        )
        if self.do_reject:
            result['LL_t'] = torch.cat(ll_chunks)                  # (T,)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # M-step
    # ─────────────────────────────────────────────────────────────────────────

    def _m_step(
        self,
        stats:    dict,
        T:        int,
        lrate:    float,
        rholrate: float,
    ) -> None:
        """Update all model parameters using the accumulators from _e_step."""
        Nv      = stats["Nv"]                          # (M,)
        safe_Nv = Nv.clamp(min=1.0)                   # avoid /0

        # ── γ  (model mixture weights) ─────────────────────────────────────
        gm_new   = (stats["dgm_numer"] / T).clamp(min=1e-30)
        self.gm_ = gm_new / gm_new.sum()

        # ── α  (per-source mixture weights) ──────────────────────────────
        alpha_new   = stats["dalpha_numer"] / safe_Nv[:, None, None]
        alpha_new   = alpha_new.clamp(min=1e-30)
        self.alpha_ = alpha_new / alpha_new.sum(dim=-1, keepdim=True)

        # ── μ  (mixture means) ── Newton-EM step ──────────────────────────
        #   Δμ = (Σ u fp) / (sbeta · Σ u fp/y)
        safe_denom  = torch.where(
            stats["dmu_denom"].abs() < 1e-30,
            torch.full_like(stats["dmu_denom"], 1e-30),
            stats["dmu_denom"])
        self.mu_ = self.mu_ + stats["dmu_numer"] / safe_denom

        # ── β  (inverse scales) ── EM step ────────────────────────────────
        #   sbeta_new = sbeta · √(Σ u / Σ u fp y)
        ratio       = (stats["dbeta_numer"] /
                       stats["dbeta_denom"].clamp(min=1e-30))
        ratio       = ratio.clamp(min=1e-30)
        self.sbeta_ = (self.sbeta_ * ratio.sqrt()).clamp(
            min=self.invsigmin, max=self.invsigmax)

        # ── ρ  (shape parameters) ── gradient step ────────────────────────
        #   ρ_new = ρ + rholrate · (1 − ρ/ψ(1+1/ρ) · numer/denom)
        psi         = torch.digamma(1.0 + 1.0 / self.rho_)         # (M, n, J)
        drho_ratio  = (stats["drho_numer"] /
                       stats["drho_denom"].clamp(min=1e-30))
        self.rho_   = (
            self.rho_ + rholrate * (1.0 - (self.rho_ / psi) * drho_ratio)
        ).clamp(min=self.minrho, max=self.maxrho)

        # ── A  (mixing matrix) ── natural-gradient step ───────────────────
        #
        #  dA was pre-computed in _e_step (read-only, A_ still unchanged).
        #  This keeps the Fortran execution order intact:
        #    accum_updates_and_likelihood  →  convergence check  →  update_params
        #
        #  The natural gradient (in mixing-matrix form):
        #    dA[m] = A[m] @ (I − (1/Nv[m]) · Σ_t g[t,m,:]ᵀ b[t,m,:])
        #    A_new = A − lr · dA
        dA = stats["dA"]                                            # (M, n, n)

        # Update mixing matrix
        self.A_ = self.A_ - lrate * dA

        # ── Column-norm rescaling of A (Fortran: doscaling) ───────────────
        #  After rescaling: ||A[:,k]|| = 1
        #  Compensate: mu *= norm, sbeta /= norm  (keeps y unchanged)
        if self.doscaling:
            col_nrm      = self.A_.norm(dim=-2, keepdim=True).clamp(min=1e-30)
            self.A_      = self.A_ / col_nrm
            scale        = col_nrm.squeeze(-2)                      # (M, n)
            self.mu_     = self.mu_   * scale[:, :, None]
            self.sbeta_  = (self.sbeta_ / scale[:, :, None]).clamp(
                min=self.invsigmin, max=self.invsigmax)

        # ── Re-derive unmixing matrix ──────────────────────────────────────
        self.W_ = torch.linalg.inv(self.A_)

    # ─────────────────────────────────────────────────────────────────────────
    # Newton's method diagonal correction (optional)
    # ─────────────────────────────────────────────────────────────────────────

    def _newton_correction(
        self, dA_dir: Tensor, stats: dict
    ) -> Tensor:
        """
        Apply the diagonal Newton correction to dA_dir.

        For each model and component (i, k) the Newton step replaces:
            dA_dir[m, i, k]  <-  dA_dir[m, i, k] / (lambda[m,i] * sigma2[m,k] - 1)
        with special handling for the diagonal (i == k).

        Uses pre-accumulated stats from _e_step (_usum, _ufp2, _ufpy2, _vbb)
        so no full (T, ...) tensors are needed.

        Falls back to the natural gradient if the Hessian is not positive definite.
        """
        _, n, _ = self.A_.shape
        Nv      = stats["Nv"].clamp(min=1.0)                       # (M,)
        usum    = stats["_usum"].clamp(min=1.0)                    # (M,n,J)
        baralpha = usum / Nv[:, None, None]                        # (M,n,J)
        sigma2  = stats["_vbb"] / Nv[:, None]                     # (M,n)

        dkap    = stats["_ufp2"] * self.sbeta_ ** 2 / usum        # (M,n,J)
        kappa   = (baralpha * dkap).sum(dim=-1)                    # (M,n)

        dlambda = stats["_ufpy2"] / usum                           # (M,n,J)
        lam     = (baralpha * (dlambda + dkap * self.mu_ ** 2)
                   ).sum(dim=-1).clamp(min=1e-5)                   # (M,n)

        # Vectorised off-diagonal Newton scaling
        # sk1[m,i,k] = sigma2[m,i] * kappa[m,k]
        # sk2[m,i,k] = sigma2[m,k] * kappa[m,i]
        sk1   = sigma2[:, :, None] * kappa[:, None, :]             # (M,n,n)
        sk2   = sigma2[:, None, :] * kappa[:, :, None]             # (M,n,n)
        denom = sk1 * sk2 - 1.0                                    # (M,n,n)

        is_diag    = torch.eye(n, dtype=torch.bool,
                               device=dA_dir.device).unsqueeze(0)  # (1,n,n)
        posdef_off = denom.masked_fill(is_diag, 1.0) > 0.0
        if not posdef_off.all():
            return dA_dir                                           # fallback

        off_diag = ((sk1 * dA_dir - dA_dir.transpose(-2, -1)) /
                    denom.clamp(min=1e-30))
        return torch.where(is_diag,
                           dA_dir / lam[:, :, None].clamp(min=1e-5),
                           off_diag)

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X: Tensor) -> "AMICA":
        """
        Fit AMICA to data X.

        Parameters
        ----------
        X : Tensor, shape (T, n_channels)
            T time points, n_channels sensors.

        Returns
        -------
        self
        """
        X = self._t(X)
        T, n_orig = X.shape

        if self.verbose:
            print(f"AMICA  T={T}  n_orig={n_orig}  "
                  f"M={self.n_models}  J={self.n_mix}")

        # ── 1. Remove mean ────────────────────────────────────────────────
        self.mean_ = X.mean(dim=0)                                  # (n_orig,)
        X          = X - self.mean_

        # ── 2. Sphere / whiten ────────────────────────────────────────────
        if self.do_sphere:
            X, self.sphere_, self.sldet_, n = self._compute_sphere(X)
        else:
            var          = (X * X).mean(dim=0).clamp(min=1e-30)
            self.sphere_ = torch.diag(1.0 / var.sqrt())             # (n, n)
            X            = X @ self.sphere_
            self.sldet_  = float(-0.5 * var.log().sum().item())
            n            = n_orig

        if self.verbose:
            print(f"  After sphering: n={n}  sldet={self.sldet_:.4f}")

        # ── 3. Initialise parameters (or resume from checkpoint) ──────────
        ckpt_path  = self.checkpoint_path
        ckpt_every = self.checkpoint_every
        if ckpt_every > 0 and ckpt_path is None:
            ckpt_path = "amica_checkpoint.npz"

        ckpt_state: dict | None = None
        if ckpt_path is not None:
            ckpt_state = self._load_checkpoint(ckpt_path)

        self._init_params(n)    # always allocates LL_, nd_, W_, A_, etc.
        if ckpt_state is not None:
            # overwrite freshly randomised params with checkpoint values
            for attr, arr in ckpt_state['_tensors'].items():
                setattr(self, attr, self._t(torch.from_numpy(arr)))
            # restore LL/nd history into the newly allocated buffers
            if ckpt_state['_LL'] is not None:
                n_prev = len(ckpt_state['_LL'])
                self.LL_[:n_prev] = torch.from_numpy(ckpt_state['_LL']).to(self.device)
            if ckpt_state['_nd'] is not None:
                n_prev = len(ckpt_state['_nd'])
                self.nd_[:n_prev] = torch.from_numpy(ckpt_state['_nd']).to(self.device)
            if self.verbose:
                print(f"  Resuming from checkpoint '{ckpt_path}' "
                      f"at iter {ckpt_state['it']}.")

            # Already complete - skip the loop entirely
            if ckpt_state['it'] >= self.max_iter:
                self.n_iter_ = ckpt_state['it']
                if self.verbose:
                    print(f"  Checkpoint already complete at iter {self.n_iter_} "
                          f"- skipping fit.")
                with torch.inference_mode():
                    self._compute_posteriors(X)
                self._sort_outputs()
                return self

        # ── CPU optimisations ─────────────────────────────────────────────
        _prev_threads: int | None = None
        _auto_chunk_t: bool = False
        if self.device.type == "cpu":
            _prev_threads = torch.get_num_threads()
            torch.set_num_threads(_physical_core_count())
            if self.chunk_t is None:
                # Target ~32 MB for the largest chunk intermediate (Tc, M, n, J)
                bps = self.n_models * n * self.n_mix * X.element_size()
                raw = max(256, (32 * 1024 * 1024) // bps)
                self.chunk_t = 1 << (raw.bit_length() - 1)  # round down to power of 2
                _auto_chunk_t = True
                if self.verbose:
                    print(f"  Auto chunk_t={self.chunk_t} (CPU, ~32 MB L3 target)")

        # ── 4. EM loop ────────────────────────────────────────────────────
        #
        # Execution order mirrors Fortran exactly:
        #   get_updates_and_likelihood   →  _e_step  (computes LL, nd, dA)
        #   accum_updates_and_likelihood →  (in _e_step)
        #   convergence check + lrate    →  fit loop (before _m_step)
        #   update_params                →  _m_step  (applies dA, other updates)
        #
        if self.compile:
            _e_step_fn = torch.compile(self._e_step)
            _m_step_fn = torch.compile(self._m_step)
            if self.verbose:
                print("  torch.compile enabled - first iter will be slow.")
        else:
            _e_step_fn = self._e_step
            _m_step_fn = self._m_step

        if ckpt_state is not None:
            lrate         = ckpt_state['lrate']
            lrate0        = ckpt_state['lrate0']
            newtrate      = ckpt_state['newtrate']
            rholrate0     = ckpt_state['rholrate0']
            numdecs       = ckpt_state['numdecs']
            numincs       = ckpt_state['numincs']
            newton_active = ckpt_state['newton_active']
            start_iter    = ckpt_state['it'] + 1
            n_rej_done    = ckpt_state.get('n_rej_done', 0)
            if ckpt_state['_rej_mask'] is not None:
                self._rej_mask_ = torch.from_numpy(
                    ckpt_state['_rej_mask']).to(device=self.device)
            else:
                self._rej_mask_ = None
        else:
            lrate         = self.lrate
            lrate0        = self.lrate0
            newtrate      = self.newtrate
            rholrate0     = self.rholrate
            numdecs       = 0
            numincs       = 0
            newton_active = False
            start_iter    = 1
            n_rej_done    = 0
            self._rej_mask_ = None

        leave         = False
        _t_iter: float = 0.0
        self.iter_times_ = []

        _im = torch.inference_mode()
        _im.__enter__()

        def _sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "xpu":
                torch.xpu.synchronize()

        for it in range(start_iter, self.max_iter + 1):
            self.n_iter_ = it

            if self.time_iters:
                _sync()
                _t_iter = time.perf_counter()

            # ── E-step: posteriors, gradient accumulators, dA, nd ─────────
            stats  = _e_step_fn(X, self.sldet_)
            LL_it  = float(stats["LL"].item())
            nd_it  = float(stats["nd"].item() ** 0.5)   # stored as sum-of-sq/n in e_step
            self.LL_[it - 1] = LL_it
            self.nd_[it - 1] = nd_it

            # ── Outlier rejection ─────────────────────────────────────────
            # Matches Fortran: compute LL_t, update the mask, then immediately
            # re-run the E-step so the M-step receives statistics that already
            # exclude the newly rejected samples.  The mask accumulates across
            # rejection events (previously rejected samples stay rejected).
            if (self.do_reject
                    and n_rej_done < self.num_reject
                    and it >= self.reject_start
                    and (it - self.reject_start) % self.reject_int == 0):
                LL_t   = stats['LL_t']                             # (T,) on device
                thresh = float(LL_t.mean().item()) - self.reject_sigma * float(LL_t.std().item())
                # Accumulate: keep samples that pass AND were not rejected before
                prev_mask = (self._rej_mask_ if self._rej_mask_ is not None
                             else torch.ones(T, dtype=torch.bool, device=X.device))
                self._rej_mask_ = prev_mask & (LL_t >= thresh)
                n_rej  = int((~self._rej_mask_).sum().item())
                n_rej_done += 1
                if self.verbose:
                    print(f"  Rejection {n_rej_done}/{self.num_reject} at iter {it}: "
                          f"{n_rej}/{T} samples excluded total "
                          f"(thresh={thresh:.4f})")
                # Re-run E-step so M-step uses statistics without rejected samples
                stats = _e_step_fn(X, self.sldet_)

            # ── Convergence checks (BEFORE applying the update) ───────────
            # This matches the Fortran, where lrate is adjusted inside
            # update_params *after* the convergence check in the main loop.
            if it > 1:
                prev_LL = float(self.LL_[it - 2].item())

                if math.isnan(LL_it):
                    if self.verbose:
                        print("  Got NaN - exiting.")
                    leave = True

                elif LL_it < prev_LL:
                    if self.verbose:
                        print(f"  LL decreased at iter {it}.")
                    if lrate <= self.minlrate or nd_it <= self.min_nd:
                        leave = True
                        if self.verbose:
                            print("  Minimum threshold met - exiting.")
                    else:
                        lrate   = lrate * self.lratefact
                        numdecs += 1
                        # rholrate0 is NOT reduced here: Fortran resets
                        # rholrate = rholrate0 at the top of every update_params
                        # call, so per-decrease reductions are immediately undone.
                        # Only permanent reductions happen at maxdecs.
                        if numdecs >= self.maxdecs:
                            lrate0  = lrate0 * self.lratefact
                            numdecs = 0
                            if newton_active:
                                # Fortran only touches rholrate0/newtrate
                                # after Newton has started (iter > newt_start)
                                rholrate0 = rholrate0 * self.rholratefact
                                newtrate  = newtrate  * self.lratefact

                if self.use_min_dll and not math.isnan(LL_it) and it > 1:
                    if (LL_it - prev_LL) < self.min_dll:
                        numincs += 1
                        if numincs > self.maxincs:
                            leave = True
                            if self.verbose:
                                print(f"  ΔLL < {self.min_dll:.1e} for "
                                      f"{self.maxincs} iters - exiting.")
                    else:
                        numincs = 0

                if self.use_grad_norm and nd_it <= self.min_nd:
                    leave = True
                    if self.verbose:
                        print(f"  Gradient norm {nd_it:.3e} ≤ "
                              f"{self.min_nd:.1e}, stopping.")

            if leave:
                if ckpt_path is not None and ckpt_every > 0:
                    self._save_checkpoint(ckpt_path, it, lrate, lrate0,
                                          newtrate, rholrate0,
                                          numdecs, numincs, newton_active,
                                          n_rej_done)
                break

            # ── Newton correction + lrate ramp (mirrors Fortran update_params)
            # Fortran runs ramp unconditionally every iter with formula:
            #   lrate = min(ceiling, lrate + min(1/newt_ramp, lrate))
            # ceiling = newtrate in Newton mode, lrate0 pre-Newton.
            # Newton correction (Hessian preconditioning) also starts here.
            if self.do_newton and it >= self.newt_start:
                if not newton_active:
                    newton_active = True
                    numdecs = 0    # Fortran: numdecs reset at iter == newt_start
                    if self.verbose:
                        print(f"  Starting Newton at iter {it} ...")
                # Apply Newton correction to dA_dir, then recompute dA = A @ dA_dir
                dA_dir_corrected = self._newton_correction(stats["dA_dir"], stats)
                stats["dA"] = torch.bmm(self.A_, dA_dir_corrected)
                lrate = min(newtrate, lrate + min(1.0 / self.newt_ramp, lrate))
            else:
                lrate = min(lrate0, lrate + min(1.0 / self.newt_ramp, lrate))

            # ── logging ───────────────────────────────────────────────────
            if self.verbose and (it == 1 or it % self.writestep == 0):
                print(f"  iter {it:5d}  lrate={lrate:.3e}  "
                      f"LL={LL_it:.10f}  nd={nd_it:.3e}")

            # ── M-step: apply dA and update all density parameters ────────
            _m_step_fn(stats, T, lrate, rholrate0)

            if self.time_iters:
                _sync()
                self.iter_times_.append(time.perf_counter() - _t_iter)

            if ckpt_path is not None and ckpt_every > 0 and it % ckpt_every == 0:
                self._save_checkpoint(ckpt_path, it, lrate, lrate0,
                                      newtrate, rholrate0,
                                      numdecs, numincs, newton_active,
                                      n_rej_done)

        _im.__exit__(None, None, None)

        if self.verbose:
            print(f"  Done - {self.n_iter_} iterations  "
                  f"final LL={float(self.LL_[self.n_iter_-1]):.8f}")

        # Compute and store per-time-point model posteriors (M, T)
        with torch.inference_mode():
            self._compute_posteriors(X)

        # Sort models by decreasing gm_, components by decreasing variance
        self._sort_outputs()

        # Restore CPU state
        if _prev_threads is not None:
            torch.set_num_threads(_prev_threads)
        if _auto_chunk_t:
            self.chunk_t = None

        return self

    def _compute_posteriors(self, X: Tensor) -> None:
        """
        Compute per-time-point model posteriors p(m|t) and store as
        ``self.posteriors_`` of shape (M, T).

        This is a lightweight forward pass - no gradient accumulation.
        Called automatically at the end of fit().  For M=1, posteriors_
        is all ones (trivial, but kept for API consistency).
        """
        T, _  = X.shape
        chunk = self.chunk_t if self.chunk_t is not None else T

        wc        = torch.einsum("mij,mj->mi", self.W_, self.c_)
        log_det_W = _slogdet(self.W_)
        log_gm    = _safe_log(self.gm_)
        rho_b     = self.rho_[None]
        log_part  = math.log(2.0) + torch.lgamma(1.0 + 1.0 / self.rho_)[None]

        v_chunks: list[Tensor] = []
        for start in range(0, T, chunk):
            Xc  = X[start : start + chunk]
            b   = torch.einsum("mij,tj->tmi", self.W_, Xc) - wc[None]
            y   = self.sbeta_[None] * (b[..., None] - self.mu_[None])

            abs_y    = y.abs().clamp(min=1e-30)
            log_abs  = abs_y.log()
            exponent = torch.exp(rho_b * log_abs)
            exponent = torch.where(rho_b.eq(1.0), abs_y, exponent)
            exponent = torch.where(rho_b.eq(2.0), y * y, exponent)

            z0          = (_safe_log(self.alpha_[None]) +
                           _safe_log(self.sbeta_[None]) -
                           exponent - log_part)
            log_p_comp  = torch.logsumexp(z0, dim=-1)               # (Tc, M, n)
            P           = ((log_det_W + log_gm + self.sldet_)[None] +
                           log_p_comp.sum(dim=-1))                   # (Tc, M)
            v_chunks.append(torch.softmax(P, dim=-1))               # (Tc, M)

        # (T, M) → (M, T) so posteriors_[m] gives the time series for model m
        self.posteriors_ = torch.cat(v_chunks, dim=0).T.contiguous()

    def _compute_svar(self, m: int) -> "Tensor":
        """
        Variance explained by each component in model m.

        Mirrors MATLAB's loadmodout15 svar computation:
        source_variance × squared column norm of the full-channel mixing matrix.

        Used by _sort_outputs() to order components highest-to-lowest after fit.
        """
        rho   = self.rho_[m]    # (n, J)
        mu    = self.mu_[m]     # (n, J)
        sbeta = self.sbeta_[m]  # (n, J)
        alpha = self.alpha_[m]  # (n, J)

        # GGD source variance: E[y²] = Σ_j α_j (μ_j² + Γ(3/ρ_j) / (Γ(1/ρ_j) β_j²))
        g3_over_g1 = torch.lgamma(3.0 / rho).exp() / torch.lgamma(1.0 / rho).exp()
        src_var = (alpha * (mu ** 2 + g3_over_g1 / sbeta ** 2)).sum(dim=-1)  # (n,)

        # Squared column norm of full-channel mixing: ||V D^{½} A[:,i]||² = Σ_k d_k A[k,i]²
        if self.pca_vals_ is not None:
            col_norm_sq = (self.pca_vals_[:, None] * self.A_[m] ** 2).sum(dim=0)  # (n,)
        else:
            col_norm_sq = (self.A_[m] ** 2).sum(dim=0)

        return src_var * col_norm_sq

    def _sort_outputs(self) -> None:
        """
        Sort models by decreasing gm_ and components by decreasing variance.

        Called once at the end of fit(), after _compute_posteriors().

        In the original Fortran implementation the binary writes parameters in
        raw fit order; sorting is applied by the MATLAB loader loadmodout15.m
        as a post-processing step.  Here the equivalent sorting is baked into
        fit() so that both AMICA and AmicaICA always return sorted outputs
        without a separate loading step.

        After this call:
        - Model 0 is always the most probable model (highest gm_).
        - Component 0 within each model is the highest-variance component.
        """
        M = self.n_models

        # ── Sort models by descending gm_ ─────────────────────────────────
        gm_order = self.gm_.argsort(descending=True)  # (M,)

        def _sm(t: "Tensor") -> "Tensor":
            return t[gm_order]

        self.gm_    = _sm(self.gm_)
        self.W_     = _sm(self.W_)
        self.A_     = _sm(self.A_)
        self.c_     = _sm(self.c_)
        self.alpha_ = _sm(self.alpha_)
        self.mu_    = _sm(self.mu_)
        self.sbeta_ = _sm(self.sbeta_)
        self.rho_   = _sm(self.rho_)
        if self.posteriors_ is not None:
            self.posteriors_ = _sm(self.posteriors_)

        # ── Sort components within each model by descending svar ──────────
        for m in range(M):
            order = self._compute_svar(m).argsort(descending=True)  # (n,)
            self.W_[m]     = self.W_[m][order]        # reorder rows of W
            self.A_[m]     = self.A_[m][:, order]     # reorder columns of A
            self.c_[m]     = self.c_[m][order]
            self.alpha_[m] = self.alpha_[m][order]
            self.mu_[m]    = self.mu_[m][order]
            self.sbeta_[m] = self.sbeta_[m][order]
            self.rho_[m]   = self.rho_[m][order]

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def transform(self, X: Tensor) -> Tensor:
        """
        Apply the learned unmixing to new data.

        Parameters
        ----------
        X : Tensor, shape (T, n_channels)

        Returns
        -------
        sources : Tensor, shape (T, M, n_components)
            Recovered source signals for each of the M models.
            For a single-model fit (M=1) you can squeeze dim 1.
        """
        if self.mean_ is None:
            raise RuntimeError("Call fit() before transform().")
        X   = self._t(X) - self.mean_
        Xs  = X @ self.sphere_                                      # (T, n)
        wc  = torch.einsum("mij,mj->mi", self.W_, self.c_)         # (M, n)
        src = torch.einsum("mij,tj->tmi", self.W_, Xs) - wc[None]  # (T,M,n)
        return src

    def fit_transform(self, X: Tensor) -> Tensor:
        """Fit the model to X and return transformed sources."""
        return self.fit(X).transform(X)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience properties / accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mixing_matrices(self) -> Tensor:
        """Mixing matrices A, shape (M, n, n).  x ≈ A @ s."""
        return self.A_

    @property
    def unmixing_matrices(self) -> Tensor:
        """Unmixing matrices W = inv(A), shape (M, n, n).  s = W @ x."""
        return self.W_

    def ll_history(self) -> Tensor:
        """Log-likelihood trace up to the last iteration, shape ``(n_iter_,)``."""
        return self.LL_[: self.n_iter_]

    def nd_history(self) -> Tensor:
        """Gradient-norm trace up to the last iteration, shape ``(n_iter_,)``."""
        return self.nd_[: self.n_iter_]

    def save(self, path: str) -> None:
        """Save the fitted model with torch.save."""
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AMICA":
        """Load a model saved with save()."""
        state = torch.load(path, map_location=device)
        obj   = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test / usage example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── Synthetic data: 3 independent Laplacian sources mixed by a random A ──
    T, n_src = 5000, 8
    A_true   = torch.randn(n_src, n_src, dtype=torch.float64)
    A_true  /= A_true.norm(dim=0, keepdim=True)

    # Laplacian sources: sample via -sign(u)*log(1 - |2u-1|)
    u = torch.rand(T, n_src, dtype=torch.float64)
    S_true = -torch.sign(u - 0.5) * torch.log(
        (1.0 - (2.0 * u - 1.0).abs()).clamp(min=1e-15))

    X = S_true @ A_true.T                              # (T, n_src)

    print("=== AMICA smoke test ===")
    model = AMICA(
        n_components=n_src,
        n_models=1,
        n_mix=3,
        max_iter=500,
        lrate=0.1,
        lrate0=0.5,
        writestep=50,
        verbose=True,
    )
    sources = model.fit_transform(X)                   # (T, 1, n_src)

    print(f"\nMixing matrix A_ shape : {model.A_.shape}")
    print(f"Log-likelihood (final) : {float(model.ll_history()[-1]):.6f}")
    print(f"Gradient norm  (final) : {float(model.nd_history()[-1]):.3e}")
    print(f"Source shape           : {sources.shape}")
