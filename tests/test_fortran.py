"""
test_fortran.py - compare pyamica against the Fortran amica15ub binary.

These tests are marked @pytest.mark.slow and require:
  - data/amica15ub   (Linux ELF binary)
  - data/Memorize.fdt  (71-channel EEG, 319 500 samples)
  - data/amicadefs.param

Run with:
    pytest tests/test_fortran.py -v -m slow
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from pyamica import AMICA

pytestmark = pytest.mark.slow

# ── Constants matching amicadefs.param ────────────────────────────────────────

N_CH     = 71
T        = 319_500
MAX_ITER = 100
N_MIX    = 3

AMICA_KWARGS = dict(
    n_components  = N_CH,
    n_models      = 1,
    n_mix         = N_MIX,
    max_iter      = MAX_ITER,
    lrate         = 0.1,
    lrate0        = 0.1,
    lratefact     = 0.5,
    rho0          = 1.5,
    minrho        = 1.0,
    maxrho        = 2.0,
    rholrate      = 0.05,
    rholratefact  = 0.5,
    do_sphere     = True,
    do_newton     = True,
    newt_start    = 50,
    newt_ramp     = 10,
    newtrate      = 1.0,
    doscaling     = True,
    min_dll       = 1e-9,
    min_nd        = 1e-6,
    use_grad_norm = True,
    use_min_dll   = True,
    maxdecs       = 3,
    minlrate      = 1e-8,
    invsigmax     = 100.0,
    invsigmin     = 1e-8,
    writestep     = 10,
    verbose       = True,
)


@pytest.fixture(scope="module")
def fortran_outputs(data_dir, fortran_binary, memorize_data, tmp_path_factory):
    """
    Run amica15ub on Memorize.fdt and return (LL_f, S_f, rho_f) arrays.
    Uses a temporary output directory; skips if binary is not executable.
    """
    param_src = data_dir / "amicadefs.param"
    fdt       = data_dir / "Memorize.fdt"

    tmp = tmp_path_factory.mktemp("fortran_out")
    out_dir = tmp / "amicaout"
    # Do NOT pre-create out_dir - the Fortran binary creates it itself.
    # Pre-creating would cause the binary's internal mkdir to print
    # "File exists" on stderr.

    # Write a tweaked param file pointing to our data and output
    tweaked = tmp / "compare.param"
    lines = param_src.read_text().splitlines(keepends=True)
    with tweaked.open("w") as f:
        for line in lines:
            key = line.split()[0] if line.split() else ""
            if key == "max_iter":
                f.write(f"max_iter {MAX_ITER}\n")
            elif key == "outdir":
                f.write(f"outdir {out_dir}/\n")
            elif key == "files":
                f.write(f"files {fdt}\n")
            else:
                f.write(line)
        f.write("write_nd 0\n")
        f.write("write_LLt 0\n")

    res = subprocess.run(
        [str(fortran_binary), str(tweaked)],
        cwd=str(data_dir),
        capture_output=False,
        timeout=600,
    )
    if res.returncode != 0:
        pytest.skip(f"amica15ub exited with code {res.returncode}")

    def _read(name, dtype="float64"):
        return np.fromfile(out_dir / name, dtype=dtype)

    LL_f  = _read("LL")
    S_f   = _read("S").reshape(N_CH, N_CH, order="F")
    rho_f = _read("rho").reshape(N_MIX, N_CH, order="F")
    return LL_f, S_f, rho_f


@pytest.fixture(scope="module")
def python_model(memorize_data):
    """Run pyamica AMICA on Memorize.fdt and return the fitted model."""
    model = AMICA(**AMICA_KWARGS)
    model.fit(memorize_data)
    return model


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ll_matches_fortran(fortran_outputs, python_model):
    LL_f, _, _ = fortran_outputs
    LL_p = python_model.ll_history().cpu().numpy()
    n_common = min(len(LL_f), len(LL_p))
    diff = abs(LL_f[n_common - 1] - LL_p[n_common - 1])
    # Both models use independent random inits so they land at nearby but not
    # identical local optima. 1e-4 checks they're in the same ballpark; the
    # sphere test verifies the fully deterministic path to machine precision.
    assert diff < 1e-4, (
        f"Final LL mismatch: Fortran={LL_f[n_common-1]:.8f}, "
        f"Python={LL_p[n_common-1]:.8f}, |diff|={diff:.3e}"
    )


def test_sphere_matches_fortran(fortran_outputs, python_model):
    _, S_f, _ = fortran_outputs
    S_p = python_model.sphere_.cpu().numpy()
    diff = np.abs(S_f - S_p).max()
    assert diff < 1e-10, f"Sphere max|diff| = {diff:.3e} (want < 1e-10)"


def test_n_iters_completed(python_model):
    """Python model should run to the expected number of iterations."""
    assert python_model.n_iter_ > 0
    assert python_model.n_iter_ <= MAX_ITER
