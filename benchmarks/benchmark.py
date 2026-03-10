"""
pyamica performance benchmark

Runs all backends available on the current machine against the Memorize.fdt
EEG dataset (71 channels, 319 500 samples, 100 EM iterations):

  - Fortran amica15ub     (if data/amica15ub and data/Memorize.fdt exist)
  - PyTorch/CPU
  - PyTorch/CPU (torch.compile)
  - PyTorch/CUDA          (if CUDA is available)
  - PyTorch/CUDA compiled (if CUDA is available)
  - PyTorch/MPS           (if MPS is available)
  - PyTorch/MPS compiled  (if MPS is available)

Results are printed as a table and saved to benchmarks/results.csv (or the
path given via --output).

Usage
-----
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --output /tmp/my_results.csv
"""

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ITER = 100
N_MODELS = 1
N_CH = 71
N_SAMPLES = 319_500
NEWT_START = 50  # matches newt_start in amicadefs.param

DATA_DIR = Path(__file__).parent.parent / "data"
FDT_FILE = DATA_DIR / "Memorize.fdt"
PARAM_FILE = DATA_DIR / "amicadefs.param"
FORTRAN_BIN = DATA_DIR / "amica15ub"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data() -> torch.Tensor:
    if not FDT_FILE.exists():
        raise FileNotFoundError(
            f"Memorize.fdt not found at {FDT_FILE}.\n"
            "Copy data/Memorize.fdt into the python-package/data/ directory to run the benchmark."
        )
    raw = np.fromfile(FDT_FILE, dtype="float32").reshape(N_SAMPLES, N_CH)
    return torch.from_numpy(raw.astype(np.float64))


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def _collect_sysinfo() -> dict[str, str]:
    import platform
    import psutil

    info: dict[str, str] = {}

    # CPU name
    cpu_name = platform.processor()
    if not cpu_name and os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    info["cpu"] = cpu_name or "unknown"
    info["cpu_physical_cores"] = str(psutil.cpu_count(logical=False) or "unknown")
    ram_gb = psutil.virtual_memory().total / (1024**3)
    info["ram_gb"] = f"{ram_gb:.1f}"

    # CUDA GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            info[f"gpu_{i}"] = f"{props.name} ({vram_gb:.1f} GB VRAM)"
    elif torch.backends.mps.is_available():
        chip = platform.processor() or platform.machine()
        info["gpu_0"] = f"Apple MPS ({chip})"

    return info


def _print_sysinfo(info: dict[str, str]) -> None:
    print("System:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _f(val: float | None, d: int = 2) -> str:
    return "n/a" if val is None else f"{val:.{d}f}"


CSV_FIELDS = [
    "backend",
    "device",
    "compiled",
    "n_iter",
    "iter_1_ms",
    "grad_ms",
    "newt_ms",
    "total_ms",
    "final_ll",
    "speedup_iter",
    "speedup_total",
]

DISPLAY_HEADERS = [
    "Backend",
    "Device",
    "Compiled",
    "n_iter",
    "iter 1 (ms)",
    f"grad ms (2-{NEWT_START})",
    f"newt ms ({NEWT_START+1}-{N_ITER})",
    "total (ms)",
    "final LL",
    "spd/iter",
    "spd/total",
]
DISPLAY_KEYS = [
    "backend",
    "device",
    "compiled",
    "n_iter",
    "iter_1_ms_fmt",
    "grad_ms_fmt",
    "newt_ms_fmt",
    "total_ms_fmt",
    "ll_fmt",
    "speedup_iter_fmt",
    "speedup_total_fmt",
]


def _print_table(rows: list[dict]) -> None:
    col_w = [
        max(len(h), max(len(str(r.get(k, ""))) for r in rows))
        for h, k in zip(DISPLAY_HEADERS, DISPLAY_KEYS)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(DISPLAY_HEADERS, col_w)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        vals = [str(r.get(k, "")) for k in DISPLAY_KEYS]
        print("| " + " | ".join(v.ljust(w) for v, w in zip(vals, col_w)) + " |")
    print(sep)


def _save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        def f2(v):
            return f"{v:.2f}" if v is not None else ""

        for r in rows:
            writer.writerow(
                {
                    "backend": r["backend"],
                    "device": r.get("device", ""),
                    "compiled": r["compiled"],
                    "n_iter": r["n_iter"],
                    "iter_1_ms": f2(r.get("iter_1_ms")),
                    "grad_ms": f2(r.get("grad_ms_raw")),
                    "newt_ms": f2(r.get("newt_ms_raw")),
                    "total_ms": f"{int(r['total_ms_raw'])}" if r.get("total_ms_raw") is not None else "",
                    "final_ll": f"{r['ll_raw']:.6f}" if r.get("ll_raw") is not None else "",
                    "speedup_iter": f2(r.get("speedup_iter_raw")),
                    "speedup_total": f2(r.get("speedup_total_raw")),
                }
            )
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# PyTorch runner
# ---------------------------------------------------------------------------


def _device_label(device: str, sysinfo: dict[str, str]) -> str:
    if device == "cpu":
        return sysinfo.get("cpu", "CPU")
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        return sysinfo.get(f"gpu_{idx}", f"CUDA:{idx}")
    if device == "mps":
        return sysinfo.get("gpu_0", "Apple MPS")
    return device


def _run_pytorch(X: torch.Tensor, device: str, compiled: bool, sysinfo: dict[str, str]) -> dict:
    from pyamica import AMICA

    label = f"PyTorch/{device.upper()}"
    compiled_lbl = "yes" if compiled else "no"
    print(f"  {label} (compiled={compiled_lbl}) ...", flush=True)

    # Warm up device runtime before timing (avoids attributing driver init to iter 1)
    if device != "cpu":
        dummy = torch.ones(64, 64, device=device, dtype=torch.float64)
        _ = dummy @ dummy

    model = AMICA(
        n_models=N_MODELS,
        max_iter=N_ITER,
        device=device,
        compile=compiled,
        time_iters=True,
        use_min_dll=False,
        use_grad_norm=False,
    )
    model.fit(X)

    times_ms = [t * 1000.0 for t in model.iter_times_]
    ll = model.ll_history()
    ll_val = float(ll[-1].item()) if ll is not None and len(ll) > 0 else None

    iter_1 = times_ms[0] if times_ms else None
    grad = times_ms[1:NEWT_START] if len(times_ms) > 1 else []
    newt = times_ms[NEWT_START:] if len(times_ms) > NEWT_START else []
    grad_mean = float(np.mean(grad)) if grad else None
    newt_mean = float(np.mean(newt)) if newt else None
    total = float(sum(times_ms)) if times_ms else None

    return {
        "backend": label,
        "device": _device_label(device, sysinfo),
        "compiled": compiled_lbl,
        "n_iter": model.n_iter_,
        "iter_1_ms": iter_1,
        "grad_ms_raw": grad_mean,
        "newt_ms_raw": newt_mean,
        "total_ms_raw": total,
        "ll_raw": ll_val,
        "iter_1_ms_fmt": _f(iter_1),
        "grad_ms_fmt": _f(grad_mean),
        "newt_ms_fmt": _f(newt_mean),
        "total_ms_fmt": f"{int(total)}" if total is not None else "n/a",
        "ll_fmt": _f(ll_val, 6),
    }


# ---------------------------------------------------------------------------
# Fortran runner
# ---------------------------------------------------------------------------


def _run_fortran(sysinfo: dict[str, str]) -> dict | None:
    if not FORTRAN_BIN.exists():
        print(f"  Fortran: skipping (binary not found: {FORTRAN_BIN})")
        return None
    if not FDT_FILE.exists():
        print(f"  Fortran: skipping (data not found: {FDT_FILE})")
        return None

    print("  Fortran ...", flush=True)

    # Write a temporary param file with absolute paths and patched settings,
    # then pass it as an explicit command-line argument (as compare_amica.py does).
    outdir = DATA_DIR / "amicaout"
    outdir.mkdir(exist_ok=True)
    tmp_param = DATA_DIR / "_benchmark.param"

    overrides = {
        "max_iter": str(N_ITER),
        "num_models": str(N_MODELS),
        "use_min_dll": "0",
        "use_grad_norm": "0",
        "files": str(FDT_FILE.resolve()),
        "outdir": str(outdir.resolve()) + "/",
        "write_LLt": "1",
    }
    lines = []
    seen = set()
    for line in PARAM_FILE.read_text().splitlines():
        key = line.split()[0] if line.strip() else ""
        if key in overrides:
            lines.append(f"{key} {overrides[key]}")
            seen.add(key)
        else:
            lines.append(line)
    for key, val in overrides.items():
        if key not in seen:
            lines.append(f"{key} {val}")

    proc = None
    elapsed_s = 0.0
    try:
        tmp_param.write_text("\n".join(lines) + "\n")
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(FORTRAN_BIN.resolve()), str(tmp_param.resolve())],
            cwd=DATA_DIR,
            capture_output=True,
            text=True,
        )
        elapsed_s = time.perf_counter() - t0
    finally:
        tmp_param.unlink(missing_ok=True)

    if proc is None or proc.returncode != 0:
        rc = proc.returncode if proc else "n/a"
        stderr = (proc.stderr[:300] if proc else "")
        print(f"  Fortran failed (rc={rc}): {stderr}")
        return None

    # Parse per-iteration times from out.txt:
    # each iter line ends with "(  X.XX s,  Y.Y h)"
    times_ms: list[float] = []
    out_txt = outdir / "out.txt"
    if out_txt.exists():
        pat = re.compile(r"iter\s+\d+.*\(\s*([\d.]+)\s*s,")
        for line in out_txt.read_text().splitlines():
            m = pat.search(line)
            if m:
                times_ms.append(float(m.group(1)) * 1000.0)

    # Read final LL from binary LL file (one float64 per iteration)
    ll_val = None
    ll_bin = outdir / "LL"
    if ll_bin.exists():
        arr = np.fromfile(ll_bin, dtype="float64")
        if arr.size > 0:
            ll_val = float(arr[-1])

    iter_1 = times_ms[0] if times_ms else None
    grad = times_ms[1:NEWT_START] if len(times_ms) > 1 else []
    newt = times_ms[NEWT_START:] if len(times_ms) > NEWT_START else []
    grad_mean = float(np.mean(grad)) if grad else None
    newt_mean = float(np.mean(newt)) if newt else None
    total = float(sum(times_ms)) if times_ms else elapsed_s * 1000.0

    return {
        "backend": "Fortran",
        "device": sysinfo.get("cpu", "CPU"),
        "compiled": "yes",
        "n_iter": len(times_ms) or N_ITER,
        "iter_1_ms": iter_1,
        "grad_ms_raw": grad_mean,
        "newt_ms_raw": newt_mean,
        "total_ms_raw": total,
        "ll_raw": ll_val,
        "iter_1_ms_fmt": _f(iter_1),
        "grad_ms_fmt": _f(grad_mean),
        "newt_ms_fmt": _f(newt_mean),
        "total_ms_fmt": f"{int(total)}" if total is not None else "n/a",
        "ll_fmt": _f(ll_val, 6),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pyamica performance benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path (default: benchmarks/results.csv)",
    )
    args = parser.parse_args()

    sysinfo = _collect_sysinfo()
    _print_sysinfo(sysinfo)

    print(f"Data: {FDT_FILE.name}  ({N_CH} ch x {N_SAMPLES:,} samples)")
    print(f"Settings: n_iter={N_ITER}, n_models={N_MODELS}\n")

    X = _load_data()
    rows: list[dict] = []

    # Fortran baseline first
    fort = _run_fortran(sysinfo)
    if fort is not None:
        rows.append(fort)

    # CPU
    rows.append(_run_pytorch(X, "cpu", compiled=False, sysinfo=sysinfo))
    rows.append(_run_pytorch(X, "cpu", compiled=True, sysinfo=sysinfo))

    # CUDA
    if torch.cuda.is_available():
        rows.append(_run_pytorch(X, "cuda", compiled=False, sysinfo=sysinfo))
        rows.append(_run_pytorch(X, "cuda", compiled=True, sysinfo=sysinfo))
    else:
        print("  PyTorch/CUDA: not available, skipping.")

    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        rows.append(_run_pytorch(X, "mps", compiled=False, sysinfo=sysinfo))
        rows.append(_run_pytorch(X, "mps", compiled=True, sysinfo=sysinfo))
    else:
        print("  PyTorch/MPS: not available, skipping.")

    # Speedup relative to Fortran (>1x = faster than Fortran)
    fort = next((r for r in rows if r["backend"] == "Fortran"), None)
    ref_newt = fort["newt_ms_raw"] if fort else None
    ref_total = fort["total_ms_raw"] if fort else None
    for r in rows:
        newt_raw = r.get("newt_ms_raw")
        if ref_newt and newt_raw:
            v = ref_newt / newt_raw
            r["speedup_iter_raw"] = v
            r["speedup_iter_fmt"] = f"{v:.2f}x"
        else:
            r["speedup_iter_raw"] = None
            r["speedup_iter_fmt"] = "n/a"

        total_raw = r.get("total_ms_raw")
        if ref_total and total_raw:
            v = ref_total / total_raw
            r["speedup_total_raw"] = v
            r["speedup_total_fmt"] = f"{v:.2f}x"
        else:
            r["speedup_total_raw"] = None
            r["speedup_total_fmt"] = "n/a"

    print()
    _print_table(rows)
    print()
    print("Notes:")
    print("  - 'iter 1' includes torch.compile tracing overhead for compiled runs.")
    print(f"  - 'grad ms' = mean of iters 2-{NEWT_START} (gradient phase, iter 1 excluded).")
    print(f"  - 'newt ms' = mean of iters {NEWT_START+1}-{N_ITER} (Newton phase).")
    print("  - 'spd/iter' = Fortran newt ms / backend newt ms.")
    print("  - 'spd/total' = Fortran total / backend total (iter 1 included).")
    print("  - Speedup >1x means faster than Fortran; <1x means slower.")
    print("  - LL values are comparable across backends (same data, same n_iter).")

    out = Path(args.output) if args.output else Path(__file__).parent / "results.csv"
    _save_csv(rows, out)


if __name__ == "__main__":
    main()
