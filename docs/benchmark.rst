Benchmark
=========

Performance comparison of all available backends on the
`Memorize <https://github.com/sccn/amica>`_ EEG dataset
(71 channels × 319 500 samples, 100 EM iterations, 1 model, ``use_min_dll=False``,
``use_grad_norm=False``).

Results
-------

Tested on:

* **CPU** - AMD Ryzen 9 7950X (16 physical cores, 62 GB RAM)
* **GPU** - NVIDIA GeForce RTX 5070 Ti (15.5 GB VRAM)

**Timing** (milliseconds per iteration phase)

.. csv-table::
   :file: _benchmark_timing.csv
   :header-rows: 1
   :widths: 20, 12, 16, 14, 14, 14

**Fit quality and speedup** (relative to Fortran baseline)

.. csv-table::
   :file: _benchmark_speedup.csv
   :header-rows: 1
   :widths: 20, 12, 20, 16, 16

Notes
-----

* **iter 1**: wall time of the very first EM iteration.  For compiled runs this
  includes ``torch.compile`` tracing overhead (graph capture + kernel
  compilation), which is a one-time cost amortised over all subsequent
  iterations.
* **grad ms**: mean of iterations 2–50 (gradient phase, iter 1 excluded).
* **newt ms**: mean of iterations 51–100 (Newton phase).  The relative
  cost of Newton vs gradient steps varies by backend: Fortran and uncompiled
  CPU are slower in the Newton phase, while compiled backends seem to be
  faster (more effective kernel fusion on the Newton correction).
* **total (ms)**: sum of all iteration times including iter 1.
* **spd/iter**: speedup vs Fortran based on Newton-phase mean ms/iter.
  Because compiled backends are faster in the Newton phase than the gradient
  phase, this speedup improves with longer runs: typical AMICA fits use
  200–2000 iterations, where nearly all compute is Newton-phase, so
  ``spd/iter`` is the more representative metric for real-world use.
* **spd/total**: speedup vs Fortran based on total time (iter 1 included);
  reflects the real-world cost of ``torch.compile`` tracing overhead.
  > 1× means faster than Fortran, < 1× means slower.
* The Fortran binary writes per-iteration timings to ``amicaout/out.txt``;
  per-phase means are derived from those values.
* LL values are comparable across backends: same dataset, same number of
  iterations, same algorithm settings.

Running the benchmark
---------------------

.. code-block:: bash

   python benchmarks/benchmark.py
   python benchmarks/benchmark.py --output /tmp/my_results.csv

The script auto-detects all available backends (Fortran, CPU, CUDA, MPS) and
saves results to ``benchmarks/results.csv`` by default.  The Fortran baseline
requires ``data/amica15ub`` and ``data/Memorize.fdt`` to be present.
