# Configuration file for the Sphinx documentation builder.
import csv
import importlib.metadata
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless rendering for sphinx-gallery

project = "pyamica"
copyright = "2025, pyamica contributors"
author = "pyamica contributors"
release = importlib.metadata.version("pyamica")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

# -- sphinx-gallery -----------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../examples",           # source scripts
    "gallery_dirs":  "_generated/examples",   # generated RST + images
    "filename_pattern": r"/plot_",
    "plot_gallery": True,
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
    "backreferences_dir": "_generated/backreferences",
    "doc_module": ("pyamica",),
}

# -- autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# -- napoleon (NumPy-style docstrings) ----------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "pyamica"
html_theme_options = {
    "sidebar_hide_name": False,
}


# -- benchmark CSV split ------------------------------------------------------
# Reads benchmark_results.csv and writes two narrower display CSVs so that
# neither table requires horizontal scrolling.

def _split_benchmark_csv(app):
    docs = Path(app.srcdir)
    src  = docs / "benchmark_results.csv"
    if not src.exists():
        return

    with src.open(newline="") as f:
        rows = list(csv.DictReader(f))

    timing_cols   = ["backend", "compiled", "iter_1_ms", "grad_ms", "newt_ms", "total_ms"]
    timing_heads  = ["Backend", "Compiled", "Iter 1 (ms)", "Grad (ms)", "Newt (ms)", "Total (ms)"]
    speedup_cols  = ["backend", "compiled", "final_ll", "speedup_iter", "speedup_total"]
    speedup_heads = ["Backend", "Compiled", "Final LL", "Spd/iter", "Spd/total"]

    def write(path, cols, heads, rows):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(heads)
            for r in rows:
                w.writerow([r[c] for c in cols])

    write(docs / "_benchmark_timing.csv",   timing_cols,   timing_heads,   rows)
    write(docs / "_benchmark_speedup.csv",  speedup_cols,  speedup_heads,  rows)


def setup(app):
    app.connect("builder-inited", _split_benchmark_csv)
