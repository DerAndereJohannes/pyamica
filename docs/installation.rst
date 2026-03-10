Installation
============

From PyPI
---------

Core package (no MNE dependency):

.. code-block:: bash

   pip install pyamica

With MNE-Python integration:

.. code-block:: bash

   pip install "pyamica[mne]"

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv add pyamica
   uv add "pyamica[mne]"

GPU support
-----------

.. warning::

   Only **CPU** and **CUDA** have been tested. Other backends (ROCm, XPU)
   may work but are untested.

   **Apple Silicon (MPS):** PyTorch's MPS backend does not support 64-bit
   floating point. When ``device="mps"`` is used, pyamica automatically
   falls back to 32-bit floats (``torch.float32``). Results will be less
   numerically precise than on CPU or CUDA, and large numerical differences
   relative to the reference Fortran implementation are expected. This was
   mainly implemented for the sake of completeness and with the hope that
   furture Apple Silicon chips will have support.

If you need a specific CUDA version (e.g. CUDA 11.8 vs 12.x), install the
matching PyTorch build **before** installing pyamica. See the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`_
for the correct index URL.

Pass ``device="cuda"`` (or ``"mps"`` on Apple Silicon) when constructing
:class:`~pyamica.AMICA` or :class:`~pyamica.AmicaICA`.

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/DerAndereJohannes/pyamica
   cd pyamica
   pip install -e ".[mne,dev]"

With uv:

.. code-block:: bash

   git clone https://github.com/DerAndereJohannes/pyamica
   cd pyamica
   uv sync --extra mne --extra dev

Running the tests:

.. code-block:: bash

   pytest tests/ -v -m "not slow and not gpu"
   uv run pytest tests/ -v -m "not slow and not gpu"   # uv

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build docs docs/_build/html

With uv:

.. code-block:: bash

   uv sync --extra docs
   uv run sphinx-build docs docs/_build/html

The generated HTML is at ``docs/_build/html/index.html``.
