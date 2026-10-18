Solver benchmarks
=================

Use stable example IDs from ``examples/catalog.py`` when selecting cases::

  uv run --extra autodiff python benchmarks/compare_solvers.py --examples minimum_energy_transfer brachistochrone

Current two-digit lesson numbers are also accepted, but may change when the
curriculum changes. New result records use stable IDs and hash the Python
sources recursively, including reusable example implementations.

``results.json`` and ``results.txt`` are historical measurements taken before
the examples were renumbered. Their numeric case IDs and source paths describe
that recorded run, not today's catalog. They are preserved with their original
hashes and measurements. To compare them with current examples:

.. list-table:: Historical number to stable ID
   :header-rows: 1

   * - Historical number
     - Stable ID
     - Current number
   * - 01
     - brachistochrone
     - 03
   * - 03
     - simple_rocket_2d
     - 06
   * - 04
     - goddard_0knot
     - 07
   * - 05
     - goddard_1knot
     - 08
   * - 12
     - minimum_energy_transfer
     - 01
