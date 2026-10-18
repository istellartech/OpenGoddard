Examples
========

Each numbered script solves one problem with one objective. Read it in order,
change the physical parameters, and run it from the repository root::

  uv run python examples/01_minimum_energy_transfer.py
  uv run python examples/02_move_stop.py

* 01: a position transfer with travel time and control effort in the objective.
* 02: minimum-time acceleration and stopping, with an acceleration limit.
* 03: the brachistochrone, using position, depth, speed, and track angle.
* 04: add a depth bound and a fixed destination to the descent problem.
* 05: use physical distances and unit scaling for a Tokyo–Osaka descent.
* 06: steer a rocket in two dimensions.
* 07–08: Goddard ascent with changing mass and drag; then split the time interval.
* 09: land vertically with constant mass and minimum impulse.
* 10–14: add vehicle classes, staging, polar coordinates, and atmosphere tables.
* 15: minimize the integral of thrust during a low-thrust orbit transfer.

The `documentation site <https://istellartech.github.io/OpenGoddard/>`_ explains
each problem, the code, its result, and exercises.

Edit ``target`` in lesson 01, ``PARAMETERS`` in lessons 02–09 and 15, or the
``Rocket`` class in lessons 10–14. There are no presets or model switches.
``make_problem`` defines the model, initial guesses, and bounds; ``solve`` runs
the optimizer and returns a ``Problem``; ``plot`` draws the result. Lesson 01
shows the solve call directly in ``main``. Imports never start optimization.

For example, change the distance in lesson 02 from Python::

  from examples.catalog import load_example

  lesson = load_example("move_stop")
  parameters = dict(lesson.PARAMETERS)
  parameters["distance"] = 20.0
  problem = lesson.solve(parameters)
  figure = lesson.plot(problem)

The later vehicle examples accept a ``Rocket`` instance in ``make_problem``.
Input CSV files for lesson 14 live in ``examples/data/``. Historical figures are
archived in ``docs/figures/<id>/``.

The website's media builder reuses each lesson's normal solve and plot functions.
It saves MP4, PNG, and sampled JSON data in ``website/public/media/`` and
intermediate figures in ``examples/output/<id>/`` (ignored by Git)::

  uv run python website/scripts/build_media.py
