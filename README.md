# OpenGoddard

OpenGoddard is a Python library for trajectory optimization and nonlinear optimal
control. It uses the Legendre–Gauss–Lobatto pseudospectral method to turn equations
of motion, constraints, and an objective into a numerical optimization problem
solved with SciPy.

[日本語](docs/README.ja.md) · [Documentation (Japanese)](https://istellartech.github.io/OpenGoddard/)

## What you can solve

Find states such as position and velocity, and controls such as thrust and
acceleration, as functions of time. Define equations and constraints with Python
functions for problems such as rocket ascent and landing, orbital transfer,
minimum-time motion, and minimum-energy control.

- Set bounds on states, controls, and final times, plus boundary and path constraints.
- Combine terminal and integral costs in the objective.
- Connect multiple phases to model changes such as stage separation.
- Use variable scaling, automatic differentiation (an optional dependency), and
  mesh refinement.

## Install

Requires Python 3.15 or later. NumPy, SciPy, and Matplotlib are installed as
dependencies.

```sh
pip install OpenGoddard
```

## Run your first example

To run the examples, clone this repository and set up the environment with
[uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/istellartech/OpenGoddard.git
cd OpenGoddard
uv sync --locked
uv run python examples/01_minimum_energy_transfer.py
```

Example 01 moves a position to a target while minimizing travel time and control
effort. It prints the numerical result alongside the analytic solution. Change
`target` in `main()` and run it again to compare results.

The 15 scripts in [examples/](examples/) are numbered in learning order, from
simple transfers to constrained motion, rocket ascent, and orbital transfer.
Each script contains its equations, constraints, objective, initial guess, and
solution; later examples also plot the results.

| Examples | Topics |
| --- | --- |
| 01–02 | Simple transfers, acceleration and stopping; states, controls, and objectives |
| 03–05 | Brachistochrones; path constraints, physical units, and spherical geometry |
| 06–09 | Rocket ascent and soft landing; changing mass, thrust limits, and phases |
| 10–14 | Single-stage and multistage rockets; separation, orbit insertion, atmosphere and drag |
| 15 | Low-thrust orbital transfer; integral cost optimization |

## Define your own problem

Start from example 01 and adapt it in this order:

1. Create a `Problem` with phases, node counts, and numbers of states and controls.
2. Write functions for the dynamics, boundary and path constraints, and objective.
3. Set variable bounds and initial guesses, and scale units as needed.
4. Assign the functions to the problem and call `solve()`.
5. Inspect the returned `success` and `message`, then check the trajectory and constraints.

The default solver is SciPy's SLSQP. Results depend on the initial guess and node
counts; as well as checking solver success, increase the node counts to see
whether the solution stabilizes. The getting-started guide and API reference
below explain the code and settings in detail.

## Learn and use

- [Getting started](https://istellartech.github.io/OpenGoddard/getting-started/):
  how to define a problem and work through the examples.
- [Pseudospectral method](https://istellartech.github.io/OpenGoddard/concepts/pseudospectral/):
  the numerical method behind the library.
- [Example tutorials](https://istellartech.github.io/OpenGoddard/examples/01-minimum-energy-transfer/):
  equations, code, results, and videos, starting with example 01.
- [API reference](https://istellartech.github.io/OpenGoddard/api/OpenGoddard/):
  classes, solver options, derivatives, and mesh refinement.

## License

[MIT](LICENSE)
