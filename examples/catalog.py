"""Learning order, stable IDs, and imports for the numbered example folders.

Each lesson owns its main.py, README, input data, reference figures and output.
Numbers define learning order; datasets and URLs should use stable IDs.
"""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class Example:
    id: str
    directory: str
    title: str

    @property
    def script(self) -> str:
        return f"{self.directory}/main.py"


EXAMPLES = (
    Example(
        "minimum_energy_transfer", "01_minimum_energy_transfer", "最小操作量の移動"
    ),
    Example("move_stop", "02_move_stop", "加速して停止する移動"),
    Example("brachistochrone", "03_brachistochrone", "最速降下"),
    Example("constrained_descent", "04_constrained_descent", "深さ制限付き最速降下"),
    Example(
        "brachistochrone_tokyo_osaka",
        "05_brachistochrone_tokyo_osaka",
        "東京・大阪の最速降下",
    ),
    Example("simple_rocket_2d", "06_simple_rocket_2d", "単純な2次元ロケット"),
    Example("goddard_0knot", "07_goddard_0knot", "Goddard問題・単一区間"),
    Example("goddard_1knot", "08_goddard_1knot", "Goddard問題・区間分割"),
    Example("soft_landing", "09_soft_landing", "月面への鉛直軟着陸"),
    Example(
        "rocket_ascent_single_stage",
        "10_rocket_ascent_single_stage",
        "単段ロケット上昇",
    ),
    Example("rocket_ascent_two_stage", "11_rocket_ascent_two_stage", "2段ロケット上昇"),
    Example(
        "rocket_ascent_polar_ssto",
        "12_rocket_ascent_polar_ssto",
        "極座標・単段軌道投入",
    ),
    Example(
        "rocket_ascent_polar_tsto", "13_rocket_ascent_polar_tsto", "極座標・2段軌道投入"
    ),
    Example("polar_tsto_taiki", "14_polar_tsto_taiki", "大樹からの2段軌道投入"),
    Example(
        "low_thrust_orbit_transfer", "15_low_thrust_orbit_transfer", "低推力軌道遷移"
    ),
)


def find_example(identifier: str) -> Example:
    """Look up a stable ID or the current two-digit lesson number."""
    for example in EXAMPLES:
        if identifier in (example.id, example.directory[:2]):
            return example
    raise ValueError(f"unknown example: {identifier}")


def load_example(identifier: str) -> ModuleType:
    """Import a lesson by stable ID, without starting optimization or plotting.

    Numeric directory names need not appear in callers' Python import syntax.
    """
    example = find_example(identifier)
    return import_module(f"examples.{example.directory}.main")
