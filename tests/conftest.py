import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import proteinstrain as psp


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    return Path(__file__).parent / "data"


@pytest.fixture
def structure(data):
    return psp.load_structure(data / "8IXA_subset.cif")


@pytest.fixture
def block():
    points = psp.synthetic.lattice((5, 5, 5), 3.0)
    F = np.array([[1.07, 0.03, 0.0], [-0.02, 0.96, 0.01], [0.0, 0.02, 1.03]])
    xyz = np.stack(
        [points, points @ F.T + [2.0, -4.0, 1.0], points @ np.diag([1.02, 0.98, 1.01])]
    )
    rows = []
    for label in ("R", "T", "U"):
        for i in range(len(points)):
            rows.append(
                dict(
                    item_id=label,
                    index=i,
                    chain_id="A",
                    hetero=" ",
                    residue_number=i + 1,
                    insertion=" ",
                    atom_name="CA",
                    element="C",
                    polymer_position=i,
                    residue_code="A",
                    alignment_group="protein",
                    column=i,
                )
            )
    return psp.AlignedCoordinates(xyz, ("R", "T", "U"), pd.DataFrame(rows))
