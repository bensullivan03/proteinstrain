"""Analytic cylinder → geometric fit, anchored frame, unrolled trace and vector plot."""

import argparse
from pathlib import Path
import numpy as np
import proteinstrain as p

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
a = parser.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
angles = np.linspace(-np.pi, np.pi, 100)
xyz = np.c_[3 * np.cos(angles), 3 * np.sin(angles), np.linspace(-5, 5, 100)]
fit = p.geometry.fit_cylinder(xyz, axis=[0, 0, 1])
frame = p.geometry.Frame.from_axis(fit.axis, fit.centre, reference=[1, 0, 0])
fig, ax = p.plotting.unrolled_cylinder(xyz, frame=frame, positions=np.arange(len(xyz)))
fig.savefig(a.output / "unrolled.png")
fig, ax = p.plotting.segments_3d(xyz[::10], xyz[::10] * 0.1, scale=1, kind="vector")
fig.savefig(a.output / "vectors.png")
print("Cylinder radius:", fit.radius, "RMSD:", fit.rmsd)
