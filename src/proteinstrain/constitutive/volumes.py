"""Molecular volumes, computed inside the package."""

import numpy as np
from numba import njit, prange
from scipy.spatial import KDTree
from .._validation import as_xyz

VDW_RADII = {
    "H": 1.2,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.8,
    "S": 1.8,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.9,
    "MG": 1.73,
    "NA": 2.27,
    "K": 2.75,
    "CA": 2.31,
    "ZN": 1.39,
    "MN": 1.97,
    "FE": 1.94,
    "CU": 1.4,
    "NI": 1.63,
    "CO": 1.92,
}
DEFAULT_RADIUS = 1.7


def atom_radii(elements, *, radii=None, default=DEFAULT_RADIUS):
    """Return the van der Waals radius of each element symbol."""
    table = dict(VDW_RADII if radii is None else radii)
    table = {str(key).upper(): float(value) for key, value in table.items()}
    return np.array(
        [table.get(str(element).upper(), float(default)) for element in elements],
        dtype=float,
    )


def sphere_volumes(radii):
    """Volume of each isolated sphere."""
    values = np.asarray(radii, dtype=float)
    return 4.0 / 3.0 * np.pi * values**3


@njit(cache=True, parallel=True)
def _argmin_power(distance, index, squared, owner):
    for sample in prange(distance.shape[0]):
        best = -1
        smallest = 0.0
        for slot in range(distance.shape[1]):
            atom = index[sample, slot]
            if atom < 0:
                continue
            power = distance[sample, slot] ** 2 - squared[atom]
            if power < 0.0 and (best < 0 or power < smallest):
                best = atom
                smallest = power
        owner[sample] = best


def _owners(points, centres, radii, tree, maximum_radius, neighbours=16):
    squared = np.ascontiguousarray(radii**2)
    wanted = min(int(neighbours), len(centres))
    while True:
        distance, index = tree.query(points, k=wanted, workers=-1)
        distance = np.asarray(distance, dtype=float).reshape(len(points), -1)
        index = (
            np.asarray(index, dtype=np.int64).reshape(len(points), -1).astype(np.int64)
        )
        index[index >= len(centres)] = -1
        if wanted >= len(centres) or distance[:, -1].min() > maximum_radius:
            break
        wanted = min(len(centres), wanted * 2)
    owner = np.full(len(points), -1, dtype=np.int64)
    _argmin_power(
        np.ascontiguousarray(distance), np.ascontiguousarray(index), squared, owner
    )
    return owner


def union_volume(
    xyz, radii, *, method="monte_carlo", samples=2000000, seed=0, spacing=0.4
):
    """Van der Waals union volume, with a power-distance partition that sums to the total."""
    centres = as_xyz(xyz, name="atom centres")
    values = np.asarray(radii, dtype=float).reshape(-1)
    if values.shape != (len(centres),) or (values <= 0).any():
        raise ValueError("One positive radius is required per atom.")
    tree = KDTree(centres)
    maximum = float(values.max())
    if method == "spheres":
        totals = sphere_volumes(values)
        return (float(totals.sum()), totals, 0.0)
    if method == "grid":
        low = centres.min(axis=0) - maximum - spacing
        high = centres.max(axis=0) + maximum + spacing
        axes = [np.arange(low[k], high[k] + spacing, spacing) for k in range(3)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        owner = _owners(mesh, centres, values, tree, maximum)
        cell = spacing**3
        per_atom = (
            np.bincount(owner[owner >= 0], minlength=len(centres)).astype(float) * cell
        )
        return (float(per_atom.sum()), per_atom, None)
    if method != "monte_carlo":
        raise ValueError("method must be 'monte_carlo', 'grid' or 'spheres'.")
    generator = np.random.default_rng(seed)
    volumes = sphere_volumes(values)
    share = volumes / volumes.sum()
    counts = np.maximum(16, np.round(share * int(samples)).astype(np.int64))
    source = np.repeat(np.arange(len(centres), dtype=np.int64), counts)
    total_samples = int(counts.sum())
    direction = generator.normal(size=(total_samples, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    fraction_of_radius = generator.random(total_samples) ** (1.0 / 3.0)
    points = (
        centres[source] + direction * (values[source] * fraction_of_radius)[:, None]
    )
    owner = _owners(points, centres, values, tree, maximum)
    owned = np.bincount(source[owner == source], minlength=len(centres)).astype(float)
    fraction = owned / counts
    per_atom = volumes * fraction
    variance = volumes**2 * fraction * (1.0 - fraction) / counts
    return (float(per_atom.sum()), per_atom, float(np.sqrt(variance.sum())))


def packing_factor(volume, radii):
    """Union volume divided by the sum of the isolated spheres: the overlap correction."""
    naive = float(sphere_volumes(radii).sum())
    return volume[0] / naive if naive > 0 else float("nan")


def structure_volume(structure, chains=None, *, radii=None, **kwargs):
    """Van der Waals union volume of every atom of the selected chains."""
    from ..selection import resolve_chains

    selected = resolve_chains(structure, chains)
    points = []
    elements = []
    for chain in selected:
        for residue in structure.chain(chain):
            for atom in residue:
                points.append((atom.pos.x, atom.pos.y, atom.pos.z))
                elements.append(atom.element.name)
    if not points:
        raise ValueError("The chain selection contains no atoms.")
    return (
        union_volume(np.asarray(points), atom_radii(elements, radii=radii), **kwargs),
        tuple(elements),
    )
