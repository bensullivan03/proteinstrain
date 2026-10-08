"""Reference→target chain maps from centroids, rigid fits or angular order."""

from itertools import permutations
from math import factorial
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..selection import resolve_chains
from ..correspondence.coordinates import _paired_coordinates
from ..geometry.transforms import fit_rigid_transform
from ..geometry.fits import fit_cylinder, cylindrical_angles


def _chains(reference, target, left, right):
    left = resolve_chains(reference, left)
    right = resolve_chains(target, right)
    if len(left) != len(right):
        raise ValueError("Both selections need the same number of chains.")
    return left, right


def assign_by_centroid(
    reference, target, *, reference_chains=None, target_chains=None, atoms="CA"
):
    """Optimal centroid assignment; structures must already share a spatial frame."""
    left, right = _chains(reference, target, reference_chains, target_chains)
    a = np.array([reference.centroid(c, atoms) for c in left])
    b = np.array([target.centroid(c, atoms) for c in right])
    rows, columns = linear_sum_assignment(
        np.linalg.norm(a[:, None] - b[None, :], axis=2)
    )
    return {left[i]: right[j] for i, j in zip(rows, columns)}


def assign_by_rigid_fit(
    reference,
    target,
    *,
    reference_chains=None,
    target_chains=None,
    atoms="CA",
    maximum_permutations=5040,
):
    """Return the chain permutation with the lowest joint rigid-fit RMSD."""
    left, right = _chains(reference, target, reference_chains, target_chains)
    if factorial(len(left)) > maximum_permutations:
        raise ValueError("Too many permutations; use centroid or angular assignment.")
    paired = {
        (a, b): _paired_coordinates(reference, a, target, b, atoms=atoms)
        for a in left
        for b in right
    }
    best = None
    cost = np.inf
    for candidate in permutations(right):
        fixed = np.concatenate([paired[a, b][0] for a, b in zip(left, candidate)])
        moving = np.concatenate([paired[a, b][1] for a, b in zip(left, candidate)])
        fit = fit_rigid_transform(moving, fixed)
        if fit.rmsd < cost:
            best = candidate
            cost = fit.rmsd
    return dict(zip(left, best))


def assign_by_angle(
    reference,
    target,
    *,
    axis=None,
    reference_chains=None,
    target_chains=None,
    atoms="CA",
    offset=None,
):
    """Preserve angular chain order; None picks the lowest-cost phase, zero fixes zero phase."""
    left, right = _chains(reference, target, reference_chains, target_chains)
    if axis is None:
        axis = fit_cylinder(reference.coordinates(left, atoms)).axis
    a = {c: reference.centroid(c, atoms) for c in left}
    b = {c: target.centroid(c, atoms) for c in right}
    order_a = np.argsort(
        cylindrical_angles(
            list(a.values()), axis=axis, origin=np.mean(list(a.values()), axis=0)
        )
    )
    order_b = np.argsort(
        cylindrical_angles(
            list(b.values()), axis=axis, origin=np.mean(list(b.values()), axis=0)
        )
    )
    left = [left[i] for i in order_a]
    right = [right[i] for i in order_b]
    if offset is not None and (
        isinstance(offset, bool) or not isinstance(offset, (int, np.integer))
    ):
        raise TypeError("offset must be an integer or None.")
    candidates = range(len(left)) if offset is None else [int(offset) % len(left)]

    def cost(shift):
        return sum(
            np.linalg.norm(a[c] - b[right[(i + shift) % len(right)]])
            for i, c in enumerate(left)
        )

    shift = min(candidates, key=cost)
    return {c: right[(i + shift) % len(right)] for i, c in enumerate(left)}
