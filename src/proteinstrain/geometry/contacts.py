"""Sparse contacts and geometry between explicitly selected chains."""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from .._validation import as_xyz, unit_vector
from ..selection import resolve_chains


def _cutoff(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError("cutoff must be positive and finite.")
    return float(value)


def contact_map(xyz, cutoff, *, minimum=0.0):
    """Return a table of unordered atom pairs i<j and distances."""
    xyz = as_xyz(xyz)
    pairs = KDTree(xyz).query_pairs(_cutoff(cutoff), output_type="ndarray")
    distance = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
    keep = distance >= minimum
    return pd.DataFrame(
        {"i": pairs[keep, 0], "j": pairs[keep, 1], "distance": distance[keep]}
    )


def contact_changes(reference_xyz, target_xyz, cutoff, *, pairs=None):
    """Return pair distance/extension/vector changes on corresponding atoms."""
    reference = as_xyz(reference_xyz)
    target = as_xyz(target_xyz, len(reference))
    contacts = contact_map(reference, cutoff) if pairs is None else pairs
    i = contacts.i.to_numpy()
    j = contacts.j.to_numpy()
    left = reference[j] - reference[i]
    right = target[j] - target[i]
    a = np.linalg.norm(left, axis=1)
    b = np.linalg.norm(right, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        extension = np.where(a > 0, b / a - 1, np.nan)
    return pd.DataFrame(
        dict(
            i=i,
            j=j,
            reference_distance=a,
            target_distance=b,
            change=b - a,
            extension=extension,
            vector_change=np.linalg.norm(right - left, axis=1),
        )
    )


def neighbour_counts(xyz, cutoff):
    points = as_xyz(xyz)
    return (
        np.asarray(
            KDTree(points).query_ball_point(points, _cutoff(cutoff), return_length=True)
        )
        - 1
    )


def external_neighbour_counts(xyz, external_xyz, cutoff):
    return np.asarray(
        KDTree(as_xyz(external_xyz)).query_ball_point(
            as_xyz(xyz), _cutoff(cutoff), return_length=True
        )
    )


def chain_interface(
    structure, first, second, *, cutoff, atoms, axis=None, normalise=None
):
    """Return centroid/minimum distances and raw contacts, with optional axial offsets."""
    first = resolve_chains(structure, first)
    second = resolve_chains(structure, second)
    if set(first) & set(second):
        raise ValueError("Interface selections must be disjoint.")
    left = structure.coordinates(first, atoms)
    right = structure.coordinates(second, atoms)
    offset = right.mean(axis=0) - left.mean(axis=0)
    tree = KDTree(right)
    contacts = int(
        np.sum(tree.query_ball_point(left, _cutoff(cutoff), return_length=True))
    )
    output = dict(
        centroid_distance=float(np.linalg.norm(offset)),
        minimum_distance=float(tree.query(left)[0].min()),
        contacts=contacts,
    )
    if axis is not None:
        axis = unit_vector(axis)
        axial = float(offset @ axis)
        output.update(
            axial_offset=axial,
            lateral_offset=float(np.linalg.norm(offset - axial * axis)),
        )
    if normalise is not None:
        if normalise != "geometric_mean":
            raise ValueError("normalise must be geometric_mean or None.")
        output["normalised_contacts"] = contacts / np.sqrt(len(left) * len(right))
    return output


def chain_contact_matrix(structure, chains=None, *, cutoff, atoms):
    ids = resolve_chains(structure, chains)
    points = {c: structure.coordinates(c, atoms) for c in ids}
    matrix = np.zeros((len(ids), len(ids)), int)
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            count = np.sum(
                KDTree(points[ids[j]]).query_ball_point(
                    points[a], _cutoff(cutoff), return_length=True
                )
            )
            matrix[i, j] = matrix[j, i] = count
    return pd.DataFrame(matrix, index=ids, columns=ids)
