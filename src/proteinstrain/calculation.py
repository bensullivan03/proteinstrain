"""One-pair and batch calculations with local graph/reference-moment reuse."""

from itertools import permutations
import numpy as np
from ._validation import as_indices
from .mechanics.neighbours import build_graph, resolve_ensemble
from .mechanics.deformation import reference_moments
from .results.models import DeformationResult


def calculate_pairs(
    coordinates,
    *,
    pairs,
    rule,
    ensemble,
    centres=None,
    neighbour_pool="centres",
    rank_tolerance=1e-8,
    det_tolerance=1e-12,
    minimum_neighbours=4,
    nonaffine=False,
):
    """Return a {(reference_id,target_id): result} dictionary, reusing fits within this call."""
    if isinstance(pairs, str) and pairs == "all":
        pairs = permutations(coordinates.item_ids, 2)
    elif isinstance(pairs, str):
        raise ValueError('pairs must be explicit pairs or "all".')
    centres = (
        np.arange(coordinates.n_atoms, dtype=np.int32)
        if centres is None
        else as_indices(centres, coordinates.n_atoms)
    )
    if isinstance(neighbour_pool, str):
        if neighbour_pool not in ("centres", "all"):
            raise ValueError("neighbour_pool must be centres/all or indices.")
        pool = (
            centres
            if neighbour_pool == "centres"
            else np.arange(coordinates.n_atoms, dtype=np.int32)
        )
    else:
        pool = as_indices(neighbour_pool, coordinates.n_atoms)
    graphs = {}
    moments = {}
    results = {}
    for reference, target in pairs:
        coordinates.coordinates(reference)
        coordinates.coordinates(target)
        ids = resolve_ensemble(coordinates, ensemble, reference, target)
        key = tuple(sorted(ids)) if rule.order_independent else tuple(ids)
        if key not in graphs:
            graphs[key] = build_graph(
                np.stack([coordinates.coordinates(i) for i in key]),
                rule,
                centre_indices=centres,
                neighbour_indices=pool,
            )
        moment_key = (key, reference)
        if moment_key not in moments:
            moments[moment_key] = reference_moments(
                graphs[key],
                coordinates.coordinates(reference),
                rank_tolerance=rank_tolerance,
                minimum_neighbours=minimum_neighbours,
            )
        F, valid, residual = moments[moment_key].solve(
            coordinates.coordinates(target),
            det_tolerance=det_tolerance,
            nonaffine=nonaffine,
        )
        results[(reference, target)] = DeformationResult(
            reference, target, F, centres.copy(), valid, residual
        )
    return results


def calculate(coordinates, reference_id, target_id, *, rule, ensemble, **kwargs):
    """Calculate one directed pair using the same implementation as batch calculations."""
    return calculate_pairs(
        coordinates,
        pairs=[(reference_id, target_id)],
        rule=rule,
        ensemble=ensemble,
        **kwargs,
    )[(reference_id, target_id)]
