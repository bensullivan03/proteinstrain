"""Weighted local affine deformation, with reusable reference moments."""

from dataclasses import dataclass
import numpy as np
from numba import njit, prange
from .._validation import as_xyz

RANK_TOLERANCE = 1e-8
DET_TOLERANCE = 1e-12
MINIMUM_NEIGHBOURS = 4


@njit(cache=True, parallel=True)
def _accumulate_reference(indptr, neighbour, weight, reference, centres, moments):
    for position in prange(len(centres)):
        centre = centres[position]
        for pair in range(indptr[position], indptr[position + 1]):
            other = neighbour[pair]
            w = weight[pair]
            dx0 = reference[other, 0] - reference[centre, 0]
            dx1 = reference[other, 1] - reference[centre, 1]
            dx2 = reference[other, 2] - reference[centre, 2]
            moments[position, 0, 0] += w * dx0 * dx0
            moments[position, 0, 1] += w * dx0 * dx1
            moments[position, 0, 2] += w * dx0 * dx2
            moments[position, 1, 1] += w * dx1 * dx1
            moments[position, 1, 2] += w * dx1 * dx2
            moments[position, 2, 2] += w * dx2 * dx2
        moments[position, 1, 0] = moments[position, 0, 1]
        moments[position, 2, 0] = moments[position, 0, 2]
        moments[position, 2, 1] = moments[position, 1, 2]


@njit(cache=True, parallel=True)
def _accumulate_mixed(indptr, neighbour, weight, reference, target, centres, mixed):
    for position in prange(len(centres)):
        centre = centres[position]
        for pair in range(indptr[position], indptr[position + 1]):
            other = neighbour[pair]
            w = weight[pair]
            dx0 = reference[other, 0] - reference[centre, 0]
            dx1 = reference[other, 1] - reference[centre, 1]
            dx2 = reference[other, 2] - reference[centre, 2]
            dy0 = target[other, 0] - target[centre, 0]
            dy1 = target[other, 1] - target[centre, 1]
            dy2 = target[other, 2] - target[centre, 2]
            mixed[position, 0, 0] += w * dy0 * dx0
            mixed[position, 0, 1] += w * dy0 * dx1
            mixed[position, 0, 2] += w * dy0 * dx2
            mixed[position, 1, 0] += w * dy1 * dx0
            mixed[position, 1, 1] += w * dy1 * dx1
            mixed[position, 1, 2] += w * dy1 * dx2
            mixed[position, 2, 0] += w * dy2 * dx0
            mixed[position, 2, 1] += w * dy2 * dx1
            mixed[position, 2, 2] += w * dy2 * dx2


@njit(cache=True, parallel=True)
def _residuals(
    indptr,
    neighbour,
    weight,
    reference,
    target,
    centres,
    gradients,
    residual,
    weight_sum,
):
    for position in prange(len(centres)):
        centre = centres[position]
        total = 0.0
        weights = 0.0
        for pair in range(indptr[position], indptr[position + 1]):
            other = neighbour[pair]
            w = weight[pair]
            dx0 = reference[other, 0] - reference[centre, 0]
            dx1 = reference[other, 1] - reference[centre, 1]
            dx2 = reference[other, 2] - reference[centre, 2]
            for row in range(3):
                predicted = (
                    gradients[position, row, 0] * dx0
                    + gradients[position, row, 1] * dx1
                    + gradients[position, row, 2] * dx2
                )
                observed = target[other, row] - target[centre, row]
                total += w * (observed - predicted) ** 2
            weights += w
        residual[position] = total
        weight_sum[position] = weights


@dataclass
class ReferenceMoments:
    """Reference coordinates and accumulated moments reused across targets."""

    graph: object
    reference: np.ndarray
    moments: np.ndarray
    valid: np.ndarray

    def solve(self, target, *, det_tolerance=DET_TOLERANCE, nonaffine=False):
        """Return (F, valid, residual); residual is None unless requested."""
        if not np.isfinite(det_tolerance) or det_tolerance < 0:
            raise ValueError("det_tolerance must be finite and nonnegative.")
        graph = self.graph
        target = as_xyz(target, graph.n_atoms)
        mixed = np.zeros_like(self.moments)
        _accumulate_mixed(
            graph.indptr(),
            graph.pair_neighbour,
            graph.pair_weight,
            self.reference,
            target,
            graph.centre_indices,
            mixed,
        )
        F = np.full_like(mixed, np.nan)
        valid = self.valid.copy()
        usable = np.flatnonzero(valid)
        if usable.size:
            solved = np.linalg.solve(
                self.moments[usable], np.swapaxes(mixed[usable], 1, 2)
            )
            F[usable] = np.swapaxes(solved, 1, 2)
            valid[usable] &= np.isfinite(F[usable]).all(axis=(1, 2)) & (
                np.linalg.det(F[usable]) > det_tolerance
            )
        F[~valid] = np.nan
        residual = None
        if nonaffine:
            total = np.zeros(graph.n_centres)
            weights = np.zeros(graph.n_centres)
            _residuals(
                graph.indptr(),
                graph.pair_neighbour,
                graph.pair_weight,
                self.reference,
                target,
                graph.centre_indices,
                np.nan_to_num(F),
                total,
                weights,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                residual = np.where(weights > 0, total / weights, np.nan)
            residual[~valid] = np.nan
        return F, valid, residual


def reference_moments(
    graph,
    reference,
    *,
    rank_tolerance=RANK_TOLERANCE,
    minimum_neighbours=MINIMUM_NEIGHBOURS,
):
    """Accumulate weighted reference moments; reject unsupported or singular centres."""
    reference = as_xyz(reference, graph.n_atoms).copy()
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be positive and finite.")
    if (
        isinstance(minimum_neighbours, bool)
        or not isinstance(minimum_neighbours, (int, np.integer))
        or minimum_neighbours < 3
    ):
        raise ValueError("minimum_neighbours must be an integer >= 3.")
    moments = np.zeros((graph.n_centres, 3, 3))
    _accumulate_reference(
        graph.indptr(),
        graph.pair_neighbour,
        graph.pair_weight,
        reference,
        graph.centre_indices,
        moments,
    )
    eigenvalues = np.linalg.eigvalsh(moments)
    scale = np.maximum(np.abs(eigenvalues).max(axis=1), 1.0)
    valid = (graph.counts() >= minimum_neighbours) & (
        np.abs(eigenvalues).min(axis=1) >= rank_tolerance * scale
    )
    return ReferenceMoments(graph, reference, moments, valid)


def deformation_gradient(
    graph,
    reference,
    target,
    *,
    rank_tolerance=RANK_TOLERANCE,
    det_tolerance=DET_TOLERANCE,
    minimum_neighbours=MINIMUM_NEIGHBOURS,
    nonaffine=False,
):
    """Fit one pair; return deformation gradients, validity and optional residuals."""
    return reference_moments(
        graph,
        reference,
        rank_tolerance=rank_tolerance,
        minimum_neighbours=minimum_neighbours,
    ).solve(target, det_tolerance=det_tolerance, nonaffine=nonaffine)
