"""Sparse spatial-tree neighbourhoods for local affine fits."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import KDTree
from .._validation import as_xyz_stack, as_indices


@dataclass
class WeightRule:
    """Choose fixed-radius, tapered, intersection or nearest-neighbour weights."""

    kind: str
    radius: float | None = None
    inner_radius: float | None = None
    neighbours: int | None = None

    def __post_init__(self):
        if self.kind not in ("fixed_radius", "tapered", "intersection", "knn"):
            raise ValueError("Unknown weight rule.")
        if self.kind == "knn":
            if (
                isinstance(self.neighbours, bool)
                or not isinstance(self.neighbours, (int, np.integer))
                or self.neighbours < 1
            ):
                raise ValueError("neighbours must be a positive integer.")
        elif self.radius is None or not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive and finite.")
        if self.kind == "tapered" and (
            self.inner_radius is None
            or not np.isfinite(self.inner_radius)
            or not 0 <= self.inner_radius < self.radius
        ):
            raise ValueError("Tapering needs 0 <= inner_radius < radius.")

    @classmethod
    def fixed_radius(cls, radius):
        return cls("fixed_radius", radius=radius)

    @classmethod
    def tapered(cls, inner_radius, radius):
        return cls("tapered", radius=radius, inner_radius=inner_radius)

    @classmethod
    def intersection(cls, radius):
        return cls("intersection", radius=radius)

    @classmethod
    def knn(cls, neighbours):
        return cls("knn", neighbours=neighbours)

    @property
    def cutoff(self):
        return self.radius

    @property
    def order_independent(self):
        return self.kind != "knn"


@dataclass
class NeighbourGraph:
    """Sparse local-centre→atom edges and their weights."""

    centre_indices: np.ndarray
    pair_centre: np.ndarray
    pair_neighbour: np.ndarray
    pair_weight: np.ndarray
    n_atoms: int

    def __post_init__(self):
        centres = np.asarray(self.centre_indices)
        self.centre_indices = as_indices(centres, self.n_atoms)
        if not np.array_equal(centres, self.centre_indices):
            raise ValueError("Graph centre indices must be distinct and increasing.")
        self.pair_centre = np.asarray(self.pair_centre).copy()
        self.pair_neighbour = np.asarray(self.pair_neighbour).copy()
        if any(
            a.size and a.dtype.kind not in "iu"
            for a in (self.pair_centre, self.pair_neighbour)
        ):
            raise TypeError("Graph edge indices must be integers.")
        self.pair_weight = np.asarray(self.pair_weight, float).copy()
        if (
            not (
                self.pair_centre.shape
                == self.pair_neighbour.shape
                == self.pair_weight.shape
            )
            or self.pair_centre.ndim != 1
        ):
            raise ValueError("Edge arrays must be equal-length vectors.")
        if len(self.pair_centre) and (
            self.pair_centre.min() < 0
            or self.pair_centre.max() >= self.n_centres
            or self.pair_neighbour.min() < 0
            or self.pair_neighbour.max() >= self.n_atoms
        ):
            raise ValueError("Edge index out of bounds.")
        if not np.isfinite(self.pair_weight).all() or (self.pair_weight < 0).any():
            raise ValueError("Weights must be finite and nonnegative.")
        self.pair_centre = self.pair_centre.astype(np.int32)
        self.pair_neighbour = self.pair_neighbour.astype(np.int32)
        order = np.argsort(self.pair_centre, kind="stable")
        self.pair_centre = self.pair_centre[order]
        self.pair_neighbour = self.pair_neighbour[order]
        self.pair_weight = self.pair_weight[order]

    @property
    def n_centres(self):
        return len(self.centre_indices)

    def counts(self):
        return np.bincount(self.pair_centre, minlength=self.n_centres)

    def indptr(self):
        return np.r_[0, np.cumsum(self.counts())]


def _pairs_within(centre_points, pool_points, radius):
    tree = KDTree(pool_points)
    lists = tree.query_ball_point(centre_points, float(radius))
    lengths = np.fromiter(
        (len(item) for item in lists), dtype=np.int64, count=len(lists)
    )
    centre = np.repeat(np.arange(len(lists), dtype=np.int64), lengths)
    pool = (
        np.concatenate([np.asarray(item, dtype=np.int64) for item in lists])
        if lengths.sum()
        else np.empty(0, dtype=np.int64)
    )
    return (centre, pool)


def _taper(distance, inner, outer):
    weight = (outer - distance) / (outer - inner)
    return np.clip(weight, 0.0, 1.0)


def build_graph(
    ensemble, rule, *, centre_indices=None, neighbour_indices=None, minimum_weight=1e-12
):
    """Build a neighbour graph from an explicit ensemble of corresponding coordinates."""
    raw = np.asarray(ensemble, dtype=float)
    coordinates = as_xyz_stack(raw[None] if raw.ndim == 2 else raw)
    if not len(coordinates):
        raise ValueError("The ensemble must have at least one member.")
    n_members, n_atoms = (coordinates.shape[0], coordinates.shape[1])
    centres = (
        np.arange(n_atoms, dtype=np.int32)
        if centre_indices is None
        else as_indices(centre_indices, n_atoms, name="Centre indices")
    )
    pool = (
        np.arange(n_atoms, dtype=np.int32)
        if neighbour_indices is None
        else as_indices(neighbour_indices, n_atoms, name="Neighbour indices")
    )
    if rule.kind == "knn":
        points = coordinates[0]
        in_pool = np.isin(centres, pool)
        available = len(pool) - int(in_pool.any())
        if available < rule.neighbours:
            raise ValueError(
                f"A {rule.neighbours}-nearest rule needs {rule.neighbours} neighbours per centre, but the pool offers at most {available}. Widen the pool or ask for fewer neighbours; returning fewer would weaken the fit without saying so."
            )
        wanted = min(rule.neighbours + int(in_pool.any()), len(pool))
        tree = KDTree(points[pool])
        _, columns = tree.query(points[centres], k=wanted)
        columns = np.asarray(columns).reshape(len(centres), -1)
        neighbour_atoms = pool[columns]
        keep_slot = neighbour_atoms != centres[:, None]
        surplus = keep_slot.sum(axis=1) > rule.neighbours
        if surplus.any():
            last = np.where(surplus, keep_slot.shape[1] - 1, -1)
            rows = np.flatnonzero(surplus)
            keep_slot[rows, last[rows]] = False
        centre_local = np.repeat(np.arange(len(centres)), keep_slot.sum(axis=1))
        neighbour_atoms = neighbour_atoms[keep_slot]
        weights = np.ones(len(centre_local))
    else:
        keys = []
        values = []
        for member in range(n_members):
            points = coordinates[member]
            centre_local, pool_local = _pairs_within(
                points[centres], points[pool], rule.cutoff
            )
            if not centre_local.size:
                continue
            neighbour_atoms = pool[pool_local]
            key = centre_local * np.int64(n_atoms) + neighbour_atoms
            if rule.kind == "tapered":
                distance = np.linalg.norm(
                    points[centres][centre_local] - points[neighbour_atoms], axis=1
                )
                value = _taper(distance, rule.inner_radius, rule.radius)
            else:
                value = np.ones(len(centre_local))
            keys.append(key)
            values.append(value)
        if not keys:
            raise ValueError(
                f"No pair lies within {rule.cutoff} A; check the rule and the units."
            )
        keys = np.concatenate(keys)
        values = np.concatenate(values)
        unique, inverse = np.unique(keys, return_inverse=True)
        totals = np.zeros(len(unique))
        occurrences = np.zeros(len(unique), dtype=np.int64)
        np.add.at(totals, inverse, values)
        np.add.at(occurrences, inverse, 1)
        if rule.kind == "intersection":
            keep = occurrences == n_members
            weights = np.ones(int(keep.sum()))
        else:
            keep = np.ones(len(unique), dtype=bool)
            weights = totals / n_members
        unique = unique[keep]
        weights = weights[keep] if rule.kind != "intersection" else weights
        centre_local = (unique // np.int64(n_atoms)).astype(np.int64)
        neighbour_atoms = (unique % np.int64(n_atoms)).astype(np.int64)
    self_pair = centres[centre_local] == neighbour_atoms
    keep = ~self_pair & (weights > minimum_weight)
    if not keep.any():
        raise ValueError("Every neighbour weight was zero; check the rule parameters.")
    return NeighbourGraph(
        centres,
        centre_local[keep].astype(np.int32),
        neighbour_atoms[keep].astype(np.int32),
        weights[keep],
        n_atoms,
    )


def resolve_ensemble(coordinates, ensemble, reference_id, target_id):
    """Return the item identifiers a named ensemble refers to. There is no default."""
    if ensemble is None:
        raise ValueError(
            "An ensemble must be named; there is no default. Use 'endpoints', 'all', 'reference', 'target', or an explicit tuple of item identifiers."
        )
    if not isinstance(ensemble, str):
        ids = tuple((str(item) for item in ensemble))
        if not ids:
            raise ValueError("An explicit ensemble must name at least one item.")
        return ids
    if ensemble == "endpoints":
        return (str(reference_id), str(target_id))
    if ensemble == "all":
        return coordinates.item_ids
    if ensemble == "reference":
        return (str(reference_id),)
    if ensemble == "target":
        return (str(target_id),)
    raise ValueError(
        f"Unknown ensemble {ensemble!r}; expected one of {['endpoints','all','reference','target']}."
    )
