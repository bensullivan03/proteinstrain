"""Shape fits: cylinders, axes, planes and inertia."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .._validation import as_xyz, as_weights, array_copy, unit_vector


@dataclass()
class CylinderFit:
    """A fitted cylinder: axis direction, a point on the axis, radius and residual."""

    axis: np.ndarray
    centre: np.ndarray
    radius: float
    rmsd: float

    def __post_init__(self):
        self.axis = array_copy(unit_vector(self.axis, "cylinder axis"))
        self.centre = array_copy(self.centre, dtype=float).reshape(3)
        self.radius = float(self.radius)
        self.rmsd = float(self.rmsd)

    def radial_distance(self, xyz):
        """Perpendicular distance of each point from the axis."""
        offset = np.asarray(xyz, dtype=float) - self.centre
        along = offset @ self.axis
        return np.linalg.norm(offset - along[:, None] * self.axis, axis=1)

    def axial_position(self, xyz):
        """Signed position of each point along the axis."""
        return (np.asarray(xyz, dtype=float) - self.centre) @ self.axis


@dataclass()
class PlaneFit:
    """A fitted plane: unit normal, a point on the plane and the residual."""

    normal: np.ndarray
    centre: np.ndarray
    rmsd: float

    def __post_init__(self):
        self.normal = array_copy(unit_vector(self.normal, "plane normal"))
        self.centre = array_copy(self.centre, dtype=float).reshape(3)
        self.rmsd = float(self.rmsd)

    def signed_distance(self, xyz):
        """Signed distance of each point from the plane."""
        return (np.asarray(xyz, dtype=float) - self.centre) @ self.normal


def _direction(theta, phi):
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def _angles(direction):
    unit = unit_vector(direction)
    return (
        float(np.arccos(np.clip(unit[2], -1.0, 1.0))),
        float(np.arctan2(unit[1], unit[0])),
    )


def inertia_axes(xyz):
    """Return the principal axes as rows, largest spread first, with the spreads."""
    points = as_xyz(xyz, minimum=3)
    centred = points - points.mean(axis=0)
    _, values, right = np.linalg.svd(centred, full_matrices=False)
    spread = values / np.sqrt(len(points))
    axes = right.copy()
    if np.linalg.det(axes) < 0:
        axes[2] = -axes[2]
    return (axes, spread)


def fit_plane(xyz):
    """Least-squares plane through a point cloud."""
    points = as_xyz(xyz, minimum=3)
    centre = points.mean(axis=0)
    axes, _ = inertia_axes(points)
    normal = axes[2]
    residual = (points - centre) @ normal
    return PlaneFit(normal, centre, float(np.sqrt((residual**2).mean())))


def fit_cylinder(xyz, *, axis=None, centre=None, max_evaluations=200):
    """Fit a cylinder by least squares over axis direction and lateral centre offset."""
    points = as_xyz(xyz, minimum=5, name="cylinder points")
    start_axis = unit_vector(axis) if axis is not None else inertia_axes(points)[0][0]
    start_centre = (
        np.asarray(centre, dtype=float).reshape(3)
        if centre is not None
        else points.mean(axis=0)
    )
    theta, phi = _angles(start_axis)
    if not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer.")

    def lateral_basis(direction):
        """Lateral basis."""
        helper = (
            np.array([0.0, 0.0, 1.0])
            if abs(direction[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        first = unit_vector(np.cross(direction, helper))
        return (first, np.cross(direction, first))

    def axis_point(parameters, direction):
        """Axis point."""
        first, second = lateral_basis(direction)
        return start_centre + parameters[2] * first + parameters[3] * second

    def residual(parameters):
        """Residual."""
        direction = _direction(parameters[0], parameters[1])
        point = axis_point(parameters, direction)
        offset = points - point
        along = offset @ direction
        distance = np.linalg.norm(offset - along[:, None] * direction, axis=1)
        return distance - distance.mean()

    solution = least_squares(
        residual, np.array([theta, phi, 0.0, 0.0]), max_nfev=max_evaluations
    )
    direction = _direction(solution.x[0], solution.x[1])
    point = axis_point(solution.x, direction)
    offset = points - point
    along = offset @ direction
    distance = np.linalg.norm(offset - along[:, None] * direction, axis=1)
    radius = float(distance.mean())
    if not solution.success:
        raise RuntimeError(f"Cylinder fit did not converge: {solution.message}")
    return CylinderFit(
        direction,
        point + float(along.mean()) * direction,
        radius,
        float(np.sqrt(((distance - radius) ** 2).mean())),
    )


def circular_mean(angles, *, degrees=False, weights=None):
    """Mean of angles on the circle, returned in the same units."""
    values = np.asarray(angles, dtype=float).reshape(-1)
    radians = np.deg2rad(values) if degrees else values
    weight = as_weights(weights, len(radians))
    if not len(radians) or not np.isfinite(radians).all():
        raise ValueError("Angles must be a nonempty finite array.")
    mean = np.arctan2(
        (weight * np.sin(radians)).sum(), (weight * np.cos(radians)).sum()
    )
    return float(np.rad2deg(mean)) if degrees else float(mean)


def cylindrical_angles(
    xyz, *, axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), reference=None, degrees=False
):
    """Azimuthal angle of each point about an axis, measured from a reference direction."""
    points = as_xyz(xyz)
    direction = unit_vector(axis, "axis")
    centre = np.asarray(origin, dtype=float).reshape(3)
    offset = points - centre
    radial = offset - (offset @ direction)[:, None] * direction
    if reference is None:
        first = radial[0]
        if np.linalg.norm(first) < 1e-09:
            raise ValueError("The first point lies on the axis; supply a reference.")
        reference = first
    reference = np.asarray(reference, dtype=float)
    if (np.linalg.norm(radial, axis=1) < 1e-09).any():
        raise ValueError("An angular position is undefined on the axis.")
    zero = unit_vector(
        reference - float(reference @ direction) * direction, "reference"
    )
    ninety = np.cross(direction, zero)
    angles = np.arctan2(radial @ ninety, radial @ zero)
    return np.rad2deg(angles) if degrees else angles


def chain_cylindrical_angles(
    structure,
    chains=None,
    *,
    axis=(0.0, 0.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    atoms="CA",
    reference=None,
    degrees=False,
):
    """Azimuthal angle of each selected chain's centroid about an axis."""
    from ..selection import resolve_chains

    selected = resolve_chains(structure, chains)
    centroids = []
    for chain in selected:
        centroids.append(structure.centroid(chains=[chain], atoms=atoms))
    angles = cylindrical_angles(
        np.asarray(centroids),
        axis=axis,
        origin=origin,
        reference=reference,
        degrees=degrees,
    )
    return dict(zip(selected, angles.tolist()))
