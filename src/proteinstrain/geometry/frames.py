"""Orthonormal frames and vector/tensor basis transformations."""

from dataclasses import dataclass
import numpy as np
from .._validation import as_xyz, direction_field, unit_vector
from .transforms import RigidTransform
from .fits import inertia_axes, fit_cylinder


@dataclass
class Frame:
    """origin and row-wise axes; angular_reference says whether cylinder longitude is anchored."""

    origin: np.ndarray
    axes: np.ndarray
    angular_reference: bool = False

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float).reshape(3).copy()
        self.axes = np.asarray(self.axes, float).copy()
        if (
            self.axes.shape != (3, 3)
            or not np.isfinite(self.origin).all()
            or not np.isfinite(self.axes).all()
            or not np.allclose(self.axes @ self.axes.T, np.eye(3), atol=1e-6)
            or np.linalg.det(self.axes) < 0
        ):
            raise ValueError("Frame needs finite, right-handed orthonormal axes.")

    @classmethod
    def cartesian(cls):
        return cls(np.zeros(3), np.eye(3), True)

    @classmethod
    def from_axis(cls, axis, origin=(0.0, 0.0, 0.0), reference=None):
        basis = _orthonormal(axis, reference)
        return cls(
            origin, np.vstack((basis[1], basis[2], basis[0])), reference is not None
        )

    @classmethod
    def from_inertia(cls, xyz):
        points = as_xyz(xyz, minimum=3)
        axes, _ = inertia_axes(points)
        return cls(points.mean(axis=0), axes)

    @classmethod
    def from_reference(cls, structure, *, chains=None, atoms="CA"):
        return cls.from_inertia(structure.coordinates(chains, atoms))

    @classmethod
    def from_chains(cls, structure, start, end, *, reference=None, atoms="CA"):
        origin = structure.centroid(start, atoms)
        helper = _reference_vector(structure, reference, origin, atoms)
        return cls.from_axis(structure.centroid(end, atoms) - origin, origin, helper)

    @classmethod
    def from_cylinder(cls, structure, chains, *, towards, reference=None, atoms="CA"):
        fit = fit_cylinder(structure.coordinates(chains, atoms))
        projection = (structure.centroid(towards, atoms) - fit.centre) @ fit.axis
        if abs(projection) < 1e-9:
            raise ValueError(
                "Towards centroid cannot determine the cylinder axis sign."
            )
        direction = fit.axis if projection > 0 else -fit.axis
        return cls.from_axis(
            direction,
            fit.centre,
            _reference_vector(structure, reference, fit.centre, atoms),
        )

    def to_local(self, xyz):
        return (np.asarray(xyz) - self.origin) @ self.axes.T

    def to_global(self, xyz):
        return np.asarray(xyz) @ self.axes + self.origin

    def transform(self):
        return RigidTransform(self.axes, -self.axes @ self.origin)


def _reference_vector(structure, reference, origin, atoms):
    if reference is None:
        return None
    try:
        vector = np.asarray(reference, float)
    except (TypeError, ValueError):
        vector = np.empty(0)
    return (
        vector
        if vector.shape == (3,)
        else structure.centroid(reference, atoms) - origin
    )


def _orthonormal(primary, secondary=None):
    first = unit_vector(primary, "primary direction")
    if secondary is None:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(first @ helper)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = unit_vector(secondary, "secondary direction")
    second = helper - float(first @ helper) * first
    norm = float(np.linalg.norm(second))
    if norm < 1e-09:
        raise ValueError("The two directions are parallel.")
    second = second / norm
    return np.vstack([first, second, np.cross(first, second)])


def cylindrical_frames(xyz, *, axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Per-atom radial, azimuthal and axial bases. Returns (n, 3, 3)."""
    points = as_xyz(xyz)
    direction = unit_vector(axis, "cylinder axis")
    centre = np.asarray(origin, dtype=float).reshape(3)
    offset = points - centre
    axial = float(1.0) * direction
    along = offset @ direction
    radial = offset - along[:, None] * direction
    norm = np.linalg.norm(radial, axis=1)
    if (norm < 1e-09).any():
        raise ValueError(
            "Some atoms lie on the axis, where no radial direction exists."
        )
    radial = radial / norm[:, None]
    azimuthal = np.cross(np.broadcast_to(axial, radial.shape), radial)
    azimuthal = azimuthal / np.linalg.norm(azimuthal, axis=1)[:, None]
    return np.stack([radial, azimuthal, np.broadcast_to(axial, radial.shape)], axis=1)


def local_frames_from_directions(primary, secondary):
    """Return vectorised (n, 3, 3) orthonormal bases from two (n, 3) direction fields."""
    first = as_xyz(primary, name="primary directions")
    second = direction_field(secondary, len(first), "secondary directions")
    first = direction_field(first, len(first), "primary directions")
    lateral = second - np.einsum("ni,ni->n", first, second)[:, None] * first
    norm = np.linalg.norm(lateral, axis=1)
    if (norm < 1e-09).any():
        raise ValueError("The two direction fields contain parallel vectors.")
    lateral /= norm[:, None]
    return np.stack([first, lateral, np.cross(first, lateral)], axis=1)


def basis_of(frame):
    """Return the (3, 3) basis of a Frame, a plain array, or per-atom bases unchanged."""
    if isinstance(frame, Frame):
        return frame.axes
    return np.asarray(frame, dtype=float)


def transport(tensors, frame_to, frame_from=None):
    """Express tensors in another basis as A E A-transpose; raises on a shape mismatch."""
    array = np.asarray(tensors, dtype=float)
    if array.shape[-2:] != (3, 3):
        raise ValueError("Tensors must have shape (..., 3, 3).")
    target = basis_of(frame_to)
    source = np.eye(3) if frame_from is None else basis_of(frame_from)
    if target.ndim == 2 and source.ndim == 2:
        change = target @ source.T
        return change @ array @ change.T
    if array.ndim != 3:
        raise ValueError("Per-atom bases require tensors of shape (n, 3, 3).")
    target = np.broadcast_to(target, (len(array), 3, 3)) if target.ndim == 2 else target
    source = np.broadcast_to(source, (len(array), 3, 3)) if source.ndim == 2 else source
    change = np.einsum("nij,nkj->nik", target, source)
    return np.einsum("nij,njk,nlk->nil", change, array, change)


def transport_vectors(vectors, frame_to, frame_from=None):
    """Express vectors in another basis. Origins are ignored; this rotates only."""
    array = np.asarray(vectors, dtype=float)
    if array.shape[-1] != 3:
        raise ValueError("Vectors must have shape (..., 3).")
    target = basis_of(frame_to)
    source = np.eye(3) if frame_from is None else basis_of(frame_from)
    if target.ndim == 2 and source.ndim == 2:
        return array @ (target @ source.T).T
    target = np.broadcast_to(target, (len(array), 3, 3)) if target.ndim == 2 else target
    source = np.broadcast_to(source, (len(array), 3, 3)) if source.ndim == 2 else source
    change = np.einsum("nij,nkj->nik", target, source)
    return np.einsum("nij,nj->ni", change, array)


def order_about_axis(structure, chains, frame, *, start, direction, atoms="CA"):
    """Return chain IDs in angular order, starting at a selection or numeric angle in radians."""
    from ..selection import resolve_chains

    selected = resolve_chains(structure, chains)
    if direction not in {"clockwise", "anticlockwise"}:
        raise ValueError("direction must be 'clockwise' or 'anticlockwise'.")
    local = frame.to_local(
        np.array([structure.centroid(chain, atoms) for chain in selected])
    )
    if (np.linalg.norm(local[:, :2], axis=1) < 1e-09).any():
        raise ValueError(
            "A chain centroid lies on the axis and has no angular position."
        )
    angle = np.arctan2(local[:, 1], local[:, 0])
    if isinstance(start, (int, float, np.number)) and (not isinstance(start, bool)):
        if not frame.angular_reference:
            raise ValueError(
                "A numeric start needs a frame with angular_reference=True."
            )
        zero = float(start)
        if not np.isfinite(zero):
            raise ValueError("The angular start must be finite.")
    else:
        local_start = frame.to_local(structure.centroid(start, atoms))
        if np.linalg.norm(local_start[:2]) < 1e-09:
            raise ValueError("The start centroid lies on the axis.")
        zero = float(np.arctan2(local_start[1], local_start[0]))
    sign = 1.0 if direction == "anticlockwise" else -1.0
    phase = np.mod(sign * (angle - zero), 2 * np.pi)
    phase[np.isclose(phase, 2 * np.pi, atol=1e-10)] = 0.0
    return tuple((selected[i] for i in np.argsort(phase, kind="stable")))
