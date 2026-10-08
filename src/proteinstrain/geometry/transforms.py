"""Proper rigid transforms, weighted fits and structure superposition."""

from dataclasses import dataclass
import gemmi
import numpy as np
from .._validation import as_xyz, as_weights, array_copy, unit_vector


@dataclass()
class RigidTransform:
    """A proper rotation and a translation, applied as x -> R x + t."""

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float | None = None

    def __post_init__(self):
        rotation = array_copy(self.rotation, dtype=float)
        translation = array_copy(self.translation, dtype=float).reshape(3)
        if rotation.shape != (3, 3):
            raise ValueError("A rotation must have shape (3, 3).")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("A transform must be finite.")
        if np.abs(rotation @ rotation.T - np.eye(3)).max() > 1e-06:
            raise ValueError("A rotation must be orthonormal.")
        if abs(np.linalg.det(rotation) - 1.0) > 1e-06:
            raise ValueError("A rotation must be proper; reflections are not allowed.")
        self.rotation = rotation
        self.translation = translation

    @classmethod
    def identity(cls):
        """The transform that changes nothing."""
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_rotvec(cls, rotvec, translation=(0.0, 0.0, 0.0)):
        """Build a transform from a rotation vector whose norm is the angle in radians."""
        vector = np.asarray(rotvec, dtype=float).reshape(3)
        angle = float(np.linalg.norm(vector))
        if angle < 1e-15:
            return cls(np.eye(3), translation)
        axis = vector / angle
        cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
        )
        return cls(rotation, translation)

    @classmethod
    def about_axis(cls, axis, angle, centre=(0.0, 0.0, 0.0), *, degrees=False):
        """Rotate by an angle about an axis through a centre."""
        unit = unit_vector(axis)
        radians = np.deg2rad(angle) if degrees else float(angle)
        rotation = cls.from_rotvec(unit * radians).rotation
        centre = np.asarray(centre, dtype=float).reshape(3)
        return cls(rotation, centre - rotation @ centre)

    def apply(self, value):
        """Apply the transform to a point, an (n, 3) array or a (k, n, 3) stack."""
        array = np.asarray(value, dtype=float)
        if array.shape[-1] != 3:
            raise ValueError("The last axis must have length three.")
        return array @ self.rotation.T + self.translation

    def inverse(self):
        """The transform that undoes this one."""
        rotation = self.rotation.T
        return RigidTransform(rotation, -rotation @ self.translation, self.rmsd)

    def then(self, second):
        """Return the transform applying this one followed by second."""
        return RigidTransform(
            second.rotation @ self.rotation,
            second.rotation @ self.translation + second.translation,
        )

    def as_rotvec(self):
        """The rotation as a vector whose norm is the angle in radians."""
        trace = np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(trace))
        if angle < 1e-12:
            return np.zeros(3)
        if abs(angle - np.pi) < 1e-07:
            values, vectors = np.linalg.eigh(self.rotation)
            axis = vectors[:, int(np.argmax(values))]
            return axis / np.linalg.norm(axis) * angle
        skew = self.rotation - self.rotation.T
        axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / (2.0 * np.sin(angle))
        return axis * angle

    def angle(self, *, degrees=False):
        """The rotation angle."""
        value = float(np.linalg.norm(self.as_rotvec()))
        return float(np.rad2deg(value)) if degrees else value

    def axis(self):
        """The unit rotation axis, or zeros when the rotation is the identity."""
        vector = self.as_rotvec()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else np.zeros(3)

    def to_gemmi(self):
        """Return the equivalent gemmi.Transform."""
        transform = gemmi.Transform()
        transform.mat.fromlist(self.rotation.tolist())
        transform.vec.fromlist(self.translation.tolist())
        return transform


def _rmsd(moving, fixed, transform, weights):
    residual = transform.apply(moving) - fixed
    squared = (residual * residual).sum(axis=1)
    return float(np.sqrt((weights * squared).sum() / weights.sum()))


def fit_rigid_transform(moving, fixed, *, weights=None):
    """Least-squares proper rigid fit taking moving onto fixed; reflections are excluded."""
    moving = as_xyz(moving, minimum=3, name="moving coordinates")
    fixed = as_xyz(fixed, minimum=3, name="fixed coordinates")
    if moving.shape != fixed.shape:
        raise ValueError("Both coordinate sets must have the same shape.")
    weight = as_weights(weights, len(moving))
    total = weight.sum()
    moving_centre = (weight[:, None] * moving).sum(axis=0) / total
    fixed_centre = (weight[:, None] * fixed).sum(axis=0) / total
    centred_moving = moving - moving_centre
    centred_fixed = fixed - fixed_centre
    correlation = centred_moving.T @ (weight[:, None] * centred_fixed)
    left, _, right = np.linalg.svd(correlation)
    sign = np.sign(np.linalg.det(right.T @ left.T))
    rotation = right.T @ np.diag([1.0, 1.0, sign if sign else 1.0]) @ left.T
    transform = RigidTransform(rotation, fixed_centre - rotation @ moving_centre, None)
    return RigidTransform(
        transform.rotation,
        transform.translation,
        _rmsd(moving, fixed, transform, weight),
    )


def apply_transform(structure, transform):
    """Apply a transform to every atom of a structure."""
    from dataclasses import replace as _replace

    target = _replace(structure, model=structure.model.clone())
    rotation = transform.rotation
    translation = transform.translation
    for chain in target.model:
        for residue in chain:
            for atom in residue:
                position = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                moved = rotation @ position + translation
                atom.pos = gemmi.Position(*moved.tolist())
    return target


def _fit_trim(moving, fixed, weights, trim):
    if trim is not None and (not np.isfinite(trim) or trim <= 0):
        raise ValueError("trim must be positive and finite.")
    transform = fit_rigid_transform(moving, fixed, weights=weights)
    if trim is not None:
        keep = np.linalg.norm(transform.apply(moving) - fixed, axis=1) <= trim
        if keep.sum() < 3:
            raise ValueError("Trimming left fewer than three points.")
        transform = fit_rigid_transform(
            moving[keep],
            fixed[keep],
            weights=None if weights is None else np.asarray(weights)[keep],
        )
    return transform


def superpose_coordinates(
    coordinates, *, moving_id, reference_id, columns=None, weights=None, trim=None
):
    """Return the rigid transform taking moving onto reference; columns are block indices."""
    from .._validation import as_indices

    indices = (
        np.arange(coordinates.n_atoms)
        if columns is None
        else as_indices(columns, coordinates.n_atoms)
    )
    return _fit_trim(
        coordinates.coordinates(moving_id)[indices],
        coordinates.coordinates(reference_id)[indices],
        weights,
        trim,
    )


def superpose_structures(
    moving,
    reference,
    *,
    moving_chains=None,
    reference_chains=None,
    atoms="CA",
    chain_map=None,
    matching=None,
    minimum_atoms=10,
    trim=None,
    **scoring,
):
    """Return (transformed_structure, transform); chain_map points reference→moving."""
    from ..correspondence.coordinates import from_pairwise

    block = from_pairwise(
        reference,
        moving,
        reference_chains=reference_chains,
        target_chains=moving_chains,
        chain_map=chain_map,
        matching=matching,
        atoms=atoms,
        item_ids=("reference", "moving"),
        **scoring,
    )
    if block.n_atoms < minimum_atoms:
        raise ValueError(f"Fit needs at least {minimum_atoms} matched atoms.")
    transform = _fit_trim(block.xyz[1], block.xyz[0], None, trim)
    return (apply_transform(moving, transform), transform)


def displacement(coordinates, reference_id, target_id, *, remove=None):
    """Return target-minus-reference vectors, optionally removing a rigid transform."""
    reference = coordinates.coordinates(reference_id)
    target = coordinates.coordinates(target_id)
    if remove is not None:
        target = remove.apply(target)
    return target - reference
