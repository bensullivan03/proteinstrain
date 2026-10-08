"""Array ownership and shape validation shared by numerical containers."""

import numpy as np


def array_copy(value, dtype=None):
    """Copy an external array into contiguous, ordinary storage of the stated dtype."""
    array = np.array(value, dtype=dtype, order="C", copy=True)
    return array


def as_xyz(value, n=None, *, name="Coordinates", minimum=0):
    """Validate finite (n, 3) coordinates; n and minimum constrain their length."""
    array = np.ascontiguousarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or (n is not None and len(array) != n):
        raise ValueError(f"{name} must have shape ({('n' if n is None else n)}, 3).")
    if len(array) < minimum:
        raise ValueError(f"{name} needs at least {minimum} points.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def as_indices(value, n, *, name="Indices", empty=False):
    """Return sorted unique integer indices; validate before narrowing the dtype."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    if not len(raw):
        if not empty:
            raise ValueError(f"{name} must not be empty.")
        return np.empty(0, dtype=np.int32)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    if raw.min() < 0 or raw.max() >= n or raw.max() > np.iinfo(np.int32).max:
        raise ValueError(f"{name} falls outside the coordinate range.")
    return np.unique(raw).astype(np.int32)


def unit_vector(value, name="vector"):
    """Return the finite, nonzero three-vector value normalised to unit length."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite three-dimensional vector.")
    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        raise ValueError(f"{name} must not be the zero vector.")
    return array / norm


def as_weights(value, n):
    """Validate n finite non-negative weights with positive total; None gives ones."""
    if value is None:
        return np.ones(n)
    array = np.asarray(value, dtype=float)
    if array.shape != (n,) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Weights must be finite, non-negative and one per point.")
    if array.sum() <= 0:
        raise ValueError("Weights must not all be zero.")
    return array


def as_xyz_stack(value):
    """Validate a finite (items, atoms, 3) coordinate stack."""
    array = np.ascontiguousarray(value, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 3 or (not np.isfinite(array).all()):
        raise ValueError("Coordinates must be a finite (items, atoms, 3) array.")
    return array


def as_tensors(value):
    """Validate an array ending in (3, 3); NaNs may represent missing tensors."""
    array = np.ascontiguousarray(value, dtype=float)
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise ValueError("Tensors must have shape (..., 3, 3).")
    return array


def direction_field(value, n, name="directions"):
    """Return n unit directions from one vector or a finite (n, 3) field."""
    array = np.asarray(value, dtype=float)
    if array.shape == (3,):
        return np.broadcast_to(unit_vector(array, name), (n, 3))
    array = as_xyz(array, n, name=name)
    norm = np.linalg.norm(array, axis=1)
    if (norm < 1e-12).any():
        raise ValueError(f"{name} contains a zero vector.")
    return array / norm[:, None]
