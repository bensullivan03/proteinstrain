"""Scalar and tensor measures derived from the deformation gradient."""

import numpy as np
from .._validation import unit_vector, direction_field, as_tensors as _as_tensors

IDENTITY = np.eye(3)


def as_tensors(value, name="tensors"):
    """Return a validated (n, 3, 3) array, accepting a single tensor."""
    array = _as_tensors(value)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (n, 3, 3).")
    return array


def _finite(tensors):
    return np.isfinite(tensors).all(axis=(1, 2))


def _empty(count, shape=()):
    return np.full((count, *shape), np.nan)


DETERMINANT_FLOOR = 1e-12
DEGENERACY_TOLERANCE = 1e-08


def _usable(deformation_gradients):
    F = as_tensors(deformation_gradients, "Deformation gradients")
    usable = _finite(F)
    if usable.any():
        determinant = np.full(len(F), np.nan)
        determinant[usable] = np.linalg.det(F[usable])
        usable &= determinant > DETERMINANT_FLOOR
    return (F, usable)


def _eigh_valid(tensors):
    tensors = as_tensors(tensors)
    finite = _finite(tensors)
    values = _empty(len(tensors), (3,))
    vectors = _empty(len(tensors), (3, 3))
    if finite.any():
        symmetric = 0.5 * (tensors[finite] + np.swapaxes(tensors[finite], 1, 2))
        found, directions = np.linalg.eigh(symmetric)
        values[finite] = found
        vectors[finite] = directions
    return (values, vectors, finite)


def right_cauchy_green(deformation_gradients):
    """Return C = F-transpose F."""
    F = as_tensors(deformation_gradients, "Deformation gradients")
    return np.einsum("nki,nkj->nij", F, F)


def left_cauchy_green(deformation_gradients):
    """Return b = F F-transpose."""
    F = as_tensors(deformation_gradients, "Deformation gradients")
    return np.einsum("nik,njk->nij", F, F)


def green_lagrange_strain(deformation_gradients):
    """Return the reference-frame strain E = (C - I) / 2."""
    return 0.5 * (right_cauchy_green(deformation_gradients) - IDENTITY)


def euler_almansi_strain(deformation_gradients):
    """Return the current-frame strain e = (I - b-inverse) / 2."""
    _, usable = _usable(deformation_gradients)
    b = left_cauchy_green(deformation_gradients)
    result = _empty(len(b), (3, 3))
    usable &= _finite(b)
    if usable.any():
        result[usable] = 0.5 * (IDENTITY - np.linalg.inv(b[usable]))
    return result


def hencky_strain(deformation_gradients):
    """Return the logarithmic strain, half the log of C in its principal basis."""
    values, vectors, finite = _eigh_valid(right_cauchy_green(deformation_gradients))
    result = _empty(len(values), (3, 3))
    usable = finite & (values > 0).all(axis=1)
    if usable.any():
        logs = 0.5 * np.log(values[usable])
        result[usable] = np.einsum(
            "nij,nj,nkj->nik", vectors[usable], logs, vectors[usable]
        )
    return result


def linear_strain(deformation_gradients):
    """Return the small-deformation strain (F + F-transpose)/2 - I."""
    F = as_tensors(deformation_gradients, "Deformation gradients")
    return 0.5 * (F + np.swapaxes(F, 1, 2)) - IDENTITY


def deviatoric(tensors):
    """Return the trace-free part of each tensor."""
    array = as_tensors(tensors)
    trace = np.trace(array, axis1=1, axis2=2)
    return array - (trace / 3.0)[:, None, None] * IDENTITY


def principal_strains(strain_tensors):
    """Return the three ordered eigenvalues of each strain tensor."""
    values, _, _ = _eigh_valid(strain_tensors)
    return values


def principal_directions(strain_tensors):
    """Return the eigenvectors of each strain tensor as rows, matching principal_strains."""
    _, vectors, _ = _eigh_valid(strain_tensors)
    return np.swapaxes(vectors, 1, 2)


def principal_stretch_system(deformation_gradients):
    """Return ascending stretches and matching row-wise directions, using one eigendecomposition."""
    values, vectors, finite = _eigh_valid(right_cauchy_green(deformation_gradients))
    stretches = _empty(len(values), (3,))
    usable = finite & (values >= 0).all(axis=1)
    stretches[usable] = np.sqrt(values[usable])
    return stretches, np.swapaxes(vectors, 1, 2)


def principal_stretches(deformation_gradients):
    """Return the three ascending principal stretches."""
    return principal_stretch_system(deformation_gradients)[0]


def stretch_directions(deformation_gradients):
    """Return row-wise directions matching principal_stretches."""
    return principal_stretch_system(deformation_gradients)[1]


def max_stretch(deformation_gradients):
    """Largest principal stretch."""
    return principal_stretches(deformation_gradients)[:, 2]


def min_stretch(deformation_gradients):
    """Smallest principal stretch."""
    return principal_stretches(deformation_gradients)[:, 0]


def corrected_max_stretch(deformation_gradients):
    """Return 1 - 1/lambda_max, a bounded transform of the largest stretch."""
    largest = max_stretch(deformation_gradients)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(largest > 0, 1.0 - 1.0 / largest, np.nan)


def principal_axis_alignment(
    deformation_gradients, direction, *, tolerance=DEGENERACY_TOLERANCE
):
    """Absolute cosine between the largest-stretch direction and a given direction."""
    axis = unit_vector(direction, "direction")
    stretches, directions = principal_stretch_system(deformation_gradients)
    directions = directions[:, 2, :]
    result = np.abs(directions @ axis)
    with np.errstate(invalid="ignore"):
        degenerate = stretches[:, 2] - stretches[:, 1] <= tolerance
    result[degenerate] = np.nan
    return result


def frobenius(tensors):
    """Frobenius norm of each tensor."""
    array = as_tensors(tensors)
    return np.sqrt(np.einsum("nij,nij->n", array, array))


def green_lagrange_frobenius(deformation_gradients):
    """Frobenius norm of the Green-Lagrange strain."""
    return frobenius(green_lagrange_strain(deformation_gradients))


def trace_strain(strain_tensors):
    """Trace of each strain tensor. Not the volume change when the strain is finite."""
    return np.trace(as_tensors(strain_tensors), axis1=1, axis2=2)


def invariants(tensors):
    """The three principal invariants of each tensor."""
    array = as_tensors(tensors)
    first = np.trace(array, axis1=1, axis2=2)
    squared = np.einsum("nij,njk->nik", array, array)
    second = 0.5 * (first**2 - np.trace(squared, axis1=1, axis2=2))
    finite = _finite(array)
    third = _empty(len(array))
    third[finite] = np.linalg.det(array[finite])
    return np.stack([first, second, third], axis=1)


def volume_ratio(deformation_gradients):
    """Return J = det F, the local affine volume ratio."""
    F = as_tensors(deformation_gradients, "Deformation gradients")
    result = _empty(len(F))
    finite = _finite(F)
    if finite.any():
        result[finite] = np.linalg.det(F[finite])
    return result


def relative_volume_change(deformation_gradients):
    """Return J - 1, the fractional local volume change."""
    return volume_ratio(deformation_gradients) - 1.0


def log_volume_change(deformation_gradients):
    """Return log J, which is additive over successive deformations."""
    ratio = volume_ratio(deformation_gradients)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ratio > 0, np.log(ratio), np.nan)


def directional_strain(strain_tensors, direction):
    """Return n-transpose E n, the axial Green-Lagrange strain along a direction."""
    E = as_tensors(strain_tensors)
    normals = direction_field(direction, len(E))
    return np.einsum("ni,nij,nj->n", normals, E, normals)


def directional_stretch(strain_tensors, direction):
    """Return the length ratio along a direction, sqrt(1 + 2 n E n)."""
    axial = directional_strain(strain_tensors, direction)
    with np.errstate(invalid="ignore"):
        return np.where(1.0 + 2.0 * axial >= 0, np.sqrt(1.0 + 2.0 * axial), np.nan)


def shear_component(strain_tensors, first_direction, second_direction):
    """Return n-transpose E m between two perpendicular directions."""
    E = as_tensors(strain_tensors)
    left = direction_field(first_direction, len(E))
    right = direction_field(second_direction, len(E))
    return np.einsum("ni,nij,nj->n", left, E, right)


def angle_change(strain_tensors, first_direction, second_direction, *, degrees=False):
    """Signed change in the angle between two material directions."""
    E = as_tensors(strain_tensors)
    left = direction_field(first_direction, len(E), "first direction")
    right = direction_field(second_direction, len(E), "second direction")
    C = 2.0 * E + IDENTITY
    numerator = np.einsum("ni,nij,nj->n", left, C, right)
    stretch_left = directional_stretch(E, left)
    stretch_right = directional_stretch(E, right)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.clip(numerator / (stretch_left * stretch_right), -1.0, 1.0)
    reference = np.arccos(np.clip(np.einsum("ni,ni->n", left, right), -1.0, 1.0))
    change = np.arccos(cosine) - reference
    return np.rad2deg(change) if degrees else change


def polar_decomposition(deformation_gradients):
    """Return (R, U) with F = R U, R proper and U symmetric positive definite."""
    F, finite = _usable(deformation_gradients)
    rotation = _empty(len(F), (3, 3))
    stretch = _empty(len(F), (3, 3))
    if finite.any():
        left, values, right = np.linalg.svd(F[finite])
        sign = np.sign(np.linalg.det(np.einsum("nij,njk->nik", left, right)))
        sign[sign == 0] = 1.0
        correction = np.broadcast_to(np.eye(3), (int(finite.sum()), 3, 3)).copy()
        correction[:, 2, 2] = sign
        rotation[finite] = np.einsum("nij,njk,nkl->nil", left, correction, right)
        scaled = values.copy()
        scaled[:, 2] = scaled[:, 2] * sign
        stretch[finite] = np.einsum("nji,nj,njk->nik", right, scaled, right)
    return (rotation, stretch)


def rotation_angle(deformation_gradients, *, degrees=False):
    """Angle of the polar rotation at each centre."""
    rotation, _ = polar_decomposition(deformation_gradients)
    finite = _finite(rotation)
    result = _empty(len(rotation))
    if finite.any():
        trace = np.clip(
            (np.trace(rotation[finite], axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0
        )
        result[finite] = np.arccos(trace)
    return np.rad2deg(result) if degrees else result


def rotation_axis(deformation_gradients, *, minimum_angle=1e-06):
    """Unit axis of the polar rotation at each centre, missing where none exists."""
    rotation, _ = polar_decomposition(deformation_gradients)
    finite = _finite(rotation)
    result = _empty(len(rotation), (3,))
    if not finite.any():
        return result
    matrices = rotation[finite]
    skew = matrices - np.swapaxes(matrices, 1, 2)
    vectors = np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=1)
    norm = np.linalg.norm(vectors, axis=1)
    found = np.full((len(matrices), 3), np.nan)
    ordinary = norm > 1e-09
    if ordinary.any():
        found[ordinary] = vectors[ordinary] / norm[ordinary, None]
    trace = np.clip((np.trace(matrices, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    half = ~ordinary & (angle > minimum_angle)
    if half.any():
        plus = matrices[half] + IDENTITY
        lengths = np.linalg.norm(plus, axis=1)
        best = np.argmax(lengths, axis=1)
        rows = np.arange(len(plus))
        chosen = plus[rows, :, best]
        found[half] = chosen / np.linalg.norm(chosen, axis=1, keepdims=True)
    result[finite] = found
    return result


def twist_swing(deformation_gradients, axis, *, degrees=False):
    """Split the polar rotation into a twist about an axis and the residual swing."""
    direction = unit_vector(axis, "axis")
    rotation, _ = polar_decomposition(deformation_gradients)
    finite = _finite(rotation)
    twist = _empty(len(rotation))
    swing = _empty(len(rotation))
    if finite.any():
        scalar, vector = _rotation_quaternion(rotation[finite])
        projected = vector @ direction
        norm = np.sqrt(scalar**2 + projected**2)
        singular = norm < 1e-12
        safe = np.where(singular, 1.0, norm)
        twist_angle = 2.0 * np.arctan2(projected / safe, scalar / safe)
        twist_angle = (twist_angle + np.pi) % (2.0 * np.pi) - np.pi
        twist_angle[singular] = np.nan
        residual = np.sqrt(np.clip(1.0 - (scalar**2 + projected**2), 0.0, 1.0))
        swing_angle = 2.0 * np.arcsin(np.clip(residual, -1.0, 1.0))
        swing_angle[singular] = np.nan
        twist[finite] = twist_angle
        swing[finite] = swing_angle
    if degrees:
        return (np.rad2deg(twist), np.rad2deg(swing))
    return (twist, swing)


def _rotation_quaternion(matrices):
    trace = np.trace(matrices, axis1=1, axis2=2)
    scalar = 0.5 * np.sqrt(np.clip(1.0 + trace, 0.0, None))
    vector = 0.5 * np.stack(
        [
            matrices[:, 2, 1] - matrices[:, 1, 2],
            matrices[:, 0, 2] - matrices[:, 2, 0],
            matrices[:, 1, 0] - matrices[:, 0, 1],
        ],
        axis=1,
    )
    small = scalar > 1e-08
    if small.any():
        vector[small] = vector[small] / (2.0 * scalar[small, None])
    if (~small).any():
        plus = matrices[~small] + IDENTITY
        best = np.argmax(np.linalg.norm(plus, axis=1), axis=1)
        chosen = plus[np.arange(len(plus)), :, best]
        vector[~small] = chosen / np.linalg.norm(chosen, axis=1, keepdims=True)
    length = np.sqrt(scalar**2 + np.einsum("ni,ni->n", vector, vector))
    length[length < 1e-12] = 1.0
    return (scalar / length, vector / length[:, None])


def nonaffine_residual(graph, reference, target, deformation_gradients):
    """Weighted mean squared motion left unexplained by the local affine fit, in square angstrom."""
    from .deformation import _residuals

    F = as_tensors(deformation_gradients, "Deformation gradients")
    reference = np.ascontiguousarray(reference, dtype=float)
    target = np.ascontiguousarray(target, dtype=float)
    total = np.zeros(graph.n_centres)
    weights = np.zeros(graph.n_centres)
    filled = np.where(np.isfinite(F), F, 0.0)
    _residuals(
        graph.indptr(),
        graph.pair_neighbour,
        graph.pair_weight,
        reference,
        target,
        graph.centre_indices,
        filled,
        total,
        weights,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.where(weights > 0, total / weights, np.nan)
    residual[~np.isfinite(F).all(axis=(1, 2))] = np.nan
    return residual


def remove_rigid(deformation_gradients):
    """Return the stretch part U of F, discarding the polar rotation."""
    _, stretch = polar_decomposition(deformation_gradients)
    return stretch
