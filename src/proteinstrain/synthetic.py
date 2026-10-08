"""Analytic geometries and deformations, for tests whose answer is known in advance."""

import numpy as np
from .geometry.transforms import RigidTransform, as_xyz, unit_vector


def lattice(shape=(6, 6, 6), spacing=4.0, *, centre=True):
    """A cubic lattice of points."""
    counts = tuple((int(value) for value in shape))
    axes = [np.arange(count) * float(spacing) for count in counts]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return points - points.mean(axis=0) if centre else points


def rod(radius=20.0, height=180.0, spacing=4.0, *, hollow=False):
    """A filled or hollow cylinder of lattice points, aligned with z."""
    radius = float(radius)
    step = float(spacing)
    grid = np.arange(-radius, radius + step, step)
    heights = np.arange(-height / 2.0, height / 2.0 + step, step)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    distance = np.hypot(x, y)
    keep = (
        distance <= radius
        if not hollow
        else (distance <= radius) & (distance >= radius - step)
    )
    disc = np.stack([x[keep], y[keep]], axis=1)
    points = np.concatenate(
        [np.column_stack([disc, np.full(len(disc), level)]) for level in heights]
    )
    return np.ascontiguousarray(points)


def sphere(radius=20.0, spacing=4.0):
    """A filled sphere of lattice points."""
    step = float(spacing)
    grid = np.arange(-float(radius), float(radius) + step, step)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    return np.ascontiguousarray(points[np.linalg.norm(points, axis=1) <= float(radius)])


def affine(xyz, gradient, *, translation=(0.0, 0.0, 0.0)):
    """Apply a uniform deformation gradient. The expected F is the gradient itself."""
    points = as_xyz(xyz)
    A = np.asarray(gradient, dtype=float)
    if A.shape != (3, 3):
        raise ValueError("A deformation gradient must have shape (3, 3).")
    return points @ A.T + np.asarray(translation, dtype=float)


def uniform_stretch(xyz, factors):
    """Stretch along the Cartesian axes. F = diag(factors)."""
    values = np.asarray(factors, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.shape != (3,) or (values <= 0).any():
        raise ValueError("Three positive stretch factors are required.")
    return affine(xyz, np.diag(values))


def simple_shear(xyz, amount, *, direction=0, normal=1):
    """Simple shear: F = I + amount e_direction (x) e_normal, with det F = 1."""
    gradient = np.eye(3)
    gradient[int(direction), int(normal)] += float(amount)
    return affine(xyz, gradient)


def spin(xyz, angle, *, axis=(0.0, 0.0, 1.0), degrees=False):
    """Rigid rotation of the whole body. The expected F is the rotation itself."""
    return RigidTransform.about_axis(axis, angle, degrees=degrees).apply(as_xyz(xyz))


def twist(xyz, rate, *, axis=(0.0, 0.0, 1.0), degrees=False):
    """Rotate each point about an axis by an angle proportional to its position along it."""
    points = as_xyz(xyz)
    direction = unit_vector(axis, "twist axis")
    along = points @ direction
    step = np.deg2rad(float(rate)) if degrees else float(rate)
    result = np.empty_like(points)
    for index, (point, position) in enumerate(zip(points, along)):
        rotation = RigidTransform.about_axis(direction, step * position)
        result[index] = rotation.apply(point)
    return result


def radial_extension(xyz, factor, *, axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Scale the radial component about an axis, leaving the axial component alone."""
    points = as_xyz(xyz)
    direction = unit_vector(axis, "axis")
    centre = np.asarray(origin, dtype=float)
    offset = points - centre
    along = (offset @ direction)[:, None] * direction
    radial = offset - along
    return centre + along + float(factor) * radial


def bend(xyz, curvature, *, axis=(0.0, 0.0, 1.0), bend_direction=(1.0, 0.0, 0.0)):
    """Bend a rod by mapping its axis onto a circular arc of the given curvature."""
    points = as_xyz(xyz)
    direction = unit_vector(axis, "axis")
    lateral = unit_vector(bend_direction, "bend direction")
    lateral = lateral - float(lateral @ direction) * direction
    lateral = lateral / np.linalg.norm(lateral)
    third = np.cross(direction, lateral)
    along = points @ direction
    across = points @ lateral
    out_of_plane = points @ third
    kappa = float(curvature)
    if abs(kappa) < 1e-12:
        return points.copy()
    radius = 1.0 / kappa
    angle = along * kappa
    new_across = (radius - across) * np.cos(angle) - radius
    new_along = (radius - across) * np.sin(angle)
    return (
        new_along[:, None] * direction
        - new_across[:, None] * lateral
        + out_of_plane[:, None] * third
    )


def noise(xyz, scale, *, seed=0):
    """Add isotropic Gaussian displacement, the floor against which strain is measured."""
    points = as_xyz(xyz)
    generator = np.random.default_rng(seed)
    return points + generator.normal(scale=float(scale), size=points.shape)


def synthetic_block(
    reference, target, *, item_ids=("reference", "target"), alignment_group="synthetic"
):
    """Wrap two coordinate arrays and their synthetic atom identities."""
    import pandas as pd
    from .correspondence.coordinates import AlignedCoordinates

    first = as_xyz(reference)
    second = as_xyz(target, len(first))
    rows = []
    for label in item_ids:
        for i in range(len(first)):
            rows.append(
                dict(
                    item_id=label,
                    index=i,
                    chain_id="A",
                    hetero=" ",
                    residue_number=i + 1,
                    insertion=" ",
                    atom_name="CA",
                    element="C",
                    polymer_position=i,
                    residue_code="A",
                    alignment_group=alignment_group,
                    column=i,
                )
            )
    return AlignedCoordinates(
        np.stack([first, second]), tuple(item_ids), pd.DataFrame(rows)
    )
