"""Stress measures. Each takes the material model first, so the assumption is explicit."""

import numpy as np
from ..mechanics.measures import (
    as_tensors,
    direction_field,
    green_lagrange_strain,
    volume_ratio,
)


def second_piola(model, deformation_gradients):
    """Second Piola-Kirchhoff stress S, referred to the reference configuration."""
    return model.second_piola(green_lagrange_strain(deformation_gradients))


def first_piola(model, deformation_gradients):
    """First Piola-Kirchhoff stress P = F S, force per unit reference area."""
    F = as_tensors(deformation_gradients, "Deformation gradients")
    return np.einsum("nij,njk->nik", F, second_piola(model, F))


def cauchy(model, deformation_gradients):
    """True stress sigma = F S F-transpose / J, force per unit current area."""
    F = as_tensors(deformation_gradients, "Deformation gradients")
    S = second_piola(model, F)
    ratio = volume_ratio(F)
    product = np.einsum("nij,njk,nlk->nil", F, S, F)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (ratio > 0)[:, None, None], product / ratio[:, None, None], np.nan
        )


def von_mises(stress_tensors):
    """Von Mises equivalent stress of each tensor, in the units of the input."""
    array = as_tensors(stress_tensors, "Stress tensors")
    trace = np.trace(array, axis1=1, axis2=2)
    deviator = array - (trace / 3.0)[:, None, None] * np.eye(3)
    return np.sqrt(1.5 * np.einsum("nij,nij->n", deviator, deviator))


def pressure(stress_tensors):
    """Mean normal stress with the sign convention that compression is positive."""
    array = as_tensors(stress_tensors, "Stress tensors")
    return -np.trace(array, axis1=1, axis2=2) / 3.0


def directional_stress(stress_tensors, direction):
    """Normal component of the stress along a direction."""
    array = as_tensors(stress_tensors, "Stress tensors")
    normal = direction_field(direction, len(array), "direction")
    return np.einsum("ni,nij,nj->n", normal, array, normal)


def traction(stress_tensors, normal):
    """Traction vector acting on a surface with the given normal."""
    array = as_tensors(stress_tensors, "Stress tensors")
    direction = direction_field(normal, len(array), "normal")
    return np.einsum("nij,nj->ni", array, direction)
