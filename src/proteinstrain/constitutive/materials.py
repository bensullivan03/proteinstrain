"""Material models. Everything in this package assumes moduli that are not measured."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np
from ..mechanics.measures import as_tensors


@runtime_checkable
class MaterialModel(Protocol):
    """The interface a material model must provide."""

    name: str

    def energy_density(self, strain_tensors):
        """Energy per unit reference volume, in pascals."""

    def second_piola(self, strain_tensors):
        """Second Piola-Kirchhoff stress, in pascals."""


def lame_from_young_poisson(young, poisson):
    """Return (first Lame parameter, shear modulus) from Young's modulus and Poisson ratio."""
    young = float(young)
    poisson = float(poisson)
    if young <= 0:
        raise ValueError("Young's modulus must be positive.")
    if not -1.0 < poisson < 0.5:
        raise ValueError("The Poisson ratio must lie in (-1, 0.5).")
    first = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    shear = young / (2.0 * (1.0 + poisson))
    return (first, shear)


def young_poisson_from_lame(first, shear):
    """Return (Young's modulus, Poisson ratio) from the Lame parameters."""
    poisson = first / (2.0 * (first + shear))
    return (2.0 * shear * (1.0 + poisson), poisson)


@dataclass()
class IsotropicSVK:
    """Saint Venant-Kirchhoff: W = (lambda/2)(tr E)^2 + mu E:E, linear in E."""

    lame_first: float
    shear_modulus: float
    name: str = "saint_venant_kirchhoff"

    def __post_init__(self):
        self.lame_first = float(self.lame_first)
        self.shear_modulus = float(self.shear_modulus)
        if self.shear_modulus <= 0:
            raise ValueError("The shear modulus must be positive.")
        if self.bulk_modulus <= 0:
            raise ValueError("The bulk modulus must be positive.")

    @classmethod
    def from_young_poisson(cls, young, poisson):
        """Build the model from Young's modulus and the Poisson ratio."""
        first, shear = lame_from_young_poisson(young, poisson)
        return cls(first, shear)

    @property
    def bulk_modulus(self):
        """K = lambda + 2 mu / 3, in pascals."""
        return self.lame_first + 2.0 * self.shear_modulus / 3.0

    @property
    def young_modulus(self):
        """Young's modulus, in pascals."""
        return young_poisson_from_lame(self.lame_first, self.shear_modulus)[0]

    @property
    def poisson_ratio(self):
        """The Poisson ratio."""
        return young_poisson_from_lame(self.lame_first, self.shear_modulus)[1]

    def energy_density(self, strain_tensors):
        """Energy per unit reference volume, in pascals."""
        E = as_tensors(strain_tensors, "Strain tensors")
        trace = np.trace(E, axis1=1, axis2=2)
        contraction = np.einsum("nij,nij->n", E, E)
        return 0.5 * self.lame_first * trace**2 + self.shear_modulus * contraction

    def energy_parts(self, strain_tensors):
        """Split the energy density into its bulk and shear parts."""
        E = as_tensors(strain_tensors, "Strain tensors")
        trace = np.trace(E, axis1=1, axis2=2)
        deviator = E - (trace / 3.0)[:, None, None] * np.eye(3)
        bulk = 0.5 * self.bulk_modulus * trace**2
        shear = self.shear_modulus * np.einsum("nij,nij->n", deviator, deviator)
        return (bulk, shear)

    def second_piola(self, strain_tensors):
        """Second Piola-Kirchhoff stress S = lambda tr(E) I + 2 mu E, in pascals."""
        E = as_tensors(strain_tensors, "Strain tensors")
        trace = np.trace(E, axis1=1, axis2=2)
        return (
            self.lame_first * trace[:, None, None] * np.eye(3)
            + 2.0 * self.shear_modulus * E
        )


@dataclass()
class NeoHookean:
    """Compressible neo-Hookean: W = (mu/2)(I1 - 3) - mu log J + (lambda/2)(log J)^2."""

    lame_first: float
    shear_modulus: float
    name: str = "neo_hookean"

    def __post_init__(self):
        self.lame_first = float(self.lame_first)
        self.shear_modulus = float(self.shear_modulus)
        if self.shear_modulus <= 0:
            raise ValueError("The shear modulus must be positive.")

    @classmethod
    def from_young_poisson(cls, young, poisson):
        """Build the model from Young's modulus and the Poisson ratio."""
        first, shear = lame_from_young_poisson(young, poisson)
        return cls(first, shear)

    def _right_cauchy_green(self, strain_tensors):
        E = as_tensors(strain_tensors, "Strain tensors")
        return 2.0 * E + np.eye(3)

    def energy_density(self, strain_tensors):
        """Energy per unit reference volume, in pascals."""
        C = self._right_cauchy_green(strain_tensors)
        first_invariant = np.trace(C, axis1=1, axis2=2)
        finite = np.isfinite(C).all(axis=(1, 2))
        result = np.full(len(C), np.nan)
        if finite.any():
            determinant = np.linalg.det(C[finite])
            usable = determinant > 0
            index = np.flatnonzero(finite)[usable]
            log_j = 0.5 * np.log(determinant[usable])
            result[index] = (
                0.5 * self.shear_modulus * (first_invariant[index] - 3.0)
                - self.shear_modulus * log_j
                + 0.5 * self.lame_first * log_j**2
            )
        return result

    def second_piola(self, strain_tensors):
        """Second Piola-Kirchhoff stress, in pascals."""
        C = self._right_cauchy_green(strain_tensors)
        result = np.full(C.shape, np.nan)
        finite = np.isfinite(C).all(axis=(1, 2))
        if finite.any():
            determinant = np.linalg.det(C[finite])
            usable = determinant > 0
            index = np.flatnonzero(finite)[usable]
            inverse = np.linalg.inv(C[index])
            log_j = 0.5 * np.log(determinant[usable])
            result[index] = (
                self.shear_modulus * (np.eye(3) - inverse)
                + (self.lame_first * log_j)[:, None, None] * inverse
            )
        return result
