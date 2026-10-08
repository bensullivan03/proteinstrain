"""The arrays returned by one directed deformation calculation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DeformationResult:
    """F at centre_indices for reference→target; invalid fits are NaN."""

    reference_id: str
    target_id: str
    F: np.ndarray
    centre_indices: np.ndarray
    valid: np.ndarray
    nonaffine: np.ndarray | None = None
