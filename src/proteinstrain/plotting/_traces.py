"""Small chain-trace and segment helpers."""

import numpy as np
from ..correspondence.coordinates import extract
from ..structure import residue_identity


def chain_trace(structure, chain, frame=None):
    native = {
        residue
        for code, residue in structure.polymer_sequences.get(chain, ())
        if residue is not None
    }
    if not any(
        residue_identity(residue) in native
        and any(atom.name == "CA" for atom in residue)
        for residue in structure.chain(chain)
    ):
        return np.empty((0, 3)), []
    xyz, atoms = extract(structure, chains=chain, atoms="CA")
    if frame is not None:
        xyz = frame.to_local(xyz)
    cuts = np.flatnonzero(np.diff(atoms.polymer_position.to_numpy()) != 1) + 1
    return xyz, np.split(xyz, cuts)


def segments(xyz, directions, scale, kind):
    xyz = np.asarray(xyz, float)
    vectors = np.asarray(directions, float) * scale
    if (
        xyz.shape != vectors.shape
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or kind not in ("vector", "axis")
    ):
        raise ValueError("xyz/directions must be (n,3), and kind vector or axis.")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite.")
    return (
        (xyz, xyz + vectors)
        if kind == "vector"
        else (xyz - 0.5 * vectors, xyz + 0.5 * vectors)
    )
