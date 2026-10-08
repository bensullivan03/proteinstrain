"""Explicit atom preparation without changing full polymer sequences."""

import numpy as np


def clean_atoms(structure, *, altloc="omit", minimum_occupancy=0.0):
    """Clone, omit nonfinite/unoccupied atoms, and omit or select highest-occupancy alternates."""
    if altloc not in ("omit", "highest"):
        raise ValueError("altloc must be omit or highest.")
    if not np.isfinite(minimum_occupancy) or minimum_occupancy < 0:
        raise ValueError("minimum_occupancy must be finite and nonnegative.")
    prepared = structure.relabel_chains({})
    for chain in prepared.model:
        for residue in chain:
            sites = {}
            for i, atom in enumerate(residue):
                sites.setdefault(atom.name, []).append(i)
            keep = set()
            for indices in sites.values():
                usable = [
                    i
                    for i in indices
                    if residue[i].occ > minimum_occupancy
                    and np.isfinite(
                        [
                            residue[i].occ,
                            residue[i].pos.x,
                            residue[i].pos.y,
                            residue[i].pos.z,
                        ]
                    ).all()
                ]
                if altloc == "highest" and usable:
                    keep.add(max(usable, key=lambda i: residue[i].occ))
                elif len(indices) == len(usable) == 1 and residue[
                    indices[0]
                ].altloc in ("\x00", " "):
                    keep.add(indices[0])
            for i in reversed(range(len(residue))):
                if i not in keep:
                    del residue[i]
            for atom in residue:
                atom.altloc = "\x00"
    return prepared
