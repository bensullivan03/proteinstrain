"""Display scalar values in the standard mmCIF B-factor column."""

import numpy as np
from ..io import save_structure
from ..structure import residue_identity


def export_field(structure, atoms, values, path, *, missing):
    """Write a model copy with native-atom B factors; exact values remain in the caller's table."""
    values = np.asarray(values, float)
    keys = ["chain_id", "hetero", "residue_number", "insertion", "atom_name"]
    if values.shape != (len(atoms),) or atoms.duplicated(keys).any():
        raise ValueError("Supply one scalar per unambiguous native atom.")
    lookup = {
        tuple(row): v
        for row, v in zip(atoms[keys].itertuples(index=False, name=None), values)
    }
    copy = structure.relabel_chains({})
    found = set()
    for chain in copy.model:
        for residue in chain:
            native = residue_identity(residue)
            for atom in residue:
                key = (
                    chain.name,
                    native.hetero,
                    native.number,
                    native.insertion,
                    atom.name,
                )
                if key in lookup:
                    if key in found:
                        raise ValueError(
                            "Model contains ambiguous alternate atoms for an exported value."
                        )
                    found.add(key)
                value = lookup.get(key, np.nan)
                atom.b_iso = float(value) if np.isfinite(value) else float(missing)
    if found != set(lookup):
        raise LookupError("Some supplied atoms are absent from the model.")
    return save_structure(copy, path)
