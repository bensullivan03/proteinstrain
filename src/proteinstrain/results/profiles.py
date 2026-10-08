"""Plain atom/residue tables and explicit alignment-position remapping."""

import numpy as np
import pandas as pd


def result_table(coordinates, result, values=None, *, on="reference", name="value"):
    """Join scalar/vector/tensor values to native centre atoms; omitted values produce F00…F22."""
    if on not in ("reference", "target"):
        raise ValueError("on must be reference or target.")
    label = result.reference_id if on == "reference" else result.target_id
    table = (
        coordinates.atom_table(label)
        .iloc[result.centre_indices]
        .copy()
        .reset_index(drop=True)
    )
    table["reference_id"] = result.reference_id
    table["target_id"] = result.target_id
    table["valid"] = result.valid
    if values is None:
        for i in range(3):
            for j in range(3):
                table[f"F{i}{j}"] = result.F[:, i, j]
    else:
        values = np.asarray(values, float)
        if values.ndim not in (1, 2, 3) or values.shape[0] != len(table):
            raise ValueError("One scalar/vector/tensor is required per centre.")
        values = values.copy()
        values[~result.valid] = np.nan
        if values.ndim == 1:
            table[name] = values
            table["valid"] &= np.isfinite(values)
        else:
            if values.shape[1:] not in ((3,), (3, 3)):
                raise ValueError("Vector/tensor values need shape (n,3) or (n,3,3).")
            for index in np.ndindex(values.shape[1:]):
                table[name + "".join(map(str, index))] = values[(slice(None), *index)]
    return table


def per_residue(table, how, *, value="value", allow_partial=False):
    """Reduce valid finite atom values; single requires exactly one per native residue."""
    if how not in ("mean", "max", "rms", "sum", "ca", "single"):
        raise ValueError("Unknown residue reduction.")
    keys = [
        k
        for k in (
            "reference_id",
            "target_id",
            "item_id",
            "alignment_group",
            "chain_id",
            "hetero",
            "residue_number",
            "insertion",
        )
        if k in table
    ]
    if not {"chain_id", "residue_number"} <= set(keys):
        raise ValueError("Table needs native residue columns.")
    rows = []
    for labels, block in table.groupby(keys, sort=False, dropna=False):
        labels = (labels,) if len(keys) == 1 else labels
        valid = np.isfinite(block[value].to_numpy(float))
        if "valid" in block:
            valid &= block.valid.to_numpy(bool)
        if how == "ca":
            valid &= block.atom_name.to_numpy() == "CA"
        selected = block.loc[valid, value].to_numpy(float)
        if how in ("single", "ca") and len(selected) > 1:
            raise ValueError(f"{how} requires at most one valid centre per residue.")
        if how == "single" and len(selected) != 1:
            raise ValueError("single requires exactly one valid centre per residue.")
        reduced = np.nan
        complete = len(selected) == len(block)
        if len(selected) and (how != "sum" or complete or allow_partial):
            reduced = {
                "mean": np.mean,
                "max": np.max,
                "sum": np.sum,
                "rms": lambda x: np.sqrt(np.mean(x * x)),
                "ca": lambda x: x[0],
                "single": lambda x: x[0],
            }[how](selected)
        row = dict(zip(keys, labels))
        row.update(
            {
                value: reduced,
                "valid": bool(np.isfinite(reduced)),
                "n_valid": len(selected),
                "n_atoms": len(block),
            }
        )
        for key in ("column", "polymer_position", "residue_code"):
            if key in block:
                if block[key].nunique(dropna=False) > 1:
                    raise ValueError(f"Residue has conflicting {key} values.")
                row[key] = block[key].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def remap_alignment(
    table, positions, *, on=("item_id", "chain_id", "polymer_position")
):
    """Join an explicit position→column map; unmapped rows retain missing columns."""
    keys = list(on)
    if not set(keys + ["column"]) <= set(positions):
        raise ValueError("Position map needs join keys and column.")
    if positions.duplicated(keys).any():
        raise ValueError("Position map assigns a site more than once.")
    return table.drop(columns=["column"], errors="ignore").merge(
        positions[keys + ["column"]],
        on=keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
