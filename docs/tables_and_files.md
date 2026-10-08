# Measurements, comparisons and standard files

## Result tables

`psp.result_table(block, result, values=None, *, on="reference", name="value")` attaches one scalar, vector `(n, 3)` or tensor `(n, 3, 3)` per centre to its native atom and alignment column.

- `on="target"` labels rows with the target's native atoms instead of the reference's.
- With `values` omitted, the nine components of **F** are returned as columns `F00` … `F22`.
- Vector and tensor values are expanded into columns `name0` … `name2` or `name00` … `name22`.
- Invalid fits become missing values, and the `valid` column records them.

## Residue reductions

`psp.per_residue(table, how, *, value="value", allow_partial=False)` groups atoms by native residue (and by comparison, if `reference_id`/`target_id` columns are present), keeping each residue's alignment column.

| `how` | Result |
|---|---|
| `"mean"`, `"max"`, `"rms"` | Reduction over the residue's valid, finite values. |
| `"sum"` | Sum over the residue's atoms. Returns NaN unless **every** atom of the residue is valid, because a partial sum underestimates an extensive quantity; pass `allow_partial=True` to sum the valid atoms only. |
| `"ca"` | The value at the residue's CA, if valid. |
| `"single"` | Requires exactly one valid centre per residue. |

The output has `n_valid` and `n_atoms` columns so you can see how many atoms contributed. Residues with no valid atoms are NaN. Choose the reduction to suit the quantity: energies are extensive and are summed; strains are intensive and are averaged.

## Remapping onto a common alignment

`psp.remap_alignment(table, positions, *, on=("item_id", "chain_id", "polymer_position"))` joins an explicit position-to-column table (for example from `psp.alignment_positions`). Unmapped rows get missing columns, and each key must map to one column. The caller chooses the common alignment; compatibility between independently built alignments is not inferred.

## Matrices and distances

After mapping to a shared alignment, use ordinary pandas operations:

```python
matrix = table.pivot(index="comparison", columns="column", values="strain")
matrix = matrix.reindex(columns=range(int(matrix.columns.min()), int(matrix.columns.max()) + 1))
mean = matrix.mean(axis=0)
counts = matrix.count(axis=0)
```

For several alignment groups, pivot on `["alignment_group", "column"]` or handle the groups separately. Decide explicitly how gaps are treated; missing positions are never filled with zero. For example, an RMS distance over shared finite positions:

```python
import numpy as np
left, right = matrix.iloc[0].to_numpy(), matrix.iloc[1].to_numpy()
shared = np.isfinite(left) & np.isfinite(right)
distance = np.sqrt(np.mean((left[shared] - right[shared]) ** 2)) if shared.any() else np.nan
```

## Saving

- `psp.save_structure(structure, path)` writes standard mmCIF, including full sequences in the entity categories. Polymer, non-polymer and water residues are written as separate entities. <!-- Requires the save_structure fix in REVIEW.md (finding 3); at v0.6.0 ligands and waters are written into the polymer entity. --> In-memory groups are not stored in the file.
- `psp.read_alignment` / `psp.write_alignment` read and write aligned FASTA.
- Save measurements and arrays with `table.to_csv(...)`, `np.save(...)` or `np.savez(...)`.
- `psp.export_field(structure, atoms_table, values, path, *, missing)` writes the values into the B-factor column of a copy of the structure. `missing` is **required**: it is the value written for invalid fits and for every atom not in the table. Choose a value that cannot be mistaken for a measurement (for example −1 for a non-negative magnitude). B-factors are written with the format's limited precision, so keep the table for exact values. The structure must have one atom per native site; run `psp.preparation.clean_atoms` first if it has alternate locations.
