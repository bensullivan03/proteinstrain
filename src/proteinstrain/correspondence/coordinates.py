"""Corresponding atom arrays and the table mapping them to native residues."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .._validation import as_xyz_stack, as_indices
from ..selection import resolve_chains
from ..structure import residue_identity
from .sequence import chain_sequence, align_pairwise
from .pairs import resolve_chain_pairs

ATOM_GROUPS = {"CA": ("CA",), "backbone": ("CA", "N", "C", "O")}
ATOM_COLUMNS = [
    "chain_id",
    "hetero",
    "residue_number",
    "insertion",
    "atom_name",
    "element",
    "polymer_position",
    "residue_code",
]


@dataclass
class AlignedCoordinates:
    """xyz[item, atom, Cartesian component], with one native atom-table row per item/atom."""

    xyz: np.ndarray
    item_ids: tuple
    atoms: pd.DataFrame

    def __post_init__(self):
        self.xyz = as_xyz_stack(self.xyz).copy()
        self.item_ids = tuple(self.item_ids)
        self.atoms = self.atoms.copy()
        if len(self.item_ids) != len(self.xyz) or len(set(self.item_ids)) != len(
            self.item_ids
        ):
            raise ValueError("Item labels must be distinct and match xyz rows.")
        required = {"item_id", "index", "alignment_group", "column", *ATOM_COLUMNS}
        if not required <= set(self.atoms):
            raise ValueError(f"Atom table needs {sorted(required)}.")
        if set(self.atoms.item_id) != set(self.item_ids):
            raise ValueError("Atom table item labels must match xyz.")
        for label in self.item_ids:
            table = self.atom_table(label)
            if (
                not np.array_equal(table["index"], np.arange(self.n_atoms))
                or table.duplicated(
                    ["chain_id", "hetero", "residue_number", "insertion", "atom_name"]
                ).any()
            ):
                raise ValueError(
                    "Atom rows must identify each array position and native atom exactly once."
                )
        first = self.atom_table(self.item_ids[0])[
            ["alignment_group", "column", "atom_name"]
        ].to_numpy()
        if any(
            not np.array_equal(
                first,
                self.atom_table(label)[
                    ["alignment_group", "column", "atom_name"]
                ].to_numpy(),
            )
            for label in self.item_ids[1:]
        ):
            raise ValueError(
                "Every item must share correspondence columns and atom names."
            )

    @property
    def n_atoms(self):
        return self.xyz.shape[1]

    def coordinates(self, item_id):
        if item_id not in self.item_ids:
            raise LookupError(f"Unknown item {item_id!r}.")
        return self.xyz[self.item_ids.index(item_id)].copy()

    def atom_table(self, item_id):
        if item_id not in self.item_ids:
            raise LookupError(f"Unknown item {item_id!r}.")
        return (
            self.atoms.loc[self.atoms.item_id == item_id]
            .sort_values("index")
            .reset_index(drop=True)
        )

    def subset_atoms(self, indices):
        index = as_indices(indices, self.n_atoms)
        tables = []
        for label in self.item_ids:
            table = self.atom_table(label).iloc[index].copy()
            table["index"] = np.arange(len(index))
            tables.append(table)
        return AlignedCoordinates(
            self.xyz[:, index], self.item_ids, pd.concat(tables, ignore_index=True)
        )


def _names(atoms):
    if isinstance(atoms, str):
        return atoms if atoms == "all" else ATOM_GROUPS.get(atoms, (atoms,))
    return tuple(atoms)


def _records(structure, chains, atoms, altloc):
    if altloc not in ("reject", "highest"):
        raise ValueError("altloc must be reject or highest.")
    names = _names(atoms)
    records = {}
    for chain_id in resolve_chains(structure, chains):
        sequence = structure.polymer_sequences.get(chain_id, ())
        positions = {
            residue: (i, code)
            for i, (code, residue) in enumerate(sequence)
            if residue is not None
        }
        for residue in structure.chain(chain_id):
            native = residue_identity(residue)
            if native not in positions:
                continue
            position, code = positions[native]
            by_name = {}
            for atom in residue:
                if names == "all" or atom.name in names:
                    by_name.setdefault(atom.name, []).append(atom)
            for name, candidates in by_name.items():
                if len(candidates) > 1 and altloc == "reject":
                    raise ValueError(
                        f'Ambiguous alternate site {chain_id}/{native}/{name}; use altloc="highest".'
                    )
                atom = max(candidates, key=lambda a: a.occ)
                key = (chain_id, position, name)
                if key in records:
                    raise ValueError("Duplicate native polymer atom.")
                records[key] = (
                    [atom.pos.x, atom.pos.y, atom.pos.z],
                    dict(
                        zip(
                            ATOM_COLUMNS,
                            (
                                chain_id,
                                native.hetero,
                                native.number,
                                native.insertion,
                                name,
                                atom.element.name,
                                position,
                                code,
                            ),
                        )
                    ),
                )
    return records


def extract(structure, *, chains=None, atoms="CA", altloc="reject"):
    """Return (xyz, atom_table) for one structure, omitting unmodelled polymer sites."""
    records = _records(structure, chains, atoms, altloc)
    if not records:
        raise ValueError("The selection contains no usable polymer atoms.")
    xyz = np.array([value[0] for value in records.values()])
    table = pd.DataFrame([value[1] for value in records.values()])
    table.insert(0, "index", np.arange(len(table)))
    return xyz, table


def _inputs(items, chains, item_ids):
    structures = []
    selectors = []
    for item in items:
        explicit = (
            isinstance(item, tuple) and len(item) == 2 and hasattr(item[0], "model")
        )
        structures.append(item[0] if explicit else item)
        selectors.append(item[1] if explicit else None)
    if not structures:
        raise ValueError("Supply at least one structure.")
    selectors[0] = selectors[0] if selectors[0] is not None else chains
    ids = (
        tuple(item_ids)
        if item_ids is not None
        else tuple(
            (
                s.structure_id
                if sum(x.structure_id == s.structure_id for x in structures) == 1
                else f'{s.structure_id}:{",".join(resolve_chains(s,selection))}'
            )
            for s, selection in zip(structures, selectors)
        )
    )
    if len(ids) != len(structures) or len(set(ids)) != len(ids):
        raise ValueError("Supply distinct item_ids for repeated items.")
    return structures, selectors, ids


def _assemble(structures, ids, positions, *, atoms, altloc, columns=None):
    # Each item's positions: (group, column, chain_id, zero-based polymer position).
    tables = []
    for structure, item_positions in zip(structures, positions):
        chains = tuple(dict.fromkeys(row[2] for row in item_positions))
        records = _records(structure, chains, atoms, altloc)
        table = {}
        for group, column, chain, position in item_positions:
            for name in (
                _names(atoms)
                if _names(atoms) != "all"
                else sorted({key[2] for key in records})
            ):
                value = records.get((chain, int(position), name))
                if value is not None:
                    table[(group, int(column), name)] = value
        tables.append(table)
    shared = set(tables[0])
    for table in tables[1:]:
        shared &= set(table)
    keys = [key for key in tables[0] if key in shared]
    if columns is not None:
        wanted = list(columns)
        selected = {
            tuple(c) if isinstance(c, (list, tuple)) else int(c) for c in wanted
        }
        keys = [k for k in keys if k[1] in selected or (k[0], k[1]) in selected]
        available = {k[1] for k in keys} | {k[:2] for k in keys}
        if not selected <= available:
            raise ValueError("Requested columns are not realised by every item.")
    if not keys:
        raise ValueError("The correspondence produced no comparable atoms.")
    xyz = []
    rows = []
    for label, table in zip(ids, tables):
        xyz.append([table[k][0] for k in keys])
        for index, key in enumerate(keys):
            rows.append(
                {
                    "item_id": label,
                    "index": index,
                    "alignment_group": key[0],
                    "column": key[1],
                    **table[key][1],
                }
            )
    return AlignedCoordinates(np.asarray(xyz), ids, pd.DataFrame(rows))


def _groups(chains, groups):
    names = tuple(chains) if groups is None else tuple(groups)
    if len(names) != len(chains) or len(set(names)) != len(names):
        raise ValueError("Each reference chain needs a distinct alignment group.")
    return names


def from_pairwise(
    reference,
    target,
    alignment=None,
    *,
    reference_chains=None,
    target_chains=None,
    chain_map=None,
    matching=None,
    atoms="backbone",
    altloc="reject",
    item_ids=None,
    alignment_groups=None,
    columns=None,
    **scoring,
):
    """Correspond two structures on reference polymer positions using sequence alignment."""
    pairs = resolve_chain_pairs(
        reference,
        target,
        reference_chains=reference_chains,
        target_chains=target_chains,
        chain_map=chain_map,
        matching=matching,
    )
    groups = _groups([a for a, b in pairs], alignment_groups)
    positions = [[], []]
    for group, (a, b) in zip(groups, pairs):
        match = alignment.get(a) if isinstance(alignment, dict) else alignment
        if match is None:
            match = align_pairwise(
                chain_sequence(reference, a), chain_sequence(target, b), **scoring
            )
        match = np.asarray(match)
        if match.ndim != 2 or match.shape[1] != 2 or match.dtype.kind not in "iu":
            raise ValueError(
                "Alignment must contain integer reference/target position pairs."
            )
        if len(match) and (
            match.min() < 0
            or match[:, 0].max() >= len(chain_sequence(reference, a))
            or match[:, 1].max() >= len(chain_sequence(target, b))
        ):
            raise ValueError("Alignment position outside a sequence.")
        if any(len(np.unique(match[:, axis])) != len(match) for axis in (0, 1)):
            raise ValueError("Alignment positions must be one-to-one.")
        for left, right in match:
            positions[0].append((group, int(left), a, int(left)))
            positions[1].append((group, int(left), b, int(right)))
    if item_ids is None:
        item_ids = (reference.structure_id, target.structure_id)
        if item_ids[0] == item_ids[1]:
            item_ids = (
                f'{item_ids[0]}:{",".join(a for a,b in pairs)}',
                f'{item_ids[1]}:{",".join(b for a,b in pairs)}',
            )
    return _assemble(
        [reference, target],
        tuple(item_ids),
        positions,
        atoms=atoms,
        altloc=altloc,
        columns=columns,
    )


def _star(
    items,
    *,
    identity,
    chains=None,
    chain_maps=None,
    item_ids=None,
    reference_index=0,
    matching=None,
    atoms="backbone",
    altloc="reject",
    alignment_groups=None,
    columns=None,
    **scoring,
):
    structures, selectors, ids = _inputs(items, None, item_ids)
    if not 0 <= reference_index < len(structures):
        raise ValueError("reference_index is outside the items.")
    selectors[reference_index] = (
        selectors[reference_index] if selectors[reference_index] is not None else chains
    )
    reference = structures[reference_index]
    left = resolve_chains(reference, selectors[reference_index])
    groups = _groups(left, alignment_groups)
    if set(chain_maps or {}) - set(ids):
        raise LookupError("chain_maps names unknown items.")
    positions = []
    for i, (structure, selector) in enumerate(zip(structures, selectors)):
        mapping = (chain_maps or {}).get(ids[i])
        if i == reference_index:
            pairs = tuple((c, c) for c in left)
        else:
            if (
                mapping is None
                and matching is None
                and selector is not None
                and selectors[reference_index] is not None
            ):
                right = resolve_chains(structure, selector)
                if len(left) == len(right) == 1:
                    mapping = {left[0]: right[0]}
            target_selection = selector
            if (
                selector is None
                and selectors[reference_index] is None
                and mapping is None
            ):
                target_selection = structure.chain_ids
            pairs = resolve_chain_pairs(
                reference,
                structure,
                reference_chains=left,
                target_chains=target_selection,
                chain_map=mapping,
                matching=matching,
            )
        rows = []
        for group, (a, b) in zip(groups, pairs):
            seq = chain_sequence(reference, a)
            other = chain_sequence(structure, b)
            if identity or i == reference_index:
                if seq != other:
                    raise ValueError(
                        "Identity correspondence requires identical full sequences."
                    )
                match = np.column_stack((np.arange(len(seq)), np.arange(len(seq))))
            else:
                match = align_pairwise(seq, other, **scoring)
            rows.extend((group, int(x), b, int(y)) for x, y in match)
        positions.append(rows)
    return _assemble(
        structures, ids, positions, atoms=atoms, altloc=altloc, columns=columns
    )


def from_identity(items, **kwargs):
    """Correspond identical full sequences; explicit maps pair differently named chains."""
    return _star(items, identity=True, **kwargs)


def from_reference_star(items, **kwargs):
    """Align every item independently to one reference; retain their common realised atoms."""
    return _star(items, identity=False, **kwargs)


def from_msa(
    items,
    alignment,
    *,
    item_ids=None,
    atoms="backbone",
    altloc="reject",
    alignment_group="protein",
    columns=None,
):
    """Use labelled aligned strings and explicit (structure, chain) items, including within-structure items."""
    structures, selectors, ids = _inputs(items, None, item_ids)
    if not set(ids) <= set(alignment) or len({len(s) for s in alignment.values()}) != 1:
        raise ValueError("MSA must include every item label and have equal row widths.")
    positions = []
    for structure, selector, label in zip(structures, selectors, ids):
        chains = resolve_chains(structure, selector)
        if len(chains) != 1:
            raise ValueError(
                "Each MSA item selects one chain; concatenate blocks for multiple groups."
            )
        row = alignment[label].upper()
        if row.replace("-", "") != chain_sequence(structure, chains[0]):
            raise ValueError("MSA row differs from its full polymer sequence.")
        position = 0
        found = []
        for column, code in enumerate(row):
            if code != "-":
                found.append((alignment_group, column, chains[0], position))
                position += 1
        positions.append(found)
    return _assemble(
        structures, ids, positions, atoms=atoms, altloc=altloc, columns=columns
    )


def from_native_keys(
    items,
    *,
    chains=None,
    chain_maps=None,
    alignment_groups=None,
    item_ids=None,
    atoms="backbone",
    altloc="reject",
    columns=None,
):
    """Assert correspondence by native residue identifiers and atom names, without claiming homology."""
    structures, selectors, ids = _inputs(items, chains, item_ids)
    if set(chain_maps or {}) - set(ids):
        raise LookupError("Chain maps refer to unknown items.")
    reference = structures[0]
    left = resolve_chains(reference, selectors[0])
    groups = _groups(left, alignment_groups)
    positions = []
    for i, (structure, selector) in enumerate(zip(structures, selectors)):
        mapping = (chain_maps or {}).get(ids[i])
        if i == 0:
            pairs = tuple((c, c) for c in left)
        else:
            if mapping is None and selector is not None and selectors[0] is not None:
                right = resolve_chains(structure, selector)
                if len(left) == len(right) == 1:
                    mapping = {left[0]: right[0]}
            pairs = resolve_chain_pairs(
                reference,
                structure,
                reference_chains=left,
                target_chains=(
                    selector
                    if selector is not None
                    else (
                        structure.chain_ids
                        if selectors[0] is None and mapping is None
                        else None
                    )
                ),
                chain_map=mapping,
            )
        found = []
        for group, (a, b) in zip(groups, pairs):
            target = {
                r: p
                for p, (code, r) in enumerate(structure.polymer_sequences[b])
                if r is not None
            }
            for column, (code, residue) in enumerate(reference.polymer_sequences[a]):
                if residue is not None and residue in target:
                    found.append((group, column, b, target[residue]))
        positions.append(found)
    return _assemble(
        structures, ids, positions, atoms=atoms, altloc=altloc, columns=columns
    )


def concat_blocks(blocks):
    """Concatenate disjoint correspondence groups with the same item order."""
    blocks = tuple(blocks)
    if not blocks or any(b.item_ids != blocks[0].item_ids for b in blocks):
        raise ValueError("Blocks need the same item order.")
    tables = []
    offset = 0
    for block in blocks:
        table = block.atoms.copy()
        table["index"] += offset
        tables.append(table)
        offset += block.n_atoms
    table = pd.concat(tables, ignore_index=True)
    if table.duplicated(["item_id", "alignment_group", "column", "atom_name"]).any():
        raise ValueError("Correspondence keys overlap between blocks.")
    return AlignedCoordinates(
        np.concatenate([b.xyz for b in blocks], axis=1), blocks[0].item_ids, table
    )


def _paired_coordinates(
    reference, reference_chain, target, target_chain, *, atoms="CA"
):
    block = from_pairwise(
        reference,
        target,
        chain_map={reference_chain: target_chain},
        reference_chains=reference_chain,
        atoms=atoms,
        item_ids=("reference", "target"),
    )
    return block.xyz[0], block.xyz[1]
