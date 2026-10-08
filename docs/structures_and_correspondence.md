# Structures and corresponding atoms

`load_structure(path, structure_id=None, chains=None, model_number=None)` reads PDB or mmCIF. The default is the first model. `load_assembly(path, assembly_name)` expands a deposited biological assembly with distinct chain labels. `list_assemblies(path)` lists available names. Downloads validate the entry and coordinates before replacing a cached file.

A `ProteinStructure` contains a Gemmi model, a readable label, chain descriptions, full deposited polymer sequences and a flat `groups` dictionary. A polymer sequence is a tuple of `(one_letter_code, native_residue_or_None)` entries. Unmodelled deposited residues remain in the sequence. Native residues retain hetero status, author number and insertion code.

Chains are selected by an ID string or sequence of IDs; `None` selects all. Resolve a group explicitly with `structure.groups['name']`. `chains_by_description(structure, pattern, regex=True)` provides description matching. `retain_chains`, `drop_chains`, `relabel_chains(mapping)` and `with_groups` return prepared structures. Relabel values of `None` drop chains. There is no preview/preparation-plan object.

`assign_groups` matches descriptions using explicit `claim='first'/'all'/'error'`. Unmatched groups remain empty tuples. Sequence group assignment requires identity/coverage thresholds and an ambiguity margin. Centroid, rigid-fit and angular assignment return reference→target mapping dictionaries. Centroid assignment assumes already superposed inputs; angular assignment's `offset=0` fixes zero phase and `None` chooses the lowest-cost phase.

## Coordinate interfaces

`extract(structure, chains=None, atoms='CA', altloc='reject')` returns `(xyz, atoms_table)` for modelled polymer atoms. Atom choices are `CA`, `backbone`, `all` or explicit names. Ambiguous alternate sites raise unless `altloc='highest'` is supplied. Non-polymer atoms remain in the Gemmi model for ordinary file output and whole-structure volume calculations.

Correspondence functions return `AlignedCoordinates`:

- `from_pairwise(reference, target, reference_chains=..., target_chains=..., chain_map=...)` uses pairwise alignment. Maps point **reference→target**.
- `from_reference_star(items, chains=..., chain_maps={item_id: mapping}, reference_index=0)` aligns each item to the reference. `chains` selects only the reference.
- `from_identity(items, ...)` requires identical full sequences.
- `from_msa(items, alignment, ...)` uses labelled aligned strings. Each item selects one chain; combine disjoint chain groups with `concat_blocks`.
- `from_native_keys(items, ...)` asserts that equal native residue/atom names correspond; matching residue numbers alone does not establish homology.

Items can be structures or `(structure, chain)` pairs, including different chains from one structure. Supply distinct `item_ids` when default labels would repeat. Multi-item maps use those IDs as keys. An explicit pair of single-chain items declares their pairing. The default all-chain route does not silently discard extra target chains.

The coordinate array has shape `(items, atoms, 3)`. `block.atom_table(item_id)` returns the mapping in array order. Its columns are `item_id`, `index`, `alignment_group`, `column`, `chain_id`, `hetero`, `residue_number`, `insertion`, `atom_name`, `element`, `polymer_position`, `residue_code`. Polymer/alignment positions are zero-based; author residue numbers are unchanged.

All items share realised atoms. Missing sites are omitted, and original column numbers are retained across gaps. `columns=[...]` selects explicit shared positions; `(group, column)` keys select across multiple groups. There is no exclusion log or support-policy object. `subset_atoms(indices)` subsets arrays and tables together.

## Sequence functions

`chain_sequence(structure, chain)` returns a full sequence string. `align_pairwise(reference_string, target_string)` returns an `(n, 2)` integer position map. Defaults preserve BLOSUM62, gap-open -11, gap-extension -1 and terminal-gap 0 scoring; these are ordinary keyword arguments.

`align_multiple({label: sequence}, executable='mafft', launcher=(), flags=('--auto',), threads=1, timeout=300)` returns `{label: aligned_string}`. Identical sequences are deduplicated by default. There is no fallback when MAFFT fails. `read_alignment`/`write_alignment` use standard aligned FASTA files.
