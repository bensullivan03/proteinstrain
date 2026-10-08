from dataclasses import replace
import numpy as np
import pytest
import proteinstrain as p
from proteinstrain.selection import resolve_chains


def test_full_sequence_missing_residues_and_native_table(structure):
    xyz, atoms = p.extract(structure, chains="I")
    assert len(p.chain_sequence(structure, "I")) == 457 and len(xyz) == 427
    assert atoms.polymer_position.iloc[0] == 1
    assert (np.diff(atoms.polymer_position) > 1).any()
    assert {"hetero", "insertion", "residue_number", "atom_name"} <= set(atoms)
    with pytest.raises(LookupError):
        resolve_chains(structure, "missing")


def test_relabel_is_direct_and_nonmutating(structure):
    prepared = structure.with_groups({"protein": ["I"], "absent": []}).relabel_chains(
        {"I": "LongChain", "Q": None}
    )
    assert prepared.chain_ids == ("LongChain",) and prepared.groups == {
        "protein": ("LongChain",),
        "absent": (),
    }
    assert structure.chain_ids == ("I", "Q")
    with pytest.raises(ValueError):
        structure.relabel_chains({"I": "Q"})
    with pytest.raises(LookupError):
        structure.relabel_chains({"bad": "X"})


@pytest.mark.parametrize(
    "constructor", [p.from_identity, p.from_reference_star, p.from_native_keys]
)
def test_mapped_reference_selection(structure, data, constructor):
    target = p.load_structure(data / "8IXB_subset.cif")
    block = constructor(
        [structure, target],
        chains="I",
        chain_maps={target.structure_id: {"I": "H"}},
        atoms="CA",
    )
    assert block.item_ids == (structure.structure_id, target.structure_id)
    assert block.atom_table(target.structure_id).chain_id.unique().tolist() == ["H"]
    assert block.n_atoms > 400
    with pytest.raises(ValueError, match="reference"):
        constructor([structure, target], chains="I", atoms="CA")


@pytest.mark.parametrize(
    "constructor", [p.from_identity, p.from_reference_star, p.from_native_keys]
)
def test_default_does_not_ignore_extra_chains(structure, constructor):
    reference = structure.retain_chains("I")
    target = structure
    with pytest.raises(ValueError):
        constructor([reference, target], item_ids=("r", "t"))
    block = constructor([reference, target], chains="I", item_ids=("r", "t"))
    assert block.n_atoms > 400


def test_pairwise_map_direction_and_plain_error(structure, data):
    target = p.load_structure(data / "8IXB_subset.cif")
    block = p.from_pairwise(
        structure, target, reference_chains="I", chain_map={"I": "H"}, atoms="CA"
    )
    assert block.n_atoms > 400
    with pytest.raises(ValueError, match="reference"):
        p.from_pairwise(structure, target, reference_chains="I", chain_map={"H": "I"})
    with pytest.raises(LookupError) as error:
        p.from_pairwise(structure, target, reference_chains="missing")
    assert not str(error.value).startswith('"')


def test_within_structure_and_msa_gaps(structure):
    second = structure.retain_chains("I").relabel_chains({"I": "Other"})
    # One combined structure with two physically distinct native chain identities.
    model = structure.retain_chains("I").model.clone()
    model.add_chain(second.chain("Other").clone())
    combined = replace(
        structure,
        model=model,
        polymer_sequences={
            "I": structure.polymer_sequences["I"],
            "Other": structure.polymer_sequences["I"],
        },
    )
    items = [(combined, "I"), (combined, "Other")]
    block = p.from_identity(items, atoms="CA")
    seq = p.chain_sequence(combined, "I")
    gapped = seq[:10] + "--" + seq[10:]
    msa = p.from_msa(items, dict(zip(block.item_ids, (gapped, gapped))), atoms="CA")
    assert msa.n_atoms == block.n_atoms
    columns = msa.atom_table(msa.item_ids[0]).column.to_numpy()
    assert not np.isin(columns, [10, 11]).any()
    np.testing.assert_allclose(msa.xyz, block.xyz)
    with pytest.raises(ValueError):
        p.from_msa(
            items,
            dict(zip(block.item_ids, (gapped, gapped.replace("R", "A", 1)))),
            atoms="CA",
        )


def test_pairwise_position_maps_and_declared_columns(structure):
    positions = p.align_pairwise("ACDEFG", "ACDXXEFG")
    np.testing.assert_array_equal(
        positions, [[0, 0], [1, 1], [2, 2], [3, 5], [4, 6], [5, 7]]
    )
    block = p.from_pairwise(
        structure,
        structure,
        reference_chains="I",
        atoms="CA",
        item_ids=("a", "b"),
        columns=[1, 2, 3],
    )
    assert block.n_atoms == 3
    with pytest.raises(ValueError):
        p.from_pairwise(
            structure, structure, reference_chains="I", item_ids=("a", "b"), columns=[0]
        )
    with pytest.raises(ValueError):
        p.from_identity([structure, structure], item_ids=("same", "same"))


def test_native_assertion_and_concat(structure):
    items = [structure, structure]
    options = {"item_ids": ("a", "b"), "atoms": "CA"}
    first = p.from_native_keys(items, chains="I", columns=[1, 2, 3], **options)
    next_block = p.from_native_keys(items, chains="I", columns=[4, 5, 6], **options)
    block = p.concat_blocks([first, next_block])
    assert block.n_atoms == 6
    with pytest.raises(ValueError):
        p.concat_blocks([first, first])
    selected = block.subset_atoms([1, 4])
    assert selected.n_atoms == 2
    assert selected.atom_table("a")["index"].tolist() == [0, 1]


@pytest.mark.parametrize("altloc", ["reject", "highest"])
def test_altloc_policy(structure, altloc):
    model = structure.model.clone()
    residue = model["I"][0]
    atom = residue[0].clone()
    atom.altloc = "B"
    atom.occ = 0.9
    residue[0].altloc = "A"
    residue[0].occ = 0.1
    residue.add_atom(atom)
    ambiguous = replace(structure, model=model)
    name = atom.name
    if altloc == "reject":
        with pytest.raises(ValueError, match="alternate"):
            p.extract(ambiguous, chains="I", atoms=name)
    else:
        xyz, table = p.extract(ambiguous, chains="I", atoms=name, altloc="highest")
        assert len(xyz) > 0


def test_insertions_are_not_collapsed(structure):
    model = structure.retain_chains("I").model.clone()
    chain = model["I"]
    first = chain[0].clone()
    first.seqid.icode = "B"
    first.label_seq = 458
    chain.add_residue(first)
    from proteinstrain.structure import residue_identity, residue_code

    seq = structure.polymer_sequences["I"] + (
        (residue_code(first.name), residue_identity(first)),
    )
    prepared = replace(structure, model=model, polymer_sequences={"I": seq})
    _, table = p.extract(prepared, chains="I")
    assert table.iloc[-1].insertion == "B"
    assert table.iloc[-1].residue_number == table.iloc[0].residue_number
    assert table.iloc[-1].polymer_position != table.iloc[0].polymer_position


def test_asserted_alignment_is_one_to_one(structure):
    with pytest.raises(ValueError, match="one-to-one"):
        p.from_pairwise(
            structure,
            structure,
            alignment=np.array([[0, 0], [1, 0]]),
            reference_chains="I",
            target_chains="I",
            item_ids=("r", "t"),
        )
