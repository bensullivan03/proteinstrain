"""Scientific data handling and plot composition regressions from the simplified review."""

import gemmi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import pytest
import proteinstrain as p


@pytest.mark.parametrize("claim", ["first", "error"])
@pytest.mark.parametrize("margin", [0.0, 0.05])
def test_ambiguous_sequence_assignment_raises(structure, claim, margin):
    sequence = p.chain_sequence(structure, "I")
    with pytest.raises(ValueError, match="ambiguous.*first.*second"):
        p.preparation.assign_groups_by_sequence(
            structure,
            {"first": sequence, "second": sequence},
            claim=claim,
            minimum_identity=0.8,
            minimum_coverage=0.9,
            margin=margin,
            chains="I",
        )


def test_sequence_assignment_clear_winner_and_all(structure):
    sequence = p.chain_sequence(structure, "I")
    # Two qualifying references, with one unambiguous winner.
    other = "A" * 50 + sequence[50:]
    kwargs = dict(minimum_identity=0.7, minimum_coverage=0.8, margin=0.01, chains="I")
    winner = p.preparation.assign_groups_by_sequence(
        structure, {"second": other, "first": sequence}, claim="error", **kwargs
    )
    assert winner.groups == {"second": (), "first": ("I",)}
    both = p.preparation.assign_groups_by_sequence(
        structure, {"first": sequence, "second": sequence}, claim="all", **kwargs
    )
    assert both.groups == {"first": ("I",), "second": ("I",)}


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 5.0])
def test_residue_sum_requires_complete_valid_input(invalid):
    table = pd.DataFrame(
        dict(
            chain_id=["A"] * 4,
            residue_number=[2] * 4,
            atom_name=["N", "CA", "C", "O"],
            value=[1.0, 2.0, 3.0, invalid],
            valid=[True, True, True, False],
        )
    )
    copy = table.copy(deep=True)
    reduced = p.per_residue(table, "sum")
    assert (reduced.n_atoms.iloc[0], reduced.n_valid.iloc[0]) == (4, 3)
    assert np.isnan(reduced.value.iloc[0]) and not reduced.valid.iloc[0]
    partial = p.per_residue(table, "sum", allow_partial=True)
    assert partial.value.iloc[0] == 6 and partial.valid.iloc[0]
    mean = p.per_residue(table, "mean")
    assert mean.value.iloc[0] == 2 and mean.n_valid.iloc[0] == 3
    pd.testing.assert_frame_equal(table, copy)


def test_comparison_labels_survive_residue_reduction(block):
    results = p.calculate_pairs(
        block,
        pairs=[("R", "T"), ("R", "U")],
        rule=p.WeightRule.fixed_radius(6),
        ensemble="all",
        centres=[0],
        neighbour_pool="all",
    )
    tables = [
        p.result_table(block, result, [i + 1.0])
        for i, result in enumerate(results.values())
    ]
    joined = p.per_residue(pd.concat(tables), "mean")
    assert joined.reference_id.tolist() == ["R", "R"]
    assert joined.target_id.tolist() == ["T", "U"]
    assert joined.value.tolist() == [1.0, 2.0]
    target = p.result_table(block, results[("R", "T")], [3.0], on="target")
    assert target.item_id.tolist() == ["T"] and target.reference_id.tolist() == ["R"]


def test_export_missing_is_explicit_and_distinct_from_zero(structure, tmp_path):
    atoms = p.extract(structure, chains="I")[1].iloc[:3]
    values = np.array([0.0, np.nan, 0.3])
    path = tmp_path / "display.cif"
    with pytest.raises(TypeError):
        p.export_field(structure, atoms, values, path)
    p.export_field(structure, atoms, values, path, missing=-1.0)
    loaded = p.load_structure(path)
    ca = [a.b_iso for r in loaded.chain("I") for a in r if a.name == "CA"]
    assert ca[:3] == pytest.approx([0.0, -1.0, 0.3])
    assert ca[3] == -1.0
    assert next(a for r in loaded.chain("I") for a in r if a.name == "N").b_iso == -1.0
    assert np.isnan(values[1])


def test_msa_accepts_extra_members_without_compressing_columns(structure, data):
    target = p.load_structure(data / "8IXB_subset.cif")
    sequence = p.chain_sequence(structure, "I")
    row = sequence[:10] + "--" + sequence[10:]
    alignment = {"r": row, "unused": "-" * len(row), "t": row}
    before = alignment.copy()
    items = [(structure, "I"), (target, "H")]
    kwargs = dict(item_ids=("r", "t"), atoms="CA")
    subset = p.from_msa(items, {k: alignment[k] for k in ("r", "t")}, **kwargs)
    full = p.from_msa(items, alignment, **kwargs)
    np.testing.assert_array_equal(full.xyz, subset.xyz)
    pd.testing.assert_frame_equal(full.atoms, subset.atoms)
    assert alignment == before
    assert full.atom_table("r").column.max() > full.n_atoms
    with pytest.raises(ValueError):
        p.from_msa(items, {"r": row, "unused": row}, **kwargs)


@pytest.mark.parametrize("bad", ["coordinates", "pair", "ensemble"])
def test_unknown_item_lookup_is_readable(block, bad):
    with pytest.raises(LookupError, match="Unknown item.*missing"):
        if bad == "coordinates":
            block.coordinates("missing")
        else:
            p.calculate(
                block,
                "missing" if bad == "pair" else "R",
                "T",
                rule=p.WeightRule.fixed_radius(6),
                ensemble=("missing",) if bad == "ensemble" else "all",
            )


def test_accessors_preserve_input_objects(block, structure):
    xyz = block.xyz.copy()
    block.coordinates("R")[:] = 999
    np.testing.assert_array_equal(block.xyz, xyz)
    grouped = structure.with_groups({"protein": ("I",)})
    before = structure.chain("I")[0][0].pos.x
    grouped.chain("I")[0][0].pos.x = 999
    assert structure.chain("I")[0][0].pos.x == before
    assert structure.groups == {}


def test_nonstandard_alignment_letters_keep_native_positions():
    pairs = p.align_pairwise("aUOcD", "AXXCD")
    np.testing.assert_array_equal(pairs, np.c_[np.arange(5), np.arange(5)])


def test_native_key_alignment_groups(structure):
    target = structure.relabel_chains({"I": "Other"})
    block = p.from_native_keys(
        [structure, target],
        chains="I",
        chain_maps={target.structure_id: {"I": "Other"}},
        item_ids=("r", target.structure_id),
        alignment_groups=("protein",),
        atoms="CA",
    )
    assert block.atoms.alignment_group.unique().tolist() == ["protein"]


def test_no_seqres_warns_about_missing_gap_information(structure, tmp_path):
    model = gemmi.Structure()
    model.add_model(structure.retain_chains("I").model.clone())
    path = tmp_path / "no_seqres.pdb"
    model.write_pdb(str(path))
    with pytest.warns(UserWarning, match="no deposited.*modelled residues"):
        loaded = p.load_structure(path)
    assert len(p.chain_sequence(loaded, "I")) < len(p.chain_sequence(structure, "I"))


def test_multi_group_heatmap_keeps_gaps_and_native_column_labels():
    columns = pd.MultiIndex.from_tuples(
        [("a", 2), ("a", 4), ("b", 10), ("b", 11)], names=["alignment_group", "column"]
    )
    table = pd.DataFrame(
        [[1.0, 2.0, 3.0, np.nan], [4.0, 5.0, 6.0, 7.0]],
        columns=columns,
        index=["r→t", "r→u"],
    )
    fig, ax = p.plotting.profile_heatmap(table, robust=False)
    array = ax.images[0].get_array()
    assert array.shape == (2, 6)  # 3 columns for a, separator, 2 columns for b.
    np.testing.assert_array_equal(
        array[:, [0, 2, 4]].data, table.iloc[:, [0, 1, 2]].to_numpy()
    )
    assert np.ma.getmaskarray(array)[:, [1, 3]].all()
    assert np.ma.getmaskarray(array)[0, 5]
    assert [label.get_text() for label in ax.child_axes[0].get_xticklabels()] == [
        "a",
        "b",
    ]
    assert "10" in [label.get_text() for label in ax.get_xticklabels()]
    assert ax.images[0].norm.vmin == 1.0 and ax.images[0].norm.vmax == 7.0
    plt.close(fig)


@pytest.mark.parametrize("values", [np.empty((0, 3)), np.empty((2, 0))])
def test_empty_heatmap_is_rejected(values):
    with pytest.raises(ValueError):
        p.plotting.profile_heatmap(values)


def test_unrolled_scale_can_disable_robust_clipping():
    angles = np.linspace(-1.0, 1.0, 100)
    xyz = np.c_[np.cos(angles), np.sin(angles), np.arange(100)]
    frame = p.geometry.Frame.from_axis([0, 0, 1], reference=[1, 0, 0])
    values = np.ones(100)
    values[-1] = 100.0
    fig, ax = p.plotting.unrolled_cylinder(xyz, values, frame=frame, robust=False)
    assert ax.collections[0].norm.vmax == 100.0
    plt.close(fig)


def test_structures_skip_chains_without_ca_without_hiding_other_errors(structure):
    prepared = structure.relabel_chains({})
    for residue in prepared.chain("Q"):
        for i in reversed(range(len(residue))):
            if residue[i].name == "CA":
                del residue[i]
    fig, ax = p.plotting.plot_structures(prepared, chains="Q")
    assert not ax.lines
    plt.close(fig)
    with pytest.raises(LookupError):
        p.plotting.plot_structures(structure, chains="missing")
    assert p.plotting.view_structures(prepared, axes=True).write_html()


def test_existing_view_grid_targets_only_requested_cell(structure):
    view = py3Dmol.view(viewergrid=(1, 2), linked=False)
    assert (
        p.plotting.view_structures(
            structure, chains="I", view=view, viewer=(0, 0), axes=True
        )
        is view
    )
    prior = view.startjs
    atoms = p.extract(structure, chains="I")[1].iloc[:3]
    assert (
        p.plotting.view_field(
            structure, atoms, [0.0, 1.0, 2.0], view=view, viewer=(0, 1), style="sphere"
        )
        is view
    )
    field = view.startjs[len(prior) :]
    assert "viewergrid_UNIQUEID[0][1].addModel" in field
    assert "viewergrid_UNIQUEID[0][0]" not in field
    assert '"model": -1' in field
    with pytest.raises(ValueError):
        p.plotting.view_structures(structure, viewer=(0, 1))


def test_existing_view_preserves_previous_model_style(structure):
    view = p.plotting.view_structures(structure, chains="I", style="sphere")
    prior = view.startjs
    atoms = p.extract(structure, chains="I")[1].iloc[:3]
    assert (
        p.plotting.view_field(
            structure, atoms, [0.0, 1.0, 2.0], view=view, style="sphere"
        )
        is view
    )
    appended = view.startjs[len(prior) :]
    assert 'setStyle({"model": -1}' in appended
    assert "mapAtomProperties(" in appended and '{"model": -1}' in appended
