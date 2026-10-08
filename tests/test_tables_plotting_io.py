from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import proteinstrain as p
from proteinstrain.mechanics import measures as m


@pytest.fixture
def residue_table():
    return pd.DataFrame(
        {
            "chain_id": ["A", "A", "A", "A"],
            "hetero": [" "] * 4,
            "residue_number": [1, 1, 2, 2],
            "insertion": [" "] * 4,
            "atom_name": ["CA", "N", "CA", "N"],
            "column": [0, 0, 2, 2],
            "value": [2.0, np.nan, 3.0, 4.0],
            "valid": [True, False, True, True],
        }
    )


@pytest.mark.parametrize(
    "how,expected",
    [
        ("mean", [2, 3.5]),
        ("max", [2, 4]),
        ("rms", [2, np.sqrt(12.5)]),
        ("sum", [np.nan, 7]),
        ("ca", [2, 3]),
    ],
)
def test_residue_reductions(residue_table, how, expected):
    reduced = p.per_residue(residue_table, how)
    np.testing.assert_allclose(reduced.value, expected)
    assert reduced.column.tolist() == [0, 2]


def test_single_counts_valid_values(residue_table):
    one = p.per_residue(residue_table.iloc[:2], "single")
    assert one.value.iloc[0] == 2
    with pytest.raises(ValueError):
        p.per_residue(residue_table, "single")
    absent = residue_table.iloc[:2].copy()
    absent["valid"] = False
    with pytest.raises(ValueError):
        p.per_residue(absent, "single")
    assert np.isnan(p.per_residue(absent, "sum").value.iloc[0])


def test_result_tables_and_alignment_remap(block):
    result = p.calculate(
        block,
        "R",
        "T",
        rule=p.WeightRule.fixed_radius(6),
        ensemble="reference",
        centres=[0, 2, 4],
        neighbour_pool="all",
    )
    values = m.green_lagrange_frobenius(result.F)
    table = p.result_table(block, result, values, name="strain")
    assert table["index"].tolist() == [0, 2, 4]
    np.testing.assert_allclose(table.strain, values)
    assert set(p.result_table(block, result)) >= {"F00", "F22"}
    tensor = p.result_table(block, result, m.green_lagrange_strain(result.F), name="E")
    assert {"E00", "E22"} <= set(tensor)
    mapping = table[["item_id", "chain_id", "polymer_position"]].copy()
    mapping["column"] = [10, 12, 14]
    remapped = p.remap_alignment(table, mapping)
    assert remapped.column.tolist() == [10, 12, 14]
    shorter = p.remap_alignment(table, mapping.iloc[:2])
    assert np.isnan(shorter.column.iloc[-1])
    with pytest.raises(ValueError):
        p.remap_alignment(table, pd.concat([mapping, mapping]))
    matrix = remapped.pivot(index="item_id", columns="column", values="strain").reindex(
        columns=range(10, 15)
    )
    assert np.isnan(matrix.iloc[0, 1])


@pytest.mark.parametrize(
    "limits,extend",
    [(None, "neither"), ((0.8, 1.2), "both"), ((0.4, 1.2), "max"), ((0.8, 2.1), "min")],
)
def test_colours_neutral_and_actual_clipping(limits, extend):
    from proteinstrain.plotting.style import colour_scale, _clipping

    values = np.array([0.5, 1, 2])
    cmap, norm = colour_scale(values, limits=limits, neutral=1, robust=False)
    assert norm(1) == pytest.approx(0.5)
    assert _clipping(values, norm)[0] == extend
    constant = np.ones(4)
    _, norm = colour_scale(constant, neutral=1)
    assert _clipping(constant, norm) == ("neither", "")
    masked = np.ma.array([0.5, 1, 2, 100], mask=[False, False, False, True])
    _, norm = colour_scale(masked, limits=(0.8, 1.2), neutral=1)
    assert "33.3% below" in _clipping(masked, norm)[1]


def test_static_plots_keep_gaps_and_unrolled_seams(structure):
    fig, ax = p.plotting.profile_plot([1, 2, 5, 6], [1, np.nan, 3, 4])
    assert len(ax.lines) == 2
    fig, ax = p.plotting.profile_heatmap(
        pd.DataFrame([[1, 2, np.nan]], columns=[1, 3, 5])
    )
    assert ax.images[0].get_array().shape == (1, 5)
    xyz = np.array([[-1, 0.1, 0], [-1, -0.1, 1], [0, -1, 2]])
    frame = p.geometry.Frame.from_axis([0, 0, 1], reference=[1, 0, 0])
    fig, ax = p.plotting.unrolled_cylinder(xyz, frame=frame)
    assert len(ax.lines) == 2
    with pytest.raises(ValueError):
        p.plotting.unrolled_cylinder(xyz, frame=p.geometry.Frame.from_axis([0, 0, 1]))
    xyz, atoms = p.extract(structure, chains="I")
    values = np.ones(len(xyz))
    values[0] = np.nan
    p.plotting.structure_scatter(xyz, values, neutral=1)
    p.plotting.projection(xyz, values, neutral=1)
    p.plotting.plot_structures(structure, chains="I")
    p.plotting.segments_3d(xyz[:3], np.ones((3, 3)), scale=2, kind="axis")
    plt.close("all")


def test_viewer_exact_values_and_shared_scale(structure, tmp_path):
    atoms = p.extract(structure, chains="I")[1].iloc[:3]
    values = np.array([0.5, 1, 2])
    viewer = p.plotting.view_field(
        structure, atoms, values, limits=(0.5, 2), neutral=1, style="sphere"
    )
    html = viewer.write_html()
    assert '"psp_value": 2.0' in html and '"psp_colour": 0.5' in html
    assert (
        '"prop": "psp_colour"' in html and '"min": 0.0' in html and '"max": 1.0' in html
    )
    assert (
        not hasattr(viewer, "_proteinstrain_notes")
        or "_proteinstrain_notes" not in viewer.__dict__
    )
    viewer = p.plotting.view_structures(
        structure.relabel_chains({"I": "LongChain", "Q": None}), axes=True
    )
    html = viewer.write_html()
    assert "LongChain" in html
    file = tmp_path / "view.html"
    viewer.write_html(str(file))
    assert file.exists()


def test_standard_mmcif_and_b_factor_export(structure, tmp_path):
    prepared = structure.retain_chains("I").relabel_chains({"I": "LongChain"})
    path = tmp_path / "prepared.cif"
    p.save_structure(prepared, path)
    assert "_psp_" not in path.read_text()
    reloaded = p.load_structure(path)
    assert reloaded.chain_ids == ("LongChain",)
    assert p.chain_sequence(reloaded, "LongChain") == p.chain_sequence(
        prepared, "LongChain"
    )
    np.testing.assert_allclose(
        reloaded.coordinates(), prepared.coordinates(), atol=1e-3
    )
    xyz, atoms = p.extract(prepared)
    values = np.linspace(0.5, 2, len(atoms))
    original = values.copy()
    output = tmp_path / "display.cif"
    p.export_field(prepared, atoms, values, output, missing=-1.0)
    exported = p.load_structure(output)
    found = [
        atom.b_iso
        for residue in exported.chain("LongChain")
        for atom in residue
        if atom.name == "CA"
    ]
    np.testing.assert_allclose(found, values, atol=1e-4)
    np.testing.assert_array_equal(values, original)
    assert "_psp_" not in output.read_text()
    with pytest.raises(ValueError):
        p.export_field(
            prepared,
            pd.concat([atoms, atoms.iloc[:1]]),
            np.r_[values, 1],
            output,
            missing=-1.0,
        )


def test_pdb_input_and_fasta(structure, tmp_path):
    import gemmi

    holder = gemmi.Structure()
    holder.name = "model"
    holder.add_model(structure.model.clone())
    path = tmp_path / "model.pdb"
    holder.write_pdb(str(path))
    with pytest.warns(UserWarning, match="no deposited full polymer sequence"):
        loaded = p.load_structure(path)
    assert loaded.chain_ids == structure.chain_ids
    assert p.extract(loaded, chains="I")[0].shape[0] == 427
    alignment = {"a": "AC--DE", "b": "ACFGDE"}
    file = tmp_path / "alignment.fasta"
    p.write_alignment(alignment, file)
    assert p.read_alignment(file) == alignment
    with pytest.raises(ValueError):
        p.write_alignment({"a": "AC", "b": "ACD"}, file)


def test_mafft_success_failure_timeout_and_output_checks(monkeypatch):
    import proteinstrain.correspondence.sequence as sequence
    from types import SimpleNamespace
    import subprocess

    def run(command, **kwargs):
        assert command[-1] == "-" and kwargs["input"].startswith(">a")
        return SimpleNamespace(returncode=0, stdout=">a\nAC-D\n>b\nACED\n", stderr="")

    monkeypatch.setattr(sequence.subprocess, "run", run)
    assert p.align_multiple({"a": "ACD", "b": "ACED"}) == {"a": "AC-D", "b": "ACED"}
    assert p.align_multiple({"a": "ACD", "b": "ACD"}) == {"a": "ACD", "b": "ACD"}
    monkeypatch.setattr(
        sequence.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=2, stdout="", stderr="test failure"
        ),
    )
    with pytest.raises(RuntimeError, match="test failure"):
        p.align_multiple({"a": "ACD", "b": "ACED"})

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired("mafft", 1)

    monkeypatch.setattr(sequence.subprocess, "run", timeout)
    with pytest.raises(RuntimeError):
        p.align_multiple({"a": "ACD", "b": "ACED"}, timeout=1)


def test_failed_download_does_not_replace_cache(tmp_path, monkeypatch):
    import proteinstrain.io as io

    path = tmp_path / "8IXA.cif"
    path.write_text("keep")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return b"data_WRONG\n_entry.id WRONG\n"

    monkeypatch.setattr(io, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError):
        p.download_mmcif("8IXA", path, overwrite=True)
    assert path.read_text() == "keep"


def test_assembly_expansion_standard_input(tmp_path):
    # A tiny deposited mmCIF with one assembly operator, independent of real proteins.
    import gemmi

    source = Path(__file__).parent / "data/8IXA_subset.cif"
    doc = gemmi.cif.read(str(source))
    block = doc.sole_block()
    block.set_mmcif_category(
        "_pdbx_struct_assembly.",
        {
            "id": ["1"],
            "details": ["author_defined_assembly"],
            "method_details": [None],
            "oligomeric_details": ["monomeric"],
            "oligomeric_count": ["1"],
        },
    )
    block.set_mmcif_category(
        "_pdbx_struct_assembly_gen.",
        {"assembly_id": ["1"], "oper_expression": ["1"], "asym_id_list": ["A"]},
    )
    block.set_mmcif_category(
        "_pdbx_struct_oper_list.",
        {
            "id": ["1"],
            "type": ["identity operation"],
            **{
                f"matrix[{i}][{j}]": [str(int(i == j))]
                for i in range(1, 4)
                for j in range(1, 4)
            },
            **{f"vector[{i}]": ["0"] for i in range(1, 4)},
        },
    )
    file = tmp_path / "assembly.cif"
    doc.write_file(str(file))
    assert p.list_assemblies(file) == ("1",)
    assembly = p.load_assembly(file, "1")
    assert len(assembly.chain_ids) > 0
    with pytest.raises(LookupError):
        p.load_assembly(file, "absent")


def test_result_table_rejects_scalar_input(block):
    result = p.calculate(
        block, "R", "T", rule=p.WeightRule.fixed_radius(6), ensemble="reference"
    )
    with pytest.raises(ValueError):
        p.result_table(block, result, 1.0)


def test_standard_cif_reindexes_native_residues_and_keeps_descriptions(
    structure, tmp_path
):
    """Prepared models may carry label_seq values from a different deposited axis."""
    import proteinstrain as psp

    chain = structure.chain_ids[0]
    prepared = structure.retain_chains(chain)
    sequence = prepared.polymer_sequences[chain]
    prepared.polymer_sequences[chain] = (("X", None),) + sequence
    prepared.chain_descriptions[chain] = "Prepared polymer chain"
    # Deliberately retain the model's old label_seq numbering.
    original_labels = [r.label_seq for r in prepared.chain(chain)]
    path = tmp_path / "prepared.cif"
    psp.save_structure(prepared, path)
    restored = psp.load_structure(path)
    assert restored.polymer_sequences == prepared.polymer_sequences
    assert restored.chain_descriptions == prepared.chain_descriptions
    assert [r.label_seq for r in prepared.chain(chain)] == original_labels
