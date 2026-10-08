"""Scientific contracts for the small helpers shared by independent analyses."""

import gemmi
import numpy as np
import pytest
import matplotlib.pyplot as plt
import proteinstrain as p
from proteinstrain.mechanics import measures as m


def test_alignment_positions_preserves_gaps_and_remaps():
    positions = p.alignment_positions({"a": "A-CD-", "b": "-AC-D"})
    assert positions[positions.item_id.eq("a")].column.tolist() == [0, 2, 3]
    assert positions[positions.item_id.eq("b")].polymer_position.tolist() == [0, 1, 2]
    mapped = p.remap_alignment(
        positions.drop(columns="column"), positions, on=("item_id", "polymer_position")
    )
    assert mapped.column.tolist() == positions.column.tolist()
    with pytest.raises(ValueError):
        p.alignment_positions({"a": "A", "b": "AA"})


def test_clean_atoms_preserves_input_and_full_sequence(structure):
    source = structure.relabel_chains({})
    residue = source.model[0][0]
    original = residue[0].clone()
    original.altloc = "A"
    original.occ = 0.8
    residue[0].altloc = "B"
    residue[0].occ = 0.2
    residue.add_atom(original)
    empty = residue[1]
    empty.occ = 0
    before = len(residue)
    omitted = p.preparation.clean_atoms(source)
    assert len(source.model[0][0]) == before
    assert omitted.polymer_sequences == source.polymer_sequences
    assert not any(a.name == original.name for a in omitted.model[0][0])
    highest = p.preparation.clean_atoms(source, altloc="highest")
    selected = [a for a in highest.model[0][0] if a.name == original.name]
    assert len(selected) == 1 and selected[0].occ == pytest.approx(0.8)
    assert selected[0].altloc == "\x00"
    assert all(a.occ > 0 for a in highest.model[0][0])
    with pytest.raises(ValueError):
        p.preparation.clean_atoms(source, minimum_occupancy=np.nan)


def test_principal_system_including_missing_and_degenerate():
    F = np.array([np.diag([0.7, 1.1, 1.4]), np.eye(3), np.full((3, 3), np.nan)])
    stretches, directions = m.principal_stretch_system(F)
    np.testing.assert_allclose(stretches[:2], [[0.7, 1.1, 1.4], [1, 1, 1]])
    reconstructed = np.einsum(
        "nji,nj,njk->nik", directions[:2], stretches[:2] ** 2, directions[:2]
    )
    np.testing.assert_allclose(reconstructed, m.right_cauchy_green(F[:2]))
    assert np.isnan(stretches[2]).all() and np.isnan(directions[2]).all()
    assert np.isnan(m.principal_axis_alignment(F, [0, 0, 1])[1:]).all()


def test_projected_trace_does_not_bridge_gap(structure):
    source = structure.relabel_chains({})
    chain = source.model[0]
    polymer = source.polymer_sequences[chain.name]
    native = next(residue for _, residue in polymer[1:-1] if residue is not None)
    for i, residue in enumerate(chain):
        if (
            residue.seqid.num == native.number
            and residue.seqid.icode == native.insertion
        ):
            del chain[i]
            break
    fig, ax = p.plotting.plot_structures(
        source, chains=chain.name, plane=(0, 1), alpha=0.4, linewidth=2, labels=False
    )
    assert len(ax.lines) >= 2
    assert all(line.get_alpha() == 0.4 for line in ax.lines)
    plt.close(fig)


def test_viewer_explicit_style_options(structure):
    viewer = p.plotting.view_structures(
        structure, opacity=0.3, label_size=19, mark_start=True
    )
    html = viewer._make_html()
    assert '"opacity": 0.3' in html and '"fontSize": 19' in html and "addSphere" in html
