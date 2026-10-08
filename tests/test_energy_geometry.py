import numpy as np
import pandas as pd
import pytest
import proteinstrain as p
from proteinstrain import constitutive as c


@pytest.mark.parametrize("model", [c.IsotropicSVK(2e6, 3e6), c.NeoHookean(2e6, 3e6)])
def test_energy_units_and_stress(model):
    F = np.diag([1.1, 0.98, 1.03])[None]
    density = c.energy_density(model, F)
    volumes = np.array([25.0])
    joules = c.total_energy(model, F, volumes, units="J")
    np.testing.assert_allclose(joules, density * 25e-30)
    np.testing.assert_allclose(
        c.total_energy(model, F, volumes, units="kT", temperature=300),
        joules / (1.380649e-23 * 300),
    )
    np.testing.assert_allclose(
        c.total_energy(model, F, volumes, units="kJ/mol"), joules * 6.02214076e23 / 1000
    )
    S = c.stress.second_piola(model, F)
    P = c.stress.first_piola(model, F)
    sigma = c.stress.cauchy(model, F)
    np.testing.assert_allclose(P, F @ S)
    np.testing.assert_allclose(
        sigma, F @ S @ F.transpose(0, 2, 1) / np.linalg.det(F)[:, None, None]
    )
    broken = F.copy()
    broken[:, 0, :] *= -1
    assert np.isnan(c.total_energy(model, broken, volumes)).all()


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_bad_energy_temperature(temperature):
    with pytest.raises(ValueError):
        c.total_energy(
            c.IsotropicSVK(2, 3), np.eye(3)[None], [1], temperature=temperature
        )


def test_material_and_explicit_volume_allocation():
    model = c.IsotropicSVK.from_young_poisson(100e6, 0.33)
    assert model.young_modulus == pytest.approx(100e6)
    assert model.poisson_ratio == pytest.approx(0.33)
    values = c.allocate_volume(100, 3, weights=[1, 2, 1])
    np.testing.assert_allclose(values, [25, 50, 25])
    values = c.allocate_group_volumes(
        {"A": 30, "B": 40}, ["A", "A", "B"], weights=[1, 2, 1]
    )
    np.testing.assert_allclose(values, [10, 20, 40])
    with pytest.raises(ValueError):
        c.allocate_volume(10, 2, weights=[0, 0])
    with pytest.raises(ValueError):
        c.allocate_group_volumes({"A": 10}, ["A", "B"])


@pytest.mark.parametrize("method", ["spheres", "grid", "monte_carlo"])
def test_single_sphere_volume(method):
    total, per_atom, error = c.union_volume(
        [[0, 0, 0]], [2], method=method, samples=10000, spacing=0.1
    )
    assert total == pytest.approx(4 * np.pi * 8 / 3, rel=0.02)
    assert per_atom.sum() == pytest.approx(total)
    if method != "grid":
        assert error == 0


def test_volume_partition_and_relabel_invariance(structure):
    _, atoms = p.extract(structure, chains="I", atoms="CA")
    atoms = atoms.iloc[:5].copy()
    volumes, error = c.volumes_from_structure(structure, atoms, method="spheres")
    assert (volumes > 0).all()
    assert error == 0
    relabelled = structure.relabel_chains({"I": "X"})
    other = atoms.copy()
    other["chain_id"] = "X"
    values, _ = c.volumes_from_structure(relabelled, other, method="spheres")
    np.testing.assert_allclose(values, volumes)
    # The same residue-union is used whether group labels partition residues or not.
    divided, _ = c.volumes_from_structure(
        structure, atoms, groups=np.arange(len(atoms)), method="spheres"
    )
    assert divided.sum() == pytest.approx(volumes.sum())
    duplicates = pd.concat([atoms, atoms.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        c.volumes_from_structure(structure, duplicates, method="spheres")


def test_rigid_fit_and_geometry_nonmutation(block):
    original = block.xyz.copy()
    fit = p.geometry.superpose_coordinates(
        block, moving_id="T", reference_id="R", weights=np.ones(block.n_atoms)
    )
    assert isinstance(fit, p.geometry.RigidTransform) and fit.rmsd > 0
    rigid = p.geometry.RigidTransform.about_axis([1, 2, 3], 0.4)
    fixed = block.xyz[0]
    moving = rigid.apply(fixed) + [4, 3, -2]
    transform = p.geometry.fit_rigid_transform(moving, fixed)
    np.testing.assert_allclose(transform.apply(moving), fixed, atol=1e-13)
    np.testing.assert_allclose(transform.inverse().apply(fixed), moving, atol=1e-13)
    np.testing.assert_array_equal(block.xyz, original)


@pytest.mark.parametrize("trim", [0, -1, np.nan, np.inf, -np.inf])
def test_shared_trim_validation(block, structure, trim):
    with pytest.raises(ValueError):
        p.geometry.superpose_coordinates(
            block, moving_id="T", reference_id="R", trim=trim
        )
    with pytest.raises(ValueError):
        p.geometry.superpose_structures(
            structure, structure, reference_chains="I", trim=trim
        )


def test_trim_refit_reduces_outlier():
    xyz = p.synthetic.lattice((3, 3, 3), 1.0)
    moving = xyz + [3, 2, 1]
    moving[0] += [2, 1, 1]
    from proteinstrain.geometry.transforms import _fit_trim

    initial = p.geometry.fit_rigid_transform(moving, xyz)
    trimmed = _fit_trim(moving, xyz, None, 0.6)
    assert trimmed.rmsd < 1e-12 and initial.rmsd > 0.1


def test_structure_superposition_returns_tuple(structure):
    shifted = p.geometry.apply_transform(
        structure, p.geometry.RigidTransform(np.eye(3), [5, 2, 1])
    )
    output, transform = p.geometry.superpose_structures(
        shifted, structure, reference_chains="I"
    )
    assert transform.rmsd < 1e-12
    np.testing.assert_allclose(
        output.coordinates("I"), structure.coordinates("I"), atol=1e-12
    )
    np.testing.assert_allclose(
        shifted.coordinates("I"), structure.coordinates("I") + [5, 2, 1]
    )


def test_cylinder_plane_and_nonconvergence():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    heights = np.linspace(-5, 5, 7)
    xyz = np.array([[3 * np.cos(a), 3 * np.sin(a), z] for a in angles for z in heights])
    fit = p.geometry.fit_cylinder(xyz)
    assert fit.radius == pytest.approx(3, abs=1e-9)
    assert fit.rmsd < 1e-9
    tilted = p.geometry.RigidTransform.about_axis([1, 1, 0], 0.5).apply(xyz)
    fit = p.geometry.fit_cylinder(tilted, axis=[0.3, 0.2, 1])
    assert fit.radius == pytest.approx(3, abs=1e-8)
    with pytest.raises(RuntimeError, match="converge"):
        p.geometry.fit_cylinder(tilted, axis=[1, 0, 1], max_evaluations=1)
    plane = p.geometry.fit_plane([[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]])
    assert plane.rmsd == pytest.approx(0)
    assert abs(plane.normal[2]) == pytest.approx(1)


def test_direct_assignments_and_groups(structure):
    target = structure.relabel_chains({"I": "X", "Q": "Y"})
    mapping = p.preparation.assign_by_centroid(structure, target)
    assert mapping == {"I": "X", "Q": "Y"}
    grouped = p.preparation.assign_groups(
        structure, {"all": ".*", "missing": "DOES_NOT_EXIST"}, claim="first"
    )
    assert (
        grouped.groups["missing"] == () and grouped.groups["all"] == structure.chain_ids
    )
    with pytest.raises(ValueError):
        p.preparation.assign_groups(structure, {"a": ".*", "b": ".*"}, claim="error")
    sequence_groups = p.preparation.assign_groups_by_sequence(
        structure,
        {"I": (structure, "I")},
        claim="first",
        minimum_identity=0.99,
        minimum_coverage=0.99,
        margin=0.05,
    )
    assert "I" in sequence_groups.groups["I"]


def test_contacts_frames_and_directions(structure):
    frame = p.geometry.Frame.from_axis([0, 0, 1], reference=[1, 0, 0])
    xyz = np.array([[1, 0, 0], [0, 1, 1], [-1, 0, 2]], float)
    np.testing.assert_allclose(frame.to_global(frame.to_local(xyz)), xyz)
    bases = p.geometry.cylindrical_frames(xyz)
    np.testing.assert_allclose(
        bases @ bases.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), bases.shape),
        atol=1e-15,
    )
    table = p.geometry.contact_map(xyz, 3)
    assert len(table) == 3
    changes = p.geometry.contact_changes(xyz, xyz * 1.1, 3)
    np.testing.assert_allclose(changes.extension, 0.1)
    interface = p.geometry.chain_interface(
        structure, "I", "Q", cutoff=8, atoms="CA", axis=[0, 0, 1]
    )
    assert isinstance(interface, dict) and interface["contacts"] >= 0


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, np.inf])
def test_direct_energy_conversion_rejects_invalid_temperature(temperature):
    from proteinstrain.constitutive.energy import convert_energy

    with pytest.raises(ValueError):
        convert_energy(np.ones(3), "kT", temperature)
