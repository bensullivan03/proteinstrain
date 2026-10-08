import numpy as np
import pytest
import proteinstrain as p
from proteinstrain.mechanics import measures as m
from proteinstrain.mechanics.deformation import (
    reference_moments,
    deformation_gradient,
)

RULES = [
    p.WeightRule.fixed_radius(6),
    p.WeightRule.tapered(3, 7),
    p.WeightRule.intersection(6),
    p.WeightRule.knn(12),
]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("subset", [False, True])
def test_affine_and_reference_reuse(block, rule, subset):
    indices = np.arange(0, block.n_atoms, 2) if subset else None
    graph = p.build_graph(block.xyz[:2], rule, centre_indices=indices)
    fit = reference_moments(graph, block.xyz[0])
    F, valid, residual = fit.solve(block.xyz[1], nonaffine=True)
    expected = np.array([[1.07, 0.03, 0.0], [-0.02, 0.96, 0.01], [0.0, 0.02, 1.03]])
    assert valid.all()
    np.testing.assert_allclose(F, np.broadcast_to(expected, F.shape), atol=2e-14)
    assert np.max(residual) < 1e-26
    other = fit.solve(block.xyz[2])[0]
    np.testing.assert_allclose(
        other, np.broadcast_to(np.diag([1.02, 0.98, 1.01]), other.shape), atol=2e-14
    )


@pytest.mark.parametrize("rule", RULES)
def test_batch_matches_single_and_inputs_unchanged(block, rule):
    original = block.xyz.copy()
    results = p.calculate_pairs(
        block,
        pairs=[("R", "T"), ("R", "U"), ("T", "R")],
        rule=rule,
        ensemble="all",
        nonaffine=True,
    )
    assert isinstance(results, dict)
    for (a, b), result in results.items():
        one = p.calculate(block, a, b, rule=rule, ensemble="all", nonaffine=True)
        np.testing.assert_allclose(result.F, one.F, equal_nan=True)
        assert not hasattr(result, "graph") and not hasattr(result, "coordinates")
    np.testing.assert_array_equal(block.xyz, original)


def test_batch_reuses_graph_and_moments(block, monkeypatch):
    import proteinstrain.calculation as calculation

    graph_builds = []
    moment_builds = []
    build = calculation.build_graph
    moments = calculation.reference_moments

    def track_graph(*a, **kw):
        graph_builds.append(1)
        return build(*a, **kw)

    def track_moments(*a, **kw):
        moment_builds.append(1)
        return moments(*a, **kw)

    monkeypatch.setattr(calculation, "build_graph", track_graph)
    monkeypatch.setattr(calculation, "reference_moments", track_moments)
    p.calculate_pairs(
        block, pairs=[("R", "T"), ("R", "U"), ("T", "R")], rule=RULES[0], ensemble="all"
    )
    assert len(graph_builds) == 1 and len(moment_builds) == 2


@pytest.mark.parametrize("rule", RULES)
def test_separate_centres_and_pool(block, rule):
    centres = np.arange(0, block.n_atoms, 3)
    result = p.calculate(
        block,
        "R",
        "T",
        rule=rule,
        ensemble="reference",
        centres=centres,
        neighbour_pool="all",
    )
    assert result.valid.all()
    np.testing.assert_array_equal(result.centre_indices, centres)
    graph = p.build_graph(
        block.xyz[:1],
        rule,
        centre_indices=centres,
        neighbour_indices=np.arange(block.n_atoms),
    )
    assert np.any(~np.isin(graph.pair_neighbour, centres))


@pytest.mark.parametrize("kind", ["line", "few", "reflection", "collapse"])
def test_invalid_neighbourhoods_are_missing(kind):
    xyz = (
        np.c_[np.arange(8), np.zeros((8, 2))]
        if kind == "line"
        else p.synthetic.lattice((3, 3, 3), 1.0)
    )
    graph = (
        p.build_graph(xyz, p.WeightRule.fixed_radius(0.1 if kind == "few" else 10.0))
        if kind != "few"
        else p.build_graph(xyz, p.WeightRule.knn(2))
    )
    target = xyz.copy()
    if kind == "reflection":
        target[:, 0] *= -1
    if kind == "collapse":
        target[:] = 0
    F, valid, residual = deformation_gradient(graph, xyz, target, nonaffine=True)
    assert not valid.any()
    assert np.isnan(F).all() and np.isnan(residual).all()


def test_nonaffine_residual_and_rigid_motion(block):
    graph = p.build_graph(block.xyz[:1], RULES[0])
    fit = reference_moments(graph, block.xyz[0])
    target = block.xyz[0].copy()
    target[:, 0] += 0.1 * np.sin(target[:, 2])
    assert fit.solve(target, nonaffine=True)[2].max() > 0
    rigid = p.geometry.RigidTransform.about_axis([0, 0, 1], 0.3)
    F, valid, _ = fit.solve(rigid.apply(block.xyz[0]))
    assert valid.all()
    assert np.abs(m.green_lagrange_strain(F)).max() < 1e-14


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_bad_radius(value):
    with pytest.raises(ValueError):
        p.WeightRule.fixed_radius(value)


@pytest.mark.parametrize("bad", [[-1], [999], [1.5], [[1]]])
def test_bad_centre_indices(block, bad):
    with pytest.raises((ValueError, TypeError)):
        p.calculate(block, "R", "T", rule=RULES[0], ensemble="reference", centres=bad)


def test_knn_exact_neighbours_with_external_pool():
    points = np.c_[np.arange(12), np.zeros((12, 2))]
    graph = p.build_graph(
        points,
        p.WeightRule.knn(3),
        centre_indices=[0, 11],
        neighbour_indices=np.arange(1, 11),
    )
    assert graph.counts().tolist() == [3, 3]
    assert graph.pair_neighbour[:3].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        p.build_graph(points, p.WeightRule.knn(20))


@pytest.mark.parametrize(
    "function",
    [
        m.volume_ratio,
        m.relative_volume_change,
        m.log_volume_change,
        m.green_lagrange_frobenius,
        m.max_stretch,
        m.min_stretch,
        m.corrected_max_stretch,
        m.rotation_angle,
    ],
)
def test_missing_measures_remain_missing(function):
    assert np.isnan(function(np.full((1, 3, 3), np.nan))).all()


def test_known_answer_measures():
    F = (1.1 * np.eye(3))[None]
    E = m.green_lagrange_strain(F)
    assert m.trace_strain(E)[0] == pytest.approx(0.315)
    assert m.relative_volume_change(F)[0] == pytest.approx(0.331)
    assert m.corrected_max_stretch(F)[0] == pytest.approx(1 - 1 / 1.1)
    assert m.hencky_strain(F)[0, 0, 0] == pytest.approx(np.log(1.1))
    F = np.diag([1.2, 1, 1])[None]
    E = m.green_lagrange_strain(F)
    assert m.directional_strain(E, [1, 0, 0])[0] == pytest.approx(0.22)
    assert m.directional_stretch(E, [1, 0, 0])[0] == pytest.approx(1.2)
    F = np.array([[[1, 0.2, 0], [0, 1, 0], [0, 0, 1]]])
    E = m.green_lagrange_strain(F)
    assert m.volume_ratio(F)[0] == pytest.approx(1)
    assert m.shear_component(E, [1, 0, 0], [0, 1, 0])[0] == pytest.approx(0.1)
    assert m.angle_change(E, [1, 0, 0], [0, 1, 0])[0] == pytest.approx(-np.arctan(0.2))


def test_rotation_and_local_tensor_basis():
    transform = p.geometry.RigidTransform.about_axis([0, 0, 1], 0.4)
    F = transform.rotation[None]
    rotation, stretch = m.polar_decomposition(F)
    np.testing.assert_allclose(rotation[0], transform.rotation, atol=1e-14)
    np.testing.assert_allclose(stretch[0], np.eye(3), atol=1e-14)
    twist, swing = m.twist_swing(F, [0, 0, 1])
    assert twist[0] == pytest.approx(0.4)
    assert swing[0] == pytest.approx(0, abs=1e-12)
    tensor = np.diag([2, 3, 4])[None]
    frame = p.geometry.Frame.from_axis([1, 0, 0], reference=[0, 1, 0])
    moved = p.geometry.transport(tensor, frame)
    back = p.geometry.transport(moved, p.geometry.Frame.cartesian(), frame)
    np.testing.assert_allclose(back, tensor)


@pytest.mark.parametrize(
    "centres,edge",
    [
        ([1, 0], [0]),
        ([0, 0], [0]),
        ([0], [0.5]),
        ([0], np.array([2**63], dtype=np.uint64)),
    ],
)
def test_raw_graph_indices_do_not_silently_change(centres, edge):
    with pytest.raises((ValueError, TypeError)):
        p.NeighbourGraph(centres, edge, [1], [1.0], 3)


def test_synthetic_workflow():
    xyz = p.synthetic.lattice((4, 4, 4), 2.0)
    target = p.synthetic.affine(xyz, np.diag([1.1, 0.9, 1.0]))
    block = p.synthetic.synthetic_block(xyz, target)
    result = p.calculate(
        block, *block.item_ids, rule=p.WeightRule.fixed_radius(5), ensemble="reference"
    )
    assert result.valid.all()
    np.testing.assert_allclose(
        result.F, np.broadcast_to(np.diag([1.1, 0.9, 1.0]), result.F.shape), atol=1e-14
    )
