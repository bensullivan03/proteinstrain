# Local deformation and direct measures

## Calculating deformation gradients

```python
result = psp.calculate(block, reference_id, target_id, *, rule, ensemble,
                       centres=None, neighbour_pool="centres",
                       rank_tolerance=1e-8, det_tolerance=1e-12,
                       minimum_neighbours=4, nonaffine=False)
```

returns a `DeformationResult(reference_id, target_id, F, centre_indices, valid, nonaffine)`:

- `F` — shape `(n_centres, 3, 3)`; `F[i]` belongs to block atom `centre_indices[i]`;
- `valid` — boolean per centre;
- `nonaffine` — weighted mean squared residual of the affine fit in Å² if `nonaffine=True`, otherwise `None`.

The result holds no coordinates, graph or fit report. The calculation is **directed**: `reference → target` and `target → reference` are different results.

`psp.calculate_pairs(block, *, pairs, rule, ensemble, ...)` returns a dictionary keyed by `(reference_id, target_id)`. `pairs="all"` requests every distinct directed pair. Within one call, neighbour graphs and reference moments are reused where the rule and ensemble allow; nothing is cached between calls.

### The fit

At each centre the fit uses the reference offsets **x** and target offsets **y** to each neighbour, with weights *w*:

- reference moments **D** = Σ *w* **x** **x**ᵀ;
- mixed moments **A** = Σ *w* **y** **x**ᵀ;
- deformation gradient **F** = **A D**⁻¹.

This is the formulation of PSA (Sartori and Leibler, *Phys. Rev. X* **14**, 011042 (2024)), following Gullett *et al.* (2008) and Zimmerman *et al.* (2009).

A fit is invalid (`valid=False`, `F` = NaN) when the centre has fewer than `minimum_neighbours` neighbours, when the smallest eigenvalue of **D** is below `rank_tolerance` × max(largest eigenvalue, 1), when **F** is not finite, or when det **F** ≤ `det_tolerance`. No per-centre reason is stored.

### Neighbourhood rules

The rule is always explicit:

| Rule | Weight of neighbour *j* of centre *i* |
|---|---|
| `psp.WeightRule.fixed_radius(r)` | 1 if ‖*j* − *i*‖ ≤ *r*, else 0, **averaged over the ensemble members**. With two members, a pair within *r* in only one member has weight 0.5. |
| `psp.WeightRule.tapered(r_inner, r)` | 1 below `r_inner`, falling linearly to 0 at `r`, averaged over the ensemble members. Equivalent to PSA's `"linear"` weights. |
| `psp.WeightRule.intersection(r)` | 1 if the pair is within *r* in **every** ensemble member, else 0. Equivalent to PSA's `"intersect"` weights. |
| `psp.WeightRule.knn(k)` | 1 for the *k* nearest neighbours in the **first** ensemble member. Equivalent to PSA's `"minimal"` weights. |

Distances are in Å.

### Ensembles

`ensemble` names the structures on which neighbourhoods are measured. There is no default:

- `"reference"` — the reference item only;
- `"target"` — the target item only;
- `"endpoints"` — reference and target, in that order;
- `"all"` — every item in the block;
- an explicit sequence of item IDs.

For `knn`, and for directed calculations in general, the order of the pair matters.

### Centres and neighbours

`centres` selects block atom indices at which **F** is fitted (default: all). `neighbour_pool="centres"` (the default) draws neighbours from those centres only; `"all"` draws them from every block atom; an index array chooses a separate pool. For example, to fit at Cα atoms using all backbone atoms as neighbours, build a backbone block, pass the Cα indices as `centres` and set `neighbour_pool="all"`.

### Low-level functions

`psp.build_graph(ensemble_xyz, rule, *, centre_indices=None, neighbour_indices=None)`, `psp.reference_moments(graph, reference_xyz)` and `psp.deformation_gradient(graph, reference_xyz, target_xyz)` accept plain arrays. A `ReferenceMoments` object's `solve(target_xyz)` returns `(F, valid, residual)` and can be reused for several targets.

## Measures

Measures are plain functions in `proteinstrain.mechanics.measures`:

```python
from proteinstrain.mechanics import measures as m

E = m.green_lagrange_strain(result.F)
strain_size = m.green_lagrange_frobenius(result.F)
axial_strain = m.directional_strain(E, direction=[0, 0, 1])
axial_stretch = m.directional_stretch(E, direction=[0, 0, 1])
stretches = m.principal_stretches(result.F)
rotation = m.rotation_angle(result.F, degrees=True)
```

Argument conventions:

- functions of a **strain tensor** (`directional_strain`, `directional_stretch`, `shear_component`, `angle_change`, `principal_strains`, `principal_directions`, `trace_strain`) take **E**;
- functions of the **deformation** (`*_strain` constructors, stretches, rotations, volume ratios, `polar_decomposition`, `remove_rigid`) take **F**.

| Group | Functions |
|---|---|
| Strain tensors | `green_lagrange_strain` (reference configuration), `euler_almansi_strain` (current configuration), `hencky_strain`, `linear_strain`, `deviatoric` |
| Scalars | `green_lagrange_frobenius`, `frobenius`, `trace_strain`, `invariants` |
| Stretch | `principal_stretches` (ascending), `stretch_directions`, `max_stretch`, `min_stretch`, `corrected_max_stretch` (1 − 1/λ<sub>max</sub>), `principal_axis_alignment` |
| Direction-resolved | `directional_strain` (**n**ᵀ**E n**), `directional_stretch` (√(1 + 2 **n**ᵀ**E n**)), `shear_component`, `angle_change` |
| Volume | `volume_ratio` (*J* = det **F**), `relative_volume_change` (*J* − 1), `log_volume_change` (ln *J*) |
| Rotation | `polar_decomposition` (**F** = **R U**), `rotation_angle`, `rotation_axis`, `twist_swing`, `remove_rigid` (returns **U**) |
| Fit quality | `nonaffine_residual(graph, reference, target, F)` (Å²) |

Tensor components are expressed in the input Cartesian basis; use `geometry.transport` to express them in another frame. `trace_strain` is not the volume change for finite strain; use `volume_ratio` or `log_volume_change` for that.

Neutral values: strain, shear and rotation are 0 for no deformation; stretch and volume ratio are 1. Set the neutral value explicitly when plotting signed quantities (see [plotting.md](plotting.md)). Magnitudes such as `green_lagrange_frobenius` are non-negative and are best shown on a sequential scale.

Invalid fits stay NaN in every measure. Some algebraic functions (for example `green_lagrange_strain`) evaluate any finite tensor, including a reflection, so physical validity is decided by the solver's `valid` flag rather than by the measure.

## Parameter studies

A parameter scan is an ordinary loop over `calculate` or `calculate_pairs`. Record the rule, ensemble and atom selection alongside each result in your own tables.
