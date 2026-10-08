# Materials, volumes and geometry

## Material models

`psp.constitutive` provides two isotropic hyperelastic models. Moduli are in pascals, and no protein-specific default is provided: elastic constants of proteins are assumptions of the analysis and should be stated with the results.

| Model | Energy density *W* |
|---|---|
| `IsotropicSVK(lame_first, shear_modulus)` | (λ/2)(tr **E**)² + μ **E**:**E** (Saint Venant–Kirchhoff) |
| `NeoHookean(lame_first, shear_modulus)` | (μ/2)(*I*₁ − 3) − μ ln *J* + (λ/2)(ln *J*)² (compressible neo-Hookean) |

Both have a `from_young_poisson(young, poisson)` constructor. `lame_from_young_poisson` and `young_poisson_from_lame` convert between parameter sets. `IsotropicSVK.energy_parts(E)` splits the density into bulk and shear parts.

For comparison with PSA, `IsotropicSVK.from_young_poisson(100e6, 0.33)` gives PSA's default λ = 7.30 × 10⁷ Pa and μ = 3.76 × 10⁷ Pa to three significant figures.

## Energies

- `energy_density(model, F)` returns one density per centre, in Pa (J m⁻³ of reference volume).
- `total_energy(model, F, volumes, *, units="kT", temperature=300.0)` returns **one energy per centre** (an array, not a sum). Volumes are in Å³. Units are `"J"`, `"kT"`, `"kJ/mol"` and `"kcal/mol"`; `temperature` (K) is used only for `"kT"`.
- `convert_energy(joules, units, temperature=300.0)` converts energies.

Invalid fits remain NaN. When summing energies over a region, check how many centres are valid: a sum over a partly invalid set is not comparable with a sum over a complete one (see `per_residue(..., "sum")`).

## Volumes

Each centre needs a volume to turn an energy density into an energy.

- `allocate_volume(total, n_centres, *, weights=None)` distributes one total volume over the centres, equally or in proportion to `weights`.
- `allocate_group_volumes(totals, groups, *, weights=None, elements=None)` distributes a total per group. With `elements`, shares are proportional to isolated van der Waals sphere volumes, as in PSA's `effective_volume`.
- `volumes_from_structure(structure, atoms_table, *, groups=None, radii=None, method="monte_carlo", samples=2000000, seed=0)` computes the van der Waals union volume of all atoms in the residues that contain the centres, partitions it by power distance, and assigns it to the centres. It returns `(volumes, standard_error)`. Pass only the centre rows, in result order. A residue cannot belong to two groups. The structure must have one atom per native site; run `preparation.clean_atoms` first if necessary.
- `union_volume(xyz, radii, *, method="monte_carlo", samples=2000000, seed=0, spacing=0.4)` returns `(total, per_atom, standard_error)`. Methods:
  - `"monte_carlo"` — unbiased sampling inside each sphere; `standard_error` is the sampling error;
  - `"grid"` — a regular grid of the given spacing (Å); `standard_error` is `None`;
  - `"spheres"` — isolated spheres, ignoring overlap; `standard_error` is 0.
- `structure_volume(structure, chains=None, *, radii=None, **kwargs)` returns `(union_volume_tuple, element_symbols)` for every atom of the selected chains, ligands and water included.
- `atom_radii(elements, *, radii=None)`, `sphere_volumes(radii)` and `packing_factor(volume_tuple, radii)` are helpers. The default radii include H 1.20, C 1.70, N 1.55, O 1.52 and S 1.80 Å (the same values as PSA) plus common ions; elements not in the table get 1.7 Å. Pass `radii={...}` to use another set.

## Stress

Functions in `psp.constitutive.stress` take the material model first, so the assumption is explicit:

| Function | Quantity | Configuration |
|---|---|---|
| `second_piola(model, F)` | **S** | reference |
| `first_piola(model, F)` | **P** = **F S** (force per reference area) | mixed |
| `cauchy(model, F)` | **σ** = **F S F**ᵀ / *J* (force per current area) | current |
| `von_mises(stress)` | von Mises equivalent stress | — |
| `pressure(stress)` | −tr(**σ**)/3, **compression positive** | — |
| `directional_stress(stress, direction)` | normal stress along a direction | as input |
| `traction(stress, normal)` | traction vector on a surface | as input |

## Geometry

In `psp.geometry`:

### Rigid transforms and superposition

- `RigidTransform(rotation, translation, rmsd=None)` applies **x** ↦ **R x** + **t**. Reflections are rejected. Methods: `apply`, `inverse`, `then`, `as_rotvec`, `angle`, `axis`, `to_gemmi`; constructors `identity`, `from_rotvec`, `about_axis`.
- `fit_rigid_transform(moving, fixed, *, weights=None)` is a weighted least-squares proper rotation and translation (Kabsch).
- `superpose_coordinates(block, *, moving_id, reference_id, columns=None, weights=None, trim=None)` returns a transform. `columns` are **block array indices**, not alignment-column labels.
- `superpose_structures(moving, reference, *, chain_map=None, atoms="CA", minimum_atoms=10, trim=None, ...)` returns `(transformed_structure, transform)`. `chain_map` points **reference→moving**.
- `trim` is a finite positive residual cut-off (Å) for one trim-and-refit pass.
- `apply_transform(structure, transform)` and `displacement(block, reference_id, target_id, *, remove=None)` return new objects.

### Fits

- `fit_plane(xyz)` returns a `PlaneFit` (normal, centre, RMSD).
- `fit_cylinder(xyz, *, axis=None, centre=None, max_evaluations=200)` returns a `CylinderFit` (axis, centre, radius, RMSD); non-convergence raises an error.
- `inertia_axes(xyz)`, `circular_mean`, `cylindrical_angles` and `chain_cylindrical_angles` are also provided.

### Frames

A `Frame` holds an origin, row-wise right-handed orthonormal axes, and an `angular_reference` flag that is `True` when an explicit reference direction fixes the zero of longitude.

- Constructors: `Frame.cartesian()`, `from_axis(axis, origin, reference=None)`, `from_inertia(xyz)`, `from_reference(structure, ...)`, `from_chains(structure, start, end, *, reference=None)` and `from_cylinder(structure, chains, *, towards, reference=None)`. In axis-based frames the third local axis is the given axis.
- Inertia frames have an arbitrary sign convention and no angular reference.
- `frame.to_local(xyz)`, `frame.to_global(xyz)`; `transport(tensors, frame_to, frame_from=None)` and `transport_vectors(...)` change the basis of tensors and vectors. `cylindrical_frames` and `local_frames_from_directions` build per-atom bases.
- `order_about_axis(structure, chains, frame, *, start, direction)` orders chains by angle about the frame's third axis. `"anticlockwise"` means increasing angle by the right-hand rule about that axis, that is, anticlockwise when viewed from the positive end of the axis. A numeric `start` (radians) requires `angular_reference=True`.
- An unrolled cylindrical plot requires a frame with an angular reference.

### Contacts

`contact_map(xyz, cutoff)`, `contact_changes(reference_xyz, target_xyz, cutoff)`, `neighbour_counts`, `external_neighbour_counts` and `chain_contact_matrix` return plain arrays or tables. `chain_interface(structure, first, second, *, cutoff, atoms, axis=None, normalise=None)` returns a dictionary of centroid distance, minimum distance, contact count and, optionally, axial/lateral offsets and normalised contacts.
