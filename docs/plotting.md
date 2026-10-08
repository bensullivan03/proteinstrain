# Essential plotting

Plotting functions are in `psp.plotting`. Static functions use Matplotlib, return `(figure, axes)` and accept an existing `ax=` to draw into. Interactive functions return a py3Dmol view; save it with `view.write_html(path)`. There is no layout framework: arrange figures with Matplotlib subplots or separate viewers.

## Structures

- `plot_structures(structures, *, chains=None, frame=None, colours=None, labels=True, plane=None)` overlays Cα traces of prepared structures in 3D, or in a Cartesian plane such as `plane=(0, 1)`. Lines break at polymer gaps.
- `view_structures(structures, *, chains=None, colours=None, style="cartoon", labels=True, axes=False)` overlays structures in one py3Dmol viewer. Chain IDs are used directly, including multi-character IDs. `axes=True` draws Cartesian axes near a corner of the bounding box.

## Scalar fields

- `structure_scatter(xyz, values, ...)` (3D) and `projection(xyz, values, *, plane=(0, 1), ...)` (2D) colour points by a scalar.
- `view_field(structure, atoms_table, values, ...)` colours the native atoms in `atoms_table` in a py3Dmol viewer. Atoms that are missing or not analysed are shown in grey. Each coloured atom carries two properties: `psp_value` (the exact value) and `psp_colour` (its normalised position on the colour scale). The structure must have one atom per native site; run `psp.preparation.clean_atoms` first if it has alternate locations.
- `unrolled_cylinder(xyz, values=None, *, frame, positions=None, ...)` plots angle about the frame's axis against axial position. Lines break at the ±π seam and, if `positions` is given, at polymer gaps. The frame must have an angular reference.

## Vectors

`segments_3d(xyz, directions, *, scale, kind="vector")` (Matplotlib) and `add_segments(view, xyz, directions, *, scale, kind="vector")` (py3Dmol) draw arrows (`kind="vector"`) or centred, unsigned axes (`kind="axis"`). `scale` is explicit, and magnitudes are preserved (vectors are not normalised). `add_axes(view, frame=None, length=10.0)` draws a frame's axes.

## Profiles

- `profile_plot(columns, values, *, label="", units="", ax=None, **kwargs)` draws one profile against alignment column, breaking the line at missing values and at non-consecutive columns. <!-- After the colour fix in REVIEW.md (finding 8): --> All segments of one profile share one colour; pass `color=` to set it.
- `profile_heatmap(matrix, ...)` accepts a pandas matrix (rows × columns) or an array. Numeric column gaps keep their true width and missing cells are grey. With `(alignment_group, column)` MultiIndex columns, groups are drawn side by side with separators.

## Colour scales

Every scalar plot takes `label`, `units`, `limits`, `neutral`, `cmap` and `robust`.

- For **signed** quantities, give the neutral value: `neutral=0` for strain, shear, rotation and pressure; `neutral=1` for stretch and volume ratio. This gives a diverging colour map (`RdBu_r` by default) centred on the neutral value.
- For **magnitudes** that cannot be negative (for example `green_lagrange_frobenius`, von Mises stress, non-affine residual), omit `neutral` to get a sequential map (`viridis` by default). A diverging map centred on 0 would leave half of the colour bar unused.
- Without `limits`, the colour range is the 2nd–98th percentile of the finite values (`robust=True`) or their full range (`robust=False`).
- When values fall outside the final limits, the colour bar gains arrows and its label reports the percentage clipped above and below. Constant data produce no clipping message.
- Matplotlib and py3Dmol plots use the same normalisation, including asymmetric ranges around a neutral value.
