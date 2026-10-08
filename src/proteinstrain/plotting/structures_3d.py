"""Static scalar/vector plots and small py3Dmol structure viewers."""

import numpy as np
import py3Dmol
import gemmi
from matplotlib.colors import to_hex
from .style import new_axes, colour_scale, mask_invalid, colour_bar, _clipping
from ._traces import segments, chain_trace


def structure_scatter(
    xyz,
    values,
    *,
    limits=None,
    neutral=None,
    cmap=None,
    robust=True,
    label="",
    units="",
    size=8,
    ax=None,
):
    """Plot Cartesian points with an explicit scalar colour scale."""
    xyz = np.asarray(xyz, float)
    values = mask_invalid(values)
    fig, ax = new_axes(ax, projection="3d")
    palette, norm = colour_scale(
        values, limits=limits, neutral=neutral, cmap=cmap, robust=robust
    )
    palette = palette.with_extremes(bad="#bdbdbd")
    image = ax.scatter(
        *xyz.T, c=values, s=size, cmap=palette, norm=norm, plotnonfinite=True
    )
    colour_bar(fig, image, ax, label=label, units=units, values=values)
    return fig, ax


def segments_3d(xyz, directions, *, scale, kind="vector", colour="#333333", ax=None):
    """Draw scaled arrows or centred unsigned axes without normalising magnitudes."""
    start, end = segments(xyz, directions, scale, kind)
    fig, ax = new_axes(ax, projection="3d")
    good = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
    if kind == "vector":
        ax.quiver(*start[good].T, *(end - start)[good].T, color=colour, normalize=False)
    else:
        for a, b in zip(start[good], end[good]):
            ax.plot(*np.array([a, b]).T, color=colour)
    return fig, ax


def _point(xyz):
    return dict(zip(("x", "y", "z"), map(float, xyz)))


def add_segments(
    view,
    xyz,
    directions,
    *,
    scale,
    kind="vector",
    colour="#333333",
    radius=0.15,
    viewer=None,
):
    cell = {} if viewer is None else {"viewer": viewer}
    start, end = segments(xyz, directions, scale, kind)
    for a, b in zip(start, end):
        if not np.isfinite([a, b]).all() or np.linalg.norm(b - a) < 1e-12:
            continue
        spec = {"start": _point(a), "end": _point(b), "color": colour, "radius": radius}
        if kind == "vector":
            view.addArrow({**spec, "radiusRatio": 2.0, "mid": 0.8}, **cell)
        else:
            view.addCylinder(spec, **cell)
    return view


def add_axes(view, frame=None, length=10.0, *, viewer=None):
    from ..geometry.frames import Frame

    cell = {} if viewer is None else {"viewer": viewer}
    frame = Frame.cartesian() if frame is None else frame
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be positive and finite.")
    for label, direction, colour in zip(
        "xyz", frame.axes, ("#d62728", "#2ca02c", "#1f77b4")
    ):
        add_segments(
            view,
            [frame.origin],
            [direction],
            scale=length,
            colour=colour,
            radius=length * 0.012,
            viewer=viewer,
        )
        view.addLabel(
            label,
            {
                "position": _point(frame.origin + length * direction),
                "fontColor": colour,
                "backgroundOpacity": 0.0,
            },
            **cell,
        )
    return view


def _cif(structure):
    holder = gemmi.Structure()
    holder.name = structure.structure_id
    holder.add_model(structure.model.clone())
    holder.setup_entities()
    return holder.make_mmcif_document().as_string()


def _style(kind, **options):
    if kind == "trace":
        return {"cartoon": {"style": "trace", **options}}
    if kind not in ("cartoon", "stick", "sphere", "line"):
        raise ValueError("Unknown rendering style.")
    return {kind: options}


def _get_view(view, viewer, width, height):
    if viewer is not None and (view is None or not view.viewergrid):
        raise ValueError("A grid cell requires an existing grid view.")
    view = py3Dmol.view(width=width, height=height) if view is None else view
    return view, ({} if viewer is None else {"viewer": viewer})


def view_structures(
    structures,
    *,
    chains=None,
    colours=None,
    style="cartoon",
    labels=True,
    label_size=14,
    opacity=1.0,
    mark_start=False,
    axes=False,
    width=900,
    height=600,
    view=None,
    viewer=None,
):
    """Overlay prepared structures in one viewer; no preparation previews or linked panels."""
    from ..selection import resolve_chains
    from ..geometry.frames import Frame

    if hasattr(structures, "model"):
        structures = (structures,)
    structures = tuple(structures)
    if not structures:
        raise ValueError("Supply at least one structure.")
    view, cell = _get_view(view, viewer, width, height)
    all_points = []
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for i, structure in enumerate(structures):
        selected = structure if chains is None else structure.retain_chains(chains)
        view.addModel(_cif(selected), "cif", {"doAssembly": False}, **cell)
        view.setStyle(
            {"model": -1},
            _style(style, color=palette[i % len(palette)], opacity=opacity),
            **cell,
        )
        for chain in resolve_chains(selected):
            color = (colours or {}).get(chain, palette[i % len(palette)])
            view.setStyle(
                {"model": -1, "chain": chain},
                _style(style, color=color, opacity=opacity),
                **cell,
            )
            points, _ = chain_trace(selected, chain)
            if not len(points):
                continue
            all_points.append(points)
            if mark_start:
                view.addSphere(
                    {"center": _point(points[0]), "radius": 1.3, "color": color}, **cell
                )
            if labels:
                view.addLabel(
                    chain,
                    {
                        "position": _point(points[len(points) // 2]),
                        "fontColor": color,
                        "fontSize": label_size,
                        "backgroundOpacity": 0.0,
                    },
                    **cell,
                )
    view.zoomTo(**cell)
    if axes and all_points:
        points = np.concatenate(all_points)
        add_axes(
            view,
            Frame(points.min(axis=0), np.eye(3)),
            max(np.ptp(points, axis=0).max() * 0.15, 1.0),
            viewer=viewer,
        )
    return view


def view_field(
    structure,
    atoms,
    values,
    *,
    limits=None,
    neutral=None,
    cmap=None,
    robust=True,
    label="",
    units="",
    missing_colour="#bdbdbd",
    style="cartoon",
    width=900,
    height=600,
    view=None,
    viewer=None,
):
    """Colour native atoms; exact psp_value is separate from normalised psp_colour."""
    keys = ["chain_id", "hetero", "residue_number", "insertion", "atom_name"]
    values = mask_invalid(values)
    if values.shape != (len(atoms),) or atoms.duplicated(keys).any():
        raise ValueError("Supply one value per unique native atom.")
    lookup = {
        tuple(row): value
        for row, value in zip(atoms[keys].itertuples(index=False, name=None), values)
    }
    palette, norm = colour_scale(
        values, limits=limits, neutral=neutral, cmap=cmap, robust=robust
    )
    cif = _cif(structure)
    category = gemmi.cif.read_string(cif).sole_block().get_mmcif_category("_atom_site.")
    properties = []
    found = set()
    for i in range(len(category["id"])):
        hetero = (
            " "
            if category["group_PDB"][i] == "ATOM"
            else "H_" + category["label_comp_id"][i]
        )
        insertion = (
            category.get("pdbx_PDB_ins_code", [None] * len(category["id"]))[i] or " "
        )
        key = (
            category["auth_asym_id"][i],
            hetero,
            int(category["auth_seq_id"][i]),
            insertion,
            category["label_atom_id"][i],
        )
        if key in lookup:
            if key in found:
                raise ValueError("Alternate atoms make a native display key ambiguous.")
            found.add(key)
            value = lookup[key]
            if not np.ma.is_masked(value) and np.isfinite(value):
                properties.append(
                    {
                        "index": i,
                        "props": {
                            "psp_value": float(value),
                            "psp_colour": float(np.clip(norm(value), 0, 1)),
                        },
                    }
                )
    if set(lookup) != found:
        raise LookupError("Some supplied native atoms are absent from the model.")
    view, cell = _get_view(view, viewer, width, height)
    view.addModel(cif, "cif", {"doAssembly": False}, **cell)
    view.setStyle({"model": -1}, _style(style, color=missing_colour), **cell)
    if properties:
        view.mapAtomProperties(properties, {"model": -1}, **cell)
        # 3Dmol CustomLinear has 256 bins; match Matplotlib's exact lookup positions.
        colors = [to_hex(palette(x)) for x in np.arange(256) / 256.0]
        view.setStyle(
            {"model": -1, "index": [p["index"] for p in properties]},
            _style(
                style,
                colorscheme={
                    "prop": "psp_colour",
                    "gradient": "linear",
                    "min": 0.0,
                    "max": 1.0,
                    "colors": colors,
                },
            ),
            **cell,
        )
    _, clipping = _clipping(values, norm)
    view.addLabel(
        label + (f" ({units})" if units else "") + clipping,
        {
            "position": {"x": 10, "y": 10, "z": 0},
            "useScreen": True,
            "fontColor": "#222222",
            "backgroundColor": "white",
        },
        **cell,
    )
    for i, fraction in enumerate(np.linspace(0, 1, 17)):
        view.addLabel(
            "█",
            {
                "position": {"x": 10 + 10 * i, "y": 40, "z": 0},
                "useScreen": True,
                "fontColor": to_hex(palette(fraction)),
                "backgroundOpacity": 0.0,
            },
            **cell,
        )
    for fraction in (0.0, 0.5, 1.0):
        view.addLabel(
            f"{norm.inverse(fraction):.4g}",
            {
                "position": {"x": 10 + 160 * fraction, "y": 60, "z": 0},
                "useScreen": True,
                "fontColor": "#222222",
                "backgroundOpacity": 0.0,
            },
            **cell,
        )
    view.addLabel(
        "■ missing",
        {
            "position": {"x": 10, "y": 80, "z": 0},
            "useScreen": True,
            "fontColor": missing_colour,
            "backgroundOpacity": 0.0,
        },
        **cell,
    )
    view.zoomTo(**cell)
    return view
