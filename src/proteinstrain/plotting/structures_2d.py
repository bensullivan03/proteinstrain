"""Static structure traces, scalar projections and cylindrical views."""

import numpy as np
from itertools import cycle
from matplotlib import rcParams
from ..selection import resolve_chains
from ._traces import chain_trace
from .style import new_axes, colour_scale, mask_invalid, colour_bar


def plot_structures(
    structures,
    *,
    chains=None,
    frame=None,
    colours=None,
    labels=True,
    plane=None,
    alpha=1.0,
    linewidth=1.0,
    label_size=8,
    mark_start=False,
    ax=None,
):
    """Overlay gap-aware CA traces in 3D or a Cartesian plane such as (0, 1)."""
    if plane is not None and (
        len(plane) != 2
        or len(set(plane)) != 2
        or any(i not in (0, 1, 2) for i in plane)
    ):
        raise ValueError("plane must select two distinct Cartesian dimensions.")
    if hasattr(structures, "model"):
        structures = (structures,)
    fig, ax = new_axes(ax, projection="3d" if plane is None else None)
    palette = cycle(rcParams["axes.prop_cycle"].by_key()["color"])
    dimensions = (0, 1, 2) if plane is None else tuple(plane)
    for structure in structures:
        for chain in resolve_chains(structure, chains):
            points, parts = chain_trace(structure, chain, frame)
            if not len(points):
                continue
            colour = (colours or {}).get(chain)
            if colour is None:
                colour = next(palette)
            for part in parts:
                ax.plot(
                    *part[:, dimensions].T,
                    color=colour,
                    alpha=alpha,
                    linewidth=linewidth,
                )
            if mark_start:
                ax.plot(*points[:1, dimensions].T, "o", color=colour, markersize=3)
            if labels:
                ax.text(
                    *points[len(points) // 2, dimensions],
                    chain,
                    color=colour,
                    fontsize=label_size,
                )
    ax.set(xlabel="xyz"[dimensions[0]] + " (Å)", ylabel="xyz"[dimensions[1]] + " (Å)")
    if plane is None:
        ax.set(zlabel="z (Å)")
    else:
        ax.set_aspect("equal", adjustable="box")
    return fig, ax


def projection(
    xyz,
    values=None,
    *,
    plane=(0, 1),
    limits=None,
    neutral=None,
    cmap=None,
    robust=True,
    label="",
    units="",
    size=8,
    ax=None,
):
    """Plot Cartesian positions with optional scalar colours."""
    xyz = np.asarray(xyz, float)
    fig, ax = new_axes(ax)
    if values is None:
        ax.scatter(xyz[:, plane[0]], xyz[:, plane[1]], s=size)
    else:
        values = mask_invalid(values)
        palette, norm = colour_scale(
            values, limits=limits, neutral=neutral, cmap=cmap, robust=robust
        )
        palette = palette.with_extremes(bad="#bdbdbd")
        artist = ax.scatter(
            xyz[:, plane[0]],
            xyz[:, plane[1]],
            c=values,
            s=size,
            cmap=palette,
            norm=norm,
            plotnonfinite=True,
        )
        colour_bar(fig, artist, ax, label=label, units=units, values=values)
    ax.set(xlabel="xyz"[plane[0]] + " (Å)", ylabel="xyz"[plane[1]] + " (Å)")
    return fig, ax


def unrolled_cylinder(
    xyz,
    values=None,
    *,
    frame,
    positions=None,
    limits=None,
    neutral=None,
    cmap=None,
    robust=True,
    label="",
    units="",
    ax=None,
):
    """Plot angle versus axial position; disconnect seams and optional polymer gaps."""
    if not frame.angular_reference:
        raise ValueError("An unrolled view needs an explicit angular reference.")
    points = frame.to_local(xyz)
    angle = np.arctan2(points[:, 1], points[:, 0])
    height = points[:, 2]
    fig, ax = new_axes(ax)
    if values is None:
        broken = np.abs(np.diff(angle)) > np.pi
        if positions is not None:
            broken |= np.diff(positions) != 1
        for part in np.split(np.arange(len(points)), np.flatnonzero(broken) + 1):
            ax.plot(angle[part], height[part])
    else:
        values = mask_invalid(values)
        palette, norm = colour_scale(
            values, limits=limits, neutral=neutral, cmap=cmap, robust=robust
        )
        palette = palette.with_extremes(bad="#bdbdbd")
        artist = ax.scatter(
            angle, height, c=values, cmap=palette, norm=norm, plotnonfinite=True
        )
        colour_bar(fig, artist, ax, label=label, units=units, values=values)
    ax.set(xlabel="angle (radians)", ylabel="axial position (Å)")
    return fig, ax
