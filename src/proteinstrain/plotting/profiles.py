"""Profile lines and heatmaps from arrays or pandas matrices."""

import numpy as np
import pandas as pd
from .style import new_axes, colour_scale, mask_invalid, colour_bar


def profile_plot(columns, values, *, label="", units="", ax=None, **kwargs):
    """Draw one profile, breaking at missing values and nonconsecutive columns."""
    columns = np.asarray(columns)
    values = mask_invalid(values)
    if columns.ndim != 1 or values.shape != columns.shape:
        raise ValueError("columns and values must be equal-length vectors.")
    fig, ax = new_axes(ax)
    for indices in np.split(
        np.arange(len(columns)), np.flatnonzero(np.diff(columns) != 1) + 1
    ):
        ax.plot(columns[indices], values[indices], **kwargs)
    ax.set(
        xlabel="alignment column (zero-based)",
        ylabel=label + (f" ({units})" if units else ""),
    )
    return fig, ax


def profile_heatmap(
    values,
    *,
    columns=None,
    rows=None,
    limits=None,
    neutral=None,
    cmap=None,
    robust=True,
    label="",
    units="",
    ax=None,
):
    """Plot a matrix; numeric column gaps occupy their actual widths."""
    if hasattr(values, "columns"):
        columns = values.columns if columns is None else columns
        rows = list(values.index) if rows is None else rows
        values = values.to_numpy()
    values = mask_invalid(values)
    if values.ndim != 2:
        raise ValueError("A heatmap requires a two-dimensional matrix.")
    columns = np.arange(values.shape[1]) if columns is None else columns
    if not all(values.shape) or len(columns) != values.shape[1]:
        raise ValueError("A heatmap needs nonempty rows and matching columns.")
    grouped = isinstance(columns, pd.MultiIndex)
    if grouped and columns.nlevels != 2:
        raise ValueError("Grouped columns need (alignment_group, column) levels.")
    groups = list(columns.get_level_values(0).unique()) if grouped else [None]
    positions = np.empty(values.shape[1], dtype=int)
    group_ticks, column_ticks, separators = [], [], []
    offset = 0
    first = 0
    for group in groups:
        selected = (
            np.flatnonzero(columns.get_level_values(0) == group)
            if grouped
            else np.arange(len(columns))
        )
        cols = np.asarray(columns.get_level_values(1)[selected] if grouped else columns)
        if (
            cols.ndim != 1
            or not np.issubdtype(cols.dtype, np.integer)
            or np.any(np.diff(cols) <= 0)
        ):
            raise ValueError(
                "Columns within each group must be increasing integer positions."
            )
        if not grouped:
            first = int(cols[0])
        width = int(cols[-1] - cols[0] + 1)
        positions[selected] = offset + cols - cols[0]
        if grouped:
            group_ticks.append((offset + (width - 1) / 2, str(group)))
            ticks = np.unique(np.linspace(cols[0], cols[-1], min(5, width), dtype=int))
            column_ticks.extend((offset + int(c - cols[0]), str(c)) for c in ticks)
            if offset:
                separators.append(offset - 1)
        offset += width + int(grouped)
    width = offset - int(grouped)
    matrix = np.ma.masked_all((values.shape[0], width))
    matrix[:, positions] = values
    palette, norm = colour_scale(
        values, limits=limits, neutral=neutral, cmap=cmap, robust=robust
    )
    palette = palette.with_extremes(bad="#bdbdbd")
    fig, ax = new_axes(ax)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=palette,
        norm=norm,
        extent=[first - 0.5, first + width - 0.5, values.shape[0] - 0.5, -0.5],
    )
    if grouped:
        ax.set_xticks(
            [x for x, label in column_ticks],
            labels=[label for x, label in column_ticks],
        )
        group_axis = ax.secondary_xaxis("top")
        group_axis.set_xticks(
            [x for x, label in group_ticks], labels=[label for x, label in group_ticks]
        )
        for x in separators:
            ax.axvline(x, color="#333333", linewidth=0.7)
    if rows is not None:
        ax.set_yticks(np.arange(len(rows)), labels=rows)
    ax.set_xlabel("alignment column (zero-based)")
    colour_bar(fig, image, ax, label=label, units=units, values=values)
    return fig, ax
