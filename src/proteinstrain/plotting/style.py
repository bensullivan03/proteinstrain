"""Explicit colour scales shared by static and interactive plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, CenteredNorm, TwoSlopeNorm

INVALID_COLOUR = "#bdbdbd"


def mask_invalid(values, valid=None):
    values = np.ma.masked_invalid(np.ma.asarray(values, float))
    return (
        values
        if valid is None
        else np.ma.masked_where(~np.asarray(valid, bool), values)
    )


def colour_scale(values, *, limits=None, neutral=None, robust=True, cmap=None):
    """Return (colormap, norm); a stated neutral value uses the diverging midpoint."""
    if neutral is not None and not np.isfinite(neutral):
        raise ValueError("neutral must be finite.")
    palette = cmap or ("viridis" if neutral is None else "RdBu_r")
    finite = mask_invalid(values).compressed()
    if limits is not None:
        low, high = map(float, limits)
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("limits need finite low < high.")
        norm = (
            TwoSlopeNorm(neutral, low, high)
            if neutral is not None and low < neutral < high
            else Normalize(low, high)
        )
    elif neutral is not None:
        bounds = (
            np.quantile(finite, [0.02, 0.98])
            if finite.size and robust
            else (
                [finite.min(), finite.max()]
                if finite.size
                else [neutral - 1, neutral + 1]
            )
        )
        span = max(
            abs(np.asarray(bounds) - neutral).max(),
            (
                1.0
                if not finite.size or np.all(finite == neutral)
                else np.finfo(float).eps
            ),
        )
        norm = CenteredNorm(neutral, span)
    else:
        low, high = (
            np.quantile(finite, [0.02, 0.98])
            if finite.size and robust
            else ([finite.min(), finite.max()] if finite.size else [0.0, 1.0])
        )
        if high <= low:
            high = low + 1.0
        norm = Normalize(low, high)
    return plt.get_cmap(palette).copy(), norm


def _clipping(values, norm):
    finite = mask_invalid(values).compressed()
    below = np.mean(finite < norm.vmin) if len(finite) else 0.0
    above = np.mean(finite > norm.vmax) if len(finite) else 0.0
    extend = (
        "both" if below and above else "min" if below else "max" if above else "neither"
    )
    text = []
    if below:
        text.append(f"{100*below:.3g}% below")
    if above:
        text.append(f"{100*above:.3g}% above")
    return extend, (" (clipped: " + ", ".join(text) + ")") if text else ""


def colour_bar(figure, mappable, axes, *, label="", units="", values=None):
    """Attach a labelled colour bar showing actual clipping in the final limits."""
    extend, suffix = _clipping(
        mappable.get_array() if values is None else values, mappable.norm
    )
    bar = figure.colorbar(mappable, ax=axes, extend=extend)
    bar.set_label(label + (f" ({units})" if units else "") + suffix)
    return bar


def new_axes(ax=None, *, projection=None):
    if ax is not None:
        return ax.figure, ax
    fig = plt.figure()
    return fig, fig.add_subplot(111, projection=projection)
