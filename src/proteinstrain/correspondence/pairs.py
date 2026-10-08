"""Explicit reference-to-target chain pairing."""

from ..selection import resolve_chains


def resolve_chain_pairs(
    reference,
    target,
    *,
    reference_chains=None,
    target_chains=None,
    chain_map=None,
    matching=None,
):
    """Return (reference_chain, target_chain) pairs; maps always point reference→target."""
    left = resolve_chains(reference, reference_chains)
    if matching not in (None, "label", "position"):
        raise ValueError("matching must be label or position.")
    if chain_map is not None:
        mapping = dict(chain_map)
        if (
            set(mapping) != set(left)
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) - set(target.chain_ids)
        ):
            raise ValueError(
                "chain_map must be one-to-one and keyed reference→target for all selected reference chains."
            )
        if target_chains is not None and set(
            resolve_chains(target, target_chains)
        ) != set(mapping.values()):
            raise ValueError(
                "Target selection conflicts with reference→target chain_map."
            )
        return tuple((c, mapping[c]) for c in left)
    if (
        target_chains is None
        and reference_chains is not None
        and set(left) <= set(target.chain_ids)
        and matching != "position"
    ):
        right = left
    else:
        right = resolve_chains(target, target_chains)
    if len(left) != len(right):
        raise ValueError(
            "Chain sets differ; give an explicit reference→target chain_map."
        )
    if matching == "position":
        return tuple(zip(left, right))
    if set(left) != set(right):
        raise ValueError(
            "Chain labels differ; give an explicit reference→target chain_map."
        )
    return tuple((c, c) for c in left)
