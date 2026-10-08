"""Chain selection by IDs, groups or descriptions."""

import re


def resolve_chains(structure, chains=None):
    """Resolve IDs in the supplied order; None selects every chain."""
    ids = (
        structure.chain_ids
        if chains is None
        else ((chains,) if isinstance(chains, str) else tuple(chains))
    )
    if not ids:
        raise ValueError("Select at least one chain.")
    missing = set(ids) - set(structure.chain_ids)
    if missing:
        raise LookupError(
            f"{structure.structure_id} has no chains {sorted(missing)}; available: {list(structure.chain_ids)}."
        )
    return tuple(dict.fromkeys(ids))


def chains_by_description(structure, pattern, *, regex=False):
    """Return chain IDs matching an exact description or regular expression."""
    match = re.compile(pattern).search if regex else lambda text: text == pattern
    return tuple(
        c for c in structure.chain_ids if match(structure.chain_descriptions.get(c, ""))
    )
