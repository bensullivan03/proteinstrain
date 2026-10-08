"""Assign plain chain groups using descriptions or sequence similarity."""

import re
from ..correspondence.sequence import chain_sequence, align_pairwise
from ..selection import resolve_chains


def _claim(groups, chain, matches, claim):
    if claim not in ("first", "all", "error"):
        raise ValueError("claim must be first, all or error.")
    if claim == "error" and len(matches) > 1:
        raise ValueError(f"{chain} matches multiple groups: {matches}.")
    for group in matches if claim == "all" else matches[:1]:
        groups[group].append(chain)


def assign_groups(structure, rules, *, claim, match="regex", keep_existing=False):
    """Return a grouped structure; unmatched leaves are empty tuples."""
    if match not in ("regex", "exact"):
        raise ValueError("match must be regex or exact.")
    patterns = {
        g: re.compile(p).search if match == "regex" else lambda text, p=p: text == p
        for g, p in rules.items()
    }
    groups = {g: [] for g in rules}
    for chain in structure.chain_ids:
        description = structure.chain_descriptions.get(chain, "")
        _claim(groups, chain, [g for g, f in patterns.items() if f(description)], claim)
    return structure.with_groups(
        {**(structure.groups if keep_existing else {}), **groups}
    )


def assign_groups_by_sequence(
    structure,
    references,
    *,
    claim,
    minimum_identity,
    minimum_coverage,
    margin,
    chains=None,
):
    """Assign groups against reference sequence strings or (structure, chain) pairs."""
    if claim not in ("first", "all", "error"):
        raise ValueError("claim must be first, all or error.")
    if not all(0 <= x <= 1 for x in (minimum_identity, minimum_coverage, margin)):
        raise ValueError("Sequence thresholds must lie between zero and one.")
    references = {
        g: chain_sequence(*s) if isinstance(s, tuple) else str(s)
        for g, s in references.items()
    }
    groups = {g: [] for g in references}
    for chain in resolve_chains(structure, chains):
        seq = chain_sequence(structure, chain)
        scores = []
        for group, reference in references.items():
            pairs = align_pairwise(reference, seq)
            identity = (
                sum(reference[i] == seq[j] for i, j in pairs) / len(pairs)
                if len(pairs)
                else 0.0
            )
            coverage = len(pairs) / len(reference)
            if identity >= minimum_identity and coverage >= minimum_coverage:
                scores.append((identity, group))
        scores.sort(key=lambda item: item[0], reverse=True)
        matches = [g for score, g in scores]
        if claim != "all" and scores:
            ambiguous = [g for score, g in scores if scores[0][0] - score <= margin]
            if len(ambiguous) > 1:
                raise ValueError(
                    f"{chain} has ambiguous sequence matches: {ambiguous}."
                )
            matches = matches[:1]
        _claim(groups, chain, matches, claim)
    return structure.with_groups(groups)
