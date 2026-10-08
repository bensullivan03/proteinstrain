"""Sequence alignments and corresponding atom arrays."""

from .sequence import (
    chain_sequence,
    alignment_positions,
    align_pairwise,
    align_multiple,
    read_alignment,
    write_alignment,
)
from .pairs import resolve_chain_pairs
from .coordinates import (
    AlignedCoordinates,
    extract,
    from_pairwise,
    from_identity,
    from_reference_star,
    from_msa,
    from_native_keys,
    concat_blocks,
)

__all__ = [
    "chain_sequence",
    "alignment_positions",
    "align_pairwise",
    "align_multiple",
    "read_alignment",
    "write_alignment",
    "resolve_chain_pairs",
    "AlignedCoordinates",
    "extract",
    "from_pairwise",
    "from_identity",
    "from_reference_star",
    "from_msa",
    "from_native_keys",
    "concat_blocks",
]
