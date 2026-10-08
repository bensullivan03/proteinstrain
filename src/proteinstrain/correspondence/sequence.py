"""Pairwise position maps and labelled FASTA alignments."""

from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices


def chain_sequence(structure, chain):
    """Full deposited polymer sequence; unmodelled positions remain in the string."""
    sequence = "".join(code for code, _ in structure.polymer_sequences[str(chain)])
    if not sequence or set(sequence) == {"X"}:
        raise ValueError("The chain has no known polymer sequence.")
    return sequence


def align_pairwise(
    reference,
    target,
    *,
    matrix="BLOSUM62",
    open_gap=-11.0,
    extend_gap=-1.0,
    end_gap=0.0,
):
    """Return an (n, 2) array of matched zero-based positions in a global alignment."""
    if not reference or not target:
        raise ValueError("Both sequences must be nonempty.")
    aligner = PairwiseAligner(mode="global")
    aligner.substitution_matrix = substitution_matrices.load(matrix)
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap
    aligner.end_gap_score = end_gap
    alphabet = set(aligner.substitution_matrix.alphabet)
    sequences = [str(reference).upper(), str(target).upper()]
    if "X" not in alphabet and any(set(seq) - alphabet for seq in sequences):
        raise ValueError("Scoring matrix has no X for unsupported sequence letters.")
    scored = ["".join(c if c in alphabet else "X" for c in seq) for seq in sequences]
    blocks = aligner.align(*scored)[0].aligned
    pairs = [
        np.column_stack((np.arange(a, b), np.arange(c, d)))
        for (a, b), (c, d) in zip(*blocks)
    ]
    return (
        np.concatenate(pairs).astype(np.int32) if pairs else np.empty((0, 2), np.int32)
    )


def _fasta(rows):
    return "".join(f">{label}\n{sequence}\n" for label, sequence in rows.items())


def _parse_fasta(text):
    rows = {}
    label = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            label = line[1:].strip()
            if not label or label in rows:
                raise ValueError("FASTA labels must be nonempty and distinct.")
            rows[label] = ""
        elif label is None:
            raise ValueError("FASTA sequence appeared before a header.")
        else:
            rows[label] += line.upper()
    if (
        not rows
        or len({len(s) for s in rows.values()}) != 1
        or any(not s for s in rows.values())
    ):
        raise ValueError("An alignment needs nonempty rows of equal length.")
    return rows


def read_alignment(path):
    """Read labelled aligned strings from a standard FASTA file."""
    return _parse_fasta(Path(path).read_text())


def write_alignment(alignment, path):
    """Write labelled aligned strings to FASTA; return the destination Path."""
    text = _fasta(alignment)
    _parse_fasta(text)
    path = Path(path)
    path.write_text(text)
    return path


def align_multiple(
    sequences,
    *,
    executable="mafft",
    launcher=(),
    flags=("--auto",),
    threads=1,
    timeout=300.0,
    deduplicate=True,
):
    """Return labelled aligned strings; run MAFFT for distinct sequences without fallback."""
    sequences = dict(sequences)
    if len(sequences) < 2 or any(not s or "-" in s for s in sequences.values()):
        raise ValueError("Supply at least two nonempty ungapped sequences.")
    unique = {}
    for label, seq in sequences.items():
        if not label or any(c.isspace() for c in label):
            raise ValueError("MAFFT labels must be nonempty without whitespace.")
        unique.setdefault(seq, label)
    query = {label: seq for seq, label in unique.items()} if deduplicate else sequences
    if len(query) == 1:
        return sequences.copy()
    with TemporaryDirectory(prefix="psp-mafft-") as directory:
        path = Path(directory) / "sequences.fasta"
        path.write_text(_fasta(query))
        # WSL receives input on stdin to avoid host/guest path translation.
        command = [*launcher, str(executable), *flags, "--thread", str(threads), "-"]
        print(command)
        try:
            result = subprocess.run(
                command,
                input=path.read_text(),
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"MAFFT failed: {error}") from error
        if result.returncode:
            raise RuntimeError(
                f"MAFFT exited {result.returncode}: {result.stderr.strip()}"
            )
        rows = _parse_fasta(result.stdout)
    if set(rows) != set(query) or any(
        rows[k].replace("-", "") != v.upper() for k, v in query.items()
    ):
        raise RuntimeError("MAFFT output does not match the input members/sequences.")
    return {
        label: rows[unique[seq]] if deduplicate else rows[label]
        for label, seq in sequences.items()
    }


def alignment_positions(alignment):
    """Return item_id, zero-based polymer_position and original column for non-gap sites."""
    import pandas as pd

    if not alignment or len({len(row) for row in alignment.values()}) != 1:
        raise ValueError("Alignment rows must have the same width.")
    rows = []
    for label, row in alignment.items():
        position = 0
        for column, code in enumerate(row):
            if code != "-":
                rows.append((label, position, column))
                position += 1
    return pd.DataFrame(rows, columns=["item_id", "polymer_position", "column"])
