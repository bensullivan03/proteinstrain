"""Shared MSA → directed profiles → ordinary pandas matrix and heatmap."""

import argparse
from pathlib import Path
import pandas as pd
import proteinstrain as p
from proteinstrain.mechanics.measures import green_lagrange_frobenius

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference")
parser.add_argument("target")
parser.add_argument("reference_chain")
parser.add_argument("target_chain")
parser.add_argument("output", type=Path)
a = parser.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
r = p.load_structure(a.reference)
t = p.load_structure(a.target)
alignment = p.align_multiple(
    {
        "reference": p.chain_sequence(r, a.reference_chain),
        "target": p.chain_sequence(t, a.target_chain),
    }
)
p.write_alignment(alignment, a.output / "alignment.fasta")
block = p.from_msa(
    [(r, a.reference_chain), (t, a.target_chain)],
    alignment,
    item_ids=("reference", "target"),
    atoms="CA",
)
results = p.calculate_pairs(
    block, pairs="all", rule=p.WeightRule.fixed_radius(8), ensemble="all"
)
profiles = []
for (reference, target), result in results.items():
    table = p.result_table(
        block, result, green_lagrange_frobenius(result.F), name="strain"
    )
    profile = p.per_residue(table, "mean", value="strain")
    profile["comparison"] = reference + "→" + target
    profiles.append(profile)
table = pd.concat(profiles, ignore_index=True)
table.to_csv(a.output / "profiles.csv", index=False)
matrix = table.pivot(index="comparison", columns="column", values="strain")
matrix = matrix.reindex(
    columns=range(int(matrix.columns.min()), int(matrix.columns.max()) + 1)
)
matrix.to_csv(a.output / "matrix.csv")
fig, ax = p.plotting.profile_heatmap(matrix, neutral=0, label="strain magnitude")
fig.savefig(a.output / "matrix.png")
print("Matrix:", matrix.shape)
