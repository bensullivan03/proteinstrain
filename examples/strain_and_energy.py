"""Structures → corresponding arrays → strain/energy → tables, figures and ordinary files."""

import argparse
from pathlib import Path
import numpy as np
import proteinstrain as p
from proteinstrain.mechanics import measures as m

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference")
parser.add_argument("target")
parser.add_argument("reference_chain")
parser.add_argument("target_chain")
parser.add_argument("output", type=Path)
a = parser.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
reference = p.load_structure(a.reference)
target = p.load_structure(a.target)
block = p.from_pairwise(
    reference,
    target,
    reference_chains=a.reference_chain,
    chain_map={a.reference_chain: a.target_chain},
    atoms="CA",
    item_ids=("reference", "target"),
)
result = p.calculate(
    block,
    "reference",
    "target",
    rule=p.WeightRule.tapered(4, 8),
    ensemble="reference",
    nonaffine=True,
)
strain = m.green_lagrange_frobenius(result.F)
table = p.result_table(block, result, strain, name="strain")
model = p.constitutive.IsotropicSVK.from_young_poisson(
    100e6, 0.33
)  # Explicit illustrative assumptions.
volumes = p.constitutive.allocate_volume(25 * len(strain), len(strain))
table["energy_kT"] = p.constitutive.total_energy(
    model, result.F, volumes, units="kT", temperature=300
)
table.to_csv(a.output / "atoms.csv", index=False)
p.per_residue(table, "mean", value="strain").to_csv(
    a.output / "residues.csv", index=False
)
np.savez(
    a.output / "deformation.npz",
    F=result.F,
    centre_indices=result.centre_indices,
    valid=result.valid,
    nonaffine=result.nonaffine,
)
fig, ax = p.plotting.profile_plot(table.column, table.strain, label="strain magnitude")
fig.savefig(a.output / "strain.png")
p.export_field(
    reference, table, table.strain.to_numpy(), a.output / "display.cif", missing=-1.0
)
p.plotting.view_field(
    reference, table, table.strain.to_numpy(), neutral=0, label="strain magnitude"
).write_html(str(a.output / "view.html"))
print("Valid fits:", int(result.valid.sum()), "of", len(result.valid))
