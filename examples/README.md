# Example workflows

Each script is a complete workflow. Run them from any directory after installing ProteinStrain:

```sh
python examples/strain_and_energy.py reference.cif target.cif A B output_directory
python examples/aligned_profiles.py reference.cif target.cif A B output_directory
python examples/geometry.py output_directory
```

`A` and `B` are the reference and target chain IDs. No file names, chains or rules for a particular protein are built into the scripts.

To try them on the test fixtures from a repository checkout:

```sh
python examples/strain_and_energy.py tests/data/8IXA_subset.cif tests/data/8IXB_subset.cif I H out1
python examples/aligned_profiles.py tests/data/8IXA_subset.cif tests/data/8IXB_subset.cif I H out2
python examples/geometry.py out3
```

| Script | What it does | Outputs |
|---|---|---|
| `strain_and_energy.py` | Pairwise correspondence of one chain pair (Cα), tapered 4–8 Å neighbourhoods on the reference, Green–Lagrange strain magnitude, non-affine residual, and Saint Venant–Kirchhoff energy with **illustrative** assumptions (Y = 100 MPa, ν = 0.33, 25 Å³ per centre). | `atoms.csv`, `residues.csv`, `deformation.npz`, `strain.png`, `display.cif` (strain in the B-factor column, −1 where missing), `view.html` |
| `aligned_profiles.py` | Multiple sequence alignment of the two chains, both directed comparisons over `ensemble="all"`, per-residue mean strain, and a comparison × column matrix. MAFFT is needed only if the two sequences differ. | `alignment.fasta`, `profiles.csv`, `matrix.csv`, `matrix.png` |
| `geometry.py` | Cylinder fit and anchored frame on an analytic helix, unrolled cylindrical trace and a 3D vector plot. | `unrolled.png`, `vectors.png` |

The volumes and moduli in `strain_and_energy.py` illustrate the interface only; replace them with values justified for your system (for example from `constitutive.volumes_from_structure`).
