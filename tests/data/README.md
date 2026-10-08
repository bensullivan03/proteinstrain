# Test structure fixtures

These files are subsets of deposited PDB entries. Each keeps the first-model atom records for selected chains, native author residue numbering, the full deposited sequences and the entity descriptions.

| File | PDB entry | Entry title | Method | Chains kept | Used to test |
|---|---|---|---|---|---|
| `8IXA_subset.cif` | [8IXA](https://www.rcsb.org/structure/8IXA) | GMPCPP-Alpha1A/Beta2A-microtubule decorated with kinesin, non-seam region | cryo-EM, 4.2 Å | I, Q (tubulin α-1A, β-2A) | multi-chain correspondence, missing residues |
| `8IXB_subset.cif` | [8IXB](https://www.rcsb.org/structure/8IXB) | GMPCPP-Alpha1A/Beta2A-microtubule decorated with kinesin, seam region | cryo-EM, 4.2 Å | H, W (tubulin α-1A, β-2A) | different chain labels for the same proteins |
| `7NJP_subset.cif` | [7NJP](https://www.rcsb.org/structure/7NJP) | *Mycobacterium smegmatis* ATP synthase state 2 | cryo-EM, 2.84 Å | A, B (α), D (β), G (γ) | a different protein family; ligands, ions, water and alternate locations |
