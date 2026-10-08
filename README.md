# ProteinStrain

ProteinStrain is a python package for finite strain analysis of protein structures. It compares corresponding atoms of (2 or more) protein structures. This package extends **PSA (Protein Strain Analysis)** by the Sartori Lab — <https://github.com/Sartori-Lab/PSA> — which introduced this formalism for proteins in:

> P. Sartori and S. Leibler, *Evolutionary Conservation of Mechanical Strain Distributions in Functional Transitions of Protein Structures*, Phys. Rev. X **14**, 011042 (2024). <https://doi.org/10.1103/PhysRevX.14.011042>

The package provides structure loading/preparation, sequence alignment, deformation gradient calculation, strain/stress/energy functions, geometry calculation and essential plots. The results can be written into `cif` structures using the B-factor column of an mmCIF copy.

## Installation

ProteinStrain needs Python 3.12 or later. [MAFFT](https://mafft.cbrc.jp/alignment/software/)  is optional and only needed for `align_multiple` in order to perform multiple sequence alignment.

### Option 1: Install a release from GitHub

```sh
python -m pip install "proteinstrain @ git+https://github.com/bensullivan03/proteinstrain.git"
```

### Option 2: Install a wheel

Install it directly from its URL (use a different version if necessary).

```sh
python -m pip install https://github.com/bensullivan03/proteinstrain/releases/download/v0.6.0/proteinstrain-0.6.0-py3-none-any.whl
```

Or download the file, then install it. Do not change the file name.

```sh
python -m pip install proteinstrain-0.6.0-py3-none-any.whl
```

## Requirements

- `numpy`
- `scipy`
- `gemmi`
- `biopython`
- `numba`
- `pandas`
- `matplotlib`
- `py3Dmol`


## Example Usage

A directed strain calculation from a reference structure to a target structure:

```python
import proteinstrain as psp
from proteinstrain.mechanics.measures import green_lagrange_frobenius

reference = psp.load_structure('reference.cif')
target = psp.load_structure('target.cif')
block = psp.from_pairwise(reference, target,
                          reference_chains='A', chain_map={'A': 'B'}, atoms='CA')
result = psp.calculate(block, *block.item_ids,
                       rule=psp.WeightRule.tapered(4., 8.), ensemble='reference')
strain = green_lagrange_frobenius(result.F)
table = psp.result_table(block, result, strain, name='strain')
residues = psp.per_residue(table, 'mean', value='strain')
residues.to_csv('strain.csv', index=False)
```

Change the chain IDs and neighbourhood rule explicitly for your inputs. The package contains no protein-specific presets. `F[i]` belongs to `result.centre_indices[i]` in the aligned block. Failed fits have `valid=False` and `NaN` values.

## Sequence alignment

Pairwise alignment (`align_pairwise`, used by `from_pairwise` and `from_reference_star`) is built in and uses Biopython's global aligner with BLOSUM62 scoring.

Multiple sequence alignment (`align_multiple`) calls an external MAFFT executable. On Windows, MAFFT is most easily installed in WSL and called through it:

```python
alignment = psp.align_multiple(sequences, launcher=('wsl',))
```

MAFFT is not called if all sequences are identical. There is no fallback when MAFFT fails; an error is raised instead.

## Parallelism

The neighbour-moment, residual and volume kernels are compiled with Numba and run in parallel across centres. Set the number of threads with the `NUMBA_NUM_THREADS` environment variable (before importing the package) or with `numba.set_num_threads(n)`. MAFFT's thread count is set with `align_multiple(..., threads=n)`.

## Detailed Documentation

The following documentation was drafted the help of AI. If there are any issues then please let me know. I intend to improve and expand it in the future.

- [Structures, preparation and correspondence](docs/structures_and_correspondence.md)
- [Deformation and direct measures](docs/calculations.md)
- [Tables, comparisons and standard files](docs/tables_and_files.md)
- [Materials, volumes and geometry](docs/energy_and_geometry.md)
- [Essential plots](docs/plotting.md)
- [Executable examples](examples/README.md)


## Citing

Cite the original paper if you use this package as part of an academic project:

```bibtex
@article{sartori2024strain,
  author  = {Sartori, Pablo and Leibler, Stanislas},
  title   = {Evolutionary Conservation of Mechanical Strain Distributions in Functional Transitions of Protein Structures},
  journal = {Physical Review X},
  volume  = {14},
  pages   = {011042},
  year    = {2024},
  doi     = {10.1103/PhysRevX.14.011042}
}
```

## Tests

The tests and their structure fixtures are also [present in this repository](tests/data/README.md); they are not included in the installed package. They can be downloaded and run separately using `pytest` if desired.