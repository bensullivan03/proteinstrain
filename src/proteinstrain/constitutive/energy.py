"""Elastic densities, centre-volume allocation and energies as NumPy arrays."""

import numpy as np
from ..mechanics.measures import _usable, green_lagrange_strain
from .volumes import atom_radii, sphere_volumes, union_volume

BOLTZMANN = 1.380649e-23
AVOGADRO = 6.02214076e23
CUBIC_ANGSTROM = 1e-30
JOULES_PER_KCAL = 4184.0
UNITS = ("J", "kT", "kJ/mol", "kcal/mol")


def convert_energy(joules, units, temperature=300.0):
    """Convert energies in joules to one of the supported units."""
    if units == "J":
        return joules
    if units == "kT":
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        return joules / (BOLTZMANN * float(temperature))
    if units == "kJ/mol":
        return joules * AVOGADRO / 1000.0
    if units == "kcal/mol":
        return joules * AVOGADRO / JOULES_PER_KCAL
    raise ValueError(f"units must be one of {list(UNITS)}.")


def energy_density(model, deformation_gradients):
    """Return energy per reference volume in Pa; unusable deformations are NaN."""
    F, valid = _usable(deformation_gradients)
    values = model.energy_density(green_lagrange_strain(F))
    values[~valid] = np.nan
    return values


def total_energy(
    model, deformation_gradients, volumes, *, units="kT", temperature=300.0
):
    """Return one energy per centre; volumes are cubic Angstroms and material moduli Pa."""
    density = energy_density(model, deformation_gradients)
    volumes = np.asarray(volumes, float)
    if (
        volumes.shape != density.shape
        or not np.isfinite(volumes).all()
        or (volumes < 0).any()
    ):
        raise ValueError("Volumes must be finite, nonnegative and one per centre.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    return convert_energy(density * volumes * CUBIC_ANGSTROM, units, temperature)


def allocate_volume(total, n_centres, *, weights=None):
    """Distribute an explicit total over centres, uniformly or proportionally to weights."""
    from .._validation import as_weights

    if n_centres < 1 or not np.isfinite(total) or total < 0:
        raise ValueError("A nonnegative total and at least one centre are required.")
    weights = as_weights(weights, n_centres)
    return float(total) * weights / weights.sum()


def allocate_group_volumes(totals, groups, *, weights=None, elements=None):
    """Distribute explicit group totals over centres; supply weights, elements or use equal shares."""
    groups = np.asarray(groups)
    weights = (
        np.ones(len(groups))
        if weights is None and elements is None
        else (
            sphere_volumes(atom_radii(elements))
            if weights is None
            else np.asarray(weights, float)
        )
    )
    if weights.shape != groups.shape or set(groups.tolist()) != set(totals):
        raise ValueError("Weights/groups/totals must cover every centre.")
    values = np.zeros(len(groups))
    for group, total in totals.items():
        mask = groups == group
        values[mask] = allocate_volume(total, int(mask.sum()), weights=weights[mask])
    return values


def volumes_from_structure(
    structure,
    atoms,
    *,
    groups=None,
    radii=None,
    method="monte_carlo",
    samples=2000000,
    seed=0,
):
    """Partition one residue-union volume onto native centres; return (volumes, standard_error)."""
    from ..structure import residue_identity

    keys = ["chain_id", "hetero", "residue_number", "insertion"]
    labels = (
        atoms.alignment_group.to_numpy()
        if groups is None and "alignment_group" in atoms
        else (np.zeros(len(atoms), int) if groups is None else np.asarray(groups))
    )
    if len(labels) != len(atoms):
        raise ValueError("One group label is required per centre.")
    residue_groups = {}
    rows = list(atoms.itertuples(index=False))
    centre_keys = []
    for i, row in enumerate(rows):
        residue = tuple(getattr(row, k) for k in keys)
        if residue in residue_groups and residue_groups[residue] != labels[i]:
            raise ValueError("A residue cannot belong to multiple volume groups.")
        residue_groups[residue] = labels[i]
        centre_keys.append((*residue, row.atom_name))
    if len(set(centre_keys)) != len(centre_keys):
        raise ValueError("Centre atoms must be unique.")
    points = []
    elements = []
    native = []
    for chain in structure.model:
        for residue in chain:
            rid = residue_identity(residue)
            key = (chain.name, rid.hetero, rid.number, rid.insertion)
            if key not in residue_groups:
                continue
            names = set()
            for atom in residue:
                if atom.name in names:
                    raise ValueError(
                        "Volume allocation requires unambiguous alternate atoms."
                    )
                names.add(atom.name)
                points.append((atom.pos.x, atom.pos.y, atom.pos.z))
                elements.append(atom.element.name)
                native.append((*key, atom.name))
    if not points or not set(centre_keys) <= set(native):
        raise LookupError("Some centre atoms are absent from the structure.")
    total, cells, error = union_volume(
        points,
        atom_radii(elements, radii=radii),
        method=method,
        samples=samples,
        seed=seed,
    )
    lookup = dict(zip(native, cells))
    group_totals = {label: 0.0 for label in set(labels.tolist())}
    for key, cell in zip(native, cells):
        group_totals[residue_groups[key[:-1]]] += cell
    centre_weights = np.array([lookup[key] for key in centre_keys])
    return allocate_group_volumes(group_totals, labels, weights=centre_weights), error
