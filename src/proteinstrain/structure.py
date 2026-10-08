"""Protein models with native residues and full deposited polymer sequences."""

from dataclasses import dataclass, field, replace
import gemmi
from .identity import ResidueId
from .selection import resolve_chains


def residue_identity(residue):
    """Return native author numbering, including insertion and hetero identifiers."""
    hetero = (
        f"H_{residue.name}"
        if residue.het_flag == "H"
        else ("W" if residue.is_water() else " ")
    )
    return ResidueId(hetero, residue.seqid.num, residue.seqid.icode)


def residue_code(name):
    """Convert a residue name to a one-letter code, or X."""
    info = gemmi.find_tabulated_residue(str(name))
    return info.one_letter_code.upper() if info.is_amino_acid() else "X"


@dataclass
class ProteinStructure:
    """One Gemmi model and the sequence/chain information needed for correspondence."""

    structure_id: str
    model: object
    chain_descriptions: dict = field(default_factory=dict)
    polymer_sequences: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)

    @property
    def chain_ids(self):
        return tuple(c.name for c in self.model)

    def chain(self, chain_id):
        resolve_chains(self, chain_id)
        return self.model[str(chain_id)]

    def relabel_chains(self, mapping):
        """Return a cloned model; mapping values of None drop chains."""
        unknown = set(mapping) - set(self.chain_ids)
        if unknown:
            raise LookupError(f"Unknown chains: {sorted(unknown)}.")
        names = [
            mapping.get(c, c) for c in self.chain_ids if mapping.get(c, c) is not None
        ]
        if any(not isinstance(c, str) or not c for c in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("New chain IDs must be nonempty and unique.")
        model = gemmi.Model(self.model.num)
        for chain in self.model:
            name = mapping.get(chain.name, chain.name)
            if name is not None:
                clone = chain.clone()
                clone.name = name
                model.add_chain(clone)

        def renamed(data):
            return {
                mapping.get(c, c): v
                for c, v in data.items()
                if mapping.get(c, c) is not None
            }

        groups = {
            g: tuple(mapping.get(c, c) for c in ids if mapping.get(c, c) is not None)
            for g, ids in self.groups.items()
        }
        return ProteinStructure(
            self.structure_id,
            model,
            renamed(self.chain_descriptions),
            renamed(self.polymer_sequences),
            groups,
        )

    def retain_chains(self, chains):
        wanted = set(resolve_chains(self, chains))
        return self.relabel_chains({c: None for c in self.chain_ids if c not in wanted})

    def drop_chains(self, chains):
        return self.relabel_chains({c: None for c in resolve_chains(self, chains)})

    def with_groups(self, groups):
        """Return a structure with a flat name-to-chain-IDs dictionary."""
        resolved = {
            name: (resolve_chains(self, ids) if ids else ())
            for name, ids in groups.items()
        }
        return replace(self.relabel_chains({}), groups=resolved)

    def coordinates(self, chains=None, atoms="CA", *, altloc="reject"):
        from .correspondence.coordinates import extract

        return extract(self, chains=chains, atoms=atoms, altloc=altloc)[0]

    def centroid(self, chains=None, atoms="CA"):
        return self.coordinates(chains, atoms).mean(axis=0)

    def chain_centroids(self, chains=None, atoms="CA"):
        return {c: self.centroid(c, atoms) for c in resolve_chains(self, chains)}

    def save(self, path):
        from .io import save_structure

        return save_structure(self, path)
