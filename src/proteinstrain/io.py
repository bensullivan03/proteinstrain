"""Standard PDB/mmCIF models, biological assemblies and downloads."""

from pathlib import Path
import gzip, os, re, tempfile
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import gemmi
import warnings
from .structure import ProteinStructure, residue_code, residue_identity

DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
SUFFIXES = (".cif", ".mmcif")


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path


def _entity_descriptions(block):
    category = block.get_mmcif_category("_entity.")
    ids = category.get("id") or []
    descriptions = category.get("pdbx_description") or []
    return {
        str(entity): "" if text is None else str(text).strip()
        for entity, text in zip(ids, descriptions)
    }


def _structure_from_block(block):
    structure = gemmi.make_structure_from_block(block)
    structure.merge_chain_parts()
    structure.setup_entities()
    return structure


def _model(structure, model_number):
    for model in structure:
        if model.num == model_number:
            return model
    raise KeyError(f"Structure has no model {model_number}.")


def _polymer_entity(structure, chain):
    polymer = chain.get_polymer()
    if len(polymer) == 0:
        return (None, polymer)
    return (structure.get_entity_of(polymer), polymer)


def _polymer_sequence(structure, chain):
    entity, polymer = _polymer_entity(structure, chain)
    modelled = {
        residue.label_seq: residue_identity(residue)
        for residue in polymer
        if residue.label_seq is not None
    }
    if entity is not None and len(entity.full_sequence):
        names = [str(name).split(",")[0] for name in entity.full_sequence]
        return tuple(
            (
                (residue_code(name), modelled.get(position + 1))
                for position, name in enumerate(names)
            )
        )
    if len(polymer):
        warnings.warn(
            f"Chain {chain.name!r} has no deposited full polymer sequence; "
            "using modelled residues only. Polymer positions may omit real gaps.",
            UserWarning,
            stacklevel=3,
        )
    return tuple(
        ((residue_code(residue.name), residue_identity(residue)) for residue in polymer)
    )


def _generator_selects(chain, residue, gen):
    return residue.subchain in set(gen.subchains) or chain.name in set(gen.chains)


def normalise_pdb_id(text):
    """Return a canonical classic ID (uppercase) or extended pdb_ ID (lowercase)."""
    value = str(text).strip()
    if re.fullmatch("[1-9][A-Za-z0-9]{3}", value):
        return value.upper()
    if re.fullmatch("pdb_[A-Za-z0-9]{8}", value, flags=re.I):
        return value.lower()
    raise ValueError(f"{text!r} is not a classic or extended PDB identifier.")


def download_url(pdb_id):
    """Return the documented download route for a normalized classic/extended ID."""
    identifier = normalise_pdb_id(pdb_id)
    if identifier.startswith("pdb_"):
        return f"https://files-beta.wwpdb.org/download/{identifier}.cif.gz"
    return DOWNLOAD_URL.format(pdb_id=identifier)


def _same_pdb_id(first, second):
    left, right = (normalise_pdb_id(first), normalise_pdb_id(second))
    expand = lambda value: (
        value if value.startswith("pdb_") else "pdb_0000" + value.lower()
    )
    return expand(left) == expand(right)


def download_mmcif(pdb_id, destination, *, overwrite=False, timeout=60.0):
    """Download, parse and validate a deposit before atomically replacing its cache."""
    identifier = normalise_pdb_id(pdb_id)
    destination = Path(destination)
    if destination.is_dir() or destination.suffix.lower() not in SUFFIXES:
        destination = destination / f"{identifier}.cif"
    if destination.exists() and (not overwrite):
        return destination
    url = download_url(identifier)
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = response.read()
            status = response.status
        if url.endswith(".gz"):
            payload = gzip.decompress(payload)
        document = gemmi.cif.read_string(payload.decode("utf-8"))
        block = document.sole_block()
        found = block.find_value("_entry.id")
        if not found or not _same_pdb_id(found, identifier):
            raise ValueError(
                f"Payload entry ID {found!r} does not match {identifier!r}."
            )
        structure = _structure_from_block(block)
        if not len(structure) or structure[0].count_atom_sites() == 0:
            raise ValueError("Payload has no modelled atoms.")
    except HTTPError as error:
        raise RuntimeError(
            f"Download {url} failed with HTTP {error.code}: {error.reason}"
        ) from error
    except (URLError, OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"Download {url} failed (HTTP {locals().get('status', 'unavailable')}): {error}"
        ) from error
    return _atomic_write(destination, lambda stream: stream.write(payload))


def _read(path):
    path = Path(path)
    if path.suffix.lower() not in (".cif", ".mmcif", ".pdb", ".ent"):
        raise ValueError("Input must be PDB or mmCIF.")
    structure = gemmi.read_structure(str(path))
    structure.merge_chain_parts()
    structure.setup_entities()
    if path.suffix.lower() in (".pdb", ".ent"):
        structure.assign_label_seq_id()
    descriptions = {}
    if path.suffix.lower() in SUFFIXES:
        descriptions = _entity_descriptions(gemmi.cif.read(str(path)).sole_block())
    return structure, descriptions


def _wrap(structure, model, descriptions, label):
    sequences = {}
    chain_descriptions = {}
    for chain in model:
        entity, polymer = _polymer_entity(structure, chain)
        chain_descriptions[chain.name] = (
            descriptions.get(entity.name, "") if entity is not None else ""
        )
        if len(polymer):
            sequences[chain.name] = _polymer_sequence(structure, chain)
    return ProteinStructure(str(label), model.clone(), chain_descriptions, sequences)


def load_structure(path, *, structure_id=None, chains=None, model_number=None):
    """Load a standard model; alternate conformers are selected when extracting atoms."""
    structure, descriptions = _read(path)
    model = structure[0] if model_number is None else _model(structure, model_number)
    loaded = _wrap(
        structure,
        model,
        descriptions,
        structure_id or structure.name or Path(path).stem,
    )
    return loaded if chains is None else loaded.retain_chains(chains)


def list_assemblies(path):
    """Return deposited biological assembly names."""
    structure, _ = _read(path)
    return tuple(a.name for a in structure.assemblies)


def load_assembly(path, assembly_name, *, structure_id=None, model_number=None):
    """Expand the named biological assembly with distinct readable chain IDs."""
    structure, descriptions = _read(path)
    source = structure[0] if model_number is None else _model(structure, model_number)
    assembly = next(
        (a for a in structure.assemblies if a.name == str(assembly_name)), None
    )
    if assembly is None:
        raise LookupError(f"No assembly {assembly_name!r}.")
    original = _wrap(structure, source, descriptions, structure.name)
    model = gemmi.Model(source.num)
    sequences = {}
    descriptions_out = {}
    used = set()
    for gen in assembly.generators:
        for operator in gen.operators:
            for chain in source:
                residues = [
                    r for r in chain.clone() if _generator_selects(chain, r, gen)
                ]
                if not residues:
                    continue
                name = chain.name
                suffix = 1
                while name in used:
                    suffix += 1
                    name = f"{chain.name}{suffix}"
                used.add(name)
                copy = gemmi.Chain(name)
                for residue in residues:
                    for atom in residue:
                        atom.pos = gemmi.Position(operator.transform.apply(atom.pos))
                    copy.add_residue(residue)
                model.add_chain(copy)
                if chain.name in original.polymer_sequences:
                    sequences[name] = original.polymer_sequences[chain.name]
                descriptions_out[name] = original.chain_descriptions.get(chain.name, "")
    if not len(model):
        raise ValueError("Assembly selected no atoms.")
    return ProteinStructure(
        structure_id or original.structure_id, model, descriptions_out, sequences
    )


def save_structure(structure, path):
    """Write ordinary mmCIF, retaining full sequences through standard entity categories."""
    holder = gemmi.Structure()
    holder.name = structure.structure_id
    holder.add_model(structure.model.clone())
    # Separate subchains/entities preserve sequences even after chain relabelling.
    for i, chain in enumerate(holder[0]):
        for residue in chain:
            residue.subchain = f"P{i+1}"
    holder.setup_entities()
    for chain in holder[0]:
        polymer = chain.get_polymer()
        if not len(polymer):
            continue
        entity = holder.get_entity_of(polymer)
        sequence = structure.polymer_sequences.get(chain.name, ())
        positions = {
            native: i + 1
            for i, (_, native) in enumerate(sequence)
            if native is not None
        }
        for residue in polymer:
            residue.label_seq = positions.get(residue_identity(residue))
        if entity is not None and sequence:
            entity.full_sequence = [
                gemmi.expand_one_letter(code, gemmi.ResidueKind.AA)
                for code, _ in sequence
            ]
    document = holder.make_mmcif_document()
    block = document.sole_block()
    entities = block.get_mmcif_category("_entity.")
    descriptions = {}
    for chain in holder[0]:
        polymer = chain.get_polymer()
        if len(polymer):
            entity = holder.get_entity_of(polymer)
            if entity is not None:
                descriptions[entity.name] = structure.chain_descriptions.get(
                    chain.name, ""
                )
    entities["pdbx_description"] = [
        descriptions.get(name, "") for name in entities.get("id", [])
    ]
    if entities:
        block.set_mmcif_category("_entity.", entities)
    return _atomic_write(
        path, lambda stream: stream.write(document.as_string().encode())
    )


def fetch_structure(
    pdb_id, directory, *, structure_id=None, chains=None, assembly=None, overwrite=False
):
    """Download if necessary, then load a model or biological assembly."""
    path = download_mmcif(pdb_id, directory, overwrite=overwrite)
    loaded = (
        load_structure(path, structure_id=structure_id)
        if assembly is None
        else load_assembly(path, assembly, structure_id=structure_id)
    )
    return loaded if chains is None else loaded.retain_chains(chains)
