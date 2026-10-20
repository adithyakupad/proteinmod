"""Parse coordinates, retain author identifiers and audit unique sequence mappings."""

import warnings
from io import StringIO
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Residue import Residue

PARENTS = {
    "MSE": "MET",
    "FME": "MET",
    "SEP": "SER",
    "TPO": "THR",
    "PTR": "TYR",
    "HYP": "PRO",
    "CSO": "CYS",
}


def letter(name, parents):
    return protein_letters_3to1.get(parents.get(name, name), "X")


def unique_alignment(reference, sequence):
    aligner = PairwiseAligner(
        mode="global", match_score=3, mismatch_score=-3, open_gap_score=-6, extend_gap_score=-0.5
    )
    alignments = iter(aligner.align(reference, sequence))
    first = next(alignments)
    if next(alignments, None) is not None:
        raise ValueError("Ambiguous optimal sequence alignment; do not guess residue indices")
    pairs = {}
    for ref_block, query_block in zip(first.aligned[0], first.aligned[1], strict=True):
        for old, new in zip(range(*ref_block), range(*query_block), strict=True):
            pairs[int(old)] = int(new)
    return pairs, float(first.score), str(first)


def pdb_metadata(path, model_index):
    seqres, parents, segments, bfactors = {}, dict(PARENTS), {}, {}
    current_model, explicit_models, segment = 0, False, 0
    for line in Path(path).read_text().splitlines():
        record = line[:6].strip()
        if record == "SEQRES":
            seqres.setdefault(line[11:12], []).extend(line[19:70].split())
        elif record == "MODRES":
            parents[line[12:15].strip()] = line[24:27].strip()
        elif record == "MODEL":
            if explicit_models:
                current_model += 1
            explicit_models = True
            segment = 0
        elif current_model == model_index and record == "TER":
            segment += 1
        elif current_model == model_index and record in ("ATOM", "HETATM"):
            key = (line[21:22], int(line[22:26]), line[26:27].strip())
            if key in segments and segments[key] != segment:
                raise ValueError("Author residue ID reused across TER segments")
            segments[key] = segment
            try:
                value = float(line[60:66])
                value = value if np.isfinite(value) else None
            except ValueError:
                value = None
            bfactors[(*key, line[12:16].strip(), line[16:17].strip())] = value
    return seqres, parents, segments, bfactors


def choose_conformer(residue):
    if residue.is_disordered() == 2:
        raise ValueError(f"Ambiguous residue microheterogeneity: {residue.full_id}")
    atoms = residue.get_unpacked_list()
    occupancy = {}
    for atom in atoms:
        alt = atom.altloc.strip()
        if alt:
            occupancy[alt] = occupancy.get(alt, 0) + (atom.occupancy or 0)
    selected = min(occupancy, key=lambda alt: (-occupancy[alt], alt)) if occupancy else ""
    chosen = []
    for name in sorted({atom.name for atom in atoms}):
        options = [
            atom for atom in atoms if atom.name == name and atom.altloc.strip() in ("", selected)
        ]
        if not options:
            continue
        if len(options) > 1:
            raise ValueError("Conflicting shared/alternate atom identifiers")
        atom = options[0].copy()
        if atom.element in ("H", "D"):
            continue
        if not np.isfinite(atom.coord).all() or atom.occupancy is None or atom.occupancy <= 0:
            continue
        atom.xtra["original_altloc"] = atom.altloc.strip()
        atom.set_altloc(" ")
        atom.disordered_flag = 0
        chosen.append(atom)
    return chosen, selected, sorted(occupancy)


def parse_structure(path, model_index=0):
    path = Path(path)
    is_cif = path.suffix.lower() in (".cif", ".mmcif")
    if not is_cif and path.suffix.lower() != ".pdb":
        raise ValueError("Local coordinates must be .pdb, .cif or .mmcif")
    parser = MMCIFParser(QUIET=False) if is_cif else PDBParser(PERMISSIVE=False, QUIET=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if is_cif:
                structure = parser.get_structure("reference", str(path))
            else:
                # Strict coordinate parsing, but missing B-fields remain unknown,
                # not false zero-confidence values. Original fields are retained below.
                lines = path.read_text().splitlines(keepends=True)
                lines = [
                    line[:60] + "  0.00" + line[66:]
                    if line.startswith(("ATOM  ", "HETATM")) and not line[60:66].strip()
                    else line
                    for line in lines
                ]
                structure = parser.get_structure("reference", StringIO("".join(lines)))
        except (PDBException, KeyError) as error:
            raise ValueError(f"Invalid coordinate file: {error}") from error
    if model_index < 0 or model_index >= len(structure):
        raise ValueError("Requested model_index absent from coordinate ensemble")
    model = structure[model_index]
    parents = dict(PARENTS)
    dictionary = MMCIF2Dict(path) if is_cif else None
    if dictionary:
        names = dictionary.get("_chem_comp.id", [])
        parent_names = dictionary.get("_chem_comp.mon_nstd_parent_comp_id", ["?"] * len(names))
        for name, parent in zip(names, parent_names, strict=True):
            if parent in protein_letters_3to1:
                parents[name] = parent
        seqres, segments, bfactors = {}, {}, {}
    else:
        seqres, parents, segments, bfactors = pdb_metadata(path, model_index)
    polymer_keys = set()
    if dictionary:
        for i, position in enumerate(dictionary["_atom_site.label_seq_id"]):
            if position in (".", "?"):
                continue
            insertion = dictionary["_atom_site.pdbx_PDB_ins_code"][i]
            polymer_keys.add(
                (
                    dictionary["_atom_site.auth_asym_id"][i],
                    int(dictionary["_atom_site.auth_seq_id"][i]),
                    "" if insertion in (".", "?") else insertion,
                )
            )
    chains, altlocs, skipped = {}, [], []
    # Work on a clean model for SASA: selected alternate conformer, no H/D/water.
    clean_model = model.copy()
    for chain in clean_model:
        for residue in list(chain):
            chain.detach_child(residue.id)
    for chain in model:
        records = []
        for residue in chain:
            if residue.id[0] == "W":
                continue
            atoms, selected, alternatives = choose_conformer(residue)
            if not atoms:
                skipped.append(
                    {
                        "chain": chain.id,
                        "author_number": residue.id[1],
                        "reason": "no occupied heavy atoms",
                    }
                )
                continue
            clean = Residue(residue.id, residue.resname, residue.segid)
            for atom in atoms:
                if not is_cif:
                    atom.xtra["raw_bfactor"] = bfactors.get(
                        (
                            chain.id,
                            residue.id[1],
                            residue.id[2].strip(),
                            atom.name,
                            atom.xtra["original_altloc"],
                        )
                    )
                clean.add(atom)
            clean_model[chain.id].add(clean)
            amino_acid = letter(residue.resname, parents)
            peptide_like = amino_acid != "X" or {"N", "CA", "C"} <= {a.name for a in atoms}
            if not peptide_like:
                continue
            if dictionary and (chain.id, residue.id[1], residue.id[2].strip()) not in polymer_keys:
                continue
            if not dictionary and residue.id[0] != " " and residue.resname not in parents:
                # Free amino-acid ligands are solvent occluders, not polymer nodes.
                continue
            author_number, insertion = residue.id[1], residue.id[2].strip()
            key = (author_number, insertion)
            if any(row["key"] == key for row in records):
                raise ValueError("Duplicate author residue identifier in protein chain")
            record = {
                "key": key,
                "author_number": author_number,
                "insertion_code": insertion,
                "resname": residue.resname,
                "letter": amino_acid,
                "residue": clean,
                "segment": segments.get((chain.id, author_number, insertion), 0),
                "altloc": selected,
                "noncanonical": residue.resname not in protein_letters_3to1,
            }
            records.append(record)
            if alternatives:
                altlocs.append(
                    {
                        "chain": chain.id,
                        "author_number": author_number,
                        "insertion_code": insertion,
                        "selected": selected,
                        "alternatives": alternatives,
                    }
                )
        if records:
            chains[chain.id] = {"observed": records}
    for chain_id, chain in chains.items():
        records = chain["observed"]
        by_key = {row["key"]: row for row in records}
        if dictionary and "_pdbx_poly_seq_scheme.seq_id" in dictionary:
            d = dictionary
            n = len(d["_pdbx_poly_seq_scheme.seq_id"])
            scheme = []
            label_chains = set()
            for index in range(n):
                if d["_pdbx_poly_seq_scheme.pdb_strand_id"][index] != chain_id:
                    continue
                label_chains.add(d["_pdbx_poly_seq_scheme.asym_id"][index])
                number = d["_pdbx_poly_seq_scheme.pdb_seq_num"][index]
                insertion = d["_pdbx_poly_seq_scheme.pdb_ins_code"][index]
                key = (
                    (int(number), "" if insertion in (".", "?") else insertion)
                    if number not in (".", "?")
                    else None
                )
                scheme.append(
                    (
                        int(d["_pdbx_poly_seq_scheme.seq_id"][index]),
                        letter(d["_pdbx_poly_seq_scheme.mon_id"][index], parents),
                        key,
                    )
                )
            if len(label_chains) > 1:
                raise ValueError(
                    "Multiple polymer label chains share an author chain; select an unambiguous file"
                )
            if scheme:
                scheme.sort()
                if len({item[0] for item in scheme}) != len(scheme):
                    raise ValueError("Ambiguous polymer sequence scheme (microheterogeneity)")
                chain["sequence"] = "".join(item[1] for item in scheme)
                # Bind label_seq_id to atom_site author identifiers, rather than
                # assuming author numbering equals polymer sequence numbering.
                atom_keys = {}
                for index, label_chain in enumerate(d["_atom_site.label_asym_id"]):
                    if label_chain not in label_chains:
                        continue
                    model_serial = d.get(
                        "_atom_site.pdbx_PDB_model_num", ["1"] * len(d["_atom_site.label_asym_id"])
                    )[index]
                    if int(model_serial) != model.serial_num:
                        continue
                    position = d["_atom_site.label_seq_id"][index]
                    if position in (".", "?"):
                        continue
                    insertion = d["_atom_site.pdbx_PDB_ins_code"][index]
                    key = (
                        int(d["_atom_site.auth_seq_id"][index]),
                        "" if insertion in (".", "?") else insertion,
                    )
                    position = int(position)
                    if position in atom_keys and atom_keys[position] != key:
                        raise ValueError("Conflicting label-to-author residue mapping")
                    atom_keys[position] = key
                chain["coordinates"] = {
                    i: by_key[atom_keys[position]]
                    for i, (position, _, _) in enumerate(scheme)
                    if atom_keys.get(position) in by_key
                }
                chain["declared_keys"] = {i: key for i, (_, _, key) in enumerate(scheme)}
                if len(chain["coordinates"]) != len(records):
                    raise ValueError(
                        "Observed protein residues absent from polymer coordinate mapping"
                    )
                chain["sequence_source"] = "mmCIF polymer scheme + atom_site label/auth mapping"
            else:
                raise ValueError("Protein chain absent from mmCIF polymer scheme")
        elif chain_id in seqres:
            sequence = "".join(letter(name, parents) for name in seqres[chain_id])
            observed = "".join(row["letter"] for row in records)
            pairs, _, _ = unique_alignment(sequence, observed)
            if any(sequence[i] != observed[j] for i, j in pairs.items()) or len(pairs) != len(
                records
            ):
                raise ValueError("PDB SEQRES/coordinate correspondence unresolved; use mmCIF")
            chain.update(
                sequence=sequence,
                coordinates={i: records[j] for i, j in pairs.items()},
                sequence_source="PDB SEQRES uniquely aligned to observed residues",
            )
        else:
            chain.update(
                sequence="".join(row["letter"] for row in records),
                coordinates=dict(enumerate(records)),
                sequence_source="observed residues only; no declared polymer sequence",
            )
        for index, row in chain["coordinates"].items():
            if chain["sequence"][index] != row["letter"]:
                raise ValueError(
                    "Declared polymer residue differs from observed coordinate identity"
                )
            row["polymer_index"] = index
    return {
        "model": clean_model,
        "chains": chains,
        "model_index": model_index,
        "model_count": len(structure),
        "parser_warnings": [str(w.message) for w in caught],
        "altlocs": altlocs,
        "skipped": skipped,
        "format": "mmCIF" if is_cif else "PDB",
        "deposition": {
            key: dictionary.get(key)
            for key in (
                "_entry.id",
                "_exptl.method",
                "_refine.ls_d_res_high",
                "_pdbx_audit_revision_history.revision_date",
            )
        }
        if dictionary
        else structure.header,
    }


def map_reference(parsed, reference, settings):
    candidates, successful = [], {}
    for chain_id, chain in parsed["chains"].items():
        try:
            polymer_range = settings.get("reference_polymer_range")
            if polymer_range is not None and chain_id == settings.get("chain"):
                # Explicit construct bounds disambiguate repeated terminal cloning tags.
                if (
                    not isinstance(polymer_range, (list, tuple))
                    or len(polymer_range) != 2
                    or any(type(v) is not int for v in polymer_range)
                ):
                    raise ValueError("Explicit polymer range requires two integer bounds")
                start, end = polymer_range
                if (
                    not (1 <= start <= end <= len(chain["sequence"]))
                    or chain["sequence"][start - 1 : end] != reference
                ):
                    raise ValueError(
                        "Explicit polymer range must exactly match the complete reference"
                    )
                aligned_positions = {i: start - 1 + i for i in range(len(reference))}
                score, alignment = (
                    3 * len(reference),
                    f"Exact declared polymer range {start}..{end}",
                )
            else:
                aligned_positions, score, alignment = unique_alignment(reference, chain["sequence"])
            identity = sum(
                reference[i] == chain["sequence"][j] for i, j in aligned_positions.items()
            ) / max(len(aligned_positions), 1)
            candidate = {
                "chain": chain_id,
                "identity": identity,
                "aligned_coverage": len(aligned_positions) / len(reference),
                "score": score,
                "sequence_source": chain["sequence_source"],
            }
            successful[chain_id] = (aligned_positions, alignment, candidate)
        except ValueError as error:
            candidate = {"chain": chain_id, "error": str(error)}
        candidates.append(candidate)
    selected = settings.get("chain")
    if selected is None:
        ranked = sorted(successful, key=lambda name: successful[name][2]["score"], reverse=True)
        if not ranked or (
            len(ranked) > 1
            and successful[ranked[0]][2]["score"] == successful[ranked[1]][2]["score"]
        ):
            return [], {
                "accepted": False,
                "errors": ["Ambiguous/absent best chain; specify chain explicitly"],
                "chains": candidates,
            }
        selected = ranked[0]
    if selected not in successful:
        return [], {
            "accepted": False,
            "errors": ["Selected chain absent or alignment ambiguous"],
            "chains": candidates,
        }
    aligned_positions, alignment, alignment_quality = successful[selected]
    chain = parsed["chains"][selected]
    errors, rows, differences = [], [], []
    allowed_mismatches = set(settings.get("allowed_mismatches", []))
    for index, amino_acid in enumerate(reference):
        polymer_index = aligned_positions.get(index)
        residue_record = chain["coordinates"].get(polymer_index)
        structure_letter = chain["sequence"][polymer_index] if polymer_index is not None else None
        mismatch = structure_letter is not None and structure_letter != amino_acid
        if mismatch:
            difference = f"{amino_acid}{index + 1}{structure_letter}"
            differences.append(difference)
            if difference not in allowed_mismatches:
                errors.append(f"Unapproved reference/construct difference: {difference}")
        declared_key = chain.get("declared_keys", {}).get(polymer_index)
        author_key = residue_record["key"] if residue_record else declared_key
        has_ca = residue_record is not None and "CA" in residue_record["residue"]
        status = (
            "absent_from_construct"
            if polymer_index is None
            else "missing_coordinates"
            if residue_record is None
            else "sequence_mismatch"
            if mismatch
            else "missing_ca"
            if not has_ca
            else "mapped"
        )
        rows.append(
            {
                "reference_position": index + 1,
                "reference_aa": amino_acid,
                "structure_aa": structure_letter,
                "chain": selected,
                "status": status,
                "author_number": author_key[0] if author_key else None,
                "insertion_code": author_key[1] if author_key else None,
                "resname": residue_record["resname"] if residue_record else None,
                "noncanonical": residue_record["noncanonical"] if residue_record else False,
                "polymer_position": polymer_index + 1 if polymer_index is not None else None,
                "record": residue_record,
                "has_ca": has_ca,
            }
        )
    coverage = sum(row["has_ca"] for row in rows) / len(reference)
    if alignment_quality["identity"] < settings.get("min_identity", 0.98):
        errors.append("Aligned sequence identity below threshold")
    if coverage < settings.get("min_coverage", 0.8):
        errors.append("C-alpha reference coverage below threshold")
    audit = {
        "accepted": not errors,
        "errors": errors,
        "chains": candidates,
        "selected_chain": selected,
        "reference_length": len(reference),
        "coordinate_coverage": coverage,
        "aligned_coverage": alignment_quality["aligned_coverage"],
        "aligned_identity": alignment_quality["identity"],
        "exact_compatible_positions": sum(row["status"] == "mapped" for row in rows),
        "differences": differences,
        "missing_positions": [r["reference_position"] for r in rows if not r["has_ca"]],
        "alignment": alignment,
        "construct_insertions": [
            j + 1 for j in range(len(chain["sequence"])) if j not in aligned_positions.values()
        ],
    }
    return rows, audit
