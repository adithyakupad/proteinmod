"""Blinded genotype decoding and common residue eligibility; no phenotype access."""

import re

from proteinmod.data import AMINO_ACIDS
from proteinmod.structure_features import parse_mutation_set


def gfp_mutations(line, reference):
    """Read only the first TSV field in the primary zero-based substitution codec."""
    genotype = line.partition("\t")[0].rstrip("\r\n")
    tokens = []
    noncanonical = False
    for text in genotype.split(":") if genotype else []:
        match = re.fullmatch(r"S([A-Z*])(\d+)([A-Z*])", text)
        if match is None:
            raise ValueError("Unsupported primary genotype codec")
        old, index, new = match.groups()
        position = int(index) + 1
        if position > len(reference) or reference[position - 1] != old:
            raise ValueError("Primary WT/reference mismatch")
        if new not in AMINO_ACIDS:
            noncanonical = True
        tokens.append(f"{old}{position}{new}")
    if noncanonical:
        return None, "noncanonical_substitution"
    return tuple(parse_mutation_set(tokens, reference)), None


def compatible_positions(residues):
    """A proposed common mask is not an accepted full-reference M2 graph."""
    return [
        r["reference_position"]
        for r in residues
        if r["status"] == "mapped"
        and r["has_ca"]
        and not r["noncanonical"]
        and r["reference_aa"] == r["structure_aa"]
    ]
