# M18.1 prospective, outcome-blind eligibility amendment

Starting revision: PR38 head3eb51fbb25325d6a9a20bccf48211cc834aa5113.
No M0–M18 record, model, threshold, H1–H5 hypothesis or denominator may change.
Candidate phenotype values remain inaccessible. No target-driven filters or decisions.

This is an **eligibility amendment**, not an original M18 rule. M18 did not define
how external profile/fold evidence adjudicates a short ambiguous alignment. The
unchanged M12 review flags remain reported even if additional evidence favors an
unrelated interpretation. A profile/fold-supported interpretation is distinct from
a literal admission-gate PASS. Unresolved mandatory gates imply BLOCKED.

Investigate exactly CspA–BBC1, CI2–BBC1, CI2–1LP1 and CI2–1A32:
- Recompute original global/local alignment, both-sided aligned coverage, identical
  residue count, identity relative to complete shorter/longer sequence, gap count,
  and optimal-alignment degeneracy. Do not alter alignment scoring or thresholds.
- Obtain primary curated UniProt/InterPro/Pfam domain boundaries and accession/version,
  including domain architectures. Distinct names alone do not prove nonhomology.
- Seek residue-mapped PDB/PDBe CATH/SCOP classifications, including homologous
  superfamily and topology. A different topology supports a nonhomologous explanation;
  absent classifications or incomplete mapping remain explicit missing evidence.
- Attempt reciprocal profile evaluation of exact assay constructs against all relevant
  family HMMs. Use official Pfam models and GA thresholds when accessible. Report
  tool/database versions, coverage and scores. Missing profile data or tools is a
  missing check, never a negative homology result. Do not replace a failed lookup
  by an invented accession or infer absence from a missing annotation.
- A significant shared homologous superfamily/profile is evidence against independence.
  Distinct mapped superfamilies plus own-family profile support and absent cross-profile
  hits favor incidental alignments, but cannot prove zero evolutionary relationship.

Seek at least one additional family with new primary experimental measurements,
explicit raw-column lineage, genotype/reference provenance, assay semantics and
license. Prefer a small study (e.g. ubiquitin or beta-lactamase); reject any historical
protein/domain aliases. Metadata, methods, table headers and code without outputs are
allowed. Never inspect numerical phenotype columns, even in candidate selection.
Acquire public metadata/profile/coordinate bytes only; log URLs and checksums.
Any target-bearing file stays quarantined and is not needed for this adjudication.

Admission requires all original gates AND at least four qualified external families
to apply the original >=3/4 directional criterion. Three families cannot silently
change its denominator: classify any future three-family study exploratory, original
confirmation BLOCKED. No final model roles or fits until full admission. If blocked,
report all evidence, exact missing links and unchanged gates; do not unblind labels.
If fully eligible, separately commit exact folds/configs/IDs/prediction-freeze protocol
before any candidate targets, then execute frozen M17 models once without adaptation.
No new model, release tag, merge or falsely independent validation claim.
