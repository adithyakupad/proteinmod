# Blinding deviation — recorded before candidate admission

The initial GET of the MaveDB abundance **metadata** endpoint returned a nonempty
`scoreDistribution` object alongside sequence/license/column metadata. Two initial
whole-metadata JSON inspections mechanically decoded that object in Python memory
before its presence was recognized. Distribution values were not printed, queried,
used for variant filtering/model selection, committed or passed to a model.
A subsequent shape-only check reported that this field was a nonempty dictionary.
No variant-level score endpoint or score rows were accessed. Calibration list was
empty. No individual external scores or prediction targets were displayed or used.

Nevertheless this fails the strict pre-decode metadata exclusion intended by this
protocol. We explicitly do **not** claim completely untouched outcome-derived
metadata. The raw response remains quarantined in the ignored local source cache;
only a whitelisted projection is eligible for committed evidence. Future metadata
reads skip excluded JSON spans before decoding. The same API's lean variant view
also contains a score field and is not an authorized genotype-only route.

Automatic approval review separately rejected the proposed bulk CSV acquisition
because its score exclusion had not yet been verified; that command was not run.
We inspected official pinned server column planning/specifications and will use
only explicitly score-excluding namespaces plus a header allowlist checked before
reading body cells. An unexpected field aborts without reading its body. Tests
must establish this boundary. This is a safer verified route, not a bypass of the
rejected full-body download.

Strict blinding status is PARTIAL / DEVIATION, while individual target-label
blinding remains intact. No external ML confirmation is authorized. Independent
scientific review must decide whether this aggregate-only decoding affects a
future SOD1 confirmation; this work cannot self-certify it away. Original M17
hypotheses and four-family criteria remain unchanged. All other qualification
work is outcome-independent. CspA/CI2 item permissions remain a separate blocker.
