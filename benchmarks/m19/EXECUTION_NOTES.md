# Execution notes, before the complete diagnostic sweep

Protocol committed at7664fd2, implementation ff8b5d5. Initial preflight stopped
before fitting because in-memory tuples differ from JSON lists; fixed canonical
JSON comparison (f234c8e). First historical replay stopped on float32 scaler
rounding (~7.6e-9); preserved M17 operation order (aad5327), without relaxing
historical tolerance. Scalar inputs were initially multiplied in float32;
changed only equivalence diagnostic inputs to float64 (2ca57ee).

A subsequent scalar solver diagnostic exceeded the implementation threshold1e-7
on case22. No architecture, penalty, split or outcome-selection change is allowed.
The fixed sweep will record all scalar failures instead of aborting on the first;
any such failure prevents overall success, regardless of predictive gain. This
is diagnostic completion after a failed control, not a successful confirmatory
rerun or relaxed tolerance. Failed attempts remain in ignored runs/m19-* paths.
No complete prior sweep or discarded favorable outcome exists. All histories
remain unchanged. No SOD1 sources are accessed.
