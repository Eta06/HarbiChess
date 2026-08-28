# DOKU fresh spatial-Q label result

Date: 2026-08-28  
Source commit: `bf56220`  
Artifact: `artifacts/diagnostics/doku-action-value-dataset-20260828-01/dataset.json`  
Decision: label gate failed; spatial learner, completed-Q search, arena, generation, and promotion remain blocked

## Frozen fresh-label result

DOKU selected 96 train and 48 validation positions that do not overlap any MIHENK/TERAZI/DEGER row. It repeated clean 512/800 search and full legal-action depth-4 verification. The run took 206.34 seconds and evaluated 181,743 neural positions.

| Validation label metric | Result | Gate |
|---|---:|---:|
| 800-Q versus verifier Spearman | 0.41284 | at least 0.35, passed |
| 512/800 Q Spearman | 0.75836 | at least 0.70, passed |
| Mean absolute Q drift | 0.01640 | at most 0.03, passed |
| Top-two Q-set overlap | **72.92%** | at least 75%, failed |
| Top-Q verified delta | +0.07981 | positive interval, passed |
| Delta 95% interval | [+0.04655, +0.12256] | passed |
| Harmful top-Q ratio | 0.00% | at most 10%, passed |
| Mean verified regret | 0.02366 | at most 0.10, passed |

The spatial learner was not run because the top-two overlap missed its frozen gate. This is independent confirmation that fixed-cardinality top-action sets are unstable even when the full Q ranking and verified strength are healthy.

## Post-failure uncertainty audit

No thresholds were changed. A descriptive audit found that 88.04% of validation actions have absolute 512/800 Q drift at most 0.03; weighted by the smaller visit count, coverage is 91.71%. When the leading action changes, the mean leader margins are only 0.0066 at 512 and 0.0093 at 800. The instability therefore comes primarily from near ties, not a globally incoherent Q surface.

The next label contract should store action-level drift as uncertainty and weight supervision accordingly. A conservative action score such as the lower of the two Q estimates can be independently verified before training. This changes the supervised representation of uncertainty rather than lowering DOKU's failed 75% set gate.
