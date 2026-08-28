# AKIS uncertainty-policy transfer result (2026-08-28)

## Decision

AKIS failed its frozen transfer gate. Search qualification, arena, generation,
and promotion remain unauthorized.

## Result

The rank-8 adapter trained only a mergeable update to the deployed policy
linear layer, leaving WDL logits bitwise unchanged. At step 60, validation
teacher Spearman was 0.1375, verified-gain 95% interval was -0.0272 to +0.0089,
harmful selection was 12.5%, and validation cross entropy improved by only
0.02%. No later checkpoint passed; step 480 also regressed 512-search tactical
solve count.

The added train-side audit separated optimization failure from generalization
failure. At step 60, train teacher Spearman rose from 0.2439 to 0.3895 and the
train verified-gain interval became +0.0015 to +0.0616, while validation did not
improve. Through step 480 train cross entropy kept falling (3.0782 to 2.9759)
as validation cross entropy worsened (3.1248 to 3.1763).

The qualified uncertainty target is learnable on observed positions, but the
96-position training set is too sparse for a transferable contextual policy
update. The next controlled test should increase fresh teacher-label coverage
from the existing qualified replay while keeping learner steps, exposure,
thresholds, and the generation lock unchanged.

## Frozen artifacts

- `artifacts/runs/akis-uncertainty-policy-20260828-01/result.json`
- `artifacts/runs/akis-uncertainty-policy-20260828-02/result.json`
