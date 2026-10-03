# UFUK: native loss is far above target entropy

The entropy-floor hypothesis does not explain the remaining native errors.
Frozen20k policy target entropy1.151859132nats versus
CE2.632099348: excess1.480240216.
WDL entropy0.133592673 versus CE0.654349148:
excess0.520756475. These are theoretical reducible loss bounds,
not achievable guarantees, Elo or evidence that larger architecture must succeed.

[Preregistration](UFUK-target-entropy-preregistration-20261003.md), exact native
observerSHA33fbb59df6b400819ccacc4393c660c8ff2de1c5e8a6328736ef962c8bdb7039.
Targets checkedfinite/nonnegative and normalizedwithin1e-6; actual maximum errors
policy4.47034835815e-08,WDL3.37604433298e-08.
Float64 H=-sum(p log p), FP32 archived observedCE; sumerrors/rounding limit strict
CE=H+KL equality slightly. Policymean support3.89145, medianH
1.345780669, maxH1.386294361;
softmax CPtemperature100/coherent four-reference MultiPV targets. WDLmean support
1.8506. Teacher WDL remains distinct from weak-policy observed
terminal targets; loss improvement did not guarantee strength in the value test.

ActualCPU1wall3.107070s, no gradients/model change/new engine query.
Every perrow entropy and exact script/stdout preserved. Earlier depth/adapter
0.03CE gates remain failed, not reclassified. Large excess makes representation,
optimization and generalization worth investigating; this diagnostic does not
choose among those causes. Next controlled hypothesis tests width with the same
initial function/data and records real latency beside strength. No main goal,
new original architecture or teacher-exceeding claim.
