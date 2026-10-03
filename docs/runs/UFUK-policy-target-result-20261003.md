# UFUK actual search target: diagnostic gate passed

On128 balanced native-validation positions, the real self-play setting64sim/
Gumbel1 soft target improved finite-engine-reference expected score over raw
policy by0.046659,16family bootstrap95%[0.031561,0.062797]. All16family means positive;
registered≥0.01/lower>0 diagnostic passed. This is not chess game strength, actual
self-learning progress or teacher-exceeding proof. Previous failures remain failed.

[Preregistration](UFUK-policy-target-preregistration-20261003.md),clean source
dd4daafe0936e1d66b6473b3376fef33002fb49a,collectorSHA838303bad4bcfff4900b6139d22cc564c561dc7f0df9c7ee1fc2a2aea1165b47.
Frozen model18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae;
datasetb34a3870…;panel1b3736f25cf70e2b6ce8aad2c7cd92ce24a9d8dd366267630d96fe7a20517916.
128distinct full histories/current positions,16 existing teacher-validation
families×8positions,4 distinct games peractorrole,2white/2black perrole/family.
First integer-index games and uniform position choice seed20261010; no quality
filter. Development validation, not independent unseen families or public Elo.

Stockfish19/one thread/Hash16MiB/nativeWDL/fresh TT **all3433 legal root choices**,
32768requested nodes each,111714215actualnodes. Coherent unbounded score packets,
15s query watchdog; mean effect uses `win + draw/2` in actual root STM. Each forced
root has finite search/reference error; all-legal coverage does not make it exact
minimax. No equal-compute oracle deployment claim.

Raw entropy2.489618→target1.472366;KL(target||raw)0.802403. Mass within0.05 of
best reference score0.711840→0.776955.69positive/13negative/46effect≈0positions.
Auxiliary tanh(CP/400) effect0.091107 is reported separately; it does not replace
the registered WDL primary. Conservative Hoeffding[−0.632392,+0.725710] remains
wide. Bootstrap conditions on these16families and cannot be merged with the
eight older diagnostic families or arena samples.

Actual wall150.346694s,parentCPU41.633676s,peakRSS501476KiB; engine childCPU not
included. Overlapped the depth learner, no isolated throughput claim. Existing
4CPU/16GiB/no new paid resources.128/128positions completed under1800s cap,
all128 full histories/legals/64visits/probability sums independently audited;
3433query packets and actual node totals accounted for exactly.

This refines the earlier16sim/Gumbel0 selected-action diagnostic rather than
changing its failed gates. Real soft-target quality on these native trajectories
is useful, while the actual weak self-play distribution remains unmeasured by
this result. Next audit **stored** raw/search probabilities from actual self-play
against alllegal references, without rerunning a different stochastic teacher.
Only if that target gate passes, isolate policy representation learning with
value protected by an independent policy branch; current shared trunk contributes
to base value, so unfreezing it would not preserve value merely by freezing heads.

[Exact text evidence](UFUK-policy-target-evidence-20261003.json) preserves every
history/target/reference packet and integrity audit. Native value semantics,
replay schemas and checkpoint provenance unchanged. Binary artifacts locally
preserved; Release400 remote-backup gap remains. Laya/LLM tasks remain secondary.
