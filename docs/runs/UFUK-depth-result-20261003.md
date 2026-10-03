# UFUK residual depth: capacity gate failed

Adding two identity-initialized residual blocks per tower did not meet the
registered improvement gate. Both arms stopped after3500updates/best2000 by
patience1500; no arena, extension or promotion. Conventional depth alone is not
a demonstrated solution to this dataset/model's limitations.

Clean fixed source95551ab01bcd0ddd9f849d9e6cc7dcdd6a4cf0ed;
[preregistration](UFUK-depth-preregistration-20261003.md). Same native dataset
manifestb34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f,
60375train/20000validation,seed20261009,batch64/LR5e-5/wd1e-4/clip5/one CPU thread,
max10000/eval250. Both started from the same function of native step9250 weights,
explicit weights-only transfer/new optimizer, then full within-task checkpoint
resume from250 in fresh processes. All15 common checkpoint RNG states0..3500
matched, establishing equal batch streams at equal cursors after the RNG-loader fix.

| Model | Params | Best policyCE | Best WDLCE | QMAE | TotalCE | Invocation wall sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen reference | 72497 | 2.632565 | 0.654617 | 0.337308 | 3.287182 | — |
| Continued control | 72497 | 2.624575 | 0.654801 | 0.334823 | 3.279376 | 503.448243s |
| Deeper | 91057 | 2.623700 | 0.655580 | 0.335182 | 3.279280 | 539.909801s |

Treatment−control policy improvement0.00087449<required0.03; treatment−reference
improvement0.00886486<required0.05. Other retention bounds did not rescue the failed
primary. Original trainer0.10/both-heads qualification flags also false, reported
unchanged rather than reinterpreted as this capacity gate.
Single seed/no multi-seed claims. Validation seed differs from the previous20k
cache20261005, so reference was evaluated directly on this exact subset.
No held-out position suite or game strength is inferred from the losses.

Control parentCPU179.290464+323.287236=502.577700s;deeper181.201305+357.962324
=539.163629s. Peak parent RSS5332580/5332716KiB,within16GiB; one model at a time,
existing4CPU quota/no GPU/no paid resources. Initial setups/tests and later
policy-target observer partly overlapped; no isolated speed comparison. Parent
counters exclude unrelated diagnostic processes. Outer command wall sums above
include both Python launches and setups; native process counters remain preserved.

Initial deep model SHAca478e3898a9b54cb769a81475e3021776dfee06a68439137dedcf20601cc8b5.
Inherited parameters identical; trained reference full/masked Torch logits
bitwise equal on8cases including bothcastling/promotion,enpassant and histories;
real MLXCPU numerical parity2e-5. New residual branches received gradients and
changed in the real test. Best controlSHA6a1d313f2756a69780dbf270c67671f27862e7f61c558708150b2bc29ab94581;
best deeperSHAb707b9b5950964676427efcafb40ab0fd3ec2bbf79ebb2ccb23869a54fbc50ff.
All16 checkpoints perarm preserved; best validation, not arena score, selected.

An archived **full optimizer/RNG** step9250 was loaded under pinned oldc2056f7
and new95551ab; same64 next sample indices and3actual native rows, real update9251.
Result weightsSHA3da0f9847e40ef9e5f0426217e3640c8e76bb843b9c9de49446377fc5f52d97b,
optimizer tensor states/RNG bitwise equal. Original checkpoint unchanged.
This proves one preserved archived update, not every future pipeline on everydevice.
Full suite496passed41.67s/zero skips; AppleMetal/CUDA device tests unavailable.

Preflight mistakes retained: the first initial-parity receipt used an old cwd
commit despite new pinned imports. That source receipt is not authoritative;
V2 ran at correct95551ab and produced identical model SHA with exact code/script
receipt. Initial resume probes refused before update due missing outputparent
and mismatched deterministic runtime; corrected runtime matches the stored true
flag. No guard was bypassed, no original metadata was rewritten and no failed
training run was relabelled successful.

[Exact text evidence](UFUK-depth-evidence-20261003.json) includes run commands,
all progress/evaluation counters, transfer/parity/native-resume audits and scripts.
Large weights/optimizer/dataset bytes remain local; Release400 remote binary
backup is incomplete. Retain original native reference, not these minor continuations.
