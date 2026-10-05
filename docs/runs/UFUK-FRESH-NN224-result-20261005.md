# UFUK-FRESH: all224 development games FAIL

All MC/SC/FULL fits failed the unchanged both-seed screen. No model promotion, formal campaign, general Elo or self-learning success. Frozen source6fcc, protocol884425f2939f8f527e8b3491741ce44c1be62ca97ebc35de7557dca250e9d4db, known8 opening/color pairs, Q512/q2/maxdepth8, full-history claim_draw, Stockfish19 requested512/Threads1/Hash16.

|Fixed method|Seed05 direct E8|Seed06 direct E8|Seed05 SF|Seed06 SF|
|---|---:|---:|---:|---:|
|MC|.53125|.5625|.03125|.0625|
|SC|.50000|.59375|.03125|0|
|FULL|.71875|.53125|.03125|0|

E8-SF controls0/0. Every method fails finalSF>=.25 and pairedSFgain>.10; both-seed direct>.60 also fails. Caps0. Root audit and independent implementation replayed all224 histories:15,505 continuation plies,5,945,575 own recursive nodes,1,994,172 actual SF nodes. SF exceeded requested512 on1,107/3,885 calls,max561. Requests are soft for SF; no equal-compute claim. Development-only known roots; no confidence/promotion threshold was lowered.

Original arena first1791228101.8651235,10800 end1791238901.8651235; allgames finished1791234397.8558388. ROOT audit first1791234502.0633478/end1791235402.0633478 (900). Original v1 incorrectly rejected SF actual node overruns; its source/error remain. Separate v2 reports nonnegative actual consumption and overruns. Separate v3 fixes a chained deadline predicate: both finished<=deadline and deadline==owner required. All224 v3 passed integrity, failed strength, finished1791235344.4462082 within SAME900. Independent all224 check and six ROOT actual NN search reproductions also passed within SAME900; six-search finish1791235024.7722652. Old24 profile omitted moves/value, so those matched actual same-board arena packets; all node/evaluation/depth/legal counts matched profile. No extra games or training.

Actual-result files preserve source/code/clock/correction/result evidence. Fit manifests retain all six original105/97 endpoints and full native0/final provenance. Public raw22-file preservation is in Release ufuk-cpu-ownplay-state-20261005 capsule66ef6b4f..., separately verified; byte preservation is not a new restored-runtime audit. All E8 anchors are old teacher-origin; no fresh external teacher labels were queried. Apple/CUDA hardware untested. The next controlled hypotheses are learned quiet ordering and own deeper-search residual targets, not reclassification of this failure.
