# UFUK-POSITION — sparse own-outcome positional critic: FAIL

CPU-only; no GPU/SSH allocation or paid resource. Source `4cae08522ac940746cf9127ca8ce4a6b63a47408`, separately SHA-bound helpers and prospective protocol under `experiments/ufuk/cpu-positional-value-v1/`. Standard existing sparse-value-v1 832 piece-square/EP plus8 metadata representation; 32channels/32hidden,28,067 trainable parameters, no novelty claim.

Weights-only architecture transfer from original teacher e8 is explicitly distinguished from subsequent full offline native-v2 training resume. Every inherited e8 tensor/policy/trunk is frozen; the new sparse output is explicitly zero initially, with a mandatory untrained same-architecture control. Fresh AdamW .003/decay.0001, game-equal completed own games then uniform positions, fixed4096updates, no loss/endpoint selection. No new teacher queries or new self-play. Four immutable Main40 own-play journals (epochs1/8/16/32) per source seed61205/06 reused:131072 historical transitions each; originalMain40 FAIL staysFAIL. Full histories, mover outcomes and UNKNOWN exclusions verified; source-game SHAmod5 split is internal validation, not board-disjoint or independent strength.

Five tests PASS/0SKIP in1.22s, including every halfmove0..100, mover/mirror/EP/repetition exact encoder feature parity; actuale8 policy bit identity and exact production critic/portable forward parity; both-mover outcomes/UNKNOWN/checksum tamper. Real whole8/pause4/freshresume8 and six fresh strict native opens PASS in127.478s on original600s. Full model/Adam/global+samplerRNG/replay-index/contracts compare bit-for-bit. Two fixed4096 fits and fresh native0/final4096 loads PASS, with immutable datasets and frozen inheritance unchanged. No online actor resume claimed.

A clerical copied arena arm/endpoint label was corrected from linear/2048 to positional/4096 BEFORE any arena game. The primary training protocol had4096 from registration. Auditor additionally asserts an exact seed-arm contrast set to prevent vacuous all([]) pass. This did not alter thresholds, training endpoints or games.

Internal game-equal validation NLL rose from uniform1.0986 to3.35755/3.68735, while sampled final train CE was.01842/.01319. This is evidence of poor generalization under this dataset/split/configuration, not a strength result or proof that positional critics universally fail.

|Seed|Trained vs e8|Trained vs untrained sparse|Trained vs SF512|e8 SF|Untrained sparse SF|
|---|---:|---:|---:|---:|---:|
|20262405|.125 (0W/4D/12L)|.40625 (1W/11D/4L)|.03125 (0W/1D/15L)|0|.03125|
|20262406|.21875 (2W/3D/11L)|.4375 (0W/14D/2L)|0 (0W/0D/16L)|0|.03125|

All160/160 actual games,12585 legal continuation plies and1660565 actualSF nodes were independently full-history replay-audited; allcaps0, protected artifacts unchanged. Both unchanged development gatesFAIL, including trained-vs-untrained criterion; no formal qualification/model promotion. Same8known root/color families, S16/root4/G0/T1,400ply, Stockfish19 nodes512/Threads1/Hash16. Descriptive paired uncertainty is recorded; no virgin confirmation/generalizedElo.

Strongest independently validated model remains teacher-origin originale8 SHA `e8fe6d4da5dd4726ff860ba760ff2830070b5e9008c123968fcee1b0f4c1af03`. Neither CPU VALUE nor POSITION meets self-learning strength goals. All failures/oldcheckpoint formats retained. CPU publicRelease checkpoint backup PENDING; local full natives under `/workspace/work/harbichess/cpu-positional-value-v1-actual/fits/`. Apple/CUDA hardware untested; existing cross-backend sparse architecture is reused, no MLX removal.

Next different control UFUK-QSEARCH: all-legal-root budgeted512-node alpha-beta with2plyquiescence and full rule history, same search for e8/untrained/previous own-trained linear2048. Six algorithm/value-adapter tests PASS/0skip; actual24fixed-move timing/legal/budget admission PASS,8.811s/original600s, not strength. Learned sparse critic is not promoted/selected; QSEARCH chooses previously registered linear2048 endpoints with better internal calibration, disclosed development choice. Search-only improvements cannot count as learning. Protocol and immutable modelSHA already registered,160 fixed games running; result unobserved at report publication.
