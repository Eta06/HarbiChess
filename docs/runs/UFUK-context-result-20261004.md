# UFUK: global policy context native result

Native learning qualification **FAILED**.
No arena, promotion or additional updates: the preregistered native gate failed. The main strong/fast/reliable self-learning and Stockfish/AlphaZero-level
goal remains unmet. No novel algorithm, general Elo or teacher-surpassing claim.
[Protocol before production](UFUK-context-preregistration-20261004.md).

Clean pinned source `d1c9ee0116f99e28388b4a87420290dbb5380575`. Ordinary pre-norm square
attention blocks2/heads4, channels16, learned64x16position features, zero output
bridges with active random QKV/FF1; only policy tokens change. Original55583
shared/value parameters frozen; control16914pair-head parameters, context23410
pair-head+context trainable. Registered model72497->78993, added6496parameters,
including unchanged historical21parameter unused material auxiliary head.
Original18f model/replay and old portable specifications remain preserved.
Context-v1 transfer SHA`e146fd4e1cc103ef42bc0fb3e7a0a7fcd3d33a2431179b9d0840757f52ec28d5` is weights-only with
optimizer reset, not full training resume or cross-backend optimizer migration.
Both native250pause->fresh-process full optimizer/RNG/input/cursor resumes passed.

Native80511dataSHA b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f,
48/16whole-family development splits/caps40000train/20000validation/seed20261020.
CPU1FP32/AdamW1e-4/wd1e-4/clip5/B64/max8000/eval250/patience1500 and1800whole-wall
seconds per arm including repeated input preparation and full resume. Frozen
value makes native totalCE and policyCE ordering equivalent. Select minimum
policyCE among frozen-state-valid checkpoints, not the legacy core's default
both-heads qualification. Older seed05entropy observer used a different capped
row sample, and is not claimed as exact entropy of these seed20panels.

| Arm | Last update | Selected | PolicyCE | ValueCE | QMAE | Charged whole wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
|control|8000|8000|2.617693473|0.654420767|0.337139172|581.913254|
|context|8000|8000|2.612676992|0.654420767|0.337139172|925.364732|

Context gain `0.019134206` versus required0.05;
context-vs-control gap `0.005016481` versus
required0.03. This result concerns the registered representation/data/frozen
trunk/budget combination; it does not prove global attention can never help.
ValueCE and QMAE are bitwise-identical measurements throughout both arms.
All native checkpoints were actually loaded with original run_config/input
checksums/model/optimizer/RNG/cursor. Frozen tensors bitwise unchanged at every
checkpoint; context QKV feature weights changed in actual updates, and common
sampling RNG matched reconstructed seed20indices. Missing Adam states belong to
frozen parameters, not missing optimizer contents.

18real full-history probes were frozen before earlier width updates. Initial
Torch full/legal-masked/WDL error0; actual MLXCPU maxpolicy error
4.29153442e-06. Trained selected portable checkpoint
parity was also executed on18full-history probes, not just identity outputs:
`{"context": {"full_policy": 5.7220458984375e-06, "legal_masked_max": 4.76837158203125e-06, "wdl": 9.5367431640625e-07}, "control": {"full_policy": 4.76837158203125e-06, "legal_masked_max": 3.337860107421875e-06, "wdl": 9.5367431640625e-07}}`, atol/rtol2e-5. Real backend
loss/grad/update, frozen value, Torch fullresume and Torch-freeMLXload tests passed
before production:522tests/zero skips45.95s. Additional MLX-native full optimizer/
sampler/next-update exact resume test passed0.26s, separately from Torch weight
conversion. Latest fullsuite527/zero skips46.19s also includes isolated tactical
control tests; these are not learned strength evidence. AppleMetal/CUDA hardware
unavailable; actual tests used LinuxCPU. Backend AdamW update equality is not
claimed; full resume remains within the original backend/profile.

Fixed18probes/CPU1/eager legal-masked B1/10warmup/200alternating-order rounds,
3600timed calls per arm. Median context/control ratio
**1.394571**, registered<=1.5:
**PASS**. Setup/warmup
0.515852s excluded; outer timed
measurement6.533034s includes finite
checks outside each timed call. No concurrent training; encoding/search/startup
excluded. Inference latency is not whole-game throughput or strength.

New24uniformlegal12ply stage stress families seed20 were frozen before updates,
bookSHA58a9bd4bc486d787b22d462c7dd7a641c11a2130034f8a9489cbf7b0407002ca,
outside1206available source receipts/30prior arena files/145prior first4histories.
They are not natural openings or complete inherited-pretraining holdouts.
Selected controlSHA`ec9498ae07a3fe90ad382e66e15d05fe6b3d318cc8b0ab3a9031a7513d200125`;
contextSHA`93b4a2649128036a300d37a48a78ac5b691c7662be1e7fea16711830d6ec35d9`.

Existing4CPUquota/16GiB/noGPU/no new paid resources. Important start log Space15
read back exactly and51untouched prior blocks verified; source12files actually
downloaded from pushed immutable Git and matched local bytes. Commit author/
committer Emir Tunahan Alim <emrtnhalim@gmail.com>, push accountEta06/admin,
no coauthor/one meaningful file per commit. Complete model/optimizer checkpoints
remain local; exact UTF8 evidence does not replace binary backup. New Releases
upload remains400BadContentLength, remote binary backup incomplete. Prior parent+
143file supplement predate these context artifacts; another supplement is needed.
The separately preregistered classical tactical-leaf diagnostic leaves this
result unchanged and cannot establish model learning.
