# UFUK: broad-history controlled pretraining protocol

Prospective registration before any broad-data optimizer update. This experiment
asks whether adding independently rooted teacher data improves held-out learning
and subsequent equal-search game strength. It is teacher pretraining, not a
closed self-learning loop, teacher superiority or a new architecture claim.

Immutable training implementation: f4c10e549c82fa35e7d595ccada90a15dccfd613,
clean work/common-validation-pinned. Initial same72,497parameter pairwise model
SHA18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae.
All used trunk/policy/native STM WDL parameters train; unused21material-readout
parameters stay frozen and hash guarded. FP32 eager CPU1, deterministic Torch,
AdamW learning_rate0.0001 with recorded LearnerConfig defaults, B64, policy CE
plus WDL CE, no architecture, loss-weight, search or dtype change.

Control uses all retained training rows of unchanged native80511 source and its
already verified exact packed cache. Candidate uses all training rows of the
version1 merged source:80511byte-preserved native rows plus586858broader rows;
9broader training overlaps removed. Dataset667369rows/21504games SHA
ca2912fc37845fe395f0e445be72de805d99ac87d5b7ce2b5d65cfe4c830eee7.
Parent586867teacher-row collection,20,480jobs,5120source games remains immutable.
No unknown game cap is transformed into a terminal result target.

Both arms use the SAME merged validation source, exclude overlaps against the
full training-key UNION before any caps, and separately report anchor families
[0,100000) and broader families[200000,300000). At most20000rows PERgroup, seed
as registered below and group-specific deterministic selection. The selected
checkpoint minimizes equal group mean(policyCE+valueCE), minimum improvement
0.0001. QMAE and top16 target coverage are separate diagnostics. CE, entropy and
agreement are not substitutes for game strength.

Two seeds20261026(primary) and20261028(replication). Each seed runs BOTHarms.
Order:primary control then candidate; replication candidate then control.
Same initial weights, B64/max30000updates/evaluation every1000/patience7500,
no train cap. Every run has3600WHOLEwall seconds including fresh imports,
complete source/array validation, startup baseline, checkpointing, BOTH fresh
processes and final result publication. No other training/benchmark runs
concurrently. Parent starts the immutable whole-run clock before invocation0;
registered process boundary1000updates is resumed in a fresh Python process
from its full native model/AdamW/RNG/cursor/sampletrace/input manifest.
Resume receives ONLY remaining wall budget; no reset or extension. Any exhausted
whole budget makes the run INCOMPLETE, even if a partial model looks favourable.
Max four runs14400whole allocated seconds plus four400second resume audits
(max16000allocated seconds, parent overhead separately recorded); no new
paid resources. Memory current including file cache>15GiB or diskfree<8GiB stops
the owned process group; partial traces are preserved.

After completing all four production runs, an additional registered isolated
400WHOLEsecond audit PERrun reloads the saved native step1000 state in a fresh
Python process and replays exactly the next1000updates without modifying the
production run. It compares every model/Adam tensor, Torch RNG and chained row
sample trace BITWISE against the actual production step2000 checkpoint. This
checks real continued production state, not two copies of an isolated toy run.
Inputs are checksum verified. These4000total audit updates do NOT replace
production checkpoints or count as learning progress. The audit time is
additional, included in stage total; it cannot extend any run's3600second budget.
Synthetic4update fresh-process resume already passed; production must pass here.

Learning gates evaluated ONCE using each run's single macro-selected checkpoint.
For EACHseed candidate must meet ALL:
- Broader policyCE improves>=0.05 from its own step0.
- Broader WDL CE improves>=0.03 and QMAE improves>=0.015 from its own step0.
- Anchor policyCE rises<=0.03, WDL CE<=0.02, QMAE<=0.015 from its own step0.
- Equal-group macro totalCE is>=0.05 below same-seed selected control.
- All source/cache/native checkpoint and resume audits pass; both runs finish
  at max updates or registered patience, not a timeout/exception/partial stop.

Both seeds must pass separately. No averaging rescues a failure, no second-seed
substitution, hyperparameter trial, longer training or architecture change within
this protocol. Primary seed ONLY supplies later arena selected checkpoints.
If either fails:FAILED/INCOMPLETE retained, no scheduled strength games or
promotion, diagnose measured old/new losses before a new prospective protocol.
If both pass:run the already committed independent48root/96game-per-arm strength
and speed protocol UFUK-broad-history-strength-preregistration-20261004.md,
including adjusted98.333333percent intervals, unknown-cap bounds and real nodes.
Any chess/speed failure remains failed, even if learning gates pass.

Completed prerequisites before registration: all667369full104input tensors are
bitwise identical to the authoritative BoardEncoder; prepared schema1 manifest
SHA8c9a06de8be2c1f15a44569cd7bfd0fbc234907242f2145dfe8f3dee5a3fb4f5,
12array bytes677584902. Whole preparation1020.3002148399974seconds. Native
cache SHAfbe4edb78a647060069330b1ec2fbe723c08e04f926021ce51e137e6bc50ec5f.
Common retained train rows control60375/candidate530036; eachgroup20000validation
rows, exact per-seed indices recorded before updates. Cold baseline SHA
7277cdb28b67f46a21c2f94d004f4dbe018d5b58626fdbc15734c196973e495c:
seed26 anchor policyCE2.6329989391/valueCE0.6543897433/QMAE0.3370882596;
broader policyCE2.8110660378/valueCE0.7636921182/QMAE0.4091092559.
Zero optimizer updates; original18f file unchanged. Seed28 has its own recorded
step0 measurements, not assumed identical to seed26's selected row subset.

Independent prospective strength book48roots SHA
d3b9a49e3c60cdce62a672fbfc8ae8ab2a5167ed38cc5781432a8559e0891765,
audit SHA4c8ea8433315da22727949ac53d2c3e8bf11ae3423a92988c9935c0ce89e833a
PASS for all48 distinct original source games, full legal histories/FEN/clocks/EP.
Excluded5120collection source-game IDs and653685old/new/prior-arena position keys.
These roots are not selected by model or reference outcomes. No games played.

Controller and audit scripts are preserved exactly in the registration evidence:
- run-ufuk-broad-v4-controlled-training.py SHA145c97fd54d815c83a3128c48f5c79b5051b0a4745d073cc06e34790e71934f2
- audit-ufuk-broad-v4-production-resume.py SHA1a121ffce88495ae82b5d189cbbae7dacc70d132c9f2c339a7e65cdc12cfa16a

Important boundaries go to Git/Space with exact commands and real full native
checkpoint state; preserve old failed evidence. New remote binary backup remains
incomplete due the authorized Releases endpoint400BadContentLength. Local verified
archives do not resolve that gap. No new paid resources, no promotion here.

Evidence: [57 exact UTF8 prerequisite and prospective script files](UFUK-broad-history-training-registration-evidence-20261004.json).
Training sourcef4c10e5 full suite572passed, zero skips; current1ab0d52 suite578passed,
zero skips; added paired-comparison9targeted tests passed. Linux MLX CPU is tested;
AppleMetal and CUDA/BF16device behavior remain untested on this CPU-only host.
