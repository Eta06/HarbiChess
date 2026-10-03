# UFUK: actual policy self-learning after measured search improvement

Before any new self-play/update. Fixed weights/narrow4vswide16 won38/7trueDraw/2,
oneunknown cap;48games score0.875,24pair bootstrap95%[0.8125,0.9375], conservative
Hoeffding[0.597779,1]. Registered search gate passed, but this is not model learning.
Only after full144game integrity audit passes start this new experiment.

Hypothesis: deeper candidate continuation with narrow search produces more useful
fresh self targets; model policy improves against its initial model **when both
use the same narrow search**. No success assumption or novel-search claim.
Earlier failed wide-search self-learning/depth/adapter tests remain failed.

Initial native bootstrap18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae,
72,497params; only16,914pair_* trainable,55,583shared/value parameters frozen.
Weights-only warm start/fresh optimizer; within-task full native optimizer/RNG/
replay/cursor resume at every generation boundary in a fresh process. No new
Stockfish label, engine actor, teacher sample or reference query during learning.

Six generations×32games =192 own-search games;64sim/max4/Gumbel1, max240plies,
four spawnedCPU1 actors, learnerCPU1/FP32/batch64/AdamW5e-5/wd1e-4/clip5,
128updates/generation=768total, seed20261015, rolling3generations/game-balanced.
Original actor train book019c5b4a49ac287ca03effb65b33ed97daa2e702ddd4a6972491969e27912b24,
48eight-ply opening families cyclinggameindex. More trajectories/iterations but
same totalupdates as previous768step experiment; no single-variable attribution
against that older run. Primary causal contrast is final versus unchanged initial
weights with the same new search in this experiment.

Total5400s wall, existing4CPU/16GiB/no GPU/new paid resources. Every generation:
native frozen20k validation observer only, no training on those rows; stop if
policy CE>initial+0.15, frozen tensor/value changed, terminal-known generated
position fraction<0.25, nonfinite, source/hash/config mismatch or wall cap.
Unknown endings staymasked; heldout terminal support counted, zero loss without
support never a value learning result. Save intermediate/full checkpoints and
immutable session summaries. No extension/luckyseed reruns if qualification fails.

Before first game freeze24 new28-ply arena roots: uniform four legal suffix moves
from the new24-ply root book, seed20261015; reject terminal/currentposition seen in
native labels/prior complete arena histories, no engine/model filter. Existing
development families reused, not independent Elo. Hash/book committed first.

Only after generation6+integrity/retention pass, three48game colour-matched arms:
finalvsinitial (both16sim/max4/Gumbel0), finalvsStockfish32, initialvsStockfish32;
samefreshroots/seed20261015/max240/CPU1/eager/batch1,900s perarm. SF19samebinary/
Threads1/Hash16MiB/32requested nodes, actualnodes recorded. Primary final score
>0.60,24pair bootstrap95% lower>0.50,capfraction<=0.10/all48completed. Report
conservative bound, matchedSFdelta and full histories/resources. Iffailed, no
promotion or self-learning success claim. Even pass is limited development
policy self-learning, not Stockfish/AlphaZero-level strength or publishable novelty.
