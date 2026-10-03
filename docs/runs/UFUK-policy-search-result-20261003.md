# UFUK direct policy: strength gate failed

Raw selection against the same weight network's16-simulation search scored
1/40/7,0.4375,pair bootstrap[0.385417,0.489583]. One unknown capped game is included
as0.5 for diagnostics only;39 draws are actual terminal draws. Primary>0.60/lower>0.50
failed. Keep search default; the favourable raw-policy point estimate on96
diagnostic positions did not establish better game play. No training or promotion.

[Preregistration](UFUK-policy-search-preregistration-20261003.md), clean source
`c2056f703dc15c68e783132d48f4659b5ef2e97f`. Fixed model18f2aae5a4dca317229b87af17ee393a4e786db3369e54aaf32c926a811cb5ae;
24 frozen20ply rootsSHA`f47d224d7911dd751440a0ac7bf536ad15351c9c4a99a7da3402cd2c4b3bacd4`,
seed20261008,both colours,one Torch thread,240plies/900s perarm. Reused development
families, fresh uniform-legal stress suffixes, no quality filter. Not new family
generalization/public Elo. Raw=max legal prior/UCI tie break/one inference;
search=FullGumbel16/max16/Gumbel0/default0.1 scale/50 maxvisit init.
Stockfish19/one thread/Hash16MiB/requested32nodes, unequal selector compute,
actual SF nodes unrecorded. Automatic claim_draw protocol retained.

| Arm | Wins/draws/losses | Score | Pair bootstrap95% | Wall s | NN evaluations |
| --- | --- | ---: | --- | ---: | ---: |
| Raw vs own search | 1/40/7 | 0.437500 | [0.385417,0.489583] | 68.814198 | 33217 |
| Raw vs SF32 | 1/19/28 | 0.218750 | [0.135417,0.302083] | 5.130052 | 1377 |
| Search vs SF32 | 5/12/31 | 0.229167 | [0.135417,0.322917] | 38.015984 | 17132 |

Raw−search SF score delta−0.010417,paired bootstrap[−0.125000,+0.104167]; no
relative SF gain proven. Direct primary conservative Hoeffding[0.160279,0.714721].
Raw compute is smaller but different played histories/game lengths mean elapsed
5.13vs38.02 is not an identical-workload speedup benchmark or a strength-speed win.
All these SF32 results remain weak and do not estimate universal Elo.

Independent audit144 histories/11,442 plies/root pairs/colours/legal moves/endings/
scores,one cap exactly240plies; primary failed retained. No repeated selection or
threshold change. Real candidate-no-search/one-inference test plus history/colour/
teacher probes6passed1.13s; full suite490passed43.25s/zero skips atc2056f7,
real Linux MLX CPU tests included. AppleMetal/CUDA hardware tests unavailable.

[Exact UTF-8 evidence](UFUK-policy-search-evidence-20261003.json) includes all
games,commands,source pin,book generator and audit with SHA256. Existing4CPU/16GiB,
no new paid resources. These arena collectors record wall/NN counts but not full
parent/engine CPU time. First arm overlapped tests; no isolated performance claim.
Binary model/replay files remain local; Release400 prevents remote-backup completion.

Decision: useful policy and low-budget search both remain weak; neither arbitrary
more simulations nor disabling search is a demonstrated solution. Next isolate
representation capacity on the fixed balanced teacher dataset with comparable
small-network control, measure qualification before any new self-learning phase.
Conventional residual/pairwise mechanisms are not claimed as original research;
publishable novelty remains a separate question requiring stronger evidence.
