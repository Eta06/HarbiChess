# ESAS mechanism audit (2026-08-28)

## Decision

HarbiChess's dominant failure is not insufficient target cleverness. The default
loop has drifted away from policy iteration in three coupled ways:

1. it asks every search target to pass a handcrafted, full-action verifier gate;
2. it trains on tiny, mostly one-generation datasets and then demands an
   immediately gateable candidate;
3. a failed candidate ends the learning chain instead of leaving a persistent
   learner/latest network and optimizer to absorb later data.

The next large generation remains blocked. Repetition/continuation, value-regret,
consensus-Q, and root-halving transforms stay available for reproducibility but
remain off by default. The highest-probability correction is to restore a simple
visit-policy/WDL loop, qualify search at the *system* level, and then introduce a
persistent learner plus a rolling replay window. This supersedes the planned
depth-2 oracle-label experiment.

## Mechanism matrix

| Mechanism | AlphaZero / OpenSpiel | Lc0 | Gumbel MuZero / Mctx | KataGo | HarbiChess now | Consequence |
|---|---|---|---|---|---|---|
| Policy target | Root MCTS visit distribution | Search policy from self-play chunks | `action_weights` from sequential halving and completed-Q | Search policy with exploration-target corrections | Visits, then many optional transforms and clean-search consensus gates | Base visit target is sound; bespoke target gates became the bottleneck |
| Value target | Final game result `z` | WDL result, optionally mixed with search Q; moves-left head | Bootstrapped reward/value in MuZero model | Outcome plus score/ownership and short-horizon auxiliary targets | Final WDL; max-ply rows now correctly masked | Semantics are sound, but sparse terminal supervision and no auxiliaries make value learning weak |
| Replay/update | Continually updated single network; large-scale self-play | Sliding chunk window, waits for new chunks per network, restores checkpoints | Online planning target, normally inside a continuing learner | Asynchronous self-play/shuffle/train/export; gatekeeper optional | 96-game KOPRU replay, generation-shaped pilots, no persistent latest learner | Data/update scale and continuity are far below the reference loops |
| Search budget | 800 simulations during training | Run-dependent, large distributed game volume | Designed for low-budget root allocation; guarantee assumes accurate action values | Mixed cheap/full searches and playout-cap randomization | 64 for KOPRU generation; 512/800 only on small diagnostics | High-budget labels cannot compensate for a weak value function and tiny replay |
| Exploration vs target | Root Dirichlet noise and visit sampling in one self-play search | Search records exploration-aware policy | Gumbel at root; deterministic interior Full Gumbel; action weights train policy | Forced exploration is pruned from target; optional root prior softening | Can run noisy behavior plus a second clean target search | Clean dual search doubles cost and introduces behavior/target mismatch; it is not required for the base loop |
| Completed-Q | Not required by base AlphaZero | Search Q can contribute to value target | Core mechanism; unvisited Q completed by a mixed value transform | Uses richer search/value corrections | Home-grown root-only halving and Q-derived targets | HarbiChess's implementation is not Full Gumbel MuZero, and poor Q invalidates the guarantee |
| Auxiliary heads | Policy + scalar value | Policy + WDL + moves-left; newer data has multiple value sources | Policy/value/reward/dynamics model | Score, ownership, short-term value/score, future policy, soft policy | Policy + WDL; experimental action-value heads are detached | Representation is under-supervised for expensive, limited data |
| Gating | AlphaZero continuously updates one network; AlphaGo Zero used a 55% best-player gate | Network publication/checkpoint cadence is data-driven | Not a champion-gating prescription | Gatekeeper is optional; ungated training is supported | Many hard pre-arena gates, then paired arena promotion | Healthy small updates are repeatedly discarded before they can compound |
| Network | Deep residual policy/value net | Large residual/SE/attention families, structured policy heads | Task-specific learned model | Large global-context residual nets | Compact CNN; KOPRU baseline 16 channels/2 blocks and dense global 4672 policy | Capacity matters later, but capacity-only ablations cannot repair the loop |

## What the reference systems actually imply

AlphaZero trains the network toward the terminal outcome and MCTS visit policy,
using the loss `(z-v)^2 - pi^T log(p)`, 800 simulations per training search, and
700,000 batches of 4,096 positions. Crucially, its chess version continually
updates one network instead of waiting for each iteration to defeat the previous
best network. See the
[AlphaZero paper](https://arxiv.org/abs/1712.01815).

OpenSpiel's illustrative implementation preserves the same topology: actors feed
a fixed-size FIFO replay buffer, the learner samples once enough new data exists,
then writes a checkpoint and refreshes actor models. It explicitly warns that
excess actors make games slower and data staler. See the
[OpenSpiel AlphaZero documentation](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md).

Lc0's current RL loader uses a sliding chunk pool and a nonzero
`chunks_per_network`, so updates follow incoming data rather than isolated pilot
runs. Its training tensors expose several value sources, while the established
trainer supports WDL, an optional result/search-Q mixture, and a moves-left head.
See the [current Lc0 training pipeline](https://github.com/LeelaChessZero/lczero-training/blob/master/docs/README.md)
and [trainer implementation](https://github.com/LeelaChessZero/lczero-training/blob/master/tf/tfprocess.py).

Mctx's Full Gumbel MuZero uses root sequential halving, deterministic interior
Gumbel selection, and completed Q values. The library's stated policy-improvement
guarantee is conditional on action values being correctly evaluated. HarbiChess's
root-only implementation launches independent child searches and has neither the
interior selection rule nor trustworthy learned action values, so the guarantee
does not transfer. See the
[Mctx implementation](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py)
and [Mctx overview](https://github.com/google-deepmind/mctx).

KataGo addresses limited-data efficiency without requiring every legal action to
match a shallow verifier: playout-cap randomization, policy-target pruning,
global context, auxiliary future-policy/value targets, and later a softened
auxiliary policy head. Its training components run continuously and its
gatekeeper is optional. See the
[KataGo paper](https://arxiv.org/abs/1902.10565),
[methods notes](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md),
and [self-play training guide](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md).

## Reinterpretation of HarbiChess evidence

### What is genuinely working

- Value perspective, terminal sign backup, legal move handling, and max-ply
  masking have dedicated tests and diagnostics. There is no current evidence of
  a sign-convention bug.
- MIHENK's 800-simulation action beat the raw-network action on the validation
  set by `+0.08319` verified value on average, with a bootstrap interval of
  `[+0.03642,+0.14328]`. Search is useful in aggregate.
- DENGE's 512/800 mixture also had positive expected-value delta
  `[+0.02774,+0.10364]` and no harmful validation rows.
- HACIM showed stable cross-budget Q (`0.7650` Spearman and `0.01556` mean
  drift). The search implementation is not random noise.

### What the failures actually say

- MIHENK rejected the teacher because only 2/48 validation rows met a brittle
  per-position confidence definition and 512/800 top-action agreement was
  66.67%. Yet aggregate verified improvement was positive. Near-equal moves
  swapping rank is expected under visit-count discretization; KataGo explicitly
  documents this sharpening effect.
- DENGE failed at 8/48 qualified rows versus a frozen 20% threshold although its
  full distribution was stronger and non-harmful. This was a valid experimental
  failure but not evidence that MCTS cannot teach.
- HACIM spent 3,563.82 seconds generating 2,304 high-budget labels and rejected
  them because depth-1 search Q correlated only `0.3096` with a depth-4
  tactical/material verifier. The learner's value task is final WDL, so demanding
  agreement with a different handcrafted scalar is a semantic mismatch.
- CIPA/KOK/capacity experiments changed transfer regularization or width while
  holding the tiny, one-shot data/update regime fixed. Their failures do not
  establish that the network can never absorb a qualified teacher; they show
  that a few hundred steps on a narrow target set do not generalize.
- The 96-game KOPRU replay contains about 19,254 positions. It is useful for a
  sanity test, not a credible substitute for a rolling self-play population.

## Fundamental assumptions to change

1. **Teacher qualification is a system property, not full-Q label purity.**
   Require clean search to improve raw play on a frozen tactical suite and paired
   search-vs-raw matches, while monitoring value calibration and harmful-action
   rate. Full legal-action verifier Spearman becomes telemetry, not a blocker.
2. **Learner/latest is not release champion.** Keep model and optimizer progress
   across data arrivals. Arena controls release promotion; only catastrophic
   regression rolls learner/latest back.
3. **Replay is a rolling, versioned stream.** Sample whole-game-disjoint
   validation and a bounded freshness window. Do not repeatedly train tiny
   diagnostic subsets as if they were generations.
4. **Use one coherent value meaning.** Main value remains terminal WDL. Search Q
   may be stored as telemetry or an explicitly weighted auxiliary target, never
   silently substituted for outcome. Short-horizon material/tactical targets, if
   added, need their own head and loss.
5. **Exploration corrections should be local and principled.** Default to normal
   noisy self-play visit targets. If forced exploration produces label noise,
   adopt fixed policy-target pruning or a soft auxiliary head, not continuation
   heuristics in the core loop.

## Highest-probability solution line

The order below is fixed before observing new results:

1. Replace the current per-action teacher gate with a frozen system-level
   qualification: raw policy versus 64/128/256/512 search on tactical solve rate,
   paired expected score, harmful-action rate, budget scaling, and calibration.
   No custom oracle target is trained.
2. Implement persisted `learner/latest` state (model, optimizer, sampler RNG,
   replay cursor, source versions) separately from `release/champion`, with
   catastrophe-only rollback.
3. Implement a bounded rolling replay window and data-ratio scheduler. Start with
   standard visit policy + terminal WDL only; keep all historic heuristic
   transforms disabled.
4. Run a small closed-loop qualification long enough to include multiple data
   arrivals and network refreshes. Gate transfer by held-out policy imitation,
   WDL calibration, tactical retention, and search-vs-raw paired strength; do not
   demand promotion from each refresh.
5. Only if value learning remains the measured blocker, add a separately headed
   short-horizon auxiliary target and/or moves-left target. Only if exploration
   mass is measured to pollute labels, add fixed target pruning.

Large generation and release promotion remain unauthorized until steps 1-4 pass.
This is a reset of the learning architecture, not a relaxation of the existing
experimental results or their preregistered thresholds.
