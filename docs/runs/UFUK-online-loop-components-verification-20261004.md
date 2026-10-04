# UFUK: real one-ply online components, targeted verification boundary

**74 targeted tests passed / zero skips. No production online trial, strength
gain, model promotion, teacher superiority or complete project success.**
The current four-run capacity experiment retains its frozen f4c10e5 source and
prospective gates. Source development and short targeted validation ran while
its CPU1 learning continued; no separate production training or speed benchmark
was launched. Its actual whole wall receipts include that elapsed time. Final
speed measurements and the forthcoming real-checkpoint online preflight must
run without another training/benchmark process.

## Implemented path

The experimental CPU learner advances a versioned actor pool by one legal ply
per update, consumes the new transitions once, makes one AdamW update, then
updates its own EMA. Current/base/EMA networks are owned separate copies; the
base stays frozen, and the inherited unused21material parameters stay frozen.
The same104history encoder,4672actions, black rank mirror, legal rules and
explicit draw-claim convention are used. Warm-starting is weights-only with
optimizer reset; subsequent native resume is a separate format.

Sampling uses the current legal policy at an explicit fixed temperature,
normalized in float64. Both actual selected pi/mu probabilities are retained;
underflow that removes nonzero policy support is rejected before cursor/RNG
changes. This version does not implement entropy-adaptive sampling or claim
calibration. The pinned prior-directed paper/code inspired the established
forward-KL/EMA idea; no external network weights or code were imported and no
algorithm novelty is claimed. Coefficients, learning rate, decay, clipping and
seed are required inputs; unit-test constants are not a production plan.

Real terminals use the mover's observed WDL. Nonterminal and capped transitions
use flipped next-player EMA WDL. Soft bootstrap targets remain distinct from
actual outcomes. A terminal at the same step as a cap preserves its real result.
Legal sorted-action support and all loss constants are validated. Torch and
MLX CPU share the tested loss semantics; the full experimental online actor/
optimizer runner is currently Torch CPU. Existing MLX paths are preserved.

Every returned batch includes full pre-history, source-game family, chosen legal
UCI action, legal action indices, current/base probabilities, current/base WDL,
next-state EMA WDL or actual terminal, target source/schema, advantage, importance,
selected sampling probabilities and cutoff. The canonical sample/loss records
form a SHA256 prefix chain. This is potential learning evidence; it is not yet
a curated decision-model/LLM dataset or an explanation-quality result.

## Native resume and bounded execution

`torch-online-native-v1` atomically stores current/base/EMA weights, Adam moments
and hyperparameters, update cursor, every active game's root/full history/
source index/game ID, termination counters, sample-chain prefix, actor RNG,
Python/NumPy/Torch CPU RNG, runtime, immutable book/config/protocol/initial model
hashes and original source commit. CPU float32 ownership, optimizer shape,
finite state, parameter group ordering and modes are checked. No legacy format
is rewritten. Loading weights cannot satisfy this full resume contract.

The runner requires a declared clean checkout containing the loaded module,
an explicit protocol/config, maximum updates, checkpoint interval and an absolute
deadline. Resume may not alter these limits or extend the deadline. It verifies
every committed compressed journal and its chain, preserves matching regenerated
uncommitted journals, rejects conflicting bytes, atomically publishes new
journals without replacement and retains interrupted temporary bytes. A live
PID/start-time owner blocks concurrent invocation; stale markers never trigger
process termination. Memory>15GiB or free disk<8GiB stop the run. There is no
automatic external engine call or promotion.

## Actual tests

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q \
  tests/test_online_targets.py tests/test_online_objective.py \
  tests/test_torch_online_checkpoint.py tests/test_online_actor.py \
  tests/test_torch_online_learner.py tests/test_online_journal.py
```

Result **74 passed in8.38s / zero skips**, pytest-reported time, not a performance
benchmark. Ruff passed for all14 new source/test files. Companion evidence
embeds their exact bytes and this test output.

Verification includes independent NumPy loss/finite-difference gradients,
actual Torch/MLX CPU autodiff, real draw/history/perspective probes, unknown cap
semantics, fixed-temperature behavior, no mutation on invalid policy input,
all-model/Adam/RNG native checkpoints and fresh-process exact next update.

The real chess learner test uses a small randomly initialized test network:
four actors, full standard/e4 prefixes, legal actual moves, bootstrap/terminal
targets, used-parameter updates, fixed base, EMA tracking and frozen material.
After two updates, a new Python process resumes and reproduces the next four
real rollout batches, every sample/target/loss/hash, all three weight files,
full actor cursor, Adam and every RNG bitwise. These are audit-only updates,
excluded from capacity and future production learning counts.

A separate forced-action mate probe validates terminal precedence and verifies
that EMA is not queried for a terminal child. It is an explicit unit override,
not evidence of a trained network finding mate. Generic checkpoint tests also
use synthetic inputs; they are distinguished from the real trajectory test.
Journal/lock helpers are tested, but the clean-source actual CLI with the real
e8 checkpoint still needs its prospective whole-budget preflight.

Existing full-suite587PASS/54.47s/zero skips predates these modules; no claim that
it is a post-change regression result. Run the updated full suite when the
registered training/arena measurements are quiescent. Apple Metal/CUDA/BF16
remain device-untested; loss parity is not cross-framework optimizer resume.

Next: freeze and execute the real e8/full-train-book short CLI pause/resume and
every-journal/checkpoint audit under one absolute budget. Only after those checks
and a separate prospective controlled-strength plan may production online
learning start. The strong/fast/self-learning chess objective remains unmet.
