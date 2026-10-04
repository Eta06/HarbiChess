# UFUK: rechecking Gumbel value scaling after failed learning

The narrow-root game improvement does not demonstrate model learning. Both the
six-generation policy update and the late-outcome value experiment failed their
played-strength gates. The current width experiment is a separate native
supervised capacity hypothesis; its outcome is pending at this note's snapshot.

Re-read the primary [Mctx Q transform at pinned commit286c37a](https://github.com/google-deepmind/mctx/blob/286c37a82204de2f92e81b2395906fc16bff98d9/mctx/_src/qtransforms.py).
Downloaded7310bytes, SHA256
`71b827f5b32b2edfd2ec67ed90f94b2d37f87ab5eb44b3290702ed178e354c99`.
The exact source and HTTP receipt stay in the dated research workspace and will
be included in the next text evidence package. This is source inspection, not a
local JAX execution or a new game-strength measurement.

Mctx's default completion uses the raw value and prior-weighted visited Q,
weighted by the total child visits. It rescales completed Q by its current
min/max range with1e-8 denominator floor, then multiplies by
`(maxvisit_init + maximum_child_visits) * value_scale`, defaults50 and0.1.
HarbiChess uses the same mathematical mechanism with alternating-player signs.
The upstream tiny-prior floor prevents underflow in prior-weighted averaging;
our zero-visited-prior fallback is a numerical edge difference, not an evaluated
strength gain. The upstream API also permits `rescale_values=False`; HarbiChess's
current public FullGumbel config does not expose that ablation.

This normalization removes absolute Q spread: for example, range0.002 and
range0.8 each span approximately5 after scaling when the visit offset dominates.
With inaccurate shallow values, relative rankings can therefore receive strong
policy influence even when the absolute differences are small. This is an
algebraic observation; no claim that it caused our failures or that disabling
normalization will improve chess. Increasing value_scale is not automatically
more useful search. The accurate-action-value assumption behind policy
improvement remains material.

The previously read [KataGo methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
offer related mechanisms: policy-surprise sampling increases learning weight on
large search/prior changes; short-horizon targets and an error head estimate
uncertainty; subtree bias correction shares relative evaluation errors rather
than absolute values. Reported Go Elo gains are author measurements, not
HarbiChess evidence. Local atari/pattern buckets cannot be transferred blindly
to chess, whose line tactics are nonlocal.

After the current width result, a bounded next hypothesis is a Q-range-floor or
upstream-style no-rescaling ablation, first against existing all-legal independent
reference scores and then fresh paired games if qualified. Preregister exact
arms, family bootstrap/multiple-comparison treatment, effect thresholds, wall
and NN-call budgets before running. Do not change the historical default or
reuse diagnostic positions for gradients. If it improves search, separately
test whether new targets produce a final-versus-initial model gain under the
same search; a better selector is not already successful self-learning.

Policy surprise alone does not certify target quality: a large KL can mean an
incorrect shallow search. A surprise-weighting follow-up needs an unchanged
uniform-sampling control and native retention/heldout/game gates. Late actual
outcomes are conditional on the weak generating policy; native Stockfish WDL
labels estimate another policy/search context. One must record that difference,
not blend them and call them identical supervision.

Laya remains a deferred external reference. The choice-scoring mathematics is
relevant, but a421M textual backbone is not justified in each CPU search node.
An eventual small compute controller must beat fixed-budget/simple-rule controls
including its own latency. Conventional Gumbel controls, residual widening,
surprise weighting or uncertainty heads are not claimed as our invention.
Publishable novelty would require a reproducible useful contribution, matched
compute and independent confirmation; neither this note nor a successful
development gate establishes that.
