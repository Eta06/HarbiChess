# UFUK: measure immutable target entropy before more capacity experiments

Before computing these new statistics. Narrow policy model learning failed and
selected late-outcome value candidate also failed its game-strength gate. Earlier
depth/adapter representation gaps remain failed; do not rewrite those outcomes.

Hypothesis: native soft-target entropy accounts for a substantial part of the
observed cross entropy. If excess CE over target entropy is small, requiring an
absolute0.03CE architecture improvement may be unattainable or disproportionate;
this must be measured, not assumed. Conversely a large excess can justify
capacity/optimization work. This is label/loss interpretation, not playing strength.

Use exact frozen native20k observer cacheSHA33fbb59df6b400819ccacc4393c660c8ff2de1c5e8a6328736ef962c8bdb7039
and existing balanced teacher dataset b34a3870b494b4cc0fe8a1df87960fe5e2406a5c9a2ed341d1b93fed0e0e094f;
no gradients, new labels, model changes, external engine queries or candidate
selection. Native original18f cached/evaluated CE2.632099348 policy and0.654349148
WDL. Compute mean perrow H=-sum(p*log(p)) in float64, probability normalization,
observed CE-minus-H, attainable absolute improvement bound, and perrow entropy
distribution. CE=H+KL for normalized soft targets with full legal support; report
floating precision and sign tolerances, avoid equating this bound with Elo.

Also inspect native data policy construction/temperature and recorded score
semantics to interpret why labels are sharp or diffuse. Do not reinterpret
observed weak-policy terminal returns as optimal teacher WDL. Existing failures
and label provenance remain unchanged. This informs a later new preregistration,
not a retrospective gate change or a new success claim. CPU1/300s/current resources.
