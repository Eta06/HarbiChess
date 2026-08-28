# KOPRU confidence-aware policy transfer preregistration

The qualified clean teacher and fresh replay are retained, but the step-30 learner is held out of
arena. Although legal teacher cross-entropy improved from 2.77165 to 2.68985, overall teacher
top-action agreement fell from 40.71% to 27.49%. On the 1,196 validation records whose 64-visit
teacher top-action margin was at least 0.20, agreement fell from 64.30% to 37.88% and selected
teacher mass fell from 0.418 to 0.271. A matched policy-only run reproduced the regression, ruling
out WDL-gradient interference as the primary cause.

The next diagnostic keeps the replay, sample order, learning rate, batch size, and 30-step compute
fixed. It adds a hard top-action auxiliary cross-entropy only for examples whose stored clean teacher
margin is at least 0.20. The existing soft visit-policy cross-entropy remains unchanged. Four weights
are fixed before results: `0.0`, `0.25`, `0.5`, and `1.0`; all runs are policy-only so value-head
movement cannot confound the policy-transfer result.

A variant is useful only if it improves legal teacher-policy cross-entropy by at least 2% versus the
baseline while preserving overall teacher top-action agreement and preserving agreement on the
margin-at-least-0.20 subset. Raw-policy tactical solves must not regress. This is a diagnostic only:
it cannot authorize arena, promotion, or a new generation. If no variant passes, the next focus is
policy representation/capacity or a longer metric-independent optimization study, not target-weight
tuning on these results.
