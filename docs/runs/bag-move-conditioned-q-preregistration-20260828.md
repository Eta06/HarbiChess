# BAG move-conditioned action-value preregistration

Date: 2026-08-28  
Parent: OLCEK spatial origin-only Q transfer failure  
Decision scope: fresh teacher qualification and move-conditioned representation transfer

## Hypothesis

OLCEK reduced validation Q MSE but failed ranking because its 1×1 spatial head sees only the origin-square trunk feature. BAG scores every action from its canonical origin feature, canonical destination feature, and move-plane identity. This supplies the missing relational inductive bias without returning to DEGER's unstructured 1.2-million-parameter action table.

The shared scorer uses an 8-dimensional embedding for each of 73 move planes, concatenates it with origin and destination trunk vectors, applies one 16-unit ReLU layer, and produces one scalar advantage. The final scalar layer is initialized to zero and the dueling `tanh(V + A)` contract is unchanged. Estimated trainable size is below 5,000 parameters.

## Fresh teacher set

Select 96 train and 48 validation positions with seed `2026082830`, excluding all TERAZI, DOKU, and OLCEK identities. Repeat the unchanged 512/800 clean search, depth-1 teacher oracle, depth-4 full legal verifier, 24/8 workers, batch cap 48, and wait 0.25 ms.

Reuse OLCEK's uncertainty label exactly: visit-weighted Q mean and weight `sqrt(min visits) * max(0, 1 - drift/0.03)`. The teacher gate is unchanged: common support at least 95%, drift-qualified visit mass at least 80%, stable Q/verifier Spearman at least 0.35, conservative min-Q action with positive verified-delta interval, harmful ratio at most 10%, and regret at most 0.10. Bootstrap seed is `2026082831`; failure blocks training.

## Frozen transfer

If the teacher passes, train only the BAG scorer using AdamW `2e-4`, zero weight decay, batch 16, 480 fixed steps, gradient clipping 5.0, seed `2026082832`, and checkpoints `0, 60, 120, 240, 480`.

Reuse every OLCEK/DEGER learner gate unchanged: 20% weighted Q-MSE improvement, teacher-Q Spearman at least 0.35 on supervised actions, positive independent verified top-action interval, harmful ratio at most 10%, regret at most 0.10, top-16 best-action coverage at least 80%, policy/WDL logit delta at most `1e-7`, exact tactical retention, and finite clipped gradients.

No architecture dimension, step count, threshold, sample count, or uncertainty cutoff can change after results appear. A pass authorizes completed-Q search qualification only; learner continuation, arena, generation, and promotion stay blocked.
