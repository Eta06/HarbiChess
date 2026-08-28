# AKTARIM train-only fit diagnostic preregistration (2026-08-28)

## Purpose

Separate adapter optimization/capacity underfitting from target quality without
using validation to choose hyperparameters.

## Frozen matrix

All arms use the qualified SIPER target, VERI train partition only, batch 16,
480 steps, checkpoints 0/60/120/240/480, seed `2026082839`, zero weight decay,
and identical sampling:

- rank 8, learning rate `2e-4` (control)
- rank 8, learning rate `1e-3`
- rank 32, learning rate `2e-4`
- rank 32, learning rate `1e-3`

An arm is train-fit capable only if cross entropy improves at least 5%, teacher
Spearman is at least 0.35, verified selected-action gain has a positive 95%
lower bound, harmful ratio is at most 10%, and mean regret is at most 0.10.
Selection uses the smallest passing rank, then the lower passing learning rate.
Validation metrics are not selection inputs.

This diagnostic cannot authorize a candidate. The chosen configuration, if
any, must be preregistered and tested on a completely fresh teacher validation
set before search qualification. If no arm fits the train target, the next
decision is representation redesign rather than more data or longer training.
