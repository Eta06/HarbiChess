# UFUK value checkpoint qualification: apply the precommitted selection rule

The500update candidate **failed native retention and stopped**, unchanged:
QMAE0.358984565 exceeds initial0.336882945+0.02. Native CE0.683131530 is
within its separate0.03 tolerance. The500checkpoint is invalid and not promoted.

The original preregistration committed before any update explicitly says:
"Select minimum heldout game-balanced observed valueCE among valid checkpoints."
It requires candidate gain>=0.10/gap over anchor control>=0.05, native retention
and exact frozen policy. It does not require every later checkpoint to stay valid.

The driver result added an unregistered whole-run-completed veto and reported
qualificationfalse despite a valid selected checkpoint. Preserve that original
false report/failed500status; create a separate selection audit applying the
original checkpoint rule, not new thresholds or an extension. This correction is
after measurement and explicitly documented; it is not a successful500run.

Selected candidate250: heldout game-balanced CE1.063226538 versusinitial/control
best0 CE1.326842336, gain and control gap0.263615799. Native CE0.667610130
(delta0.013260981<=0.03), QMAE0.349700419(delta0.012817474<=0.02).
Both criteria pass. Policy/shared tensors unchanged and actual resume/cache/
matching sample+TorchRNG traces must pass independent integrity audit before arena.

Use only this already selected250candidate and anchor selected0control, with
the fresh32ply book frozen before updates, same16sim/max4/Gumbel0 search.
The three48game arms and their strength/cap/statistical gates remain exactly
as preregistered. No further training, alternate seed or candidate selection.
Heldout CE qualification is not playing strength;500retention failure remains
a material limit on sustainable value learning. Main objective is still unmet.
