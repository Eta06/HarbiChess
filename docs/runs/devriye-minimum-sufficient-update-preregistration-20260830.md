# DEVRIYE minimum-sufficient update preregistration

## Evidence from run 03

`devriye-continuous-pilot-20260830-03` accepted two sequential latest-network updates. Update 3
passed every numeric gate, improved fresh teacher CE `2.51946 -> 2.39428`, preserved WDL and scored
`56.25%` against update 2, but the selected step-40 checkpoint lost `forced-defense-a`. Full Gumbel
still solved 5/8 because it gained a different case, so the no-lost-case tactical guard correctly
rolled the update back.

The selection rule always minimized teacher CE among eligible checkpoints and therefore preferred the
largest policy drift. In a continuous learner, once the preregistered minimum improvement is reached,
additional within-update drift has no demonstrated benefit.

## Frozen correction

- Keep the same 40-step maximum, validation every 10, optimizer continuation, learning rate, batch,
  rolling buffer, target/search compute, WDL training, and every gate.
- Select the earliest checkpoint that passes all numeric policy/WDL gates. Later checkpoints cannot
  replace it merely for lower teacher CE.
- Run continuation, tactical, and mini-arena once on that selected checkpoint. No downstream retry.
- Repeat on fresh targets/openings with seed `2026083301`.

This is a checkpoint policy correction, not reduced required quality: the same `0.01` policy CE gain,
top-action, WDL, material, continuation, tactical, and arena thresholds remain unchanged. Passing all
three updates still requires the frozen final MIHVER comparison; promotion remains unauthorized.
