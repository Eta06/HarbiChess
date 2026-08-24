# Model quality measurement

Training loss is a diagnostic, not the promotion criterion. HarbiChess measures
playing strength in a candidate-versus-champion arena where colors alternate
and both models receive the same search budget and opening distribution.

Each win, draw, and loss contributes `1`, `0.5`, and `0` points. The dashboard
reports the resulting score rate and converts it to an Elo difference:

```text
score = (wins + 0.5 * draws) / games
elo = 400 * log10(score / (1 - score))
```

A 95% confidence interval is estimated from the observed game-score variance.
A candidate is marked promotion-ready only after the configured game budget is
complete and the interval's lower Elo bound clears the promotion threshold.
This prevents a lucky short match from replacing the champion.

The telemetry history is capped at 240 points and should be sampled at a low
frequency. It carries Elo and its confidence band, total loss, games/hour,
positions/second, training step, elapsed training time, and lifetime game count.
The browser renders these points locally; it does not query or block the trainer.

For a production arena, openings must be paired with reversed colors, search
settings must be identical, and draws/timeouts/crashes must be recorded. The
normal interval implemented now is suitable for the first pipeline. A later
evaluation phase can add SPRT or bootstrap intervals when enough real match data
exists to calibrate promotion risk.
