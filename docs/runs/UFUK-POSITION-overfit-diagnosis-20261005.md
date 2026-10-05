# UFUK-POSITION internal calibration diagnosis

Registered post-hoc diagnosis, not new training, endpoint selection, optimal-play evaluation or strength success. Source and all immutable model/journal SHA bindings are in the diagnostic protocol and complete receipt. Each of nine fixed checkpoints per seed was read; no old result changed or promoted.

|Seed|Step|Game-equal train NLL|Game-equal validation NLL|
|---|---:|---:|---:|
|20262405|0|1.09861|1.09861|
|20262405|512|0.19989|1.39921|
|20262405|1024|0.08847|1.90823|
|20262405|1536|0.05532|2.33808|
|20262405|2048|0.03917|2.59620|
|20262405|2560|0.03321|2.79307|
|20262405|3072|0.02706|3.05348|
|20262405|3584|0.02260|3.17859|
|20262405|4096|0.02134|3.35755|
|20262406|0|1.09861|1.09861|
|20262406|512|0.19859|1.42569|
|20262406|1024|0.08830|2.09319|
|20262406|1536|0.05227|2.51178|
|20262406|2048|0.03946|2.95564|
|20262406|2560|0.03340|3.13469|
|20262406|3072|0.02475|3.31660|
|20262406|3584|0.02245|3.56361|
|20262406|4096|0.01997|3.68735|

Both validation curves worsen by step512, while train curves keep decreasing. This supports an overfit diagnosis for this model/data/configuration, not a causal explanation of every historical failure or a universal failure of positional critics. Prior selected4096 endpoint and160game FAIL remain unchanged. A separately prospective lower-capacity spatial critic with positional shrinkage is a next hypothesis; not implemented/trained/successful at publication. GPU remains revoked, CPU only; no new teacher queries.
