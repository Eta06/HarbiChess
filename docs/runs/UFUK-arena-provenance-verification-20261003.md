# UFUK self-play arena provenance verification: before replay

The original generation3 arena runs at declared source43376c3. During its CLI
sequence the optional compiler adapter was edited in the shared working tree.
Default inference still calls the identical eager numerical method, but source
commit alone does not capture that dirty adapter. This is a provenance flaw;
do not silently rewrite historical source fields or choose a nicer rerun score.

Preserve all original outputs and the dirty adapter patch snapshot. Repeat exactly
all3arms/48games, same fixed model weights,24frozen roots/colours,seed20261006,
16simulations/Gumbel0/240plies andSF32/one thread/Hash16MiB, in a detached **clean**
43376c3 worktree. Same existing environment; explicit PYTHONPATH selects pinned
source, models/outputs are absolute paths. No new paid resources, no source edits
in that checkout.900s ceiling/arm. Both numerical paths have the same default math.

Verification requires all144complete move histories, openings, colours, endings
and scores to match originals exactly. Original scores remain the strength test;
reproduction is integrity evidence, not an additional sample or model selection.
Any mismatch rejects the claimed source equivalence and strength qualification;
keep both sets, diagnose, no threshold change or repeat until lucky. Preserve the
initial preflight command that ran before the audit file existed: it refused before
playing and created no game results. This was a sequencing error, not a chess result.
