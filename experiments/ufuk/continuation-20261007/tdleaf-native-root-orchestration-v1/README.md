# TDLeaf H0 native orchestration (source-only)

This directory provides a ROOT-invoked wrapper around the already closed
TDLeaf v4 self-play receipts. It preserves the original actor registrations,
receipts, 1,024 selected actions, current H0 parent, TDLeaf lambda 0.5 target,
and the existing native proof/fresh-fit implementation. It does not make a
strength claim.

The wrapper has four one-shot phases for each seed: `convert`, `six`, `proof`,
and `fresh-fit`. Each phase records its own first-start/deadline before work;
failed or expired phases cannot be resumed with a new clock. Proof is capped
at 600 seconds and fresh fit at 1,800 seconds, also bounded by the registered
operator end. Fresh fit requires passing proof receipts for both seeds. The
phase-registration `inputs` field is a flat mapping from canonical full path
to `{path, sha256}`; the contract still retains its complete nested raw input
structure, including semantic lists and strings.

ROOT commands (the ROOT operator supplies the original actor directory and
chooses available CPU cores):

```sh
python controller.py --root /workspace/work/harbichess/continuation-20261007/TDLeaf-collection-v4-actual-20262905 --seed 20262905 --core 0 --phase convert
python controller.py --root /workspace/work/harbichess/continuation-20261007/TDLeaf-collection-v4-actual-20262905 --seed 20262905 --core 0 --phase six
python controller.py --root /workspace/work/harbichess/continuation-20261007/TDLeaf-collection-v4-actual-20262905 --seed 20262905 --core 0 --phase proof
python controller.py --root /workspace/work/harbichess/continuation-20261007/TDLeaf-collection-v4-actual-20262905 --seed 20262905 --core 0 --phase fresh-fit
```

Repeat for seed `20262906` with a ROOT-selected CPU. Do not start `fresh-fit`
until both seed proof controller records pass. All generated datasets, native
payloads, and phase outputs are required to stay under `/dev/shm`; compact
registrations and receipts are published once under the continuation record
directory. The known-160 adapter is a separate later read-only admission step
that requires both complete fresh-fit bundles and replays the raw collection
conversion before accepting model refs.

The original producer inventory is pinned at
`750b2f64e4261704138d208a5dbd96a649d159942ca3462d196c032a3648481`. The
integration derivative's original inventory remains
`a3f754a966820abe03c4c98de35286adc447ce9a84b9699ab379da890793c44b`; this
orchestration inventory separately binds the files in this directory and
records that derivative as its ancestor. No conversion, six-PV audit, native
proof, fresh fit, or arena was executed from this orchestration directory.
