# Ancestry-conditional confirmation: version 2 proposal only

**Decision required.** This version corrects v1's comparator and interval plan. V1 remains unchanged at SHA `755e20206c952a19452d3efb7e44580b7094f893cedd2b43f22beba61e39113c`. Nothing here amends a protocol, authorizes books/games, or resolves the missing source6/source8 archive gap.

## Controlling scope and decision

The confirmation review memo `/workspace/work/harbichess/classical-independent-confirmation-proposal/protocol-review-v2.txt` (SHA `b6e58037e2c579a2254f4b1510281869bef5a535c277b81fd9788228c513a325`) says: “Exclude ALL historical arena/teacher/teacher-source roots, all registered heldouts (including unused6/7/8), all observed own train/dev/preflight positions including both final journals’ UNKNOWN and active tails, all NN-MC/SC/FULL observed data and curriculum roots, and known8 development roots.” It requires “zero-root/state-overlap proof before games” against the “complete observed-history union.” The review says source6/8 remain blockers. The frozen prereg separately requires “Allformal2/old685k training/teacher/arena/fullprefix exclusions retained”; the closure memo adds source6 original900 and source8 whole/split/partial records. This is a prospective review, not a signed registration. Changing this scope requires user approval.

Until approval, retain the global INCOMPLETE finding. If approved, the claim becomes conditional: *incremental learned improvement of one fixed descendant against its exact frozen ancestor, at the registered search budget and on the newly scoped roots*. Do not call it globally independent, globally virgin, general Elo, or free of historical teacher influence.

## One candidate and two distinct parent paths

Before book selection, freeze exactly one candidate recipe and two independently seeded final checkpoints. Candidate selection uses only fixed development eligibility/lineage evidence; no new confirmation outcomes. There is no fallback after an incomplete or failed campaign.

- **Fresh human-prior → own learning:** name/hash the manual human-prior checkpoint and all PGN/game/site/alias inputs; child updates from its own completed games and own-search data. Select exactly one of OWNQ, C80, or selective-Q. Compare against that exact human-prior parent and against E8.
- **Teacher-once NNUE → own-only updates:** name/hash the NNUE parent and every teacher model/label/source input; after initialization the child uses only its own outcomes/search experience. “Own-only” applies to the child updates, not to its teacher-derived ancestry. Compare against that exact NNUE parent and against E8.

These paths are not pooled replications. The human-prior parent and teacher-once NNUE parent have different ancestors; preregister one path only. E8’s use solely as an arena comparator does not make it a training ancestor, but any E8 warm-start, anchor, label, or data makes that explicit in the DAG. Freeze code, data, optimizer initialization, train/validation split, both final endpoints and evaluation/search source before books.

## Exposure boundary and honest book claim

Hash-bind the chosen DAG’s parent/teacher or human-PGN sources and aliases; all actual lineage played/train/validation/proof/dev, completed/UNKNOWN/capped/tail states; model/evaluation roots; logged search nodes; and known8 roots/prefixes. UNKNOWN is not a WDL label but is exposure. Exclude trajectories touching protected states. Remove already viewed known8 roots/prefixes from books and validation; never inspect outcomes.

Log each new search evaluation’s full history/FEN4, trajectory/source, phase and model SHA; reconcile against counters/RNG. Missing source6/8 roots, expansions and member maps provide no finite, verified neighborhood for unlogged leaves; a guessed ply radius is not coverage. Thus **globally virgin** remains BLOCKED until recovery. Under an approved conditional claim, disclose source6/8 and inherited-E8 as unresolved; if the selected parent’s search trace is incomplete, do not claim full-lineage disjointness.

Only after the candidate and amendment are frozen: select two source-block books, 48 roots per seed, both colors, ECO-matched under existing time-control/source/legality rules. Seed 2 excludes seed 1 source IDs, aliases, roots and full prefixes. Independently replay/audit each PGN before games. No model/engine queries during selection, no result peeking, no source relaxation or reselection. Shared START/opening ancestors and unknown inherited exposure remain disclosed limitations.

## Preserve all strength gates; strengthen only joint confidence

Retain the original arms and thresholds. For **each** seed require both E8-primary and exact-parent comparisons:

- Candidate vs E8 direct point score `>0.60`, with its registered direct lower bound `>0.50`.
- Candidate’s paired SF512 gain over E8 `>0.10`, with lower bound `>0`.
- Candidate vs its exact parent direct point score `>0.60`, with lower bound `>0.50`.
- Candidate’s paired SF512 gain over that parent `>0`, with lower bound `>0`.
- Candidate absolute SF512 score `>=0.25`; caps `<=0.05` in every arm; same adverse-cap recoding gates; final/parent median latency `<=1.10`.

Keep registered game counts, both seeds/colors, Q512/q2/depth/search, SF controls, 50,000 source-root bootstrap and no pooled rescue. E8 remains primary; the parent arm does **not** replace it.

The new campaign’s joint alpha ceiling is `.00625`; old+new disclosure stays `<=.05625`, not global `.05`, with no refund. Original 98.75% bounds do not give simultaneous `.00625` coverage across the added tests. Bonferroni over **eight** lower bounds (four comparisons × two seeds) gives one-sided alpha `.00625/8 = .00078125` each; symmetric two-sided intervals use total alpha `.0015625` (99.84375%). Keep thresholds unchanged. Freeze bootstrap/blocking/CI implementation before books; include any extra test in K and use `.00625/K`. This is stricter inference, not a gate change. The earlier memo’s six-test IUT bound is not this simultaneous-interval claim.

## Limits and required user choice

This estimates improvement conditional on the named parent/books, not zero global exposure, broad strength or independent discovery. Source overlap, completion/selection, unknown E8 pretraining and unlogged leaves limit it. Human-prior and teacher-once paths are distinct lineages.

Choose explicitly: **(A)** keep the current complete-global-exposure rule and remain BLOCKED pending source6/source8 artifacts and reconciliation; or **(B)** approve the ancestry-conditional claim above, including its unresolved exposure disclosure and stricter eight-bound confidence rule. Until choice B is authorized and the one-candidate inventory is frozen, no books or confirmation are authorized. No old MAX8 reopening, endpoint substitution, gate changes, alpha reset, or repeat-after-failure is proposed.
