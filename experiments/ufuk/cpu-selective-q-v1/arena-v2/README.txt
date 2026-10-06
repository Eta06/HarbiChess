SELECTIVE-Q 192-GAME DEVELOPMENT ADAPTER — UNEXECUTED

ROOT steps, after actual48 fits complete:
1 Freeze fresh48 registrations/current native0/native48/candidate/result files.
   Required layouts output /dev/shm/harbichess-selective-q-fit-v1/<seed>/
   initial.json, native.json, candidate.json, result.json, exactly originaltrain.
2 python <stage>/seal.py --draft-sha256 <verification draft hash>
   --fit05-registration <actual fresh48 reg> --fit05-registration-sha256 <SHA>
   --fit06-registration <actual fresh48 reg> --fit06-registration-sha256 <SHA>
   --profile-first <NEW first> --arena-first <NEW arena first>
   --output <stage>/protocol.json
   Seal accepts only completed48, unchanged source/dataset/split and BOTH actual
   proof6 fresh native receipts; strict all-model/Adam/global+samplerRNG roundtrip.
   No trainingclock resets, no weights-only admission. Output publish-once.
3 Actualtrained24 profile (new600): python <stage>/profile.py --protocol
   <stage>/protocol.json --output <NEW>/profile.json --first-epoch <profile first>.
   Fixed2seeds x learned/ORIGINALfastprior/E8 x4knownroots. Sum4 actual search
   wall ratios perseed against each ORIGINALcontrol <=1.10. Newzero not falsely
   substituted as latency baseline. Whole includes admission/model construction.
4 python <stage>/run.py --checkout <cleanCORE6fcc> --study <stage>
   --qualification <profile> --output <NEW> --first-epoch <arena first>.
   Exactly2seeds x6arms x8knownopeningfamilies x2colors=192, newwhole7200.
   Arms learnedSF,newzeroSF,oldpriorSF,E8SF,learnedE8,learnednewzero.
   Same SF512/8knownbook/E8/prior inputSHA and unchanged developmentgates.
5 Independent read-only fullhistory audit900 via audit_arena_v2.py --protocol
   <protocol> --arena <NEW> --output <NEW/audit.json>. ROOT owns externaldeadline.
   Records search-only newzero-minus-oldprior SF contrast separately; learning
   gates use learned vs IDENTICALnewzero controller, not originaloldprior.

Run ROOT CLI cwd cleanCORE6fcc with PYTHONPATH=<cleanCORE>/src, CPU1 env.
Resources guardian15GiB, diskfloor256MiB, noGPU; clocks<=1791273600.
Arenafirst>=profilefirst and arenaend+900<=operatorend. ROOT may freeze future
arenafirst but must wait until it starts; owner rejects early/future/expiredstart.

Exact semantics: effort16 controls optional extra2ply q-search up to64 nodes
per selectedhorizon WITHIN globallycharged512, never externalunbillednodes.
Both newzero/learned extend IN-CHECK horizons mandatorily; bounded incomplete
check evasion censors unfinished pass. Newzero sigmoid.5, strict>.5 means no
optional uncheckedextension. All scalar values unchanged humanprior18.
Original prior/E8 retain old originalBudgetSearch. Genericwrappedcontroller
resets only diagnostic extensionreceipt list per search; algorithm/RNG unchanged.
Ledger records per-move controller and completed/censored extension node slices.
Audit rejects >64 or extension sums exceeding actual globalnodes; source pins
bind the checked-horizon algorithm. It does not re-run planning or infer which
optional states a classifier would select; full search reconstruction is outside
this independent rules-only audit. Pure source/value changes neverclaimednovel.

All preparation tests are static/fake-routing/receiptcorruption. No actual
forward/search/SGD/game/job/network/sourceMAIN changes performed by agent.
Final registration/source/helper hashes must be frozen by ROOT beforeexecution.
