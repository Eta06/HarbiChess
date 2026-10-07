from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['collector','run_collection','metadata_factory','convert','contracts','specs','audit_collection_six','prepare_collection','ledger']:
 f=p/(name+'.py');s=f.read_text()
 for identity in ['own-collection-build-seal','own-collection-registration','own-collection-receipt','own-closed-ledger-seal','own-closed-ledger']:
  s=s.replace('human-prior-'+identity+'-v1','human-randomstarts-'+identity+'-v2')
 for category in ['data','contracts','proof','fit']:
  s=s.replace('harbichess-human-prior-own-'+category+'-v1','harbichess-human-randomstarts-own-'+category+'-v2')
 f.write_text(s)
f=p/'run_collection.py';s=f.read_text();s=s.replace('    guard()\n    out.mkdir', '''    from root_bank import verify

    selected, bank = verify(reg["procedural_bank_receipt"], reg["protected_aliases"], guard)
    pool = json.loads(root.read_bytes())
    if (pool["rows"] != selected["rows"]
            or pool["source_selection_sha256"] != bank["selection"]["sha256"]
            or pool["procedural_receipt_sha256"] != reg["procedural_bank_receipt"]["sha256"]):
        raise ValueError("exact procedural pool before model/search loading")
    guard()
    out.mkdir''');f.write_text(s)
# Every next-phase registry binds entire bank source closure via raw input reference maps.
f=p/'specs.py';s=f.read_text().replace('            registration=ref(a.registration),','            registration=ref(a.registration),\n            procedural_bank_receipt=reg["procedural_bank_receipt"],');f.write_text(s)
f=p/'convert.py';s=f.read_text().replace('    original_selection, bank = verify(', '    if spec["procedural_bank_receipt"] != reg["procedural_bank_receipt"]:\n        raise ValueError("explicit procedural source seal")\n    original_selection, bank = verify(');f.write_text(s)
