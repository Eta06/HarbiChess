from pathlib import Path
p=Path(__file__).resolve().parent
# Change source/distribution protocol identities, never numerical native/model phase identities.
names=['collector','run_collection','metadata_factory','convert','contracts','specs','audit_collection_six','prepare_collection','ledger']
identities=['collection-build-seal','collection-registration','collection-receipt','ownq-labels','own1024-dataset-conversion-seal','own1024-data-provenance','own-contract-build-seal','own-registry-plan','own-training-orchestration','own-replay-audit-set','own-six-root-audit-result','own-six-root-audit-clock']
for n in names:
    f=p/(n+'.py'); s=f.read_text()
    for k in identities:
        old='human-prior-'+k+'-v1'
        new='human-randomstarts-'+k+'-v2'
        s=s.replace(old,new)
    s=s.replace('/dev/shm/harbichess-human-prior-ownq-v1','/dev/shm/harbichess-human-randomstarts-ownq-v2')
    f.write_text(s)
f=p/'collector.py';s=f.read_text().replace('human-prior-own-train-roots-v1','human-randomstarts-train-roots-v2').replace('ancestral_teacher_labels_sha256','procedural_receipt_sha256');f.write_text(s)
f=p/'metadata_factory.py';s=f.read_text(); start=s.index('    source = json.loads');end=s.index('    protected = pinned',start)
s=s[:start]+'''    from root_bank import verify

    selected, bank = verify(spec["procedural_bank_receipt"], spec["protected_aliases"])
    selection_ref = bank["selection"]
    pool = dict(
        schema="human-randomstarts-train-roots-v2", selection_status="pass", train_only=True,
        source_selection_sha256=selection_ref["sha256"],
        procedural_receipt_sha256=spec["procedural_bank_receipt"]["sha256"],
        rows=selected["rows"],
    )
    pool_path = Path(spec["root_pool_output"])
    publish(pool_path, pool)
'''+s[end:]
s=s.replace('ancestral_teacher_labels_sha256=pool["ancestral_teacher_labels_sha256"]','procedural_receipt_sha256=pool["procedural_receipt_sha256"],\n            procedural_bank_receipt=spec["procedural_bank_receipt"]')
s=s.replace('ancestral_pool=spec["ancestral_root_pool"],','procedural_bank_receipt=spec["procedural_bank_receipt"],')
s=s.replace('("collector.py", "run_collection.py", "metadata_factory.py")','("collector.py", "run_collection.py", "metadata_factory.py", "root_bank.py")')
s=s.replace('    "90c85ba8e3401cba57c82e6c2a34764d25aa999a03b244a4f2838124a8b46357",\n','')
f.write_text(s)
f=p/'run_collection.py';s=f.read_text();start=s.index('            "ancestral_selection_path"');end=s.index('            "selected_root_order_sha256"',start)
s=s[:start]+'''            "procedural_selection_path": reg["root_pool"]["selection_path"],
            "procedural_selection_sha256": reg["root_pool"]["selection_sha256"],
            "procedural_bank_receipt": reg["procedural_bank_receipt"],
            "source_selection_sha256": reg["root_pool"]["selection_sha256"],
'''+s[end:]
# Additional helper must be hashed on the runtime boundary.
s=s.replace('("collector.py", "run_collection.py", "metadata_factory.py")','("collector.py", "run_collection.py", "metadata_factory.py", "root_bank.py")')
# verify before model or search module loading
marker='    evaluator, m, n = parent(reg)'
print('parent call marker',marker in s)
f.write_text(s)
f=p/'convert.py';s=f.read_text();start=s.index('    original_selection = json.loads');end=s.index('    from parent_bridge',start)
s=s[:start]+'''    from root_bank import verify

    original_selection, bank = verify(reg["procedural_bank_receipt"], reg["protected_aliases"])
    if (
        bank["selection"] != selection_ref
        or pool["source_selection_sha256"] != selection_ref["sha256"]
        or receipt["procedural_selection_path"] != selection_ref["path"]
        or receipt["procedural_selection_sha256"] != selection_ref["sha256"]
        or pool["procedural_receipt_sha256"] != reg["procedural_bank_receipt"]["sha256"]
        or receipt["procedural_bank_receipt"] != reg["procedural_bank_receipt"]
    ):
        raise ValueError("immutable rule-only procedural TRAIN source")
'''+s[end:]
f.write_text(s)
f=p/'prepare_collection.py';s=f.read_text().replace('        ancestral_root_pool=ref(a.ancestral_root_pool),\n        ancestral_selection=ref(a.ancestral_selection),','        procedural_bank_receipt=ref(a.procedural_bank_receipt),').replace('        "ancestral-root-pool",\n        "ancestral-selection",','        "procedural-bank-receipt",');f.write_text(s)
f=p/'contracts.py';s=f.read_text().replace('            "audit_collection_six.py",','            "audit_collection_six.py",\n            "root_bank.py",').replace('        phase="own-learning",','        phase="own-learning",\n        root_source_type="procedural-uniform-legal-walk-v2",');f.write_text(s)
