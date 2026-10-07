from pathlib import Path
p=Path(__file__).resolve().parent
f=p/'root_bank.py';s=f.read_text()
s=s.replace('def generate(seed, protected, recipe=None, guard=lambda: None):','''class BankExhausted(ValueError):
    def __init__(self, selection, trace):
        super().__init__("attempt cap exhausted; no relaxation or alternate seed")
        self.selection, self.trace = selection, trace


def generate(seed, protected, recipe=None, guard=lambda: None):''')
s=s.replace("    if len(rows) != recipe['count']:\n        raise ValueError('attempt cap exhausted; no relaxation or alternate seed')\n    selection =",'    selection =')
s=s.replace('    return selection, trace\n\n\ndef verify', "    if len(rows) != recipe['count']:\n        raise BankExhausted(selection, trace)\n    return selection, trace\n\n\ndef verify")
s=s.replace("    selected, trace = generate(receipt['seed'], protection(receipt['protected_aliases']), guard=guard)",'''    if (receipt["teacher_labels_used"] is not False
            or not receipt["first"] < receipt["finished_epoch"] <= receipt["deadline"]
            or receipt["deadline"] > min(receipt["first"] + 600, receipt["operator_end_epoch"], 1791448916.685839)):
        raise ValueError("complete prospective procedural source clock")
    registration = json.loads(pinned(receipt["registration"]).read_bytes())
    if (registration["schema"] != "human-procedural-root-bank-registration-v2"
            or registration["status"] != "registered"
            or any(registration[k] != receipt[k] for k in
                   ("seed", "recipe", "generator_sha256", "protected_aliases", "first", "deadline", "operator_end_epoch"))):
        raise ValueError("exact ROOT source registration")
    selected, trace = generate(receipt['seed'], protection(receipt['protected_aliases']), guard=guard)''')
s=s.replace("    selection, trace = generate(spec['seed'], protection(spec['protected_aliases']), guard=guard)",'''    try:
        selection, trace = generate(spec['seed'], protection(spec['protected_aliases']), guard=guard)
    except BankExhausted as error:
        with (directory / "failed-attempts.jsonl").open("xb") as f:
            f.write(b"".join(canonical(x) for x in error.trace))
        with (directory / "failure.json").open("xb") as f:
            f.write(canonical(dict(schema=RECEIPT, status="FAIL-attempt-cap", seed=spec["seed"],
                                  accepted_count=len(error.selection["rows"]), attempt_count=len(error.trace),
                                  recipe=RECIPE, first=first, deadline=end, operator_end_epoch=op)))
        raise''')
s=s.replace("protected_aliases=spec['protected_aliases'], registration_sha256=hashlib.sha256(canonical(spec)).hexdigest(),", "protected_aliases=spec['protected_aliases'], registration=ref(directory/'registration.json'),")
s=s.replace('    directory.mkdir(parents=True, exist_ok=False)\n    def guard():', '''    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "registration.json").open("xb") as f:
        f.write(canonical(spec))
    def guard():''')
f.write_text(s)
f=p/'metadata_factory.py';s=f.read_text().replace('selected, bank = verify(spec["procedural_bank_receipt"], spec["protected_aliases"])','''def guard():
        if time.time() >= end:
            raise TimeoutError("original collection source-validation deadline")

    selected, bank = verify(spec["procedural_bank_receipt"], spec["protected_aliases"], guard)''');f.write_text(s)
f=p/'run_collection.py';s=f.read_text().replace('    for name in ("collector.py",', '''    if reg["search_helper"]["sha256"] != "de53c14728a67b7772f18b396ac8ef5c35a144d4e4e616fec40099cd461a6670":
        raise ValueError("original de53 search only")
    for name in ("collector.py",''');f.write_text(s)
