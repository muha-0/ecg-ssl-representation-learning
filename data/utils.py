from pathlib import Path

def list_patients(root: Path):
    # patient folders like p00/p00000
    patients = []
    for sub in sorted(root.glob("p*/p*")):
        if sub.is_dir() and sub.name.startswith("p"):
            patients.append(sub)
    return patients

def list_record_bases(patient_dir: Path):
    heads = sorted(patient_dir.glob("*.hea"))
    recs = []
    for h in heads:
        base = h.with_suffix("")
        if base.with_suffix(".dat").exists() and base.with_suffix(".atr").exists():
            recs.append(base)
    return recs