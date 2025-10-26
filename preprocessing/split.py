# Split DAIC-WOZ .pt files into train/validate/test folders based on CSVs
# Run in Python 3 (Jupyter or script)

from pathlib import Path
import csv
import shutil
import re

# ====== CONFIG: set your paths here ======
BASE_DIR = Path("/media/popsatorn/timeshift_backup/DAIC-WOZ/DAIC_embeddings")
CSV_DIR  = Path("/home/popsatorn/depressionFinalProject/label")  # folder that has train.csv/validate.csv/test.csv

SPLIT_CSV = {
    "train":   CSV_DIR / "train.csv",
    "validate":CSV_DIR / "validate.csv",
    "test":    CSV_DIR / "test.csv",
}
ID_COL_NAME = "Participant_ID"   # case-insensitive; falls back to first column if not found
# ========================================

def read_ids_from_csv(csv_path: Path, id_col=ID_COL_NAME):
    """Return a sorted, de-duplicated list of participant IDs as strings, e.g. ['300','301']."""
    ids = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        # Try to sniff delimiter/header; fall back gracefully
        try:
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            dialect = csv.excel
            has_header = True

        reader = list(csv.reader(f, dialect))
        if not reader:
            return []

        header = [h.strip() for h in reader[0]] if has_header else []
        if has_header and any(h.lower() == id_col.lower() for h in header):
            idx = next(i for i, h in enumerate(header) if h.lower() == id_col.lower())
            rows = reader[1:]
            col_vals = [r[idx] for r in rows if len(r) > idx]
        else:
            rows = reader[1:] if has_header else reader
            col_vals = [r[0] for r in rows if r]

    for v in col_vals:
        s = str(v).strip()
        if not s:
            continue
        # Normalize to an integer-like string (handles '300', '300.0', 'ID=300', etc.)
        try:
            ids.append(str(int(float(s))))
        except ValueError:
            m = re.search(r"\d+", s)
            if m:
                ids.append(str(int(m.group())))
    # unique + numeric sort
    return sorted(set(ids), key=lambda x: int(x))

def move_split(ids, split_name, base_dir: Path):
    """Move 'ID.pt' from base_dir to base_dir/split_name. Returns (moved_ids, missing_ids, already_there_ids)."""
    dest = base_dir / split_name
    dest.mkdir(parents=True, exist_ok=True)

    moved, missing, already = [], [], []
    for sid in ids:
        src = base_dir / f"{sid}.pt"
        dst = dest / f"{sid}.pt"
        if dst.exists():
            already.append(sid)
            continue
        if src.exists():
            shutil.move(str(src), str(dst))
            moved.append(sid)
        else:
            # Could be already moved by a previous split or truly missing
            # Check other split folders to decide which message to print later
            missing.append(sid)
    return moved, missing, already

# ---- Safety check: warn if duplicates across splits ----
split_id_sets = {split: set(read_ids_from_csv(path)) for split, path in SPLIT_CSV.items()}
dupe_report = []
all_seen = set()
for split, ids in split_id_sets.items():
    overlap = ids & all_seen
    if overlap:
        dupe_report.append((split, sorted(overlap, key=int)))
    all_seen |= ids

if dupe_report:
    print("⚠️  Warning: some IDs appear in multiple CSVs (first split processed will get them):")
    for split, overlaps in dupe_report:
        print(f"  - {split}: {', '.join(overlaps)}")

# ---- Do the moves ----
summary = {}
for split, csv_path in SPLIT_CSV.items():
    if not csv_path.exists():
        print(f"❌ CSV not found for {split}: {csv_path}")
        summary[split] = {"csv_missing": True}
        continue

    ids = read_ids_from_csv(csv_path)
    moved, missing, already = move_split(ids, split, BASE_DIR)
    summary[split] = {
        "n_ids_in_csv": len(ids),
        "moved": len(moved),
        "already_there": len(already),
        "missing_or_in_other_split": len(missing),
    }
    print(f"\n[{split}] {csv_path.name}")
    print(f"  IDs in CSV: {len(ids)}")
    print(f"  Moved:      {len(moved)}")
    print(f"  Already in {split}/: {len(already)}")
    if missing:
        # Try to tell whether each missing file is actually in another split folder
        actually_elsewhere = []
        truly_missing = []
        for sid in missing:
            found_elsewhere = any((BASE_DIR / other / f"{sid}.pt").exists()
                                  for other in SPLIT_CSV.keys() if other != split)
            (actually_elsewhere if found_elsewhere else truly_missing).append(sid)

        if actually_elsewhere:
            print(f"  Skipped (already moved to a different split): {', '.join(sorted(actually_elsewhere, key=int))}")
        if truly_missing:
            print(f"  Missing .pt in BASE_DIR: {', '.join(sorted(truly_missing, key=int))}")

# ---- Leftover .pt files that were not referenced by any CSV ----
leftovers = sorted([p.name for p in BASE_DIR.glob("*.pt")], key=lambda s: int(s.split(".")[0]) if s.split(".")[0].isdigit() else s)
if leftovers:
    print("\nLeftover .pt files still in BASE_DIR (not listed in any CSV):")
    print(", ".join(leftovers))
else:
    print("\nNo leftover .pt files in BASE_DIR.")
