#!/usr/bin/env python3
from pathlib import Path

# ====== CONFIG ======
BASE_DIR = Path("/media/popsatorn/timeshift_backup/DAIC-WOZ")
SESSION_MIN, SESSION_MAX = 300, 492
# If True: rename all .txt files -> .csv.
# If False: only rename when the file clearly looks CSV-ish (commas/semicolons).
RENAME_ALL_TXT = True
# ====================

def looks_like_csv(p: Path, sniff_bytes: int = 4096) -> bool:
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(sniff_bytes)
        # Heuristics: has commas/semicolons and multiple lines
        return (head.count(",") > 5 or head.count(";") > 5) and ("\n" in head)
    except Exception:
        return False

def unique_with_suffix(target: Path) -> Path:
    """Return a non-colliding path by adding _1, _2, ... before the suffix."""
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    parent = target.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def process_session_folder(folder: Path):
    for txt in folder.glob("*.txt"):
        if not RENAME_ALL_TXT and not looks_like_csv(txt):
            # Skip non-CSV-looking .txt files when in safe mode
            continue
        new_path = txt.with_suffix(".csv")
        new_path = unique_with_suffix(new_path)
        try:
            txt.rename(new_path)
            print(f"[renamed] {txt.relative_to(BASE_DIR)}  ->  {new_path.name}")
        except Exception as e:
            print(f"[error]   {txt}: {e}")

def main():
    if not BASE_DIR.exists():
        print(f"Base not found: {BASE_DIR}")
        return

    # Traverse only digit-named session dirs within range
    for d in sorted(BASE_DIR.iterdir()):
        if d.is_dir() and d.name.isdigit():
            sid = int(d.name)
            if SESSION_MIN <= sid <= SESSION_MAX:
                process_session_folder(d)

    print("Done.")

if __name__ == "__main__":
    main()
