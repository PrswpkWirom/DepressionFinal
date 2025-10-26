#!/usr/bin/env python3
from pathlib import Path

# ==== CONFIG ====
BASE_DIR = Path("/media/popsatorn/timeshift_backup/DAIC-WOZ")
EXPECTED_START = 300
EXPECTED_END   = 492   # inclusive
EXPECTED_FILES_PER_SESSION = 10
# ================

def count_files_only(folder: Path) -> int:
    """Count regular files (not directories) in folder (non-recursive)."""
    return sum(1 for p in folder.iterdir() if p.is_file())

def main():
    if not BASE_DIR.exists():
        print(f"Base directory not found: {BASE_DIR}")
        return

    # Collect existing session folders that are pure digits and in range.
    existing_sessions = {}
    for p in BASE_DIR.iterdir():
        if p.is_dir() and p.name.isdigit():
            sid = int(p.name)
            if EXPECTED_START <= sid <= EXPECTED_END:
                existing_sessions[sid] = p

    expected_set = set(range(EXPECTED_START, EXPECTED_END + 1))
    present_set  = set(existing_sessions.keys())

    # 1) Missing sessions
    missing = sorted(expected_set - present_set)

    # 2) Sessions with fewer than EXPECTED_FILES_PER_SESSION files
    fewer_than_expected = []
    for sid, path in sorted(existing_sessions.items()):
        n_files = count_files_only(path)
        if n_files < EXPECTED_FILES_PER_SESSION:
            fewer_than_expected.append((sid, n_files, str(path)))

    # 3) (Optional) unexpected folders (digits but outside expected range)
    unexpected = sorted(
        int(p.name) for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
        and (int(p.name) < EXPECTED_START or int(p.name) > EXPECTED_END)
    )

    # ---- Report ----
    print(f"\nScan: {BASE_DIR}")
    print(f"Expected sessions: {EXPECTED_START}-{EXPECTED_END} "
          f"({len(expected_set)} total)")
    print(f"Present sessions : {len(present_set)}")
    print(f"Missing sessions : {len(missing)}")
    if missing:
        print("Missing list:\n  " + ", ".join(map(str, missing)))

    print("\nFolders with fewer than "
          f"{EXPECTED_FILES_PER_SESSION} files:")
    if fewer_than_expected:
        for sid, n, path in fewer_than_expected:
            print(f"  {sid}: {n} files  ({path})")
    else:
        print("  None")

    if unexpected:
        print("\nUnexpected digit-named folders (outside expected range):")
        print("  " + ", ".join(map(str, unexpected)))

if __name__ == "__main__":
    main()
