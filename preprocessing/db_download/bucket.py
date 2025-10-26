#!/usr/bin/env python3
import re
import os
import shutil
from pathlib import Path
from zipfile import ZipFile, BadZipFile

# ---- CONFIG ----
BASE_DIR = Path("/media/popsatorn/timeshift_backup/DAIC-WOZ")
ID_REGEX = re.compile(r"^(\d{3,4})[_\-]")   # matches "301_" or "301-"
ZIP_ID_REGEX = re.compile(r"^(\d{3,4})[_\-].*\.zip$", re.IGNORECASE)
# ----------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def extract_zip_to_id_folder(zip_path: Path) -> None:
    """Extract 301_P.zip -> BASE_DIR/301/, then delete the zip."""
    m = ZIP_ID_REGEX.match(zip_path.name)
    if not m:
        print(f"  [skip zip: no ID] {zip_path.name}")
        return
    sid = m.group(1)
    dest_dir = BASE_DIR / sid
    ensure_dir(dest_dir)

    try:
        with ZipFile(zip_path, "r") as zf:
            # Extract directly into the ID folder
            zf.extractall(dest_dir)
        print(f"  [unzipped] {zip_path.name} -> {dest_dir}")
        zip_path.unlink()  # permanent delete
        print(f"  [deleted zip] {zip_path.name}")
    except BadZipFile:
        print(f"  [bad zip] {zip_path.name} (skipped)")
    except Exception as e:
        print(f"  [error unzip] {zip_path.name}: {e}")

def move_file_to_id_folder(file_path: Path) -> None:
    """Move 301_AUDIO.wav -> BASE_DIR/301/301_AUDIO.wav"""
    m = ID_REGEX.match(file_path.name)
    if not m:
        # ignore html/listing or unrelated files
        return
    sid = m.group(1)
    dest_dir = BASE_DIR / sid
    ensure_dir(dest_dir)
    dest_path = dest_dir / file_path.name

    if dest_path.exists():
        # If same size, assume duplicate and remove source; otherwise, skip
        try:
            if dest_path.stat().st_size == file_path.stat().st_size:
                file_path.unlink()
                print(f"  [duplicate removed] {file_path.name}")
            else:
                print(f"  [exists diff] {dest_path.name} (kept original, skipped moving {file_path.name})")
        except Exception as e:
            print(f"  [error compare] {file_path.name}: {e}")
        return

    try:
        shutil.move(str(file_path), str(dest_path))
        print(f"  [moved] {file_path.name} -> {dest_dir.name}/")
    except Exception as e:
        print(f"  [error move] {file_path.name}: {e}")

def main():
    if not BASE_DIR.exists():
        print(f"Base directory not found: {BASE_DIR}")
        return

    # 1) Handle zips first (so later moves don’t fight with extracted files)
    for p in sorted(BASE_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() == ".zip":
            extract_zip_to_id_folder(p)

    # 2) Move any remaining flat files into their ID folders
    for p in sorted(BASE_DIR.iterdir()):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if name_lower.endswith(".zip"):
            continue
        if name_lower.endswith(".html") or name_lower.startswith("index.html"):
            continue
        move_file_to_id_folder(p)

    print("\nDone.")

if __name__ == "__main__":
    main()
