import os
import zipfile

# Path where your .zip files are stored
DATA_DIR = "/media/popsatorn/timeshift_backup/DAIC-WOZ"

def unzip_and_delete(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted: {os.path.basename(zip_path)}")

        # Permanently delete the zip file
        os.remove(zip_path)
        print(f"🗑️ Deleted: {os.path.basename(zip_path)}")

    except Exception as e:
        print(f"❌ Failed with {zip_path}: {e}")

def main():
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(DATA_DIR, file_name)
            unzip_and_delete(zip_path, DATA_DIR)

if __name__ == "__main__":
    main()
