import os
import shutil
import argparse

def merge_pngs(source_dir, target_dir):
    """
    Copy all PNG files from source_dir and its subdirectories to target_dir.
    """
    os.makedirs(target_dir, exist_ok=True)  # Ensure the target directory exists
    file_count = 0  # To keep track of how many files have been copied

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".png"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(target_dir, file)
                
                # Avoid overwrites by renaming duplicates
                if os.path.exists(dest_path):
                    base, extension = os.path.splitext(file)
                    i = 1
                    # Find a new name by appending a number
                    while os.path.exists(os.path.join(target_dir, f"{base}_{i}{extension}")):
                        i += 1
                    dest_path = os.path.join(target_dir, f"{base}_{i}{extension}")
                
                shutil.copy(src_path, dest_path)
                file_count += 1

    print(f"Total PNG files copied: {file_count}")

def main():
    parser = argparse.ArgumentParser(description="Merge PNG files from a source directory to a target directory.")
    parser.add_argument("source_dir", type=str, help="Path to the source directory containing PNG files.")
    parser.add_argument("target_dir", type=str, help="Path to the target directory where PNG files will be merged.")

    args = parser.parse_args()

    merge_pngs(args.source_dir, args.target_dir)

if __name__ == "__main__":
    main()
