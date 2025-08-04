import os
import shutil

source_root = "calls"
dest_root = "wer+jiwer/calls"

# Ensure destination root exists
os.makedirs(dest_root, exist_ok=True)

for call_folder in os.listdir(source_root):
    source_path = os.path.join(source_root, call_folder)
    dest_path = os.path.join(dest_root, call_folder)

    if not os.path.isdir(source_path):
        continue

    # Create destination call folder
    os.makedirs(dest_path, exist_ok=True)

    for filename in ["ref_transcript.json", "gt_transcript.json"]:
        src_file = os.path.join(source_path, filename)
        dst_file = os.path.join(dest_path, filename)

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} → {dst_file}")
        else:
            print(f"Missing: {src_file}")