import os
import zipfile
import subprocess
import glob

TASK_ID = 26  # hard-sdk task
GT_JSON = r"D:\VinUni\Day05\data\tiers\hard_panoptic\groundtruth\panoptic.json"
GT_PNG_DIR = r"D:\VinUni\Day05\data\tiers\hard_panoptic\groundtruth\png"
ZIP_FILE = r"D:\VinUni\Day05\data\tiers\hard_panoptic\groundtruth_import.zip"

def run():
    print(f"1. Packing groundtruth into COCO Panoptic ZIP format at {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        # CVAT expects COCO Panoptic JSON inside an 'annotations' folder named panoptic_default.json
        zf.write(GT_JSON, arcname="annotations/panoptic_default.json")
        
        # And the masks inside annotations/panoptic_default/
        for png_file in glob.glob(os.path.join(GT_PNG_DIR, "*.png")):
            fname = os.path.basename(png_file)
            zf.write(png_file, arcname=f"annotations/panoptic_default/{fname}")
            
    print(f"2. Running cvat-cli task import-dataset for Task {TASK_ID}...")
    
    cmd = [
        r".\.venv\Scripts\cvat-cli.exe",
        "--auth", "hoap:1toi9a",
        "--server-host", "http://localhost:8080",
        "task", "import-dataset",
        str(TASK_ID),
        ZIP_FILE,
        "--format", "COCO 1.0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Import successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error during import: {e}")

if __name__ == "__main__":
    run()

