import os
import zipfile
import subprocess

TASK_ID = 23
GT_JSON = r"D:\VinUni\Day05\data\tiers\medium_instance\groundtruth\instances.json"
ZIP_FILE = r"D:\VinUni\Day05\data\tiers\medium_instance\groundtruth_import.zip"

def run():
    print(f"1. Packing groundtruth into COCO ZIP format at {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zf:
        # CVAT expects COCO JSON inside an 'annotations' folder
        zf.write(GT_JSON, arcname="annotations/instances_default.json")
        
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

