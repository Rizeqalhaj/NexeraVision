#!/usr/bin/env python3
import shutil
from pathlib import Path

trash = Path("/root/.local/share/Trash/files")
backup = trash / "organized_dataset_backup"
output = Path("/workspace/organized_dataset")

print("="*80)
print("RECOVERING FROM TRASH")
print("="*80)

if backup.exists():
    print(f"Found backup: {backup}")
    print()
    print("RECOVERING...")

    if output.exists():
        shutil.rmtree(output)

    shutil.move(str(backup), str(output))

    print()
    print("="*80)
    print("✅ RECOVERED!")
    print("="*80)
    print()

    # Count videos
    for p in [output/"train"/"violent", output/"train"/"nonviolent",
              output/"val"/"violent", output/"val"/"nonviolent"]:
        if p.exists():
            count = len(list(p.glob('*.mp4')))
            print(f"{p.relative_to(output)}: {count} videos")
else:
    print("ERROR: Backup not found in trash")
