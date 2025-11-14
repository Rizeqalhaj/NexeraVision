#!/usr/bin/env python3
"""
Recover Dataset from Trash
Moves all videos back from trash to organized_dataset
"""

import shutil
from pathlib import Path

def find_trash():
    """Find trash directory"""
    possible_trash = [
        Path.home() / ".local/share/Trash/files",
        Path("/root/.local/share/Trash/files"),
        Path("/workspace/.Trash-0/files"),
        Path("/workspace/.Trash-1000/files"),
    ]

    for trash in possible_trash:
        if trash.exists():
            print(f"✅ Found trash: {trash}")
            return trash

    return None

def recover_dataset():
    print("="*80)
    print("RECOVER DATASET FROM TRASH")
    print("="*80)
    print()

    # Find trash
    trash_dir = find_trash()

    if not trash_dir:
        print("❌ Could not find trash directory")
        print()
        print("Checking common locations:")
        print("  ~/.local/share/Trash/files")
        print("  /root/.local/share/Trash/files")
        print("  /workspace/.Trash-0/files")
        print()
        manual_path = input("Enter trash path manually (or 'cancel'): ").strip()
        if manual_path.lower() == 'cancel':
            return
        trash_dir = Path(manual_path)
        if not trash_dir.exists():
            print(f"❌ Path does not exist: {trash_dir}")
            return

    print()
    print(f"📁 Trash directory: {trash_dir}")
    print()

    # Look for dataset folders in trash
    print("🔍 Searching for dataset folders in trash...")
    print()

    folders_found = []

    # Look for violent/nonviolent folders
    for item in trash_dir.iterdir():
        if item.is_dir():
            name = item.name.lower()
            if 'violent' in name or 'nonviolent' in name or 'train' in name or 'val' in name:
                folders_found.append(item)
                print(f"   Found: {item.name}")

    if not folders_found:
        print("❌ No dataset folders found in trash")
        print()
        print("Contents of trash:")
        for item in list(trash_dir.iterdir())[:20]:
            print(f"   {item.name}")
        return

    print()
    print(f"✅ Found {len(folders_found)} dataset folders")
    print()

    # Count videos
    total_videos = 0
    for folder in folders_found:
        videos = list(folder.glob('*.mp4'))
        total_videos += len(videos)
        print(f"   {folder.name}: {len(videos)} videos")

    print()
    print(f"📊 Total videos to recover: {total_videos}")
    print()

    response = input("Proceed with recovery? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        return

    # Restore to organized_dataset
    output_base = Path("/workspace/organized_dataset")
    output_base.mkdir(parents=True, exist_ok=True)

    print()
    print("📋 Recovering files...")
    print()

    for folder in folders_found:
        # Determine destination based on folder name
        folder_name = folder.name.lower()

        if 'train' in folder_name and 'violent' in folder_name and 'non' not in folder_name:
            dest = output_base / "train" / "violent"
        elif 'train' in folder_name and 'nonviolent' in folder_name:
            dest = output_base / "train" / "nonviolent"
        elif 'val' in folder_name and 'violent' in folder_name and 'non' not in folder_name:
            dest = output_base / "val" / "violent"
        elif 'val' in folder_name and 'nonviolent' in folder_name:
            dest = output_base / "val" / "nonviolent"
        else:
            # Can't determine - ask user
            print(f"⚠️  Unknown folder: {folder.name}")
            dest_str = input(f"   Destination path (or 'skip'): ").strip()
            if dest_str.lower() == 'skip':
                continue
            dest = Path(dest_str)

        dest.mkdir(parents=True, exist_ok=True)

        # Move all videos
        videos = list(folder.glob('*.mp4'))
        print(f"   Moving {len(videos)} videos from {folder.name} to {dest}")

        for video in videos:
            dst_path = dest / video.name
            shutil.move(str(video), str(dst_path))

        print(f"   ✅ Recovered {len(videos)} videos to {dest}")

    print()
    print("="*80)
    print("✅ RECOVERY COMPLETE!")
    print("="*80)
    print()

    # Verify recovery
    print("📊 Verification:")

    train_violent = output_base / "train" / "violent"
    train_nonviolent = output_base / "train" / "nonviolent"
    val_violent = output_base / "val" / "violent"
    val_nonviolent = output_base / "val" / "nonviolent"

    for path in [train_violent, train_nonviolent, val_violent, val_nonviolent]:
        if path.exists():
            count = len(list(path.glob('*.mp4')))
            print(f"   {path.relative_to(output_base)}: {count} videos")

    print()
    print("="*80)

if __name__ == "__main__":
    recover_dataset()
