#!/usr/bin/env python3
"""
Investigate UCF_Crime Dataset Structure
Find out why videos aren't being detected
"""

from pathlib import Path
import os

def investigate_ucf_crime():
    """Investigate UCF_Crime directory structure"""

    print("=" * 80)
    print("UCF_Crime Dataset Investigation")
    print("=" * 80)
    print()

    ucf_path = Path("/workspace/datasets/tier1/UCF_Crime")

    if not ucf_path.exists():
        print(f"❌ Directory not found: {ucf_path}")
        return

    print(f"📂 Directory: {ucf_path}")
    print(f"✅ Directory exists\n")

    # Get total size
    total_size = sum(f.stat().st_size for f in ucf_path.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)
    print(f"💾 Total Size: {size_gb:.2f} GB\n")

    # List all items in directory (first level)
    print("=" * 80)
    print("📁 DIRECTORY STRUCTURE (First Level):")
    print("=" * 80)

    items = list(ucf_path.iterdir())
    print(f"\nTotal items: {len(items)}\n")

    for item in sorted(items)[:20]:  # Show first 20 items
        if item.is_dir():
            item_count = len(list(item.iterdir()))
            print(f"📁 {item.name}/ ({item_count} items)")
        else:
            size_mb = item.stat().st_size / (1024**2)
            print(f"📄 {item.name} ({size_mb:.2f} MB)")

    if len(items) > 20:
        print(f"\n... and {len(items) - 20} more items")

    # Count files by extension
    print("\n" + "=" * 80)
    print("📊 FILE TYPES:")
    print("=" * 80 + "\n")

    extensions = {}
    for file in ucf_path.rglob('*'):
        if file.is_file():
            ext = file.suffix.lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
            else:
                extensions['<no extension>'] = extensions.get('<no extension>', 0) + 1

    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"{ext:20} {count:,} files")

    # Check for video files specifically
    print("\n" + "=" * 80)
    print("🎬 VIDEO FILE SEARCH:")
    print("=" * 80 + "\n")

    video_exts = ['.mp4', '.avi', '.mkv', '.webm', '.mov', '.m4v', '.flv', '.wmv']

    for ext in video_exts:
        files = list(ucf_path.rglob(f'*{ext}'))
        if files:
            print(f"✅ Found {len(files):,} {ext} files")
            # Show sample paths
            for f in files[:3]:
                rel_path = f.relative_to(ucf_path)
                print(f"   - {rel_path}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")
        else:
            files_upper = list(ucf_path.rglob(f'*{ext.upper()}'))
            if files_upper:
                print(f"✅ Found {len(files_upper):,} {ext.upper()} files")

    # Check for compressed files
    print("\n" + "=" * 80)
    print("📦 COMPRESSED FILES:")
    print("=" * 80 + "\n")

    compressed_exts = ['.zip', '.tar', '.gz', '.7z', '.rar']
    found_compressed = False

    for ext in compressed_exts:
        files = list(ucf_path.rglob(f'*{ext}'))
        if files:
            found_compressed = True
            print(f"📦 Found {len(files):,} {ext} files")
            for f in files[:5]:
                size_gb = f.stat().st_size / (1024**3)
                print(f"   - {f.name} ({size_gb:.2f} GB)")

    if not found_compressed:
        print("No compressed files found")

    # Show sample directory tree
    print("\n" + "=" * 80)
    print("🌳 DIRECTORY TREE (Sample):")
    print("=" * 80 + "\n")

    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return

        items = sorted(list(directory.iterdir()))[:10]  # Show first 10 items per level

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "

            if item.is_dir():
                print(f"{prefix}{current_prefix}{item.name}/")
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
            else:
                size_mb = item.stat().st_size / (1024**2)
                print(f"{prefix}{current_prefix}{item.name} ({size_mb:.1f} MB)")

    print_tree(ucf_path)

    print("\n" + "=" * 80)
    print("✅ Investigation Complete")
    print("=" * 80)

if __name__ == "__main__":
    investigate_ucf_crime()
