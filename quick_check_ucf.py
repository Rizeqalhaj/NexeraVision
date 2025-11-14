#!/usr/bin/env python3
"""
Quick UCF_Crime Check - Lightweight version that won't hang
"""

import subprocess
from pathlib import Path

def quick_check():
    """Quick check using bash commands"""

    print("=" * 80)
    print("UCF_Crime Quick Check")
    print("=" * 80)
    print()

    ucf_path = "/workspace/datasets/tier1/UCF_Crime"

    # Check if directory exists
    print(f"📂 Checking: {ucf_path}\n")

    # Count items in top level (fast)
    print("1️⃣ Top-level contents (first 30 items):")
    print("-" * 80)
    result = subprocess.run(
        f"ls -lh {ucf_path} | head -30",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)

    # Count total files (fast)
    print("\n2️⃣ File count:")
    print("-" * 80)
    result = subprocess.run(
        f"find {ucf_path} -type f | wc -l",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    print(f"Total files: {result.stdout.strip()}")

    # Count directories (fast)
    print("\n3️⃣ Directory count:")
    print("-" * 80)
    result = subprocess.run(
        f"find {ucf_path} -type d | wc -l",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    print(f"Total directories: {result.stdout.strip()}")

    # Find video files (with timeout)
    print("\n4️⃣ Video files (.mp4):")
    print("-" * 80)
    result = subprocess.run(
        f"find {ucf_path} -type f -iname '*.mp4' | head -20",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.stdout.strip():
        print(result.stdout)
        # Count total mp4
        result2 = subprocess.run(
            f"find {ucf_path} -type f -iname '*.mp4' | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"\nTotal .mp4 files: {result2.stdout.strip()}")
    else:
        print("No .mp4 files found")

    # Check for .avi files
    print("\n5️⃣ Video files (.avi):")
    print("-" * 80)
    result = subprocess.run(
        f"find {ucf_path} -type f -iname '*.avi' | head -20",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.stdout.strip():
        print(result.stdout)
        result2 = subprocess.run(
            f"find {ucf_path} -type f -iname '*.avi' | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"\nTotal .avi files: {result2.stdout.strip()}")
    else:
        print("No .avi files found")

    # Check for compressed files
    print("\n6️⃣ Compressed files:")
    print("-" * 80)
    result = subprocess.run(
        f"find {ucf_path} -type f \\( -iname '*.zip' -o -iname '*.tar' -o -iname '*.gz' \\) | head -10",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.stdout.strip():
        print(result.stdout)
    else:
        print("No compressed files found")

    # Check directory structure depth
    print("\n7️⃣ Directory structure (2 levels deep):")
    print("-" * 80)
    result = subprocess.run(
        f"tree -L 2 -d {ucf_path} 2>/dev/null || find {ucf_path} -maxdepth 2 -type d",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    print(result.stdout[:1000])  # Limit output

    # Disk usage
    print("\n8️⃣ Disk usage:")
    print("-" * 80)
    result = subprocess.run(
        f"du -sh {ucf_path}",
        shell=True,
        capture_output=True,
        text=True,
        timeout=30
    )
    print(result.stdout)

    print("\n" + "=" * 80)
    print("✅ Quick check complete")
    print("=" * 80)

if __name__ == "__main__":
    try:
        quick_check()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
