#!/usr/bin/env python3
"""
Find video files and run benchmark
Auto-detects your dataset location on Vast.ai
"""

import os
import sys
from pathlib import Path

def find_videos_recursive(start_path="/workspace"):
    """Recursively search for video files"""
    print(f"🔍 Searching for videos in {start_path}...")

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    found_videos = []

    try:
        for root, dirs, files in os.walk(start_path):
            # Skip cache and checkpoint directories
            if 'cache' in root or 'checkpoint' in root or 'model' in root:
                continue

            for file in files:
                if Path(file).suffix.lower() in video_extensions:
                    video_path = os.path.join(root, file)
                    found_videos.append(video_path)

                    if len(found_videos) >= 5:  # Find up to 5 videos
                        return found_videos

    except PermissionError:
        pass

    return found_videos


if __name__ == "__main__":
    print("=" * 80)
    print("🎬 AUTO-DETECT VIDEO PATHS FOR BENCHMARKING")
    print("=" * 80 + "\n")

    # Search for videos
    videos = find_videos_recursive("/workspace")

    if not videos:
        print("❌ No videos found in /workspace")
        print("\n📝 Please check where your dataset is located:")
        print("   ls -R /workspace | grep -i mp4")
        sys.exit(1)

    print(f"✅ Found {len(videos)} video(s):")
    for i, video in enumerate(videos, 1):
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"   {i}. {video} ({size_mb:.1f} MB)")

    print("\n" + "=" * 80)
    print("🚀 RUNNING BENCHMARK WITH FIRST VIDEO")
    print("=" * 80 + "\n")

    # Run benchmark with first video
    test_video = videos[0]
    print(f"Using: {test_video}\n")

    # Import and run benchmark
    from benchmark_video_loading import benchmark_loader

    results = benchmark_loader(test_video, n_iterations=20)

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)
