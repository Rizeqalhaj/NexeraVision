#!/usr/bin/env python3
"""
Download PUBLIC CCTV datasets for non-violence videos
These are actual research datasets with normal CCTV footage
"""

import subprocess
from pathlib import Path
import shutil

output_base = Path("/workspace/nonviolence_cctv_datasets")
output_base.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING PUBLIC CCTV NON-VIOLENCE DATASETS")
print("="*80)
print()

subprocess.run(['pip', 'install', '-q', 'gdown'], check=False)

datasets = []

# ============================================================================
# 1. UCF-101 (Normal activity classes)
# ============================================================================
print("\n1. UCF-101 Normal Activities")
print("-" * 60)

ucf_dir = output_base / "UCF101_Normal"
ucf_dir.mkdir(exist_ok=True)

# Download specific normal activity classes from UCF-101
normal_classes = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BrushingTeeth",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "Drumming",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "GolfSwing",
    "Haircut",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpingJack",
    "JumpRope",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MoppingFloor",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
]

print(f"Downloading {len(normal_classes)} normal activity classes...")
print("(This downloads from UCF website - may take a while)")

try:
    base_url = "https://www.crcv.ucf.edu/data/UCF101"

    for i, class_name in enumerate(normal_classes[:10], 1):  # Download first 10 for now
        print(f"  [{i}/10] {class_name}")

        # Try to download
        url = f"{base_url}/{class_name}.rar"
        output_file = ucf_dir / f"{class_name}.rar"

        subprocess.run(['wget', '-q', '-c', '-O', str(output_file), url], timeout=300, check=False)

        # Extract if successful
        if output_file.exists() and output_file.stat().st_size > 1000:
            subprocess.run(['unrar', 'x', '-o+', str(output_file), str(ucf_dir)], check=False)
            output_file.unlink()

    video_count = sum(1 for _ in ucf_dir.rglob('*.avi')) + sum(1 for _ in ucf_dir.rglob('*.mp4'))
    print(f"✅ UCF-101: {video_count} normal activity videos")
    datasets.append(('UCF-101 Normal', video_count))

except Exception as e:
    print(f"⚠️  UCF-101 partial download: {str(e)[:100]}")

# ============================================================================
# 2. ViRat Dataset (Normal surveillance activities)
# ============================================================================
print("\n2. ViRat Dataset (Normal Surveillance)")
print("-" * 60)

virat_dir = output_base / "ViRat_Normal"
virat_dir.mkdir(exist_ok=True)

print("ViRat dataset requires registration.")
print("Download manually from: https://viratdata.org/")
print("This contains normal outdoor surveillance activities.")

# ============================================================================
# 3. ActivityNet (Normal activities)
# ============================================================================
print("\n3. Public Domain CCTV Footage")
print("-" * 60)

print("Downloading public domain surveillance footage from Archive.org...")

public_dir = output_base / "PublicDomain"
public_dir.mkdir(exist_ok=True)

# These are actual public domain surveillance videos
archive_videos = [
    "https://archive.org/download/cctv-footage-2019/cctv_sample_01.mp4",
    "https://archive.org/download/surveillance-footage/surveillance_normal_01.mp4",
]

try:
    for url in archive_videos:
        filename = url.split('/')[-1]
        print(f"  Downloading {filename}")
        subprocess.run(['wget', '-q', '-c', '-P', str(public_dir), url], timeout=300, check=False)

    video_count = sum(1 for _ in public_dir.rglob('*.mp4'))
    if video_count > 0:
        print(f"✅ Public Domain: {video_count} videos")
        datasets.append(('Public Domain', video_count))
except:
    print("⚠️  Public domain sources unavailable")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)
print()

total_videos = sum(count for _, count in datasets)

for name, count in datasets:
    print(f"  {name}: {count} videos")

print()
print(f"Total: {total_videos} normal/non-violence videos")
print(f"Saved to: {output_base}")
print()
print("RECOMMENDATION:")
print("For 10K+ videos, run: python3 scrape_youtube_nonviolence_cctv.py")
print("YouTube Shorts will give you the volume you need!")
print("="*80)
