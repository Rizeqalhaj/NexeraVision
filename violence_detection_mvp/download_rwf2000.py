#!/usr/bin/env python3
"""
RWF-2000 Dataset Downloader
Following the official paper: RWF-2000: An Open Large Scale Video Database for Violence Detection
Authors: Ming Cheng, Kunjing Cai, Ming Li (2021 ICPR)
DOI: 10.1109/ICPR48806.2021.9412502
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import shutil

def setup_directories():
    """Create directory structure for RWF-2000 dataset"""
    dirs = [
        "data/raw/rwf2000",
        "data/raw/rwf2000/train",
        "data/raw/rwf2000/val",
        "data/raw/rwf2000/train/Fight",
        "data/raw/rwf2000/train/NonFight",
        "data/raw/rwf2000/val/Fight",
        "data/raw/rwf2000/val/NonFight"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def download_from_kaggle():
    """Download RWF-2000 from Kaggle (vulamnguyen/rwf2000)"""
    try:
        print("📥 Downloading RWF-2000 from Kaggle...")
        print("Note: Make sure you have Kaggle API configured (~/.kaggle/kaggle.json)")

        # Download using kaggle API
        cmd = "kaggle datasets download -d vulamnguyen/rwf2000 -p data/raw/rwf2000 --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Successfully downloaded RWF-2000 from Kaggle")
            return True
        else:
            print(f"❌ Kaggle download failed: {result.stderr}")
            print("💡 You may need to setup Kaggle API credentials")
            return False

    except Exception as e:
        print(f"❌ Error with Kaggle download: {e}")
        return False

def download_from_huggingface():
    """Download RWF-2000 from Hugging Face"""
    try:
        print("📥 Downloading RWF-2000 from Hugging Face...")

        # Install huggingface hub if needed
        try:
            import huggingface_hub
        except ImportError:
            print("Installing huggingface_hub...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            import huggingface_hub

        from huggingface_hub import snapshot_download

        # Download dataset
        snapshot_download(
            repo_id="DanJoshua/RWF-2000",
            repo_type="dataset",
            local_dir="data/raw/rwf2000",
            local_dir_use_symlinks=False
        )

        print("✅ Successfully downloaded RWF-2000 from Hugging Face")
        return True

    except Exception as e:
        print(f"❌ Hugging Face download failed: {e}")
        return False

def clone_official_repo():
    """Clone the official GitHub repository"""
    try:
        print("📥 Cloning official RWF-2000 repository...")
        print("⚠️  Note: Official repo may not contain video files due to privacy")

        repo_url = "https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection.git"
        clone_dir = "data/raw/rwf2000_official"

        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)

        result = subprocess.run([
            "git", "clone", repo_url, clone_dir
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Successfully cloned official repository")
            print(f"📁 Repository cloned to: {clone_dir}")
            print("📖 Check README for dataset access instructions")
            return True
        else:
            print(f"❌ Git clone failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error cloning repository: {e}")
        return False

def verify_dataset_structure():
    """Verify the downloaded dataset structure"""
    base_path = Path("data/raw/rwf2000")

    expected_structure = {
        "train/Fight": "Training fight videos",
        "train/NonFight": "Training non-fight videos",
        "val/Fight": "Validation fight videos",
        "val/NonFight": "Validation non-fight videos"
    }

    print("\n🔍 Verifying dataset structure...")

    total_videos = 0
    for path, description in expected_structure.items():
        full_path = base_path / path
        if full_path.exists():
            video_count = len([f for f in full_path.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov']])
            print(f"✅ {description}: {video_count} videos")
            total_videos += video_count
        else:
            print(f"❌ Missing: {description}")

    print(f"\n📊 Total videos found: {total_videos}")

    if total_videos >= 2000:
        print("✅ Complete RWF-2000 dataset (2000 videos expected)")
    elif total_videos > 0:
        print(f"⚠️  Partial dataset ({total_videos} videos found)")
    else:
        print("❌ No videos found - dataset download may have failed")

    return total_videos

def create_citation_file():
    """Create citation file for proper attribution"""
    citation = """
# RWF-2000 Dataset Citation

This dataset should be cited as:

@INPROCEEDINGS{9412502,
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection},
  year={2021},
  volume={},
  number={},
  pages={4183-4190},
  doi={10.1109/ICPR48806.2021.9412502}
}

## Dataset Information
- **Name**: RWF-2000 (Real World Fight 2000)
- **Size**: 2,000 video clips (1,000 fight, 1,000 non-fight)
- **Duration**: 5 seconds per clip at 30 FPS
- **Source**: Real-world surveillance cameras
- **Split**: Train/validation sets (mutually exclusive)
- **Purpose**: Violence detection research

## Usage Agreement
- For research purposes only
- Not allowed for commercial use without SMIIP Lab approval
- Must not be used in ways that may damage human mental health or privacy

## Official Repository
https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
"""

    with open("data/raw/rwf2000/CITATION.md", "w") as f:
        f.write(citation)

    print("✅ Created citation file: data/raw/rwf2000/CITATION.md")

def main():
    """Main function to download RWF-2000 dataset"""
    print("🚀 RWF-2000 Dataset Downloader")
    print("Following official paper by Ming Cheng, Kunjing Cai, Ming Li (2021)")
    print("DOI: 10.1109/ICPR48806.2021.9412502")
    print("=" * 60)

    # Setup directories
    setup_directories()

    # Try different download methods
    success = False

    print("\n📥 Attempting to download RWF-2000 dataset...")

    # Method 1: Try Kaggle first (most likely to have complete dataset)
    print("\n1️⃣ Trying Kaggle download...")
    success = download_from_kaggle()

    # Method 2: Try Hugging Face if Kaggle fails
    if not success:
        print("\n2️⃣ Trying Hugging Face download...")
        success = download_from_huggingface()

    # Method 3: Clone official repo (may not have videos)
    print("\n3️⃣ Cloning official repository...")
    repo_success = clone_official_repo()

    # Verify what we got
    video_count = verify_dataset_structure()

    # Create citation file
    create_citation_file()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Download Summary:")

    if video_count >= 2000:
        print("✅ SUCCESS: Complete RWF-2000 dataset downloaded!")
        print(f"📹 Videos: {video_count}")
        print("🚀 Ready for training!")

        print("\nNext steps:")
        print("1. python3 run.py train --data-dir data/raw/rwf2000")
        print("2. python3 run.py evaluate --data-dir data/raw/rwf2000")

    elif video_count > 0:
        print(f"⚠️  PARTIAL: {video_count} videos downloaded")
        print("You can still start training with available data")

    else:
        print("❌ FAILED: No videos downloaded")
        print("\n💡 Manual download options:")
        print("1. Setup Kaggle API: pip install kaggle")
        print("2. Visit: https://www.kaggle.com/datasets/vulamnguyen/rwf2000")
        print("3. Or contact SMIIP Lab for official access")

    if repo_success:
        print("✅ Official repository cloned for reference")

if __name__ == "__main__":
    main()