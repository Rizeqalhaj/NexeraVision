#!/usr/bin/env python3
"""
Dataset Download Script for Violence Detection MVP
Downloads and prepares multiple violence detection datasets
"""

import os
import sys
import requests
import zipfile
import subprocess
from pathlib import Path
import kaggle

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_directories():
    """Create necessary directories for datasets"""
    dirs = [
        "data/raw/datasets",
        "data/raw/real_life_violence",
        "data/raw/hockey_fight",
        "data/raw/rwf2000",
        "data/raw/ucf_crime",
        "data/processed"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def download_kaggle_dataset(dataset_name, output_dir):
    """Download dataset from Kaggle"""
    try:
        print(f"📥 Downloading {dataset_name} from Kaggle...")

        # Use kaggle API to download
        cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Successfully downloaded {dataset_name}")
            return True
        else:
            print(f"❌ Failed to download {dataset_name}: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return False

def download_ucf_crime():
    """Download UCF-Crime dataset"""
    try:
        print("📥 Downloading UCF-Crime dataset...")

        url = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
        output_path = "data/raw/datasets/UCF_Crimes.zip"

        print("⏳ This is a large dataset (~2GB), please wait...")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Progress: {progress:.1f}%", end="", flush=True)

        print(f"\n✅ Downloaded UCF-Crime dataset to {output_path}")

        # Extract the zip file
        print("📂 Extracting UCF-Crime dataset...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw/ucf_crime")

        print("✅ UCF-Crime dataset extracted successfully")
        return True

    except Exception as e:
        print(f"❌ Error downloading UCF-Crime: {e}")
        return False

def download_sample_videos():
    """Create sample videos for immediate testing"""
    try:
        print("🎬 Creating sample videos for testing...")

        sample_script = """
import cv2
import numpy as np
import os

def create_sample_video(filename, label, duration=3, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))

    total_frames = fps * duration
    for i in range(total_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if label == "violence":
            # Red-dominant frames with movement patterns
            frame[:, :, 2] = 200 + int(50 * np.sin(i * 0.3))  # Red channel
            frame[:, :, 0] = 50  # Blue channel
            # Add some movement simulation
            cv2.circle(frame, (320 + int(100*np.sin(i*0.2)), 240 + int(50*np.cos(i*0.3))), 30, (0, 0, 255), -1)
            cv2.rectangle(frame, (200 + int(50*np.sin(i*0.1)), 150), (400, 350), (255, 255, 255), 2)
        else:
            # Blue-dominant frames with calm patterns
            frame[:, :, 0] = 200 + int(30 * np.sin(i * 0.1))  # Blue channel
            frame[:, :, 1] = 150  # Green channel
            # Add calm movement
            cv2.circle(frame, (320, 240), 50, (255, 150, 0), 2)
            cv2.putText(frame, "Normal", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Created {filename}")

# Create sample datasets
os.makedirs("data/raw/sample_dataset/violence", exist_ok=True)
os.makedirs("data/raw/sample_dataset/non_violence", exist_ok=True)

# Create 10 violence videos
for i in range(10):
    create_sample_video(f"data/raw/sample_dataset/violence/violence_{i:03d}.mp4", "violence")

# Create 10 non-violence videos
for i in range(10):
    create_sample_video(f"data/raw/sample_dataset/non_violence/normal_{i:03d}.mp4", "non_violence")

print("🎬 Sample dataset created successfully!")
"""

        # Write and execute sample creation script
        with open("create_samples.py", "w") as f:
            f.write(sample_script)

        # Execute the script
        result = subprocess.run([sys.executable, "create_samples.py"], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Sample videos created successfully")
            os.remove("create_samples.py")  # Clean up
            return True
        else:
            print(f"❌ Error creating samples: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error creating sample videos: {e}")
        return False

def main():
    """Main function to download all datasets"""
    print("🚀 Violence Detection Dataset Downloader")
    print("=" * 50)

    # Setup directories
    setup_directories()

    print("\n📥 Starting dataset downloads...")

    # Track download results
    results = {}

    # 1. Create sample videos first (immediate testing)
    print("\n1️⃣ Creating sample videos for immediate testing...")
    results['samples'] = download_sample_videos()

    # 2. Download Real Life Violence dataset from Kaggle
    print("\n2️⃣ Downloading Real Life Violence dataset...")
    results['real_life'] = download_kaggle_dataset(
        "mohamedmustafa/real-life-violence-situations-dataset",
        "data/raw/real_life_violence"
    )

    # 3. Download Hockey Fight dataset from Kaggle
    print("\n3️⃣ Downloading Hockey Fight dataset...")
    results['hockey'] = download_kaggle_dataset(
        "yassershrief/hockey-fight-vidoes",
        "data/raw/hockey_fight"
    )

    # 4. Download RWF2000 dataset from Kaggle
    print("\n4️⃣ Downloading RWF-2000 dataset...")
    results['rwf2000'] = download_kaggle_dataset(
        "vulamnguyen/rwf2000",
        "data/raw/rwf2000"
    )

    # 5. Download UCF-Crime dataset (direct download)
    print("\n5️⃣ Downloading UCF-Crime dataset...")
    results['ucf_crime'] = download_ucf_crime()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Download Summary:")
    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {dataset.capitalize()}: {status}")

    successful_downloads = sum(results.values())
    total_datasets = len(results)

    print(f"\n🎯 Total: {successful_downloads}/{total_datasets} datasets downloaded successfully")

    if successful_downloads > 0:
        print("\n🚀 Ready to start training!")
        print("Next steps:")
        print("  1. Run: python3 run.py train --data-dir data/raw/sample_dataset")
        print("  2. Or: python3 run.py demo --data-dir data/raw/real_life_violence")
    else:
        print("\n⚠️  No datasets downloaded. Check your internet connection and Kaggle setup.")

if __name__ == "__main__":
    main()