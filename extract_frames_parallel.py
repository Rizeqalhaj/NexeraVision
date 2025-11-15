#!/usr/bin/env python3
"""
Parallel Frame Extraction for NexaraVision
Utilizes all 44 CPU cores for maximum speed
"""

import cv2
import numpy as np
from pathlib import Path
import json
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import time
from datetime import datetime

class ParallelFrameExtractor:
    """Extract frames from all videos using parallel processing"""

    def __init__(self,
                 splits_file="/workspace/processed/splits.json",
                 output_dir="/workspace/processed/frames",
                 frames_per_video=20,
                 img_size=(224, 224),
                 num_workers=None):
        """
        Initialize parallel extractor

        Args:
            splits_file: Path to splits JSON file
            output_dir: Where to save extracted frames
            frames_per_video: Number of frames to extract per video
            img_size: Target image size (height, width)
            num_workers: Number of parallel workers (default: all CPUs)
        """
        self.splits_file = Path(splits_file)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        self.img_size = img_size

        # Use all available CPUs if not specified
        if num_workers is None:
            self.num_workers = cpu_count()
        else:
            self.num_workers = num_workers

        print(f"🚀 Using {self.num_workers} CPU cores for parallel extraction")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_single_video(self, args):
        """
        Extract frames from a single video

        Args:
            args: Tuple of (video_path, video_id, label)

        Returns:
            dict: Result information
        """
        video_path, video_id, label = args

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {
                    'video_id': video_id,
                    'status': 'failed',
                    'error': 'Could not open video'
                }

            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < self.frames_per_video:
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)

            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # Resize and normalize
                    frame = cv2.resize(frame, self.img_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                else:
                    # Add black frame if read fails
                    frames.append(np.zeros((*self.img_size, 3), dtype=np.float32))

            cap.release()

            # Save frames as numpy array
            frames_array = np.array(frames)
            output_file = self.output_dir / f"{video_id}.npy"
            np.save(output_file, frames_array)

            return {
                'video_id': video_id,
                'status': 'success',
                'output_file': str(output_file),
                'shape': frames_array.shape,
                'label': label
            }

        except Exception as e:
            return {
                'video_id': video_id,
                'status': 'failed',
                'error': str(e)
            }

    def extract_all(self):
        """Extract frames from all videos in parallel"""

        print("\n" + "=" * 80)
        print("Parallel Frame Extraction")
        print("=" * 80)

        # Load splits
        if not self.splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {self.splits_file}")

        with open(self.splits_file) as f:
            splits_data = json.load(f)

        # Collect all videos
        all_tasks = []
        video_id = 0

        for split_name in ['train', 'val', 'test']:
            videos = splits_data[split_name]['videos']
            labels = splits_data[split_name]['labels']

            for video_path, label in zip(videos, labels):
                all_tasks.append((video_path, video_id, label))
                video_id += 1

        total_videos = len(all_tasks)

        print(f"\n📊 Extraction Plan:")
        print(f"   Total videos: {total_videos:,}")
        print(f"   Frames per video: {self.frames_per_video}")
        print(f"   Target size: {self.img_size}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Parallel workers: {self.num_workers}")
        print(f"   Total frames to extract: {total_videos * self.frames_per_video:,}")

        # Estimate time
        # Assuming ~0.5 seconds per video with 44 cores
        estimated_time_seconds = (total_videos * 0.5) / self.num_workers
        estimated_time_minutes = estimated_time_seconds / 60

        print(f"\n⏱️  Estimated time: {estimated_time_minutes:.1f} minutes")
        print(f"   (vs {total_videos * 45 / 60:.1f} minutes serial)")
        print("\n" + "=" * 80)

        # Start extraction
        start_time = time.time()

        print(f"\n🚀 Starting parallel extraction with {self.num_workers} workers...")
        print("   This will max out all CPU cores!\n")

        results = []

        # Use multiprocessing Pool with progress bar
        with Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for progress tracking
            for result in tqdm(
                pool.imap_unordered(self.extract_single_video, all_tasks),
                total=total_videos,
                desc="Extracting frames",
                unit="videos",
                ncols=100
            ):
                results.append(result)

        elapsed_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        # Calculate statistics
        total_frames = sum(r.get('shape', (0,))[0] for r in successful)
        total_size_bytes = sum(
            (self.output_dir / Path(r['output_file']).name).stat().st_size
            for r in successful
            if 'output_file' in r
        )
        total_size_gb = total_size_bytes / (1024**3)

        # Print summary
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)

        print(f"\n✅ Successful: {len(successful):,}/{total_videos:,} videos ({len(successful)/total_videos*100:.1f}%)")
        if failed:
            print(f"❌ Failed: {len(failed):,} videos")

        print(f"\n📊 Statistics:")
        print(f"   Total frames extracted: {total_frames:,}")
        print(f"   Total size: {total_size_gb:.2f} GB")
        print(f"   Average per video: {total_size_gb/len(successful)*1000:.1f} MB")

        print(f"\n⏱️  Performance:")
        print(f"   Total time: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
        print(f"   Videos/second: {total_videos/elapsed_time:.2f}")
        print(f"   Speedup vs serial: {total_videos * 0.5 / elapsed_time:.1f}x")

        print(f"\n📁 Output:")
        print(f"   Directory: {self.output_dir}")
        print(f"   Files: {len(successful):,} .npy files")

        # Save extraction metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': total_videos,
            'successful': len(successful),
            'failed': len(failed),
            'total_frames': total_frames,
            'total_size_gb': total_size_gb,
            'extraction_time_seconds': elapsed_time,
            'workers': self.num_workers,
            'frames_per_video': self.frames_per_video,
            'img_size': self.img_size,
            'failed_videos': [r['video_id'] for r in failed] if failed else []
        }

        metadata_file = self.output_dir / 'extraction_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Metadata saved: {metadata_file}")

        if failed:
            print(f"\n⚠️  Failed Videos:")
            for r in failed[:10]:  # Show first 10 failures
                print(f"   Video {r['video_id']}: {r.get('error', 'Unknown error')}")
            if len(failed) > 10:
                print(f"   ... and {len(failed) - 10} more")

        print("\n" + "=" * 80)

        return metadata


def main():
    """Main extraction function"""

    print("=" * 80)
    print("NexaraVision Parallel Frame Extraction")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Create extractor (uses all available CPUs)
    extractor = ParallelFrameExtractor(
        splits_file="/workspace/processed/splits.json",
        output_dir="/workspace/processed/frames",
        frames_per_video=20,
        img_size=(224, 224),
        num_workers=44  # Use all 44 CPU cores!
    )

    # Extract all frames
    metadata = extractor.extract_all()

    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Verify extracted frames:")
    print("     ls -lh /workspace/processed/frames/ | head -20")
    print("  2. Check metadata:")
    print("     cat /workspace/processed/frames/extraction_metadata.json")
    print("  3. Start optimized training:")
    print("     python3 train_model_optimized.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
