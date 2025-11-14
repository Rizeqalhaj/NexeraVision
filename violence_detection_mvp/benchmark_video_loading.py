#!/usr/bin/env python3
"""
Benchmark Video Loading Performance
Compare OpenCV CPU vs TensorFlow GPU vs PyAV

Run this on your Vast.ai instance to see the speedup!
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Configure GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from gpu_video_loader import GPUVideoLoader

def benchmark_loader(video_path: str, n_iterations: int = 20):
    """Benchmark different video loading backends"""

    print("=" * 80)
    print("🎬 VIDEO LOADING PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Iterations: {n_iterations}")
    print(f"Target: 30 frames @ 224x224")
    print("=" * 80 + "\n")

    results = {}

    # Test each backend
    backends = ['tensorflow', 'opencv']

    # Check if PyAV is available
    try:
        import av
        backends.insert(1, 'pyav')
    except ImportError:
        pass

    for backend in backends:
        print(f"Testing {backend.upper()}...")

        try:
            loader = GPUVideoLoader(backend=backend)

            # Warmup
            for _ in range(3):
                frames = loader.load_video_gpu(video_path, n_frames=30, frame_size=(224, 224))

            # Benchmark
            times = []
            for i in range(n_iterations):
                start = time.time()
                frames = loader.load_video_gpu(video_path, n_frames=30, frame_size=(224, 224))
                elapsed = time.time() - start
                times.append(elapsed)

                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i+1}/{n_iterations} iterations")

            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = 1.0 / avg_time

            results[backend] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput,
                'frames_shape': frames.shape if frames is not None else None
            }

            print(f"  ✅ Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"  ✅ Throughput: {throughput:.1f} videos/sec")
            print(f"  ✅ Frames shape: {frames.shape}\n")

        except Exception as e:
            print(f"  ❌ Failed: {e}\n")
            results[backend] = None

    # Summary
    print("=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)

    # Find baseline (OpenCV)
    baseline = results.get('opencv')

    if baseline:
        baseline_time = baseline['avg_time']
        baseline_throughput = baseline['throughput']

        print(f"\n{'Backend':<15} {'Time (ms)':<15} {'Throughput':<20} {'Speedup':<15}")
        print("-" * 80)

        for backend in backends:
            if results[backend]:
                r = results[backend]
                speedup = baseline_time / r['avg_time']

                time_str = f"{r['avg_time']*1000:.2f}ms"
                throughput_str = f"{r['throughput']:.1f} videos/sec"
                speedup_str = f"{speedup:.1f}x"

                # Add emoji for best performer
                emoji = "🚀" if backend != 'opencv' and speedup > 5 else ""

                print(f"{backend:<15} {time_str:<15} {throughput_str:<20} {speedup_str:<15} {emoji}")

        print("-" * 80)

        # Calculate time savings
        best_backend = min(
            [(b, r['avg_time']) for b, r in results.items() if r],
            key=lambda x: x[1]
        )[0]

        if best_backend != 'opencv':
            best_time = results[best_backend]['avg_time']
            speedup = baseline_time / best_time

            print(f"\n🎯 BEST: {best_backend.upper()}")
            print(f"   Speedup: {speedup:.1f}x faster than OpenCV")
            print(f"   Time saved per video: {(baseline_time - best_time)*1000:.2f}ms")

            # Extrapolate to full dataset
            total_videos = 176_780  # 17,678 videos * 10x augmentation

            baseline_total = baseline_time * total_videos
            best_total = best_time * total_videos
            time_saved = baseline_total - best_total

            print(f"\n💰 TIME SAVINGS FOR FULL TRAINING:")
            print(f"   OpenCV (CPU):  {baseline_total/3600:.1f} hours")
            print(f"   {best_backend.upper()} (GPU): {best_total/3600:.1f} hours")
            print(f"   Time saved:    {time_saved/3600:.1f} hours ({time_saved/60:.0f} minutes)")
            print(f"   Speedup:       {speedup:.1f}x faster!")

    print("\n" + "=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80)

    return results


def find_test_video(data_dir: str = "/workspace/data"):
    """Find a test video"""
    data_path = Path(data_dir)

    # Try to find a video
    for split in ['train', 'val', 'test']:
        for label in ['Fight', 'NonFight', 'Normal']:
            search_path = data_path / split / label
            if search_path.exists():
                videos = list(search_path.glob('*.mp4'))
                if videos:
                    return str(videos[0])

    return None


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 GPU-ACCELERATED VIDEO LOADING BENCHMARK")
    print("=" * 80)
    print("This will compare OpenCV CPU vs TensorFlow GPU vs PyAV")
    print("=" * 80 + "\n")

    # Find test video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("🔍 Searching for test video...")
        video_path = find_test_video()

        if video_path is None:
            print("❌ No test video found!")
            print("\nUsage: python3 benchmark_video_loading.py /path/to/video.mp4")
            sys.exit(1)

        print(f"✅ Found test video: {video_path}\n")

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    # Run benchmark
    results = benchmark_loader(video_path, n_iterations=20)

    print("\n💡 TIP: If TensorFlow GPU is not available:")
    print("   pip install --upgrade tensorflow>=2.13.0")
    print("\n💡 To enable PyAV:")
    print("   pip install av")
