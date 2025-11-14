#!/usr/bin/env python3
"""
Video Deduplication Pipeline
Removes duplicate videos based on content hashing and perceptual similarity
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2
import numpy as np

def get_file_hash(filepath):
    """Get MD5 hash of entire file"""
    hasher = hashlib.md5()

    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def get_video_fingerprint(video_path, num_frames=10):
    """
    Extract perceptual fingerprint from video
    Samples frames and computes average hash
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None

        # Sample frames evenly throughout video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        hashes = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Resize to 8x8 for perceptual hash
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)

                # Compute average hash
                avg = resized.mean()
                hash_bits = (resized > avg).flatten()
                hashes.append(hash_bits)

        cap.release()

        if not hashes:
            return None

        # Combine all frame hashes
        combined = np.concatenate(hashes)
        return combined.tobytes()

    except Exception as e:
        print(f"Error extracting fingerprint from {video_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two byte hashes"""
    if hash1 is None or hash2 is None:
        return float('inf')

    xor = bytes(a ^ b for a, b in zip(hash1, hash2))
    return bin(int.from_bytes(xor, byteorder='big')).count('1')

def find_duplicates(video_dir, method='file_hash', similarity_threshold=10):
    """
    Find duplicate videos using specified method

    Methods:
    - 'file_hash': Exact file hash (fastest, only finds identical files)
    - 'content_hash': Perceptual hash (slower, finds similar videos)
    - 'both': Use both methods (most thorough)
    """
    print(f"\n{'='*80}")
    print(f"FINDING DUPLICATES: {method}")
    print(f"{'='*80}\n")

    video_dir = Path(video_dir)
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm'}

    # Find all videos
    print("📹 Scanning for videos...")
    videos = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                videos.append(Path(root) / file)

    print(f"Found {len(videos)} videos\n")

    if method in ['file_hash', 'both']:
        print("🔍 Computing file hashes...")
        file_hashes = {}
        hash_to_files = defaultdict(list)

        for video in tqdm(videos, desc="Hashing files"):
            file_hash = get_file_hash(video)
            if file_hash:
                file_hashes[video] = file_hash
                hash_to_files[file_hash].append(video)

        # Find exact duplicates
        exact_duplicates = {
            hash_val: files
            for hash_val, files in hash_to_files.items()
            if len(files) > 1
        }

        print(f"✅ Found {len(exact_duplicates)} groups of exact duplicates")

    if method in ['content_hash', 'both']:
        print("\n🎬 Computing perceptual fingerprints...")
        print("⚠️  This may take a while for large datasets...\n")

        fingerprints = {}
        for video in tqdm(videos, desc="Computing fingerprints"):
            fingerprint = get_video_fingerprint(video)
            if fingerprint:
                fingerprints[video] = fingerprint

        print(f"✅ Computed fingerprints for {len(fingerprints)} videos")

        # Find similar videos
        print(f"\n🔍 Finding similar videos (threshold: {similarity_threshold})...")
        similar_groups = []
        processed = set()

        video_list = list(fingerprints.keys())
        for i, video1 in enumerate(tqdm(video_list, desc="Comparing videos")):
            if video1 in processed:
                continue

            similar = [video1]
            for video2 in video_list[i+1:]:
                if video2 in processed:
                    continue

                distance = hamming_distance(
                    fingerprints[video1],
                    fingerprints[video2]
                )

                if distance <= similarity_threshold:
                    similar.append(video2)
                    processed.add(video2)

            if len(similar) > 1:
                similar_groups.append(similar)
                processed.add(video1)

        print(f"✅ Found {len(similar_groups)} groups of similar videos")

    # Compile results
    duplicates = {}

    if method == 'file_hash':
        duplicates = exact_duplicates
    elif method == 'content_hash':
        duplicates = {i: group for i, group in enumerate(similar_groups)}
    elif method == 'both':
        # Combine both methods
        duplicates.update(exact_duplicates)
        duplicates.update({f"similar_{i}": group for i, group in enumerate(similar_groups)})

    return duplicates

def remove_duplicates(duplicates, keep='first', dry_run=True):
    """
    Remove duplicate videos

    keep: 'first', 'largest', 'smallest', 'newest', 'oldest'
    dry_run: If True, only report what would be deleted
    """
    print(f"\n{'='*80}")
    print(f"REMOVING DUPLICATES (keep={keep}, dry_run={dry_run})")
    print(f"{'='*80}\n")

    total_to_remove = 0
    removed_count = 0

    for group_id, files in duplicates.items():
        if len(files) <= 1:
            continue

        # Select file to keep based on strategy
        if keep == 'first':
            keep_file = files[0]
        elif keep == 'largest':
            keep_file = max(files, key=lambda f: f.stat().st_size)
        elif keep == 'smallest':
            keep_file = min(files, key=lambda f: f.stat().st_size)
        elif keep == 'newest':
            keep_file = max(files, key=lambda f: f.stat().st_mtime)
        elif keep == 'oldest':
            keep_file = min(files, key=lambda f: f.stat().st_mtime)
        else:
            keep_file = files[0]

        # Remove others
        for file in files:
            if file != keep_file:
                total_to_remove += 1

                if dry_run:
                    print(f"Would remove: {file}")
                else:
                    try:
                        os.remove(file)
                        removed_count += 1
                        print(f"Removed: {file}")
                    except Exception as e:
                        print(f"❌ Error removing {file}: {e}")

        print(f"Keeping: {keep_file}\n")

    if dry_run:
        print(f"📊 DRY RUN SUMMARY:")
        print(f"Would remove {total_to_remove} duplicate files")
        print(f"\nRun with --execute to actually remove files")
    else:
        print(f"📊 REMOVAL SUMMARY:")
        print(f"Removed {removed_count} of {total_to_remove} duplicate files")

    return removed_count if not dry_run else total_to_remove

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Video Deduplication Pipeline')
    parser.add_argument('video_dir', help='Directory containing videos')
    parser.add_argument('--method', choices=['file_hash', 'content_hash', 'both'],
                       default='file_hash', help='Deduplication method')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Similarity threshold for content_hash method')
    parser.add_argument('--keep', choices=['first', 'largest', 'smallest', 'newest', 'oldest'],
                       default='first', help='Which duplicate to keep')
    parser.add_argument('--execute', action='store_true',
                       help='Actually remove files (default is dry run)')
    parser.add_argument('--output', help='Save duplicate report to JSON file')

    args = parser.parse_args()

    # Find duplicates
    duplicates = find_duplicates(
        args.video_dir,
        method=args.method,
        similarity_threshold=args.threshold
    )

    # Save report if requested
    if args.output:
        report = {
            str(k): [str(f) for f in v]
            for k, v in duplicates.items()
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Duplicate report saved to: {args.output}")

    # Remove duplicates
    if duplicates:
        remove_duplicates(duplicates, keep=args.keep, dry_run=not args.execute)
    else:
        print("\n✅ No duplicates found!")

if __name__ == "__main__":
    main()
