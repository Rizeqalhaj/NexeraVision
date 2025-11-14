#!/usr/bin/env python3
"""
Dataset Cleaner
Removes suspicious/non-violent videos based on validation report
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

def clean_dataset(validation_report_path, action='move', dry_run=False):
    """
    Clean dataset based on validation report

    Args:
        validation_report_path: Path to validation_report.json
        action: 'move' (to quarantine) or 'delete' (permanent removal)
        dry_run: If True, only show what would be done without doing it
    """

    with open(validation_report_path, 'r') as f:
        results = json.load(f)

    # Create quarantine directory
    quarantine_dir = Path(validation_report_path).parent / 'quarantine'
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Count videos by suspicion level
    suspicious_videos = [r for r in results if r['suspicion_level'] == 'SUSPICIOUS']
    review_videos = [r for r in results if r['suspicion_level'] == 'REVIEW']

    print("="*80)
    print("DATASET CLEANING")
    print("="*80)
    print(f"Validation report: {validation_report_path}")
    print(f"Action: {action}")
    print(f"Dry run: {dry_run}")
    print("")
    print(f"📊 Statistics:")
    print(f"  Total videos: {len(results)}")
    print(f"  OK: {len(results) - len(suspicious_videos) - len(review_videos)}")
    print(f"  SUSPICIOUS: {len(suspicious_videos)} (will be removed)")
    print(f"  REVIEW: {len(review_videos)} (will be kept for manual review)")
    print("")

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("")

    # Process suspicious videos
    removed_count = 0
    error_count = 0

    if len(suspicious_videos) > 0:
        print("🗑️  Removing SUSPICIOUS videos...")
        print("")

        for r in tqdm(suspicious_videos, desc="Processing"):
            video_path = Path(r['video'])

            if not video_path.exists():
                print(f"  ⚠️  File not found: {video_path.name}")
                error_count += 1
                continue

            try:
                if dry_run:
                    print(f"  [DRY RUN] Would {action}: {video_path.name}")
                else:
                    if action == 'move':
                        # Move to quarantine with preserved structure
                        dest = quarantine_dir / video_path.name
                        shutil.move(str(video_path), str(dest))
                    elif action == 'delete':
                        video_path.unlink()

                removed_count += 1
            except Exception as e:
                print(f"  ❌ Error processing {video_path.name}: {e}")
                error_count += 1

    print("")
    print("="*80)
    print("CLEANING COMPLETE")
    print("="*80)
    print(f"✅ Removed: {removed_count} videos")
    print(f"❌ Errors: {error_count}")
    print(f"🔍 Review: {len(review_videos)} videos (kept for manual review)")
    print("")

    if action == 'move' and removed_count > 0 and not dry_run:
        print(f"📁 Quarantined videos: {quarantine_dir}")
        print("   You can review these and delete manually if confirmed non-violent")
        print("")

    # Generate clean dataset report
    clean_videos = [r for r in results if r['suspicion_level'] != 'SUSPICIOUS']

    print("📊 CLEAN DATASET STATISTICS:")
    print(f"  Total videos: {len(clean_videos)}")
    print(f"  Average duration: {sum(r['duration'] for r in clean_videos) / len(clean_videos):.1f}s")
    print(f"  Average motion score: {sum(r['motion_score'] for r in clean_videos) / len(clean_videos):.1f}")
    print("")

    print("💡 NEXT STEPS:")
    if len(review_videos) > 0:
        print(f"1. Manually review {len(review_videos)} videos marked for REVIEW")
        print("   Open validation_report.html in browser to see thumbnails")
    print("2. Re-run validation to confirm dataset quality")
    print("3. Proceed with training on clean dataset")
    print("")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Clean Dataset Based on Validation Report')
    parser.add_argument('--validation-report', required=True,
                       help='Path to validation_report.json')
    parser.add_argument('--action', choices=['move', 'delete'], default='move',
                       help='Action to take: move to quarantine or delete (default: move)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')

    args = parser.parse_args()

    clean_dataset(args.validation_report, args.action, args.dry_run)

if __name__ == "__main__":
    main()
