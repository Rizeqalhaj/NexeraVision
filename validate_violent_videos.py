#!/usr/bin/env python3
"""
Violent Video Validator
Verifies downloaded videos actually contain violence
Generates preview thumbnails and motion analysis for manual verification
"""

import cv2
import numpy as np
from pathlib import Path
import subprocess
import json
from tqdm import tqdm
import random

def extract_keyframes(video_path, num_frames=9):
    """Extract evenly-spaced keyframes from video"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < num_frames:
        num_frames = total_frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize for thumbnail
            frame = cv2.resize(frame, (320, 240))
            frames.append(frame)

    cap.release()
    return frames

def detect_motion(video_path):
    """Calculate average motion/action intensity in video"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_scores = []

    # Sample 30 frames throughout video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, total_frames - 1, min(30, total_frames), dtype=int)

    for idx in sample_indices[1:]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference (motion)
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.mean(diff)
        motion_scores.append(motion_score)

        prev_gray = gray

    cap.release()

    if motion_scores:
        return np.mean(motion_scores)
    return 0.0

def create_thumbnail_grid(frames, output_path):
    """Create 3x3 grid of thumbnails"""
    if not frames or len(frames) == 0:
        return False

    # Ensure we have 9 frames
    while len(frames) < 9:
        frames.append(frames[-1])
    frames = frames[:9]

    # Create 3x3 grid
    rows = []
    for i in range(0, 9, 3):
        row = np.hstack(frames[i:i+3])
        rows.append(row)

    grid = np.vstack(rows)

    # Add border and text
    grid = cv2.copyMakeBorder(grid, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    cv2.imwrite(str(output_path), grid)
    return True

def get_video_info(video_path):
    """Get video metadata using ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        info = json.loads(result.stdout)

        duration = float(info['format'].get('duration', 0))

        for stream in info['streams']:
            if stream['codec_type'] == 'video':
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                fps = eval(stream.get('r_frame_rate', '0/1'))
                return {
                    'duration': duration,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'resolution': f"{width}x{height}"
                }

        return {'duration': duration, 'width': 0, 'height': 0, 'fps': 0, 'resolution': 'unknown'}
    except Exception as e:
        return {'duration': 0, 'width': 0, 'height': 0, 'fps': 0, 'resolution': 'unknown', 'error': str(e)}

def validate_video(video_path, output_dir):
    """Validate single video and generate report"""
    video_path = Path(video_path)

    # Get metadata
    info = get_video_info(video_path)

    # Extract keyframes
    frames = extract_keyframes(video_path)

    # Calculate motion
    motion_score = detect_motion(video_path)

    # Create thumbnail
    thumbnail_path = output_dir / "thumbnails" / f"{video_path.stem}.jpg"
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

    thumbnail_created = False
    if frames:
        thumbnail_created = create_thumbnail_grid(frames, thumbnail_path)

    # Determine suspicion level
    suspicion_level = "OK"
    reasons = []

    # Too short (< 3 seconds likely not violent)
    if info['duration'] < 3:
        suspicion_level = "SUSPICIOUS"
        reasons.append("Too short (< 3s)")

    # Very low motion (static image or slideshow)
    if motion_score < 5.0:
        suspicion_level = "SUSPICIOUS"
        reasons.append(f"Low motion ({motion_score:.1f})")

    # Very long (> 10 minutes, likely not a fight clip)
    if info['duration'] > 600:
        suspicion_level = "REVIEW"
        reasons.append("Very long (> 10 min)")

    return {
        'video': str(video_path),
        'thumbnail': str(thumbnail_path) if thumbnail_created else None,
        'duration': info['duration'],
        'resolution': info['resolution'],
        'fps': info['fps'],
        'motion_score': motion_score,
        'suspicion_level': suspicion_level,
        'reasons': reasons
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate Violent Video Dataset')
    parser.add_argument('--dataset-dir', required=True,
                       help='Directory containing violent videos to validate')
    parser.add_argument('--output-dir', default='./validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='Number of videos to validate (0 = all)')
    parser.add_argument('--random-sample', action='store_true',
                       help='Randomly sample videos instead of first N')

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("VIOLENT VIDEO VALIDATION")
    print("="*80)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("")

    # Find all videos
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'}
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(dataset_dir.rglob(f'*{ext}'))

    print(f"Found {len(all_videos)} videos")

    # Sample if needed
    if args.sample_size > 0 and args.sample_size < len(all_videos):
        if args.random_sample:
            all_videos = random.sample(all_videos, args.sample_size)
            print(f"Randomly sampling {args.sample_size} videos")
        else:
            all_videos = all_videos[:args.sample_size]
            print(f"Validating first {args.sample_size} videos")

    print("")
    print("🔍 Validating videos...")
    print("")

    results = []
    suspicious_count = 0
    review_count = 0

    for video_path in tqdm(all_videos, desc="Processing"):
        try:
            result = validate_video(video_path, output_dir)
            results.append(result)

            if result['suspicion_level'] == 'SUSPICIOUS':
                suspicious_count += 1
            elif result['suspicion_level'] == 'REVIEW':
                review_count += 1
        except Exception as e:
            print(f"\n  ❌ Error processing {video_path.name}: {e}")

    # Save results
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate HTML report for manual review
    html_report_path = output_dir / 'validation_report.html'
    generate_html_report(results, html_report_path)

    # Summary
    print("")
    print("="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Total videos validated: {len(results)}")
    print(f"OK: {len(results) - suspicious_count - review_count}")
    print(f"SUSPICIOUS: {suspicious_count} ⚠️")
    print(f"REVIEW: {review_count} 🔍")
    print("")

    if suspicious_count > 0:
        print("⚠️  SUSPICIOUS VIDEOS (likely not violent):")
        print("-"*60)
        for r in results:
            if r['suspicion_level'] == 'SUSPICIOUS':
                video_name = Path(r['video']).name
                reasons = ', '.join(r['reasons'])
                print(f"  • {video_name}: {reasons}")
        print("")

    if review_count > 0:
        print("🔍 VIDEOS NEEDING MANUAL REVIEW:")
        print("-"*60)
        for r in results:
            if r['suspicion_level'] == 'REVIEW':
                video_name = Path(r['video']).name
                reasons = ', '.join(r['reasons'])
                print(f"  • {video_name}: {reasons}")
        print("")

    print(f"📊 JSON Report: {report_path}")
    print(f"🌐 HTML Report: {html_report_path}")
    print(f"🖼️  Thumbnails: {output_dir / 'thumbnails'}")
    print("")
    print("💡 NEXT STEPS:")
    print("1. Open HTML report in browser to review thumbnails")
    print("2. Remove suspicious videos from dataset:")
    print("   python3 clean_dataset.py --validation-report validation_results/validation_report.json")
    print("")

def generate_html_report(results, output_path):
    """Generate HTML report with thumbnails for manual review"""

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .stats { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .video-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }
        .video-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .video-card.suspicious { border-left: 4px solid #ff9800; }
        .video-card.review { border-left: 4px solid #2196f3; }
        .video-card.ok { border-left: 4px solid #4caf50; }
        .thumbnail { width: 100%; border-radius: 4px; margin-bottom: 10px; }
        .video-name { font-weight: bold; margin-bottom: 8px; word-break: break-all; }
        .video-meta { font-size: 12px; color: #666; }
        .suspicion-tag { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-top: 8px; }
        .suspicion-tag.suspicious { background: #ff9800; color: white; }
        .suspicion-tag.review { background: #2196f3; color: white; }
        .suspicion-tag.ok { background: #4caf50; color: white; }
        .reasons { font-size: 12px; color: #f44336; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>📹 Video Validation Report</h1>
    <div class="stats">
        <h2>Summary</h2>
        <p><strong>Total Videos:</strong> """ + str(len(results)) + """</p>
        <p><strong>OK:</strong> """ + str(sum(1 for r in results if r['suspicion_level'] == 'OK')) + """</p>
        <p><strong>Suspicious:</strong> """ + str(sum(1 for r in results if r['suspicion_level'] == 'SUSPICIOUS')) + """</p>
        <p><strong>Review:</strong> """ + str(sum(1 for r in results if r['suspicion_level'] == 'REVIEW')) + """</p>
    </div>

    <h2>Videos (showing suspicious first)</h2>
    <div class="video-grid">
"""

    # Sort: suspicious first, then review, then OK
    sorted_results = sorted(results, key=lambda r: {'SUSPICIOUS': 0, 'REVIEW': 1, 'OK': 2}[r['suspicion_level']])

    for r in sorted_results:
        video_name = Path(r['video']).name
        level = r['suspicion_level'].lower()

        thumbnail_html = ""
        if r['thumbnail'] and Path(r['thumbnail']).exists():
            # Use relative path for thumbnail
            thumb_rel = Path(r['thumbnail']).relative_to(output_path.parent)
            thumbnail_html = f'<img class="thumbnail" src="{thumb_rel}" alt="Thumbnail">'

        reasons_html = ""
        if r['reasons']:
            reasons_html = f'<div class="reasons">⚠️ {", ".join(r["reasons"])}</div>'

        html += f"""
        <div class="video-card {level}">
            {thumbnail_html}
            <div class="video-name">{video_name}</div>
            <div class="video-meta">
                Duration: {r['duration']:.1f}s | Resolution: {r['resolution']} | Motion: {r['motion_score']:.1f}
            </div>
            <span class="suspicion-tag {level}">{r['suspicion_level']}</span>
            {reasons_html}
        </div>
        """

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    main()
