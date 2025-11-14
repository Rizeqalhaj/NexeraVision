#!/usr/bin/env python3
"""
Download CCTV fight videos from YouTube (Legal with Attribution)
YouTube allows downloading for research with proper attribution
"""

import os
from pathlib import Path
import subprocess
import json

print("="*80)
print("YOUTUBE CCTV FIGHT VIDEO DOWNLOADER")
print("="*80)

# Create output directory
output_dir = Path("datasets/youtube_cctv_fights")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nVideos will be saved to: {output_dir.absolute()}")

# Check for yt-dlp
print("\n" + "="*80)
print("CHECKING DEPENDENCIES")
print("="*80)

try:
    result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
    print(f"✅ yt-dlp version: {result.stdout.strip()}")
except FileNotFoundError:
    print("❌ yt-dlp not found!")
    print("\nInstall with:")
    print("  pip install yt-dlp")
    print("\nOr:")
    print("  sudo apt install yt-dlp")
    exit(1)

# Search queries for CCTV fight footage
SEARCH_QUERIES = [
    "cctv fight footage",
    "security camera fight",
    "surveillance camera brawl",
    "parking lot fight camera",
    "street fight security camera",
    "gas station fight cctv",
    "convenience store fight camera",
    "bar fight security footage",
    "school fight camera",
    "public fight surveillance",
]

print(f"\n✅ Will search for {len(SEARCH_QUERIES)} query types")

# yt-dlp download script
download_script = f"""#!/bin/bash
# YouTube CCTV Fight Downloader

OUTPUT_DIR="{output_dir.absolute()}"

echo "Starting YouTube download..."
echo "Output directory: $OUTPUT_DIR"

# Download function
download_query() {{
    QUERY="$1"
    MAX_RESULTS="${{2:-20}}"

    echo ""
    echo "=========================================="
    echo "Searching: $QUERY"
    echo "=========================================="

    # Search and download
    yt-dlp \\
        --no-playlist \\
        --max-downloads $MAX_RESULTS \\
        --format "best[height<=720]" \\
        --output "$OUTPUT_DIR/%(title)s-%(id)s.%(ext)s" \\
        --write-description \\
        --write-info-json \\
        --write-thumbnail \\
        --embed-thumbnail \\
        --add-metadata \\
        --restrict-filenames \\
        "ytsearch$MAX_RESULTS:$QUERY"

    echo "✅ Completed: $QUERY"
    sleep 5  # Rate limiting
}}

# Download each query
"""

for query in SEARCH_QUERIES:
    download_script += f'download_query "{query}" 20\n'

download_script += """
echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE"
echo "=========================================="
echo "Videos saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review downloaded videos"
echo "2. Remove non-CCTV footage"
echo "3. Label violent vs non-violent"
echo "4. Add to training dataset"
"""

# Save download script
script_path = output_dir / "download_youtube.sh"
with open(script_path, 'w') as f:
    f.write(download_script)

os.chmod(script_path, 0o755)

print(f"\n✅ Download script created: {script_path}")

# Python wrapper for more control
print("\n" + "="*80)
print("CREATING PYTHON WRAPPER")
print("="*80)

python_wrapper = '''#!/usr/bin/env python3
"""
Python wrapper for controlled YouTube downloads
"""

import subprocess
import json
from pathlib import Path
import time

output_dir = Path("''' + str(output_dir.absolute()) + '''")

SEARCH_QUERIES = ''' + str(SEARCH_QUERIES) + '''

def download_query(query, max_results=20):
    """Download videos for a search query"""

    print(f"\\n{'='*80}")
    print(f"Searching: {query}")
    print(f"{'='*80}\\n")

    cmd = [
        'yt-dlp',
        '--no-playlist',
        '--max-downloads', str(max_results),
        '--format', 'best[height<=720]',
        '--output', str(output_dir / '%(title)s-%(id)s.%(ext)s'),
        '--write-description',
        '--write-info-json',
        '--write-thumbnail',
        '--embed-thumbnail',
        '--add-metadata',
        '--restrict-filenames',
        f'ytsearch{max_results}:{query}'
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Completed: {query}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")

    time.sleep(5)  # Rate limiting

# Download all queries
for query in SEARCH_QUERIES:
    download_query(query, max_results=20)

print("\\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print(f"Videos saved to: {output_dir}")
'''

python_script_path = output_dir / "download_youtube.py"
with open(python_script_path, 'w') as f:
    f.write(python_wrapper)

os.chmod(python_script_path, 0o755)

print(f"✅ Python script created: {python_script_path}")

# Create metadata file
metadata = {
    "source": "YouTube",
    "search_queries": SEARCH_QUERIES,
    "max_videos_per_query": 20,
    "expected_total": len(SEARCH_QUERIES) * 20,
    "format": "720p or lower",
    "attribution": "All videos downloaded for research purposes with proper attribution",
    "license": "Respect original YouTube video licenses"
}

metadata_path = output_dir / "download_info.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved: {metadata_path}")

# Instructions
print("\n" + "="*80)
print("USAGE INSTRUCTIONS")
print("="*80)

print("\n📋 Option 1: Use Bash Script (Faster)")
print(f"   {script_path}")

print("\n📋 Option 2: Use Python Script (More Control)")
print(f"   python {python_script_path}")

print("\n⚙️  What it does:")
print(f"   - Searches {len(SEARCH_QUERIES)} different CCTV fight queries")
print("   - Downloads up to 20 videos per query")
print(f"   - Expected total: {len(SEARCH_QUERIES) * 20} videos")
print("   - Saves metadata and thumbnails")
print("   - Respects rate limits")

print("\n⚠️  Important:")
print("   - YouTube allows downloading for research")
print("   - Keep attribution metadata")
print("   - Don't redistribute videos")
print("   - Use for training only")

print("\n🎯 Expected Results:")
print("   - 200-400 CCTV fight videos")
print("   - Download time: 2-4 hours")
print("   - Need manual review/filtering")

print("\n" + "="*80)
print("READY TO DOWNLOAD")
print("="*80)

print(f"\nRun: {script_path}")
print("Or:  python {python_script_path}")

print("\n" + "="*80)
