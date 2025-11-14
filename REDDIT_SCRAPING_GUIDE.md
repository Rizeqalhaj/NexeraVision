# Reddit Infinite Scroll Scraping Guide

Complete guide for scraping and downloading violent videos from Reddit to replace corrupted data.

---

## Overview

**Goal**: Collect 7,000+ fight videos from Reddit to replace corrupted dataset

**Two-Step Process**:
1. **Scrape**: Collect video URLs with infinite scroll (undetected)
2. **Download**: Download actual videos from collected URLs

**Expected Timeline**: 2-3 days
- Scraping: 3-6 hours (for 10,000 links)
- Downloading: 1-2 days (depends on Reddit rate limits)

---

## Step 1: Install Dependencies

### On Local Machine (Ubuntu)
```bash
# Install Playwright
pip install --break-system-packages playwright
python3 -m playwright install chromium

# Install yt-dlp for downloading
pip install --break-system-packages yt-dlp

# Or if you have virtualenv
python3 -m venv venv
source venv/bin/activate
pip install playwright yt-dlp
python3 -m playwright install chromium
```

### On Vast.ai (if running there)
```bash
pip install playwright yt-dlp
python3 -m playwright install chromium
```

---

## Step 2: Scrape Reddit URLs (Infinite Scroll)

### Basic Usage

**Run the scraper**:
```bash
cd /home/admin/Desktop/NexaraVision
python3 scrape_reddit_infinite_scroll.py
```

**What happens**:
- Opens Chrome browser (visible by default, see below for headless)
- Loads Reddit search: https://www.reddit.com/search/?q=fights&type=media
- Scrolls infinitely with human-like behavior
- Extracts video links as it scrolls
- Saves to `reddit_fight_videos.json` every 5 minutes
- Stops when: 10,000 videos OR 3 hours OR reached end of results

**Stealth Features**:
- ✅ Random scroll distances (300-800px)
- ✅ Random delays (2-5 seconds between scrolls)
- ✅ Sometimes scrolls back up (human behavior)
- ✅ Random mouse movements
- ✅ Occasional "reading pauses" (3-8 seconds)
- ✅ Breaks every 10 scrolls (10-20 seconds)
- ✅ Removes automation indicators (`webdriver` flag)
- ✅ Realistic user agent (Chrome 120, Windows 10)

### Configuration Options

**Edit the script** (`scrape_reddit_infinite_scroll.py`) to customize:

```python
# At the bottom of the file, in async def main():

REDDIT_URL = "https://www.reddit.com/search/?q=fights&type=media..."  # Change search query
OUTPUT_FILE = "reddit_fight_videos.json"  # Change output filename
MAX_VIDEOS = 10000   # Stop after this many videos
MAX_RUNTIME = 180    # Stop after this many minutes (180 = 3 hours)
```

**Run headless** (no visible browser):
```python
# Line ~147 in scrape_reddit_infinite_scroll.py
browser = await p.chromium.launch(
    headless=True,  # Change False to True
    args=[...]
)
```

### Monitoring Progress

**While running**:
```
🚀 Starting Reddit scraper...
📊 Target: 10000 videos or 180 minutes
📁 Output: reddit_fight_videos.json
✅ Already have: 0 videos

🌐 Loading: https://www.reddit.com/search/?q=fights...
📹 Found 15 new videos | Total: 15 | Runtime: 0.1min
📹 Found 12 new videos | Total: 27 | Runtime: 0.3min
⏳ Scrolling... (1/20 no new) | Total: 27
💤 Taking a short break...
📹 Found 8 new videos | Total: 35 | Runtime: 1.2min
...
```

**Resume after interruption**:
- Script automatically loads existing `reddit_fight_videos.json`
- Skips already scraped URLs
- Continues from where it left off

```bash
# Just run again - it will resume
python3 scrape_reddit_infinite_scroll.py
```

### Output Format

**`reddit_fight_videos.json`**:
```json
[
  {
    "url": "https://www.reddit.com/r/fightporn/comments/xyz123/title",
    "title": "Street fight caught on camera",
    "subreddit": "fightporn",
    "scraped_at": "2025-10-14T10:30:45.123456"
  },
  {
    "url": "https://www.reddit.com/r/PublicFreakout/comments/abc456/title",
    "title": "Fight at mall",
    "subreddit": "PublicFreakout",
    "scraped_at": "2025-10-14T10:31:12.789012"
  },
  ...
]
```

---

## Step 3: Download Videos

### Basic Usage

**After scraping, download videos**:
```bash
python3 download_reddit_scraped_videos.py reddit_fight_videos.json downloaded_reddit_fights 8
```

**Arguments**:
1. `reddit_fight_videos.json` - Input JSON from scraper
2. `downloaded_reddit_fights` - Output directory
3. `8` - Number of parallel downloads (adjust based on CPU)

**What happens**:
- Reads JSON file with URLs
- Downloads videos using yt-dlp
- Saves to `downloaded_reddit_fights/` directory
- Filenames: `{subreddit}_{post_id}_{title}.mp4`
- Logs success to `download_success.txt`
- Logs failures to `download_failed.txt`

### Download Output

**While running**:
```
╔════════════════════════════════════════════════════════════════════════════╗
║                    REDDIT VIDEO DOWNLOADER                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Total videos in JSON: 10000
✅ Already downloaded: 0
📥 To download: 10000
🔧 Parallel workers: 8
📁 Output directory: downloaded_reddit_fights

Downloading: 45%|████████████▌             | 4523/10000 [2:15:30<3:45:12, ✅:4321, ❌:202]
```

**Resume capability**:
- Automatically skips already downloaded videos
- Safe to stop and restart (Ctrl+C)
- Progress tracked in `download_success.txt`

```bash
# Resume after interruption
python3 download_reddit_scraped_videos.py reddit_fight_videos.json downloaded_reddit_fights 8
```

### Handling Failures

**Common failures**:
- Deleted posts (video no longer available)
- Private subreddits
- Age-restricted content
- Rate limiting (temporary)

**Retry failed downloads**:
```bash
# Check failed.txt
cat downloaded_reddit_fights/download_failed.txt

# Manually investigate specific URLs
yt-dlp "https://www.reddit.com/r/..."

# Or reduce parallel workers to avoid rate limits
python3 download_reddit_scraped_videos.py reddit_fight_videos.json downloaded_reddit_fights 4
```

---

## Step 4: Organize Downloaded Videos

### Move to Dataset

**After downloading, organize into your dataset**:

```bash
# Create violent directory if needed
mkdir -p /workspace/organized_dataset/train/violent

# Move downloaded videos
cp downloaded_reddit_fights/*.mp4 /workspace/organized_dataset/train/violent/

# Verify count
ls /workspace/organized_dataset/train/violent/*.mp4 | wc -l
```

### Verify Video Quality

**Run corruption check on new videos**:
```bash
# Copy checker script to downloaded directory
python3 clean_corrupted_videos.py downloaded_reddit_fights

# Remove any corrupted
python3 clean_corrupted_videos.py downloaded_reddit_fights --remove
```

**Expected success rate**: 85-95% (some videos may be corrupted on Reddit's end)

---

## Advanced Usage

### Multiple Search Queries

**Scrape different keywords**:

```python
# Edit scrape_reddit_infinite_scroll.py, change REDDIT_URL to:

# Query 1: "fights"
"https://www.reddit.com/search/?q=fights&type=media"

# Query 2: "street fight"
"https://www.reddit.com/search/?q=street%20fight&type=media"

# Query 3: "brawl"
"https://www.reddit.com/search/?q=brawl&type=media"

# Query 4: "knockout"
"https://www.reddit.com/search/?q=knockout&type=media"
```

**Run multiple scrapers in parallel**:
```bash
# Terminal 1
python3 scrape_reddit_infinite_scroll.py  # fights
# Output: reddit_fight_videos.json

# Terminal 2
# Edit script to use different URL and output file
# REDDIT_URL = "...street%20fight..."
# OUTPUT_FILE = "reddit_street_fight.json"
python3 scrape_reddit_infinite_scroll.py

# Terminal 3
# REDDIT_URL = "...brawl..."
# OUTPUT_FILE = "reddit_brawl.json"
python3 scrape_reddit_infinite_scroll.py
```

**Combine JSON files**:
```bash
python3 << 'EOF'
import json
from pathlib import Path

# Load all JSON files
all_videos = []
seen_urls = set()

for json_file in Path('.').glob('reddit_*.json'):
    with open(json_file) as f:
        videos = json.load(f)
        for v in videos:
            if v['url'] not in seen_urls:
                all_videos.append(v)
                seen_urls.add(v['url'])

# Save combined
with open('reddit_all_combined.json', 'w') as f:
    json.dump(all_videos, f, indent=2)

print(f"Combined {len(all_videos)} unique videos")
EOF
```

### Optimize Download Speed

**Faster downloads** (if Reddit doesn't rate limit):
```bash
# Increase parallel workers
python3 download_reddit_scraped_videos.py reddit_fight_videos.json downloaded_reddit_fights 16

# Run multiple downloaders on chunks
# Split JSON into chunks first, then run parallel downloaders
```

**Avoid rate limits** (slower but more reliable):
```bash
# Reduce parallel workers
python3 download_reddit_scraped_videos.py reddit_fight_videos.json downloaded_reddit_fights 2
```

---

## Troubleshooting

### Playwright Issues

**Error**: `playwright._impl._api_types.Error: Executable doesn't exist`
```bash
python3 -m playwright install chromium
```

**Error**: `Could not find browser`
```bash
# Re-install playwright
pip uninstall playwright
pip install --break-system-packages playwright
python3 -m playwright install chromium
```

### Reddit Detection

**If scraper gets blocked**:
1. Increase delays in script:
   ```python
   self.scroll_delay = (5, 10)  # Slower scrolling
   self.action_delay = (2, 4)   # Longer pauses
   ```

2. Run with visible browser (headless=False) to verify behavior

3. Add random IP rotation (VPN or proxy)

### yt-dlp Issues

**Error**: `yt-dlp: command not found`
```bash
pip install --break-system-packages yt-dlp
```

**Error**: `Unable to download video`
```bash
# Update yt-dlp
pip install --upgrade --break-system-packages yt-dlp

# Try single video manually
yt-dlp "https://www.reddit.com/r/..." -v  # verbose mode
```

**Rate limiting**:
```bash
# Add delays between downloads (edit download script):
time.sleep(0.5)  # Change from 0.1 to 0.5 seconds
```

---

## Expected Results

### Realistic Expectations

**Scraping (3-6 hours)**:
- Collect: 8,000-12,000 URLs
- Success rate: 95%+ (URL extraction)
- Output: JSON file with links

**Downloading (1-2 days)**:
- Success rate: 75-85% (some videos deleted/private)
- Speed: 10-30 videos/minute (depends on rate limits)
- Expected yield: 6,000-10,000 actual videos

**Quality**:
- 85-95% playable videos
- Need to filter out corrupted (use `clean_corrupted_videos.py`)
- Final yield: 5,000-8,000 usable violent videos ✅

### Timeline

```
Day 1: Scraping
00:00 - Start scraper
03:00 - Reach 10,000 URLs
03:30 - Start downloader (parallel)

Day 2: Downloading
00:00 - ~4,000 videos downloaded
12:00 - ~7,000 videos downloaded
18:00 - ~9,000 videos downloaded

Day 3: Verification
00:00 - All downloads complete
02:00 - Run corruption check
04:00 - Move to dataset
06:00 - Start feature extraction 🎯
```

---

## Integration with Training Pipeline

**After collecting videos**:

```bash
# 1. Verify no corruption
python3 clean_corrupted_videos.py /workspace/organized_dataset --remove

# 2. Check new counts
ls /workspace/organized_dataset/train/violent/*.mp4 | wc -l
# Should show: 11,000+ videos (4,038 existing + 7,000 new)

# 3. Start feature extraction
python3 train_HYBRID_OPTIMAL_FAST_EXTRACTION.py

# 4. Expected results after training:
#    - 90-92% TTA accuracy ✅
#    - Violent detection: 88-90% ✅
#    - Ready for 110-camera deployment 🎯
```

---

## Quick Reference

```bash
# Install dependencies
pip install --break-system-packages playwright yt-dlp
python3 -m playwright install chromium

# Scrape URLs (3-6 hours)
python3 scrape_reddit_infinite_scroll.py
# Output: reddit_fight_videos.json (10,000 URLs)

# Download videos (1-2 days)
python3 download_reddit_scraped_videos.py reddit_fight_videos.json downloaded_reddit_fights 8
# Output: downloaded_reddit_fights/*.mp4 (6,000-8,000 videos)

# Verify quality
python3 clean_corrupted_videos.py downloaded_reddit_fights --remove
# Removes corrupted, keeps 5,000-7,000 usable videos ✅

# Move to dataset
cp downloaded_reddit_fights/*.mp4 /workspace/organized_dataset/train/violent/

# Continue with training pipeline
python3 train_HYBRID_OPTIMAL_FAST_EXTRACTION.py
```
