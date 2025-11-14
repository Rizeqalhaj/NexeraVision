# YouTube Shorts Fight Video Scraper Guide

Complete guide for collecting fight videos from YouTube Shorts to supplement Reddit data.

---

## Why YouTube Shorts?

**Advantages**:
- ✅ Short duration (15-60 seconds) - perfect for training
- ✅ Vertical format (9:16) - works great for surveillance cameras
- ✅ High quality (1080p)
- ✅ Recent uploads - fresh content
- ✅ Different from Reddit - more diversity
- ✅ Fast to download (small file sizes)

**Expected Yield**:
- 20 queries × 200-300 shorts = **4,000-6,000 unique shorts**
- Combined with Reddit: **10,000-12,000 total videos** ✅

---

## Step 1: Install Dependencies

```bash
# Already have playwright from Reddit scraping
# Just need yt-dlp
pip install --break-system-packages yt-dlp
```

---

## Step 2: Scrape YouTube Shorts URLs

### Run the Scraper

```bash
cd /workspace/violence_detection_mvp
python3 scrape_youtube_shorts_fights.py
```

**What Happens**:
```
╔════════════════════════════════════════════════════════════════════════════╗
║              YOUTUBE SHORTS FIGHT VIDEO SCRAPER                            ║
║              Collect short-form fight videos from YouTube                 ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Total queries: 20
📊 Expected: ~200-300 shorts per query
📊 Total expected: 4,000-6,000 unique shorts

================================================================================
📍 Query 1/20: "fight"
================================================================================

🌐 Loading: https://www.youtube.com/results?search_query=fight&sp=EgIYAQ...
Scrolling and collecting Shorts...
📹 +15 shorts | Query total: 15 | Overall: 15
📹 +12 shorts | Query total: 27 | Overall: 27
📹 +18 shorts | Query total: 45 | Overall: 45
...
📹 +8 shorts | Query total: 237 | Overall: 237
⏳ Scrolling... (1/10) | Got: 237
⏳ Scrolling... (5/10) | Got: 237
   ✅ Got 237 shorts, moving on

✅ Query complete: Found 237 new shorts from "fight"

💤 Resting 32s before next query...

================================================================================
📍 Query 2/20: "street fight"
================================================================================
...
```

**Timeline**: 20 queries × 4 min = **~1.5 hours**

**Output**: `youtube_shorts_fights.json`

---

## Step 3: Download Shorts Videos

### Run the Downloader

```bash
python3 download_youtube_shorts.py youtube_shorts_fights.json /workspace/youtube_shorts 8
```

**What Happens**:
```
╔════════════════════════════════════════════════════════════════════════════╗
║                    YOUTUBE SHORTS DOWNLOADER                               ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Total shorts in JSON: 5,247
✅ Already downloaded: 0
📥 To download: 5,247
🔧 Parallel workers: 8

Downloading: 45%|████████████▌             | 2,361/5,247 [1:23:15<1:42:31, ✅:2,103, ❌:258]
```

**Timeline**: 5,000 shorts × ~3 seconds = **~4 hours**

**Why So Fast?**:
- Shorts are small (15-60 seconds, 10-30 MB)
- YouTube has fast servers
- 8 parallel downloads

---

## Step 4: Combine with Reddit Videos

### Option A: Simple Combination

```bash
# Copy YouTube Shorts to dataset
cp /workspace/youtube_shorts/*.mp4 /workspace/organized_dataset/train/violent/

# Copy Reddit videos
cp /workspace/reddit_videos/*.mp4 /workspace/organized_dataset/train/violent/

# Check final count
ls /workspace/organized_dataset/train/violent/*.mp4 | wc -l
# Should show: 10,000-12,000 videos!
```

### Option B: Verify Quality First

```bash
# Check YouTube Shorts quality
python3 clean_corrupted_videos.py /workspace/youtube_shorts

# Check Reddit videos quality
python3 clean_corrupted_videos.py /workspace/reddit_videos

# Remove corrupted from both
python3 clean_corrupted_videos.py /workspace/youtube_shorts --remove
python3 clean_corrupted_videos.py /workspace/reddit_videos --remove

# Then combine
cp /workspace/youtube_shorts/*.mp4 /workspace/organized_dataset/train/violent/
cp /workspace/reddit_videos/*.mp4 /workspace/organized_dataset/train/violent/
```

---

## Complete Timeline

### Parallel Collection (RECOMMENDED)

**Run both scrapers simultaneously**:

```bash
# Terminal 1: Reddit scraper
python3 scrape_reddit_multi_query.py
# → 2.5 hours, 6,000 URLs

# Terminal 2: YouTube Shorts scraper
python3 scrape_youtube_shorts_fights.py
# → 1.5 hours, 5,000 URLs

# Both complete in ~2.5 hours (limited by Reddit)
```

**Then download both**:

```bash
# Terminal 1: Reddit downloader
python3 download_reddit_scraped_videos.py reddit_fight_videos_all.json /workspace/reddit_videos 8
# → ~12 hours, 5,000 videos

# Terminal 2: YouTube Shorts downloader
python3 download_youtube_shorts.py youtube_shorts_fights.json /workspace/youtube_shorts 8
# → ~4 hours, 4,500 videos

# Both complete in ~12 hours (limited by Reddit)
```

### Sequential Collection

```
Hour 0-2.5:   Reddit scraping (6,000 URLs)
Hour 2.5-4:   YouTube scraping (5,000 URLs)
Hour 4-16:    Reddit downloading (5,000 videos)
Hour 4-8:     YouTube downloading (4,500 videos) [parallel]
Hour 16-17:   Quality check both
Hour 17:      Combine datasets
Hour 17-22:   Feature extraction
Hour 22-34:   Training
Hour 34:      90-92% accuracy → Deploy! 🎯
```

**Total: ~34 hours (~1.5 days) from start to deployment**

---

## Expected Final Dataset

```
Original dataset:
- Violent: 10,995 videos
- Corrupted: 6,957 videos (63%)
- Remaining: 4,038 videos ❌

After collection:
- Reddit videos: ~5,000
- YouTube Shorts: ~4,500
- Total new: ~9,500
- Final: 4,038 + 9,500 = 13,538 videos ✅

Result: 123% of original dataset! 🎉
```

---

## Why This Combination is Perfect

**Diversity**:
- Reddit: User-uploaded real fights (street, school, bars)
- YouTube Shorts: Mix of real + compilation + viral clips
- Combined: Maximum pattern diversity for model training

**Quality**:
- Reddit: Raw, unedited footage (realistic)
- YouTube Shorts: High-quality, well-lit (better than surveillance)
- Combined: Model learns both low and high quality

**Format**:
- Reddit: Various aspect ratios
- YouTube Shorts: Vertical 9:16 (perfect for mobile/surveillance)
- Combined: Model handles all formats

**Duration**:
- Reddit: 10-180 seconds (variable)
- YouTube Shorts: 15-60 seconds (consistent)
- Combined: Training data covers all durations

---

## Troubleshooting

### YouTube Shorts Not Found

**Problem**: Search returns 0 results

**Solution**: YouTube may require cookies/login

```python
# Update scraper to accept cookies
# Add to context creation:
context = await browser.new_context(
    user_agent='...',
    storage_state='youtube_cookies.json'  # Export from your browser
)
```

### "Sign in to confirm you're not a bot"

**Problem**: YouTube blocks automated access

**Solution 1**: Add delays
```python
await asyncio.sleep(10)  # Wait longer between queries
```

**Solution 2**: Use residential proxies (if needed)

### Download Fails: "Video unavailable"

**Problem**: Some shorts are region-locked or removed

**Expected**: 10-15% failure rate is normal

**Solution**: Already handled - failed URLs logged to `download_failed.txt`

---

## Run Both Scrapers Now

### Quick Start

```bash
cd /workspace/violence_detection_mvp

# Terminal 1 (or tmux/screen session 1)
python3 scrape_reddit_multi_query.py

# Terminal 2 (or tmux/screen session 2)
python3 scrape_youtube_shorts_fights.py
```

**Let them run for 2-3 hours**, then start downloading!

---

## Output Files

```
/workspace/violence_detection_mvp/
├── reddit_fight_videos_all.json          # Reddit URLs
├── youtube_shorts_fights.json            # YouTube URLs
├── reddit_videos/                        # Downloaded Reddit videos
│   ├── *.mp4
│   ├── download_success.txt
│   └── download_failed.txt
├── youtube_shorts/                       # Downloaded YouTube Shorts
│   ├── short_*.mp4
│   ├── download_success.txt
│   └── download_failed.txt
└── organized_dataset/
    └── train/
        └── violent/
            ├── [4,038 existing]
            ├── [5,000 from Reddit]
            └── [4,500 from YouTube]
            = 13,538 total ✅
```

---

## Next Steps After Collection

```bash
# 1. Verify total count
ls /workspace/organized_dataset/train/violent/*.mp4 | wc -l

# 2. Start feature extraction
python3 train_HYBRID_OPTIMAL_FAST_EXTRACTION.py

# 3. Expected results
#    - Training: 88-92% validation accuracy
#    - TTA: 90-92% test accuracy
#    - Ready for 110-camera deployment! 🎯
```
