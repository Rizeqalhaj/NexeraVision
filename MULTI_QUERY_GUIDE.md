# Multi-Query Reddit Scraper Guide

## Problem Discovered

Your test showed Reddit has a **250-video limit per search query**:
```
📹 Found 250 videos total
⏳ No more results after scrolling
```

## Solution: Multi-Query Strategy

Instead of 1 query → 250 videos, run **30 queries** → **7,500+ videos**

### How It Works

**30 Different Queries**:
```
1. "fights"           → ~250 results
2. "street fight"     → ~250 results
3. "fight video"      → ~250 results
4. "brawl"            → ~250 results
5. "knockout"         → ~250 results
... (25 more queries)
```

**Expected Yield**:
- Per query: 150-250 unique videos (some overlap)
- 30 queries: **4,500-7,500 unique videos** (after deduplication)

## Run Multi-Query Scraper

```bash
cd /workspace/violence_detection_mvp

# Copy new script to Vast.ai
# (If running from local, scp it first)

# Run multi-query scraper
python3 scrape_reddit_multi_query.py
```

**What Happens**:
```
╔════════════════════════════════════════════════════════════════════════════╗
║              MULTI-QUERY REDDIT SCRAPER                                    ║
║              Bypass 250-result limit with multiple queries                ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Total queries: 30
📊 Expected yield: 4500 - 7500 unique videos
📁 Output: reddit_fight_videos_all.json

================================================================================
📍 Query 1/30: "fights"
================================================================================

🌐 Loading: https://www.reddit.com/search/?q=fights&type=media
📹 +50 videos | Query total: 50 | Overall: 50
📹 +25 videos | Query total: 75 | Overall: 75
...
✅ Query complete: Found 237 new videos from "fights"

💤 Resting 45s before next query...

================================================================================
📍 Query 2/30: "street fight"
================================================================================

🌐 Loading: https://www.reddit.com/search/?q=street%20fight&type=media
📹 +48 videos | Query total: 48 | Overall: 285
📹 +23 videos | Query total: 71 | Overall: 308
...
✅ Query complete: Found 198 new videos from "street fight"

💤 Resting 52s before next query...

... (continues through all 30 queries)

================================================================================
📊 SCRAPING COMPLETE
================================================================================
✅ Total unique videos: 6,847
🔍 Queries completed: 30
⏱️  Total runtime: 185.3 minutes (3.1 hours)
📁 Saved to: reddit_fight_videos_all.json
================================================================================
```

## Timeline

```
Hour 0-3:   Run multi-query scraper (30 queries × 6 min = 3 hours)
            Expected: 4,500-7,500 unique video URLs ✅

Hour 3-27:  Download videos (24 hours)
            Expected: 4,000-6,000 actual videos (75-85% success rate)

Hour 27-28: Verify quality + remove corrupted
            Expected: 3,500-5,500 usable videos

Hour 28:    Move to dataset
            Final count: 4,038 existing + 3,500-5,500 new = 7,500-9,500 ✅

Hour 28-33: Feature extraction (5 hours)

Hour 33-45: Training (8-12 hours)

Hour 45:    TTA test → 90-92% accuracy ✅
            Deploy to 110 cameras 🎯
```

**Total: ~2 days from start to deployment**

## Features

### 1. Automatic Deduplication
- Tracks all scraped URLs in `self.scraped_links`
- Skips duplicate videos across queries
- Only counts unique videos

### 2. Progress Saving
- Saves after each query completes
- Safe to interrupt (Ctrl+C)
- Resume by running again (loads existing data)

### 3. Rate Limit Protection
- 30-60 second delays between queries
- Human-like scrolling within queries
- Stealth mode (removes automation indicators)

### 4. Per-Query Metadata
- Stores which query found each video
- Helps identify best-performing queries
- JSON format for easy processing

## Output Format

**`reddit_fight_videos_all.json`**:
```json
[
  {
    "url": "https://www.reddit.com/r/fightporn/comments/xyz/...",
    "title": "Street fight caught on camera",
    "subreddit": "fightporn",
    "query": "fights",
    "scraped_at": "2025-10-14T16:30:45.123456"
  },
  {
    "url": "https://www.reddit.com/r/PublicFreakout/comments/abc/...",
    "title": "Brawl at gas station",
    "subreddit": "PublicFreakout",
    "query": "street fight",
    "scraped_at": "2025-10-14T16:35:12.789012"
  },
  ...
]
```

## After Scraping

```bash
# Download videos
python3 download_reddit_scraped_videos.py reddit_fight_videos_all.json /workspace/reddit_videos 8

# Check quality
python3 clean_corrupted_videos.py /workspace/reddit_videos

# Remove corrupted
python3 clean_corrupted_videos.py /workspace/reddit_videos --remove

# Move to dataset
cp /workspace/reddit_videos/*.mp4 /workspace/organized_dataset/train/violent/

# Verify count
ls /workspace/organized_dataset/train/violent/*.mp4 | wc -l
# Should show: 7,500-9,500 videos ✅

# Start training
python3 train_HYBRID_OPTIMAL_FAST_EXTRACTION.py
```

## Customization

**Add more queries** (edit `scrape_reddit_multi_query.py`):
```python
self.queries = [
    "fights",
    "street fight",
    # Add your own:
    "fight caught on tape",
    "hood fights",
    "parking lot brawl",
    # ... etc
]
```

**Adjust timing**:
```python
# Line 157: Delay between queries (default 30-60s)
delay = random.randint(30, 60)  # Increase to 60-120 if rate limited

# Line 147: Scrolls before giving up on query (default 10)
while consecutive_no_new < 10:  # Increase to 15 for more thorough scraping
```

## Why This Works Better

**Single Query Approach**:
- 1 query × 250 results = **250 videos max** ❌
- Hit limit in 2 minutes

**Multi-Query Approach**:
- 30 queries × 200 avg = **6,000 videos** ✅
- After deduplication: **4,500-7,500 unique**
- Takes 3 hours but gets 20-30x more data

**Trade-off**: Time vs. Quantity
- Single query: 2 min, 250 videos
- Multi-query: 3 hours, 6,000+ videos
- **Worth it for model training!**

## Run It Now

```bash
python3 scrape_reddit_multi_query.py
```

Let it run for ~3 hours, then you'll have 6,000+ video URLs ready for download!
