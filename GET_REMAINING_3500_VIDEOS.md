# Get 3,500 More Non-Violence Videos

## 🎯 Current Status

**Violence:** 14,000 videos ✅
**Non-violence:** 10,454 videos ⚠️
**Gap:** Need **3,546 more** non-violence videos for balance

---

## 🚀 Strategy: Reddit Scraping

### Why Reddit?
- ✅ **Massive content** - Millions of videos
- ✅ **Easy filtering** - Subreddit-based organization
- ✅ **High success rate** - Videos actually exist
- ✅ **Free API** - No rate limiting issues
- ✅ **Diverse content** - Many non-violence categories

### Expected Yield
**Target subreddits:** 30+ subreddits
**Videos per subreddit:** 50-200
**Expected total:** 2,000-4,000 URLs
**Success rate:** ~85%
**Final videos:** **1,700-3,400** ✅

---

## 📋 Method 1: gallery-dl (Recommended - Faster)

### Installation
```bash
pip install gallery-dl
```

### Usage
```bash
cd /home/admin/Desktop/NexaraVision

python scrape_reddit_nonviolence.py gallery-dl reddit_nonviolence_urls.txt
```

### Expected Output
```
Reddit Non-Violence Video Scraper
======================================================================
Subreddits: 30
Time filter: all
Limit per subreddit: 100
======================================================================

[1/30] Processing r/MadeMeSmile
✓ Found: 87 | New unique: 87 | Total: 87

[2/30] Processing r/HumansBeingBros
✓ Found: 124 | New unique: 112 | Total: 199

[3/30] Processing r/sports
✓ Found: 156 | New unique: 143 | Total: 342

...

[30/30] Processing r/happycrowds
✓ Found: 45 | New unique: 38 | Total: 2,847

SCRAPING COMPLETE
Total unique URLs: 2,847
```

### Timeline
- **Scraping:** 30-45 minutes (30 subreddits × 1-1.5 min each)
- **Total:** ~45 minutes to get 2,000-3,000 URLs

---

## 📋 Method 2: Pushshift API (Slower but More Detailed)

### No Installation Needed
Uses Python `requests` (already installed)

### Usage
```bash
python scrape_reddit_nonviolence.py pushshift reddit_nonviolence_urls.txt
```

### Features
- ✅ More filtering (by title, score, date)
- ✅ Saves post metadata (title, subreddit, score)
- ✅ Better violence filtering
- ⏱️ Slower (2-3 hours for 30 subreddits)

---

## 🎯 Target Subreddits (30 total)

### Wholesome/Positive (Expected: 400-600 videos)
- r/MadeMeSmile
- r/HumansBeingBros
- r/wholesome
- r/UpliftingNews
- r/aww

### Sports (Expected: 600-900 videos)
- r/sports
- r/basketball
- r/soccer
- r/baseball
- r/football
- r/tennis
- r/golf
- r/hockey

### Activities (Expected: 300-500 videos)
- r/dancing
- r/music
- r/concerts
- r/festivals

### Interesting Content (Expected: 400-600 videos)
- r/BeAmazed
- r/Damnthatsinteresting
- r/oddlysatisfying
- r/nextfuckinglevel
- r/toptalent

### Animals (Expected: 300-400 videos)
- r/AnimalsBeingBros
- r/AnimalsBeingDerps
- r/Eyebleach

---

## 🔧 Alternative Sources (If Reddit Not Enough)

### Option 1: Kaggle Datasets
```bash
# Search for non-violence video datasets
kaggle datasets list -s "normal activities"
kaggle datasets list -s "daily life"
kaggle datasets list -s "cctv normal"

# Example datasets:
# - UCF101 (action recognition - filter non-violent actions)
# - Kinetics-700 (filter sports, daily activities)
# - ActivityNet (filter non-violent categories)
```

### Option 2: More Websites
```bash
# Use your existing scrapers on different sites
python scrape_final_working.py 'https://another-video-site.com' 'peaceful activities' 100

# Or scrape tag pages
python scrape_tag_page_fixed.py 'https://site.com' '/tag/wholesome/' 150
```

### Option 3: YouTube (Specific Channels)
```bash
# Scrape specific YouTube channels with non-violence content
# Use yt-dlp to download channel playlists

yt-dlp --get-id "https://www.youtube.com/c/channelname/videos" | \
    head -500 > youtube_channel_ids.txt

# Convert to URLs
while read id; do
    echo "https://www.youtube.com/watch?v=$id"
done < youtube_channel_ids.txt > youtube_urls.txt
```

---

## 📊 Combined Strategy for 3,500+ Videos

### Phase 1: Reddit (Main Source)
**Method:** gallery-dl
**Expected:** 2,000-3,000 URLs
**Time:** 45 minutes
**Command:**
```bash
python scrape_reddit_nonviolence.py gallery-dl reddit_urls.txt
```

### Phase 2: Download Reddit Videos
**Expected success:** 85%
**Videos:** 1,700-2,550
**Time:** 3-4 hours (with 30 workers)
**Command:**
```bash
python download_videos_parallel_robust.py reddit_urls.txt reddit_videos/ 30
```

### Phase 3: Check Balance
```bash
# Count downloaded videos
ls reddit_videos/*.mp4 | wc -l

# Calculate remaining needed
# If you get 2,000 from Reddit:
# 10,454 + 2,000 = 12,454
# Still need: 14,000 - 12,454 = 1,546 more
```

### Phase 4: Additional Sources (If Needed)
If Reddit gives less than 3,500:

**Option A: Kaggle Datasets**
```bash
# Download UCF101 or Kinetics datasets
# Filter non-violent categories
# Expected: 1,000-2,000 videos
```

**Option B: More Websites**
```bash
# Use existing scrapers on new sites
# Expected: 500-1,500 videos
```

---

## 🎯 Quick Win: Start Reddit Scraping NOW

```bash
cd /home/admin/Desktop/NexaraVision

# Install gallery-dl if needed
pip install gallery-dl

# Start scraping (45 minutes)
python scrape_reddit_nonviolence.py gallery-dl reddit_nonviolence_urls.txt

# While scraping runs, prepare download command
# After scraping completes:
python download_videos_parallel_robust.py \
    reddit_nonviolence_urls.txt \
    reddit_nonviolence_videos/ \
    30
```

**Expected Timeline:**
- ⏱️ **Reddit scraping:** 45 min → 2,000-3,000 URLs
- ⏱️ **Download videos:** 3-4 hours → 1,700-2,550 videos
- ✅ **Total:** 4-5 hours to get most/all remaining videos

---

## 📈 Final Dataset Projection

**Best Case (Reddit gives 2,500 videos):**
- Violence: 14,000
- Non-violence: 10,454 + 2,500 = **12,954**
- Still need: 1,046 more (easy with Kaggle datasets)

**Good Case (Reddit gives 2,000 videos):**
- Violence: 14,000
- Non-violence: 10,454 + 2,000 = **12,454**
- Still need: 1,546 more (Kaggle + more scraping)

**Minimum Case (Reddit gives 1,500 videos):**
- Violence: 14,000
- Non-violence: 10,454 + 1,500 = **11,954**
- Still need: 2,046 more (multiple sources needed)

---

## ✅ Recommended Action Plan

**RIGHT NOW:**
```bash
# 1. Install gallery-dl
pip install gallery-dl

# 2. Start Reddit scraping (45 min)
python scrape_reddit_nonviolence.py gallery-dl reddit_urls.txt

# 3. Start download (3-4 hours)
python download_videos_parallel_robust.py reddit_urls.txt reddit_videos/ 30

# 4. Check results
ls reddit_videos/*.mp4 | wc -l

# 5. If still need more: Use Kaggle or scrape more sites
```

**Expected result:** **2,000-2,500 videos** from Reddit → Close to balance! 🎯

Let's get those final 3,500 videos and hit **93-95% accuracy**! 🚀
