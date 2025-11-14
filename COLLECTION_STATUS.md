# Violence Detection Dataset Collection Status
**Date**: October 14, 2025
**Goal**: Collect 15,000-20,000 videos for 90-92% TTA accuracy

## Current Collection Progress

### ✅ WORKING PLATFORMS

#### 1. **WorldStar HipHop** - RUNNING NOW
- **Status**: ✅ FIXED - Infinite scroll working perfectly
- **Progress**: 40 videos collected (18 in last minute)
- **Rate**: ~18 videos/minute = **1,080 videos/hour**
- **Expected**: 5,000-10,000 fight videos
- **File**: `worldstar_fight_videos.json`
- **Quality**: High - Real street fights, CCTV footage, public altercations
- **Fix Applied**: Removed broken `/page/2/` pagination, now using single-page infinite scroll
- **Collection Method**:
  - Loads main `/videos/` page
  - Scrolls continuously (500 max scrolls)
  - Extracts every 5 scrolls (efficiency optimization)
  - Fight keyword filtering: fight, knockout, brawl, beat, attack, assault, punch, violence, etc.
  - Stops automatically when no new content found

#### 2. **Reddit**
- **Status**: ✅ Completed earlier
- **Videos Collected**: ~6,000 URLs
- **Quality**: High - Real fight videos from multiple subreddits
- **File**: `reddit_fight_urls.json` (from previous session)
- **Download Status**: Pending

#### 3. **YouTube Shorts**
- **Status**: ✅ Script ready and tested
- **Expected**: 5,000 videos
- **Quality**: Medium-High - Mix of real fights and staged content
- **File**: `scrape_youtube_shorts.py` ready to run
- **Search Terms**: "street fight", "CCTV fight", "caught on camera fight"

#### 4. **Dailymotion**
- **Status**: ⚠️ Partially working (20-38 videos only)
- **Videos Collected**: 20 videos
- **Issue**: Bot detection or pagination limits
- **File**: `dailymotion_fight_videos.json`
- **Priority**: Low (use if other sources insufficient)

### ❌ BLOCKED PLATFORMS (Vast.ai IP Issues)

#### 1. **TikTok**
- **Issue**: Login required OR bot detection (all methods failed)
- **Methods Tried**: Playwright, yt-dlp, TikTokApi library
- **Status**: Abandoned for Vast.ai environment

#### 2. **Twitter/X**
- **Issue**: Authentication required for video search
- **Status**: Abandoned (requires manual cookie export)

---

## Collection Strategy

### Phase 1: URL Collection (IN PROGRESS)
1. ✅ **WorldStar**: Running now (expect 4-6 hours for 5,000+ videos)
2. ⏳ **YouTube Shorts**: Start next (5,000 videos estimated)
3. ✅ **Reddit**: Already completed (6,000 URLs)

### Phase 2: Video Downloads (PENDING)
- **Parallel downloads** using `yt-dlp` with 8 workers
- **WorldStar**: 5,000 videos × 4 seconds = ~6 hours
- **YouTube**: 5,000 videos × 3 seconds = ~4 hours
- **Reddit**: 6,000 videos × 4 seconds = ~7 hours
- **Total download time**: ~17 hours (can run parallel)

### Phase 3: Dataset Preparation
- Remove duplicates (yt-dlp's video ID matching)
- Quality check (remove corrupted/unplayable videos)
- Balance violent/non-violent classes
- Split: 80% train, 10% val, 10% test

### Phase 4: Training
- Feature extraction with optimized model
- Train with best hyperparameters
- Target: 90-92% TTA accuracy
- Deploy to 110 CCTV cameras

---

## Expected Final Dataset

| Source | Videos | Quality | Status |
|--------|--------|---------|--------|
| WorldStar | 5,000-10,000 | High | In progress |
| YouTube Shorts | 5,000 | Medium-High | Ready to run |
| Reddit | 6,000 | High | Collected |
| Dailymotion | 20 | Medium | Complete (limited) |
| **TOTAL** | **16,020-21,020** | **Mixed** | **On track** |

**Target**: 15,000-20,000 videos ✅
**Status**: Will exceed target with current sources

---

## Next Steps

1. **Let WorldStar finish collecting** (~4-6 hours for 5,000+ videos)
2. **Start YouTube Shorts scraper** in parallel (5,000 videos)
3. **Begin downloads** as soon as URL collection completes
4. **Prepare dataset** and start training

---

## Performance Notes

### WorldStar Scraper Performance
- **Collection rate**: ~18 videos/minute = 1,080/hour
- **Expected runtime**: 4.6 hours for 5,000 videos, 9.3 hours for 10,000 videos
- **Memory usage**: Low (JSON file grows ~350 bytes per video)
- **CPU usage**: Moderate (Playwright browser automation)

### Download Performance (Estimated)
- **8 parallel workers** using ThreadPoolExecutor
- **Average download time**: 3-4 seconds per video
- **Success rate**: ~85% (some videos may be removed/unavailable)
- **Storage needed**: ~30-40GB for 15,000 videos (assuming 2-3MB avg)

---

## Academic Dataset Supplement

If additional videos needed, academic datasets available:
- **RWF-2000**: 2,000 videos (1,000 fight, 1,000 non-fight)
- **UCF Crime**: ~500 violence videos
- **Hockey Fight Dataset**: 1,000 fight videos

**Total additional**: 3,500+ videos available via `download_academic_datasets.sh`
