# Additional Video Platforms for Fight/Violence Data Collection

Comprehensive guide to collecting CCTV-style fight videos from multiple platforms.

---

## Platform Priority Matrix

### Tier 1: High Priority (Build These) ✅

#### 1. **TikTok** 🔥 **HIGHEST PRIORITY**
**Why**:
- ✅ Massive CCTV fight video library
- ✅ Short-form (15-60 sec) - perfect for training
- ✅ High engagement = popular content surfaces
- ✅ Vertical format (matches surveillance)
- ✅ Hashtags: #cctv #securitycamera #caughtoncamera

**Expected Yield**: 5,000-10,000 unique videos

**Hashtags to Scrape**:
```
#cctv #cctvcamera #securitycamera #securityfootage
#caughtoncamera #surveillance #fightcaughtoncamera
#streetfight #fight #brawl #knockout #fighting
#cctvfight #cctvfootage #surveillancecamera
```

**Scraping Method**: Playwright + infinite scroll
**Difficulty**: Medium (requires handling video player)
**Build Priority**: 🔥 #1

---

#### 2. **Twitter/X** 🔥 **HIGH PRIORITY**
**Why**:
- ✅ Real-time fight footage (news, viral clips)
- ✅ CCTV footage frequently shared
- ✅ High quality, diverse sources
- ✅ Easy to scrape with API/web scraping

**Expected Yield**: 3,000-5,000 unique videos

**Search Terms**:
```
"cctv fight" "security camera"
"caught on camera" "surveillance footage"
"fight video" "street fight cctv"
from:worldstar OR from:fightcompilation
```

**Scraping Method**: Twitter API v2 or Playwright
**Difficulty**: Easy-Medium
**Build Priority**: 🔥 #2

---

#### 3. **Instagram Reels** 🔥 **HIGH PRIORITY**
**Why**:
- ✅ Similar to TikTok (short-form)
- ✅ CCTV fight compilations popular
- ✅ High quality videos
- ✅ Vertical format

**Expected Yield**: 4,000-6,000 unique videos

**Hashtags**:
```
#cctv #securitycamera #caughtoncamera #fight
#streetfight #fightvideos #surveillance
```

**Scraping Method**: Playwright + scroll
**Difficulty**: Medium-Hard (Instagram has strong anti-bot)
**Build Priority**: 🔥 #3

---

### Tier 2: Good Sources (Build If Time)

#### 4. **Telegram Channels**
**Why**:
- ✅ Uncensored fight content
- ✅ CCTV footage channels exist
- ✅ High-quality, curated collections

**Expected Yield**: 2,000-4,000 videos

**Channels to Join**:
```
- Street Fights
- CCTV Footage
- Security Camera Videos
- Fight Compilations
```

**Scraping Method**: Telethon (Telegram API)
**Difficulty**: Easy
**Build Priority**: #4

---

#### 5. **Vimeo**
**Why**:
- ✅ High-quality videos
- ✅ Documentary/news CCTV footage
- ✅ Less crowded = unique content

**Expected Yield**: 1,000-2,000 videos

**Search Terms**:
```
"cctv fight" "security camera violence"
"surveillance footage" "caught on camera"
```

**Scraping Method**: Vimeo API or Playwright
**Difficulty**: Easy
**Build Priority**: #5

---

#### 6. **Dailymotion**
**Why**:
- ✅ Alternative to YouTube
- ✅ Less restrictive content policy
- ✅ Fight compilations available

**Expected Yield**: 1,500-3,000 videos

**Scraping Method**: Dailymotion API or Playwright
**Difficulty**: Easy
**Build Priority**: #6

---

### Tier 3: Niche/Lower Priority

#### 7. **LiveLeak Alternatives** (Kaotic, Documenting Reality)
**Why**: Extreme violence footage (use carefully)
**Expected Yield**: 500-1,000 videos
**Build Priority**: #7 (ethical considerations)

#### 8. **Facebook/Meta**
**Why**: Public pages share CCTV footage
**Expected Yield**: 1,000-2,000 videos
**Difficulty**: Hard (strong anti-scraping)
**Build Priority**: #8

#### 9. **Imgur**
**Why**: GIF/video hosting with fight content
**Expected Yield**: 500-1,000 videos
**Build Priority**: #9

---

## Recommended Build Order

### Phase 1: Quick Wins (This Week)
**Goal**: Collect 15,000-20,000 videos

```
1. Reddit (done) ✅          → 6,000 videos
2. YouTube Shorts (done) ✅  → 5,000 videos
3. TikTok (build now) 🔥     → 8,000 videos
4. Twitter (build now) 🔥    → 4,000 videos

Total: 23,000 videos ✅
After dedup + quality: 18,000-20,000 usable
```

### Phase 2: Expansion (Next Week)
**Goal**: Add diversity

```
5. Instagram Reels           → 5,000 videos
6. Telegram                  → 3,000 videos
7. Vimeo                     → 1,500 videos

Total: +9,500 videos
Combined: 27,000+ raw videos
```

---

## Platform-Specific Implementation

### TikTok Scraper (Priority #1)

**Key Features**:
```python
# TikTok-specific selectors
video_selector = 'div[class*="DivVideoWrapper"]'
hashtag_url = f"https://www.tiktok.com/tag/{hashtag}"

# Infinite scroll logic
while videos_collected < target:
    scroll_down()
    extract_video_urls()
    await asyncio.sleep(2)
```

**Challenges**:
- Video URLs are dynamic (need to extract from page)
- Rate limiting (use delays)
- Login may be required for full access

**Solution**: Playwright with stealth mode

---

### Twitter/X Scraper (Priority #2)

**Option A: Twitter API v2** (Recommended)
```python
import tweepy

# Search for videos
query = "cctv fight -is:retweet has:videos"
tweets = client.search_recent_tweets(
    query=query,
    max_results=100,
    expansions=['attachments.media_keys'],
    media_fields=['url', 'variants']
)
```

**Option B: Web Scraping** (If no API access)
```python
# Use Playwright to scrape search results
url = f"https://twitter.com/search?q=cctv%20fight&f=video"
```

**Advantages**:
- Real-time content
- High quality
- Easy to filter by engagement

---

### Instagram Reels Scraper (Priority #3)

**Challenges**:
- Instagram has strong anti-bot detection
- Requires login
- Rate limits

**Solution**:
```python
# Use instaloader library
from instaloader import Instaloader, Hashtag

L = Instaloader()
# L.login(user, passwd)  # May be required

hashtag = Hashtag.from_name(L.context, 'cctv')
for post in hashtag.get_posts():
    if post.is_video:
        L.download_post(post, target='downloads')
```

**Alternative**: Playwright with cookies from real browser

---

## Expected Total Yield

### Conservative Estimate (Phase 1 Only)
```
Reddit:         5,000 usable videos
YouTube Shorts: 4,500 usable videos
TikTok:         7,000 usable videos
Twitter:        3,500 usable videos

Total: 20,000 unique videos
After quality check: 17,000-18,000 ✅
```

### Aggressive Estimate (Phase 1 + 2)
```
Phase 1:        18,000 videos
Instagram:      4,500 videos
Telegram:       2,500 videos
Vimeo:          1,200 videos

Total: 26,200 unique videos
After quality check: 22,000-24,000 ✅
```

---

## Quick Decision Matrix

| Platform | Difficulty | Yield | CCTV Focus | Build It? |
|----------|-----------|-------|------------|-----------|
| Reddit | Easy | 6K | Medium | ✅ Done |
| YouTube Shorts | Easy | 5K | High | ✅ Done |
| **TikTok** | Medium | 8K | **High** | 🔥 **YES** |
| **Twitter** | Easy | 4K | **High** | 🔥 **YES** |
| Instagram | Hard | 5K | Medium | Maybe |
| Telegram | Easy | 3K | Medium | Maybe |
| Vimeo | Easy | 2K | Low | Later |
| Others | Varies | <2K | Low | Skip |

---

## Recommendation: Build TikTok + Twitter Next

### Why These Two?

**TikTok**:
- ✅ Massive CCTV fight video library (#cctv hashtag is huge)
- ✅ Short-form = perfect for training
- ✅ Vertical format = matches surveillance cameras
- ✅ 8,000+ unique videos expected

**Twitter**:
- ✅ Real-time fight footage (news, viral)
- ✅ High engagement content surfaces quickly
- ✅ Easy to scrape (API available)
- ✅ 4,000+ unique videos expected

### Combined Impact

```
Current:
- Reddit: 6,000 videos
- YouTube: 5,000 videos
- Subtotal: 11,000 videos

After TikTok + Twitter:
- TikTok: 8,000 videos
- Twitter: 4,000 videos
- Total: 23,000 videos

Final dataset:
- Existing: 4,038 videos
- New: 19,000 videos (after quality check)
- Total: 23,000+ violent videos ✅

Result: 2x original dataset size!
```

---

## Next Steps

### Immediate (Today)
1. ✅ Let Reddit scraper finish
2. ✅ Run YouTube Shorts scraper
3. 🔥 Build TikTok scraper
4. 🔥 Build Twitter scraper

### Short-term (This Week)
5. Run all 4 scrapers in parallel
6. Download videos from all sources
7. Quality check and combine
8. Train model with 20,000+ videos
9. Achieve 92%+ accuracy
10. Deploy to 110 cameras! 🎯

---

## Want Me To Build These?

I can create:
1. **TikTok scraper** - Hashtag-based infinite scroll
2. **Twitter scraper** - Search + video extraction
3. **Combined downloader** - Downloads from all platforms

**Should I start with TikTok (highest priority) or Twitter (easiest)?**
