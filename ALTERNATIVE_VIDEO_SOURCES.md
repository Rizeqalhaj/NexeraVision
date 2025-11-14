# Alternative Video Sources for Fight/Violence Collection

## Platforms That DON'T Require Login (Vast.ai Compatible)

---

## ⭐ Tier 1: High-Yield, No Authentication Required

### 1. **Dailymotion** 🔥 **HIGHEST PRIORITY**

**Why This Is Perfect**:
- ✅ NO login required for search/viewing
- ✅ Less restrictive than YouTube
- ✅ Lots of fight/violence content
- ✅ Easy to scrape with Playwright
- ✅ Similar to YouTube but more permissive

**Expected Yield**: 3,000-5,000 videos

**Search Queries**:
```
cctv fight
security camera fight
street fight
fight caught on camera
surveillance fight
knockout
brawl
real fight
violent fight
```

**Scraping Method**: Playwright (same as Reddit/YouTube)

**URL Format**: `https://www.dailymotion.com/search/{query}/videos`

---

### 2. **Vimeo** 🔥 **HIGH PRIORITY**

**Why This Works**:
- ✅ NO login required for public videos
- ✅ High-quality content
- ✅ Documentary/news footage
- ✅ CCTV compilations available
- ✅ Less crowded = unique content

**Expected Yield**: 1,500-2,500 videos

**Search Queries**:
```
cctv fight
security camera violence
surveillance footage
caught on camera fight
street fight
```

**Scraping Method**: Vimeo API (free tier) or Playwright

**URL Format**: `https://vimeo.com/search?q={query}`

---

### 3. **Bitchute** 🔥 **GOOD ALTERNATIVE**

**Why This Is Valuable**:
- ✅ NO login required
- ✅ Less restrictive content policy
- ✅ Alternative to YouTube
- ✅ Uncensored fight content
- ✅ Easy scraping

**Expected Yield**: 2,000-3,000 videos

**Search Queries**:
```
cctv fight
security camera
street fight
fight video
real fight
```

**Scraping Method**: Playwright

**URL Format**: `https://www.bitchute.com/search/?query={query}`

---

### 4. **Rumble** 🔥 **GOOD ALTERNATIVE**

**Why This Works**:
- ✅ NO login required
- ✅ YouTube alternative
- ✅ Less restrictive
- ✅ Trending fight content
- ✅ API available

**Expected Yield**: 1,500-2,500 videos

**Scraping Method**: Rumble API or Playwright

---

### 5. **Odysee/LBRY** 🔥 **DECENTRALIZED OPTION**

**Why This Is Good**:
- ✅ NO login required
- ✅ Decentralized platform
- ✅ Less censorship
- ✅ API available
- ✅ Growing fight content library

**Expected Yield**: 1,000-2,000 videos

**Scraping Method**: LBRY API or Playwright

---

## ⭐ Tier 2: Archive/Dataset Sites

### 6. **Internet Archive** 🔥 **MASSIVE RESOURCE**

**Why This Is Excellent**:
- ✅ NO login required
- ✅ Massive video archive
- ✅ News footage, documentaries
- ✅ Historical CCTV footage
- ✅ Public domain content
- ✅ API available

**Expected Yield**: 2,000-4,000 videos

**Search Queries**:
```
cctv fight
security camera violence
surveillance footage
street fight
fight video
violence caught on camera
```

**Scraping Method**: Internet Archive API (free)

**URL Format**: `https://archive.org/search.php?query={query}&and[]=mediatype:movies`

---

### 7. **Kaggle Datasets** 🔥 **READY-TO-USE**

**Why This Is Perfect**:
- ✅ Pre-collected datasets
- ✅ NO scraping needed
- ✅ High quality
- ✅ Academic validation
- ✅ Easy download with API

**Available Datasets**:
```
- Violence Detection Dataset
- UCF Crime Dataset (1,900 videos)
- RWF-2000 (2,000 fight videos)
- Hockey Fight Dataset (1,000 videos)
- Street Fight Dataset
- CCTV Violence Dataset
```

**Expected Yield**: 5,000-10,000 videos (combined)

**Download Method**: Kaggle CLI
```bash
kaggle datasets download -d <dataset-id>
```

---

### 8. **Academic Datasets (Direct Download)** 🔥 **HIGH QUALITY**

**Why This Is Valuable**:
- ✅ Pre-labeled data
- ✅ High quality
- ✅ Research-validated
- ✅ Direct download links
- ✅ NO scraping needed

**Available Datasets**:

#### UCF Crime Dataset
- 1,900+ violent videos
- 14 crime categories including fights
- Direct download: http://www.crcv.ucf.edu/projects/real-world/

#### RWF-2000 (Real World Fight)
- 2,000 videos (1,000 fight, 1,000 non-fight)
- Surveillance camera quality
- GitHub: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

#### Violent Flows Dataset
- 246 violent videos
- Action recognition focus
- Available on request from researchers

#### Movie Fight Dataset
- 200+ movie fight scenes
- High quality choreographed fights
- Useful for augmentation

**Expected Yield**: 4,000-6,000 videos (combined)

---

## ⭐ Tier 3: News/Media Sites

### 9. **LiveLeak Alternatives (Kaotic, WatchPeopleDie Archives)**

**Why This Works**:
- ✅ Extreme violence footage
- ✅ Real-world scenarios
- ✅ CCTV content
- ✅ NO login required

**Expected Yield**: 1,000-2,000 videos

**⚠️ WARNING**: Extremely graphic content, use ethically

---

### 10. **News Site Video Archives**

**CNN, BBC, Reuters, AP Video Archives**:
- ✅ Public video archives
- ✅ News footage of violence/riots
- ✅ Professional quality
- ✅ API access available

**Expected Yield**: 500-1,000 videos

---

## RECOMMENDED BUILD ORDER

### Phase 1: Quick Wins (This Week)

**Build These 3 Scrapers** (No authentication needed):

```
1. Dailymotion scraper → 4,000 videos
2. Vimeo scraper → 2,000 videos
3. Bitchute scraper → 2,500 videos

Total: 8,500 additional videos
```

**Combined with Reddit + YouTube**:
```
Reddit:      5,400 videos ✅
YouTube:     4,500 videos ✅
Dailymotion: 4,000 videos 🔥 NEW
Vimeo:       2,000 videos 🔥 NEW
Bitchute:    2,500 videos 🔥 NEW

Total: 18,400 videos ✅ (168% of original!)
Final dataset: 22,438 videos
```

### Phase 2: Datasets (Instant Download)

**Download These 3 Datasets** (No scraping needed):

```
1. RWF-2000 → 1,000 fight videos
2. UCF Crime → 500 violence videos
3. Kaggle datasets → 1,000 videos

Total: 2,500 additional videos
```

**Grand Total**: 24,938 videos (227% of original!)

---

## IMMEDIATE ACTION PLAN

### Option A: Build Dailymotion + Vimeo + Bitchute Scrapers

**Advantages**:
- ✅ NO authentication required
- ✅ Will work on Vast.ai (no IP blocks)
- ✅ 8,500 additional videos
- ✅ Similar to YouTube scraping (proven working)

**Timeline**:
- Build scrapers: 1 hour
- Run scrapers: 4-6 hours
- Download: 8-10 hours
- **Total: ~18 hours to collect 8,500 videos**

### Option B: Download Academic Datasets

**Advantages**:
- ✅ Instant access (no scraping)
- ✅ Pre-labeled data
- ✅ High quality
- ✅ Research-validated

**Timeline**:
- Find datasets: 30 minutes
- Download: 2-4 hours
- **Total: ~4 hours to get 2,500 videos**

### Option C: Both (RECOMMENDED)

**Do datasets first (fast), then build scrapers**:

```
Hour 0-4:   Download datasets (2,500 videos)
Hour 4-5:   Build 3 scrapers
Hour 5-11:  Run 3 scrapers (8,500 video URLs)
Hour 11-21: Download all (10 hours)
Hour 21:    Combine everything

Total: 11,000 new videos in ~21 hours
Final: 15,938 existing + 11,000 = 26,938 videos ✅
```

---

## MY STRONG RECOMMENDATION

### Build These 3 Now:

1. **Dailymotion scraper** (HIGHEST priority - 4,000 videos)
2. **Vimeo scraper** (2,000 videos)
3. **Academic datasets download** (2,500 videos)

**Skip**:
- Bitchute (if time constrained)
- News sites (too niche)
- LiveLeak alternatives (ethical concerns)

**This gives you**:
```
Reddit:             5,400 videos ✅
YouTube:            4,500 videos ✅
Dailymotion:        4,000 videos 🔥 NEW
Vimeo:              2,000 videos 🔥 NEW
Academic datasets:  2,500 videos 🔥 NEW

Total: 18,400 videos ✅
Final dataset: 22,438 videos (204% of original!)
```

---

## Platform Comparison Matrix

| Platform | Auth Required? | Expected Yield | Difficulty | Vast.ai Compatible? |
|----------|----------------|----------------|------------|---------------------|
| Reddit | ❌ No | 5,400 | Easy | ✅ YES |
| YouTube Shorts | ❌ No | 4,500 | Easy | ✅ YES |
| **Dailymotion** | ❌ **No** | **4,000** | **Easy** | ✅ **YES** |
| **Vimeo** | ❌ **No** | **2,000** | **Easy** | ✅ **YES** |
| **Bitchute** | ❌ **No** | **2,500** | **Easy** | ✅ **YES** |
| Internet Archive | ❌ No | 3,000 | Medium | ✅ YES |
| Academic Datasets | ❌ No | 2,500 | Easy | ✅ YES |
| TikTok | ✅ Yes | 7,500 | Hard | ❌ NO |
| Twitter | ✅ Yes | 4,000 | Hard | ❌ NO |

---

## What Should I Build First?

**Option 1**: Dailymotion scraper (4,000 videos, no login, will work on Vast.ai) 🔥 **RECOMMENDED**

**Option 2**: Vimeo scraper (2,000 videos, no login, high quality)

**Option 3**: Academic datasets download script (2,500 videos, instant)

**Option 4**: All three in parallel

**Which one do you want me to build now?**

I recommend starting with **Dailymotion** since it's the highest yield and will definitely work on Vast.ai!
